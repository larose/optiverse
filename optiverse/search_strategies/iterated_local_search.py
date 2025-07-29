from abc import ABC, abstractmethod
import random
from typing import Any, Dict, List, Optional, Union, cast

from .models import SearchStrategy, SearchContext, SearchResult, SolutionWithTitle
from ..store import Solution


def get_initial_solution(solutions: List[Solution]) -> Solution:
    for solution in solutions:
        if solution.is_initial:
            return solution

    raise ValueError("No initial solution found.")


def select_best_solutions_from_groups(
    solutions: List[Solution],
) -> List[SolutionWithTitle]:
    valid_solutions = [s for s in solutions if s.score is not None]
    if not valid_solutions:
        return []

    # Group solutions by their group tag
    groups: Dict[Union[str, int], List[Solution]] = {}
    for solution in valid_solutions:
        group_id = solution.tags.get("group")
        if group_id is None:
            continue

        if group_id not in groups:
            groups[group_id] = []

        groups[group_id].append(solution)

    # Pick the best solution from each group (lowest score)
    best_from_each_group: List[Solution] = []
    for group_solutions in groups.values():
        best_solution = min(group_solutions, key=lambda s: cast(float, s.score))
        best_from_each_group.append(best_solution)

    # Randomly pick up to 3 solutions from these best solutions
    num_to_pick = min(3, len(best_from_each_group))
    if num_to_pick == 0:
        return []

    parents = random.sample(best_from_each_group, num_to_pick)

    return [
        SolutionWithTitle(solution=s, title=f"Solution {i + 1}")
        for i, s in enumerate(parents)
    ]


class PerturbationMethod(ABC):
    @abstractmethod
    def perturb(
        self, solutions: List[Solution], tags: Dict[str, Union[int, str]]
    ) -> SearchResult:
        pass


def normalize_perturbation_weights(
    perturbation_methods: Dict[PerturbationMethod, float],
) -> Dict[PerturbationMethod, float]:
    """Normalize weights so they sum to 1.0.

    Args:
        perturbation_methods: Dictionary mapping perturbation methods to weights

    Returns:
        Dictionary with normalized weights
    """
    total_weight = sum(perturbation_methods.values())
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")

    return {
        method: weight / total_weight for method, weight in perturbation_methods.items()
    }


class InitialSolutionPerturbation(PerturbationMethod):
    def perturb(
        self, solutions: List[Solution], tags: Dict[str, Union[int, str]]
    ) -> SearchResult:
        initial_solution = get_initial_solution(solutions)
        result_tags = tags.copy()
        result_tags["move"] = "perturb_restart"
        return SearchResult(
            solutions=[SolutionWithTitle(solution=initial_solution, title="Parent")],
            tags=result_tags,
            task="""Generate a fresh starting solution for this problem, independent of prior solutions.

Construct the solution from scratch based on the problem requirements, constraints, and any relevant domain knowledge.
""",
        )


class BestSolutionPerturbation(PerturbationMethod):
    def perturb(
        self, solutions: List[Solution], tags: Dict[str, Union[int, str]]
    ) -> SearchResult:
        selected_solutions = select_best_solutions_from_groups(solutions)

        if not selected_solutions:
            return InitialSolutionPerturbation().perturb(solutions, tags)

        result_tags = tags.copy()
        result_tags["move"] = "perturb_exploit"
        return SearchResult(
            solutions=selected_solutions,
            tags=result_tags,
            task="""Develop an improved solution by analysing the provided parent solutions to identify their most effective features, methods, or structures.

Where beneficial, combine complementary elements from multiple parents to create a cohesive hybrid solution that leverages their collective strengths.

Ensure that the resulting solution is logically consistent, feasible, and represents a meaningful improvement or integration beyond any individual parent.
""",
        )


class DiverseBestSolutionPerturbation(PerturbationMethod):
    def perturb(
        self, solutions: List[Solution], tags: Dict[str, Union[int, str]]
    ) -> SearchResult:
        selected_solutions = select_best_solutions_from_groups(solutions)

        if not selected_solutions:
            return InitialSolutionPerturbation().perturb(solutions, tags)

        result_tags = tags.copy()
        result_tags["move"] = "perturb_explore"
        return SearchResult(
            solutions=selected_solutions,
            tags=result_tags,
            task="""Identify and implement a novel solution strategy distinct from the approaches represented in the provided parent solutions.

First, review known algorithms, heuristics or data structures relevant to this problem type or related optimization problems.

Then, select or adapt an approach that diversifies the solution set by introducing new perspectives or mechanisms not yet explored in the current search.

If appropriate, draw inspiration from other domains or interdisciplinary methodologies to maximize solution diversity and conceptual coverage.
""",
        )


class IteratedLocalSearch(SearchStrategy):
    def __init__(
        self,
        max_iterations_without_improvements: int,
        perturbation_methods: Optional[Dict[PerturbationMethod, float]] = None,
    ):
        self._group = 0
        self._best_score = float("inf")
        self._num_iterations_without_improvements = 0
        self._max_iterations_without_improvements = max_iterations_without_improvements

        if perturbation_methods is None:
            perturbation_methods = {
                BestSolutionPerturbation(): 0.1,
                DiverseBestSolutionPerturbation(): 0.8,
                InitialSolutionPerturbation(): 0.1,
            }
        else:
            perturbation_methods = perturbation_methods

        self._perturbation_methods = normalize_perturbation_weights(
            perturbation_methods
        )

    def _improve(self, solutions: List[Solution]) -> SearchResult:
        solutions_in_current_group = [
            s
            for s in solutions
            if s.score is not None and s.tags.get("group") == self._group
        ]

        if not solutions_in_current_group:
            return InitialSolutionPerturbation().perturb(
                solutions, {"group": self._group}
            )

        sorted_solutions = sorted(
            solutions_in_current_group, key=lambda x: cast(float, x.score)
        )

        return SearchResult(
            solutions=[SolutionWithTitle(solution=sorted_solutions[0], title="Parent")],
            tags={
                "move": "local_search",
                "group": self._group,
            },
            task="Apply focused local improvements to enhance the current solution quality",
        )

    def _perturb(self, solutions: List[Solution]) -> SearchResult:
        self._best_score = float("inf")
        self._group += 1
        self._num_iterations_without_improvements = 0

        # Select perturbation method using weighted random selection
        methods = list(self._perturbation_methods.keys())
        weights = list(self._perturbation_methods.values())

        selected_method = random.choices(methods, weights=weights)[0]

        # Get the perturbation result
        result = selected_method.perturb(solutions, {"group": self._group})

        return result

    def apply(self, context: SearchContext) -> SearchResult:
        solutions = context.store.get_all_solutions()

        if (
            self._num_iterations_without_improvements
            < self._max_iterations_without_improvements
        ):
            return self._improve(solutions)

        return self._perturb(solutions)

    def result(self, iteration: int, score: Optional[float]) -> None:
        if score is not None and score < self._best_score:
            self._best_score = score
            self._num_iterations_without_improvements = 0
        else:
            self._num_iterations_without_improvements += 1

    def serialize(self) -> Dict[str, Any]:
        return {
            "group": self._group,
            "best_score": self._best_score,
            "num_iterations_without_improvements": self._num_iterations_without_improvements,
        }

    def deserialize(self, state: Dict[str, Any]) -> None:
        self._group = state["group"]
        self._best_score = state["best_score"]
        self._num_iterations_without_improvements = state[
            "num_iterations_without_improvements"
        ]
