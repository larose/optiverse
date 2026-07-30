import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

from . import codebase as codebase_helpers
from .config import OptimizerConfig
from .evaluator import SCORE, VALIDATE, EvaluatorError, ScoreResult
from .generator import GenerationContext, GenerationResult
from .prompt_generator import DefaultPromptGenerator, PromptGeneratorContext
from .search_strategies import SearchContext, SearchResult
from .store import CODE_DIRECTORY_NAME, FileSystemStore, Solution

logger = logging.getLogger(__name__)

REFERENCE_METADATA_NAME = "metadata.txt"


class Optimizer:
    def __init__(self, config: OptimizerConfig) -> None:
        self._config = config

        self._store = FileSystemStore(directory=config.directory)
        self._prompt_generator = DefaultPromptGenerator()
        self._evaluator = config.problem.evaluator()
        self._generator = config.generator
        self._search_strategy = config.search_strategy
        self._checkpoint_file = Path(config.directory) / "checkpoint.json"

    def _do_iteration(self, iteration: int) -> None:
        strategy_result = self._search_strategy.apply(
            context=SearchContext(iteration=iteration, store=self._store)
        )

        prompt = self._prompt_generator.generate(
            PromptGeneratorContext(
                problem=self._config.problem, strategy_result=strategy_result
            )
        )

        # Allocate first, so the agent works directly in the solution's final
        # home. There is no scratch directory and nothing to copy back.
        solution_id = self._store.allocate()
        codebase = self._store.codebase_path(solution_id)

        generation_result = self._generator.generate(
            GenerationContext(
                codebase=codebase,
                log_path=self._store.agent_log_path(solution_id),
                prompt=prompt,
                references_directory=self._copy_references(
                    strategy_result, solution_id
                ),
                validate=lambda: self._evaluator.validate(codebase),
                validate_shell_command=self._evaluator.shell_command(
                    VALIDATE, codebase
                ),
            )
        )

        score_result = self._score(codebase)

        tags = self._tags(strategy_result, generation_result)

        self._store.commit(
            solution_id,
            is_initial=False,
            metrics={**score_result.metrics, **generation_result.metrics},
            score=score_result.score,
            tags=tags,
        )

        self._search_strategy.result(iteration=iteration, score=score_result.score)

        if score_result.score is None:
            logger.info(f"Saved unscoreable solution {solution_id} for inspection")
        else:
            logger.info(f"Saved solution {solution_id}, score: {score_result.score}")

    def _copy_references(self, strategy_result: SearchResult, solution_id: str) -> Path:
        """Give the agent its own copy of every parent, named by solution id.

        Copies rather than paths into the population: the agent can then read,
        edit or throw them away without any of that reaching a stored solution.
        The codebase it starts from is empty, so what it takes from a parent is
        its decision rather than ours.
        """
        references_directory = self._store.references_path(solution_id)
        references_directory.mkdir(parents=True, exist_ok=True)

        for solution_with_title in strategy_result.solutions:
            solution = solution_with_title.solution
            reference = references_directory / solution.id

            codebase_helpers.materialize(
                solution.codebase, reference / CODE_DIRECTORY_NAME
            )
            (reference / REFERENCE_METADATA_NAME).write_text(_render_metadata(solution))

        return references_directory

    def _tags(
        self, strategy_result: SearchResult, generation_result: GenerationResult
    ) -> Dict[str, Union[int, str]]:
        tags: Dict[str, Union[int, str]] = {
            **strategy_result.tags,
            **generation_result.tags,
        }

        for index, solution_with_title in enumerate(strategy_result.solutions, 1):
            tags[f"parent_id_{index}"] = solution_with_title.solution.id
            tags[f"parent_title_{index}"] = solution_with_title.title

        return tags

    def _score(self, codebase: Path) -> ScoreResult:
        """Score a candidate, treating a broken evaluator as unscoreable.

        A non-zero exit from the evaluator is a different problem from a bad
        candidate, so it is logged loudly rather than silently folded into the
        population as another failure.
        """
        try:
            result = self._evaluator.score(codebase)
        except EvaluatorError:
            logger.error(f"Evaluator failed on {codebase}", exc_info=True)
            return ScoreResult(score=None, metrics={}, log="")

        if result.score is None:
            # The log is not persisted: it is reproducible by re-running the
            # evaluate command against this codebase, which is still on disk.
            logger.info(
                f"Candidate did not score. Reproduce with:\n"
                f"  {self._evaluator.shell_command(SCORE, codebase)}\n"
                f"{result.log.strip()}"
            )

        return result

    def _save_checkpoint(self, iteration: int) -> None:
        checkpoint_data = {
            "iteration": iteration,
            "search_strategy_state": self._search_strategy.serialize(),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "search_strategy_class": self._search_strategy.__class__.__name__,
            },
        }

        with open(self._checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def _save_checkpoint_safely(self, iteration: int) -> None:
        try:
            self._save_checkpoint(iteration)
        except Exception as e:
            logger.warning(
                f"Failed to save checkpoint at iteration {iteration}: {e}",
                exc_info=True,
            )

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        if not self._checkpoint_file.exists():
            return None

        with open(self._checkpoint_file, "r") as f:
            checkpoint_data = cast(Dict[str, Any], json.load(f))

        required_keys = ["iteration", "search_strategy_state", "metadata"]
        if not all(key in checkpoint_data for key in required_keys):
            raise ValueError("Invalid checkpoint format")

        expected_class = self._search_strategy.__class__.__name__
        actual_class = cast(Dict[str, Any], checkpoint_data["metadata"]).get(
            "search_strategy_class"
        )
        if actual_class != expected_class:
            raise ValueError(
                f"Search strategy class mismatch: expected {expected_class}, "
                f"got {actual_class}"
            )

        return checkpoint_data

    def _restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> int:
        self._search_strategy.deserialize(
            cast(Dict[str, Any], checkpoint["search_strategy_state"])
        )
        iteration = cast(int, checkpoint["iteration"])
        logger.info(f"Resuming from checkpoint at iteration {iteration + 1}")
        return iteration

    def _initialize_fresh_optimization(self) -> None:
        logger.info("Evaluating and saving initial solution...")

        solution_id = self._store.allocate()
        codebase = self._store.codebase_path(solution_id)
        codebase_helpers.materialize(self._config.problem.initial_codebase, codebase)

        score_result = self._score(codebase)

        self._store.commit(
            solution_id,
            is_initial=True,
            metrics=score_result.metrics,
            score=score_result.score,
            tags={},
        )

        logger.info(
            f"Initial solution saved with ID: {solution_id}, "
            f"score: {score_result.score}"
        )

    def run(self) -> None:
        checkpoint = self._load_checkpoint()

        if checkpoint is not None:
            start_iteration = self._restore_from_checkpoint(checkpoint)
        else:
            logger.info("Starting fresh optimization...")
            self._initialize_fresh_optimization()
            start_iteration = 0

        for iteration in range(start_iteration, self._config.max_iterations):
            logger.info(
                f"Starting iteration {iteration + 1}/{self._config.max_iterations}"
            )

            try:
                self._do_iteration(iteration=iteration)
            except Exception as e:
                logger.info(
                    f"Iteration {iteration} failed with error: {e}", exc_info=True
                )
                continue

            self._save_checkpoint_safely(iteration)

        self._report_best_solution()

    def _report_best_solution(self) -> None:
        logger.info("\n" + "=" * 50)
        logger.info("BEST SOLUTION:")
        logger.info("=" * 50)

        valid_solutions = [
            s for s in self._store.get_all_solutions() if s.score is not None
        ]

        if not valid_solutions:
            logger.info("No valid solutions found - all solutions failed evaluation")
            return

        best_solution: Solution = min(
            valid_solutions, key=lambda s: cast(float, s.score)
        )

        logger.info(f"ID: {best_solution.id}")
        logger.info(f"Score: {best_solution.score}")
        logger.info(f"Codebase: {best_solution.codebase}")
        logger.info(f"Files:\n{codebase_helpers.describe(best_solution.codebase)}")


def _render_metadata(solution: Solution) -> str:
    """What a parent looks like to the agent, now that the prompt says nothing.

    Score and metrics only. The id is the directory's own name, and the tags
    describe the search's bookkeeping rather than the solution.
    """
    score = "unscored" if solution.score is None else solution.score

    lines = [
        f"Solution: {solution.id}",
        "",
        f"Score: {score}",
        "Lower is better.",
    ]

    if solution.metrics:
        lines.append("")
        lines.append("Metrics:")
        lines.extend(f"  {name}: {value}" for name, value in solution.metrics.items())

    return "\n".join(lines) + "\n"
