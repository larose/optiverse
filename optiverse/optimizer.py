import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, cast

from . import codebase as codebase_helpers
from .config import OptimizerConfig
from .evaluator import SCORE, EvaluatorError, ScoreResult
from .generator import GenerationContext, GenerationResult
from .prompt_generator import DefaultPromptGenerator, PromptGeneratorContext
from .search import Search, SearchResult
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
        self._search = Search(
            directory=config.directory,
            playbook=config.playbook,
            store=self._store,
            strategist=config.strategist,
        )

    def _do_iteration(self, iteration: int) -> None:
        search_result = self._search.decide(
            iteration=iteration,
            problem_description=self._config.problem.description,
        )

        # Allocate first, so the agent works directly in the solution's final
        # home. There is no scratch directory and nothing to copy back.
        started_at = datetime.now().isoformat(timespec="seconds")
        solution_id = self._store.allocate()
        codebase = self._store.codebase_path(solution_id)

        # Before the prompt, which names the copies.
        references_directory = self._copy_references(search_result, solution_id)

        prompt = self._prompt_generator.generate(
            PromptGeneratorContext(
                problem=self._config.problem,
                references_directory=os.path.relpath(references_directory, codebase),
                search_result=search_result,
            )
        )

        generation_result = self._generator.generate(
            GenerationContext(
                codebase=codebase,
                log_path=self._store.agent_log_path(solution_id),
                prompt=prompt,
                references_directory=references_directory,
                validate=lambda: self._evaluator.validate(codebase),
            )
        )

        score_result = self._score(codebase)

        solution = self._store.commit(
            solution_id,
            is_initial=False,
            metrics={**score_result.metrics, **generation_result.metrics},
            score=score_result.score,
            started_at=started_at,
            tags=self._tags(search_result, generation_result),
        )

        self._search.record(iteration=iteration, solution=solution)

        if score_result.score is None:
            logger.info(f"Saved unscoreable solution {solution_id} for inspection")
        else:
            logger.info(f"Saved solution {solution_id}, score: {score_result.score}")

    def _copy_references(self, search_result: SearchResult, solution_id: str) -> Path:
        """Give the agent its own copy of every parent, named by solution id.

        Copies rather than paths into the population: the agent can then read,
        edit or throw them away without any of that reaching a stored solution.
        The codebase it starts from is empty, so what it takes from a parent is
        its decision rather than ours.
        """
        references_directory = self._store.references_path(solution_id)
        references_directory.mkdir(parents=True, exist_ok=True)

        for solution_with_title in search_result.solutions:
            solution = solution_with_title.solution
            reference = references_directory / solution.id

            codebase_helpers.materialize(
                solution.codebase, reference / CODE_DIRECTORY_NAME
            )
            (reference / REFERENCE_METADATA_NAME).write_text(_render_metadata(solution))

        return references_directory

    def _tags(
        self, search_result: SearchResult, generation_result: GenerationResult
    ) -> Dict[str, Union[int, str]]:
        """What describes this solution, and nothing about how it was chosen.

        Lineage and the branch it was built under are the journal's business:
        they are recorded there per iteration, in full, and duplicating them here
        cost six columns of solutions.csv without answering anything the join
        cannot.
        """
        return {**search_result.tags, **generation_result.tags}

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

    def _initialize_fresh_optimization(self) -> None:
        logger.info("Evaluating and saving initial solution...")

        started_at = datetime.now().isoformat(timespec="seconds")
        solution_id = self._store.allocate()
        codebase = self._store.codebase_path(solution_id)
        codebase_helpers.materialize(self._config.problem.initial_codebase, codebase)

        score_result = self._score(codebase)

        self._store.commit(
            solution_id,
            is_initial=True,
            metrics=score_result.metrics,
            score=score_result.score,
            started_at=started_at,
            tags={},
        )

        logger.info(
            f"Initial solution saved with ID: {solution_id}, "
            f"score: {score_result.score}"
        )

    def run(self) -> None:
        """Run until the iteration budget is spent, resuming if there is a run here.

        Where to resume is the journal's length: it has one line per finished
        iteration, so there is no checkpoint file that could disagree with it, and
        an iteration that died partway through is simply re-run rather than
        skipped or repeated.

        Iterations are numbered from 1, and it is the loop that counts that way
        rather than each place the number is displayed. The same value reaches the
        console, the strategist's log filename and the journal, so there is no
        `+ 1` left to forget at a new call site. Counting journal lines still
        works: a count does not care what the entries are numbered.
        """
        completed = self._search.completed_iterations()

        if completed == 0 and not self._store.get_all_solutions():
            logger.info("Starting fresh optimization...")
            self._initialize_fresh_optimization()
        elif completed > 0:
            logger.info(f"Resuming after {completed} completed iterations")

        for iteration in range(completed + 1, self._config.max_iterations + 1):
            logger.info(f"Starting iteration {iteration}/{self._config.max_iterations}")

            try:
                self._do_iteration(iteration=iteration)
            except Exception as e:
                logger.info(
                    f"Iteration {iteration} failed with error: {e}", exc_info=True
                )
                continue

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
