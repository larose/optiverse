import logging
from datetime import datetime
from pathlib import Path
from typing import cast

from . import codebase as codebase_helpers
from .config import OptimizerConfig
from .evaluator import SCORE, EvaluatorError, ScoreResult
from .generator import GenerationContext
from .graph import ROOT_NODE_ID
from .prompt_generator import DefaultPromptGenerator, PromptGeneratorContext
from .search import Search
from .store import FileSystemStore, Solution


logger = logging.getLogger(__name__)


class Optimizer:
    def __init__(self, config: OptimizerConfig) -> None:
        self._config = config

        self._store = FileSystemStore(directory=config.directory)
        self._prompt_generator = DefaultPromptGenerator()
        self._evaluator = config.problem.evaluator()
        self._generator = config.generator
        self._search = Search(
            directory=config.directory,
            director=config.director,
            playbook=config.playbook,
            store=self._store,
        )

    def _do_iteration(self, iteration: int) -> None:
        search_result = self._search.decide(
            iteration=iteration,
            problem_description=self._config.problem.description,
        )
        plan = search_result.plan

        if plan is None:
            logger.warning(f"Iteration {iteration} has nothing to build on; skipping")
            return

        # Allocate first, so the agent works directly in the solution's final
        # home. There is no scratch directory and nothing to copy back.
        started_at = datetime.now().isoformat(timespec="seconds")
        solution_id = self._store.allocate()
        codebase = self._store.codebase_path(solution_id)

        # The agent opens on the code it is improving rather than copying it in
        # itself, which is one thing fewer to get wrong and makes "you changed
        # nothing" a fact the environment can check.
        codebase_helpers.materialize(
            self._store.codebase_path(plan.parent_solution_id), codebase
        )

        prompt = self._prompt_generator.generate(
            PromptGeneratorContext(
                constraints=self._search.constraints(plan.node_id),
                memory=plan.memory,
                problem=self._config.problem,
            )
        )
        self._search.write_generator_prompt(iteration, prompt)

        generation_result = self._generator.generate(
            GenerationContext(
                codebase=codebase,
                log_path=self._search.generator_log_path(iteration),
                prompt=prompt,
                remember=lambda text: self._search.remember(
                    iteration, "generator", text
                ),
                validate=lambda: self._evaluator.validate(codebase),
            )
        )

        score_result = self._score(codebase)

        self._store.commit(
            solution_id,
            iteration=iteration,
            metrics={**score_result.metrics, **generation_result.metrics},
            node_id=plan.node_id,
            parent_solution_id=plan.parent_solution_id,
            score=score_result.score,
            started_at=started_at,
            tags={**search_result.tags, **generation_result.tags},
        )

        if score_result.score is None:
            logger.info(f"Saved unscoreable solution {solution_id} for inspection")
        else:
            logger.info(f"Saved solution {solution_id}, score: {score_result.score}")

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

        # Under the root, with no constraints and no parent solution: it is where
        # the search starts rather than something the search did.
        self._store.commit(
            solution_id,
            iteration=None,
            metrics=score_result.metrics,
            node_id=ROOT_NODE_ID,
            parent_solution_id=None,
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

        Where to resume is the highest iteration among committed solutions.
        `metadata.json` is written atomically and last, and every iteration
        produces exactly one solution, so a committed solution is the record that
        its iteration finished — there is no checkpoint file that could disagree
        with it, and an iteration that died partway through is simply re-run.

        Iterations are numbered from 1, and it is the loop that counts that way
        rather than each place the number is displayed. The same value reaches the
        console, the iteration's directory name and its solution's metadata, so
        there is no `+ 1` left to forget at a new call site.
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
        logger.info(f"Node: {best_solution.node_id}")
        logger.info(f"Codebase: {best_solution.codebase}")
        logger.info(f"Files:\n{codebase_helpers.describe(best_solution.codebase)}")
