import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, cast

from .solution import codebase as codebase_helpers
from .config import OptimizerConfig
from .evaluator import SCORE, EvaluatorError, ScoreResult
from .programmer import ProgrammerContext
from .programmer import prompt as programmer_prompt
from .search import Phase, Search
from .search.graph import ROOT_NODE_ID
from .solution import FileSystemStore, Solution


logger = logging.getLogger(__name__)


class IterationFailed(Exception):
    """The iteration produced no solution. It is set aside and tried again."""


class Optimizer:
    def __init__(self, config: OptimizerConfig) -> None:
        self._config = config

        self._store = FileSystemStore(directory=config.directory)
        self._evaluator = config.problem.evaluator()
        self._programmer = config.programmer
        self._search = Search(
            directory=config.directory,
            director=config.director,
            kicks=config.kicks,
            store=self._store,
        )

    def _do_iteration(self, iteration: int) -> None:
        """One iteration, in the order the design depends on.

        The move is settled before anything is spent: which node is worked, and
        which solution's code the programmer opens on. Only a perturbation costs
        a director call, and only a perturbation can fail here — a local search
        has nothing to decide and so has nothing to get wrong.
        """
        self._search.begin(iteration)

        move = self._search.next_move(iteration)
        node_id = move.base.id

        metrics: Dict[str, Union[int, float]] = {}
        tags: Dict[str, Union[int, str]] = {"phase": move.phase.value}

        if move.phase is Phase.PERTURB:
            perturbation = self._search.perturb(
                iteration, move, self._config.problem.description
            )

            metrics.update(perturbation.metrics)
            tags.update(perturbation.tags)

            if perturbation.node_id is None:
                # Raised before anything is allocated, so an iteration with no
                # constraint leaves no half-made solution directory behind.
                status = tags.get("director_exit_status", "unknown")
                raise IterationFailed(
                    f"the director wrote no usable constraint ({status})"
                )

            node_id = perturbation.node_id

        # Allocate first, so the programmer works directly in the solution's final
        # home. There is no scratch directory and nothing to copy back.
        started_at = datetime.now().isoformat(timespec="seconds")
        solution_id = self._store.allocate()
        codebase = self._store.codebase_path(solution_id)

        # The programmer opens on the code it improves rather than copying it in
        # itself, which is one thing fewer to get wrong and makes "you changed
        # nothing" a fact the environment can check.
        codebase_helpers.materialize(move.parent_solution.codebase, codebase)

        prompt = programmer_prompt.build(
            programmer_prompt.PromptContext(
                constraints=self._search.constraints(node_id),
                problem_description=self._config.problem.description,
            )
        )
        self._search.write_programmer_prompt(iteration, prompt)

        programmer_result = self._programmer.write(
            ProgrammerContext(
                codebase=codebase,
                log_path=self._search.programmer_log_path(iteration),
                prompt=prompt,
                validate=lambda: self._evaluator.validate(codebase),
            )
        )

        score_result = self._score(codebase)

        self._store.commit(
            solution_id,
            iteration=iteration,
            metrics={
                **metrics,
                **score_result.metrics,
                **programmer_result.metrics,
            },
            node_id=node_id,
            parent_solution_id=move.parent_solution.id,
            score=score_result.score,
            started_at=started_at,
            tags={**tags, **programmer_result.tags},
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
        with it.

        The number therefore only advances once a solution has committed. An
        attempt that did not get that far did not happen: it is set aside under a
        name that says so and the same number is tried again, which is already
        what resume assumes. That is deliberately unbounded — a director that is
        broken rather than unlucky will retry forever, in plain sight, rather
        than quietly spending the budget on something else. A retry decides the
        same way as the attempt it replaces, because the policy draws from the
        iteration number.

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

        iteration = completed + 1

        while iteration <= self._config.max_iterations:
            logger.info(f"Starting iteration {iteration}/{self._config.max_iterations}")

            if self._attempt(iteration):
                iteration += 1

        self._report_best_solution()

    def _attempt(self, iteration: int) -> bool:
        """One attempt at an iteration. True once its solution has committed."""
        try:
            self._do_iteration(iteration=iteration)
            return True
        except IterationFailed as error:
            reason = str(error)
        except Exception as error:
            # Not a failure the loop knows about, so the traceback goes out. What
            # happens next is the same either way.
            logger.warning(f"Iteration {iteration} raised", exc_info=True)
            reason = f"{type(error).__name__}: {error}"

        marked = self._search.mark_crashed(iteration)
        location = f" The attempt is at {marked.name}." if marked else ""

        logger.warning(f"Iteration {iteration} crashed: {reason}.{location} Retrying.")

        return False

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
