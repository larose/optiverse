import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast

from .search_strategies import SearchContext
from .prompt_generator import DefaultPromptGenerator, PromptGeneratorContext
from .store import FileSystemStore
from .config import OptimizerConfig
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class Optimizer:
    def __init__(self, config: OptimizerConfig):
        self._config = config

        self._store = FileSystemStore(directory=config.directory)
        self._prompt_generator = DefaultPromptGenerator()
        self._llm_client = LLMClient(llm_config=config.llm)

        self._evaluator = config.problem.evaluator
        self._search_strategy = config.search_strategy
        self._checkpoint_file = Path(config.directory) / "checkpoint.json"

    def _do_iteration(self, iteration: int):
        strategy_context = SearchContext(
            iteration=iteration,
            store=self._store,
        )

        strategy_result = self._search_strategy.apply(context=strategy_context)

        prompt_generator_context = PromptGeneratorContext(
            strategy_result=strategy_result, problem=self._config.problem
        )

        prompt = self._prompt_generator.generate(prompt_generator_context)

        # Generate solution using the LLM client
        solution_response = self._llm_client.generate(prompt)

        if not solution_response.code:
            logger.info("No code generated by solution generator")
            return

        file_content = solution_response.code
        description = solution_response.description

        logger.info("Evaluating solution")

        # Call evaluator on the temporary directory
        result = self._evaluator.evaluate(file_content)

        # Output the result of the evaluation
        if result.score is None:
            logger.info(
                "Evaluation failed - solution did not compile or produce valid results"
            )
        else:
            logger.info(f"Evaluation result: {result.score}")

        # Add parent solution information to tags
        enhanced_tags = strategy_result.tags.copy()
        for i, solution_with_title in enumerate(strategy_result.solutions, 1):
            enhanced_tags[f"parent_id_{i}"] = solution_with_title.solution.id
            enhanced_tags[f"parent_title_{i}"] = solution_with_title.title

        solution_id = self._store.add_solution(
            artifacts=result.artifacts,
            code=file_content,
            description=description,
            is_initial=False,
            metrics=result.metrics,
            prompt=prompt,
            score=result.score,
            tags=enhanced_tags,
        )
        self._search_strategy.result(iteration=iteration, score=result.score)

        if result.score is None:
            logger.info(f"Saved failed solution with ID: {solution_id} for debugging")
        else:
            logger.info(f"Saved solution with ID: {solution_id}")

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
            checkpoint_data = json.load(f)

        required_keys = ["iteration", "search_strategy_state", "metadata"]
        if not all(key in checkpoint_data for key in required_keys):
            raise ValueError("Invalid checkpoint format")

        expected_class = self._search_strategy.__class__.__name__
        actual_class = checkpoint_data["metadata"].get("search_strategy_class")
        if actual_class != expected_class:
            raise ValueError(
                f"Search strategy class mismatch: expected {expected_class}, got {actual_class}"
            )

        return checkpoint_data

    def _restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> int:
        self._search_strategy.deserialize(checkpoint["search_strategy_state"])
        iteration = checkpoint["iteration"]
        logger.info(f"Resuming from checkpoint at iteration {iteration + 1}")
        return iteration

    def _initialize_fresh_optimization(self) -> None:
        logger.info("Evaluating and saving initial solution...")

        initial_solution_result = self._evaluator.evaluate(
            self._config.problem.initial_solution
        )

        initial_id = self._store.add_solution(
            artifacts=initial_solution_result.artifacts,
            code=self._config.problem.initial_solution,
            description=None,
            is_initial=True,
            metrics=initial_solution_result.metrics,
            prompt="",
            score=initial_solution_result.score,
            tags={},
        )

        logger.info(
            f"Initial solution saved with ID: {initial_id}, score: {initial_solution_result.score}"
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

        # Show the best solution at the end
        logger.info("\n" + "=" * 50)
        logger.info("BEST SOLUTION:")
        logger.info("=" * 50)

        all_solutions = self._store.get_all_solutions()
        valid_solutions = [s for s in all_solutions if s.score is not None]

        if valid_solutions:
            # Sort by score (best first) and get the top solution
            sorted_solutions = sorted(
                valid_solutions, key=lambda x: cast(float, x.score)
            )
            best_solution = sorted_solutions[0]
            logger.info(f"ID: {best_solution.id}")
            logger.info(f"Score: {best_solution.score}")

            if best_solution.description:
                logger.info("\nExplanation:")
                logger.info("-" * 30)
                logger.info(f"\n{best_solution.description}")
                logger.info("-" * 30)

            logger.info("\nSource code:")
            logger.info("-" * 30)
            logger.info(f"\n{best_solution.code}")
            logger.info("-" * 30)
        else:
            logger.info("No valid solutions found - all solutions failed evaluation")
