import logging
import re
from .prompt_generator import Context, PromptGenerator
from .evaluator import Evaluator
from .store import Store
from .config import Config

logger = logging.getLogger(__name__)


SOLUTION_FILE_NAME = "main.py"


class Optimizer:
    def __init__(
        self,
        config: Config,
        evaluator: Evaluator,
        prompt_generator: PromptGenerator,
        store: Store,
    ):
        self._config = config
        self._evaluator = evaluator
        self._prompt_generator = prompt_generator
        self._store = store

    def _do_iteration(self, iteration: int):
        context = Context(
            iteration=iteration,
            max_iterations=self._config.max_iterations,
            problem=self._config.problem,
            store=self._store,
        )
        prompt_result = self._prompt_generator.generate(context=context)

        prompt = prompt_result.text

        prompt += """
# Response

Your response must include:

1. A plain-text description in bullet-point style (no formatting, no headline). Assume no prior context.
2. Your solution

Example:

- Description line 1
- Description line 2
...

```
Solution text here
```
"""

        messages = [{"role": "user", "content": prompt}]

        logger.debug("=" * 60)
        logger.debug("=== LLM INPUT ===")
        logger.debug("=" * 60)
        logger.debug("Prompt being sent to LLM:")
        logger.debug(prompt)
        logger.debug("=" * 60)

        response = self._config.llm.client.chat.completions.create(
            model=self._config.llm.model,
            messages=messages,
        )

        response_content = response.choices[0].message.content

        # Debug log: LLM output
        logger.debug("=" * 60)
        logger.debug("=== LLM OUTPUT ===")
        logger.debug("=" * 60)
        logger.debug("Response received from LLM:")
        logger.debug(response_content)
        logger.debug("=" * 60)

        # Parse the response to extract explanation and code
        # Look for the first ``` to separate explanation from code
        code_blocks = re.findall(r"```(.*?)```", response_content, re.DOTALL)

        if not code_blocks:
            logger.info("No code blocks found in LLM response")
            return

        # Get the first code block and remove any text before the first newline
        raw_content = code_blocks[0]
        first_newline = raw_content.find("\n")
        if first_newline != -1:
            file_content = raw_content[first_newline + 1 :].strip()
        else:
            file_content = raw_content.strip()

        # Extract description (everything before the first ```)
        description = None
        first_code_block_start = response_content.find("```")
        if first_code_block_start > 0:
            description_text = response_content[:first_code_block_start].strip()
            if description_text:
                description = description_text

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

        solution_id = self._store.add_solution(
            artifacts=result.artifacts,
            code=file_content,
            description=description,
            group=prompt_result.group,
            score=result.score,
        )

        if result.score is None:
            logger.info(f"Saved failed solution with ID: {solution_id} for debugging")
        else:
            logger.info(f"Saved solution with ID: {solution_id}")

    def run(self) -> None:
        logger.info("Evaluating and saving initial solution...")

        initial_solution_result = self._evaluator.evaluate(
            self._config.problem.initial_solution
        )

        # Save the initial solution
        initial_id = self._store.add_solution(
            artifacts=initial_solution_result.artifacts,
            code=self._config.problem.initial_solution,
            description=None,
            group=0,
            score=initial_solution_result.score,
        )

        logger.info(
            f"Initial solution saved with ID: {initial_id}, score: {initial_solution_result.score}"
        )

        for iteration in range(1, self._config.max_iterations + 1):
            try:
                logger.info(
                    f"Starting iteration {iteration}/{self._config.max_iterations}"
                )
                self._do_iteration(iteration=iteration - 1)
            except Exception as e:
                logger.info(
                    f"Iteration {iteration} failed with error: {e}", exc_info=True
                )
                continue

        # Show the best solution at the end
        logger.info("\n" + "=" * 50)
        logger.info("BEST SOLUTION:")
        logger.info("=" * 50)

        all_solutions = self._store.get_all_solutions()
        valid_solutions = [s for s in all_solutions if s.score is not None]

        if valid_solutions:
            # Sort by score (best first) and get the top solution
            sorted_solutions = sorted(valid_solutions, key=lambda x: x.score)
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
