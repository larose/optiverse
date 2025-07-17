import logging
import re
from dataclasses import dataclass
from typing import List, Optional
from openai.types.chat import ChatCompletionMessageParam
from .config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class SolutionResponse:
    code: str
    description: Optional[str]


def create_system_prompt() -> str:
    return """You are an expert software developer. Follow the response format exactly as specified - no more, no less.

Your response must include only the following two parts, in this order:

1. A concise, self-contained plain-text description of the solution in bullet points. Use simple bullet points without any additional formatting, headlines, or references to prior solutions.
2. The complete solution code enclosed strictly within triple backticks (```). Do not include any additional text inside or outside the code block, apart from the bullet-point description above.

Example:

- Description bullet point 1
- Description bullet point 2
- Description bullet point 3

```
Solution code
```"""


class LLMClient:
    def __init__(self, llm_config: LLMConfig):
        self._llm_config = llm_config

    def generate(self, prompt: str) -> SolutionResponse:
        system_prompt = create_system_prompt()

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        logger.debug("=" * 60)
        logger.debug("=== LLM INPUT ===")
        logger.debug("=" * 60)
        logger.debug("System prompt:")
        logger.debug(system_prompt)
        logger.debug("User prompt:")
        logger.debug(prompt)
        logger.debug("=" * 60)

        completion_stream = self._llm_config.client.chat.completions.create(
            model=self._llm_config.model,
            messages=messages,
            stream=True,
        )

        response_buffer: List[str] = []

        for chunk in completion_stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                response_buffer.append(content)

        response_content = "".join(response_buffer)

        # Debug log: LLM output
        logger.debug("=" * 60)
        logger.debug("=== LLM OUTPUT ===")
        logger.debug("=" * 60)
        logger.debug("Response received from LLM:")
        logger.debug(response_content)
        logger.debug("=" * 60)

        return self._parse_response(response_content)

    def _parse_response(self, response_content: str) -> SolutionResponse:
        # Parse the response to extract explanation and code
        # Look for the first ``` to separate explanation from code
        code_blocks = re.findall(r"```(.*?)```", response_content, re.DOTALL)

        if not code_blocks:
            logger.info("No code blocks found in LLM response")
            return SolutionResponse(code="", description=None)

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

        return SolutionResponse(code=file_content, description=description)
