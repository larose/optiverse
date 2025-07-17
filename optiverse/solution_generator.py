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


class SolutionGenerator:
    def __init__(self, llm_config: LLMConfig):
        self._llm_config = llm_config

    def generate(self, prompt: str) -> SolutionResponse:
        # Enhance prompt with response format instructions
        enhanced_prompt = (
            prompt
            + """
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
        )

        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": enhanced_prompt}
        ]

        logger.debug("=" * 60)
        logger.debug("=== LLM INPUT ===")
        logger.debug("=" * 60)
        logger.debug("Prompt being sent to LLM:")
        logger.debug(enhanced_prompt)
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
