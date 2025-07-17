from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI
from .evaluator import Evaluator
from .strategies import Strategy


@dataclass
class Problem:
    description: str
    initial_solution: str
    evaluator: Evaluator


@dataclass
class LLMConfig:
    model: str
    client: OpenAI


@dataclass
class OptimizerConfig:
    llm: LLMConfig
    max_iterations: int
    problem: Problem
    search_strategy: Strategy
    directory: Path
