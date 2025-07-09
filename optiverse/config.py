from dataclasses import dataclass
from openai import OpenAI


@dataclass
class Problem:
    description: str
    initial_solution: str


@dataclass
class LLM:
    model: str
    client: OpenAI


@dataclass
class Config:
    llm: LLM
    max_iterations: int
    problem: Problem
