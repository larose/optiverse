from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict


@dataclass
class EvaluatorResult:
    artifacts: Dict[str, str]  # name, file content
    score: float


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, code: str) -> EvaluatorResult:
        pass
