from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from ..store import Solution, Store


@dataclass
class SearchContext:
    iteration: int
    store: Store


@dataclass
class SolutionWithTitle:
    solution: Solution
    title: str


@dataclass
class SearchResult:
    solutions: List[SolutionWithTitle]
    tags: Dict[str, Union[int, str]]
    task: str


class SearchStrategy(ABC):
    @abstractmethod
    def apply(self, context: SearchContext) -> SearchResult:
        pass

    @abstractmethod
    def result(self, iteration: int, score: Optional[float]) -> None:
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def deserialize(self, state: Dict[str, Any]) -> None:
        pass
