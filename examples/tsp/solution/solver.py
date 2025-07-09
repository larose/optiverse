from abc import ABC, abstractmethod
from datetime import timedelta
import random
from typing import List, Tuple


class Context(ABC):
    @property
    @abstractmethod
    def instance(self) -> List[Tuple[float, float]]:
        """
        TSP instance representing city locations.

        Returns:
            List of (x, y) coordinate tuples representing the cities to visit.
            Each tuple contains the x and y coordinates of a city in the TSP instance.
            The indices of this list correspond to city identifiers used in solutions.
        """
        pass

    @abstractmethod
    def remaining_time(self) -> timedelta:
        """Returns remaining time"""
        pass

    @abstractmethod
    def report_new_best_solution(self, solution: List[int]) -> None:
        """Report a new best solution as a list of city indices.

        This method should be called when a better solution is found.
        The most recently reported solution will be used as the final answer
        when the time runs out.

        Args:
            solution: List of city indices representing the tour order
        """
        pass


def solve(context: Context) -> None:
    num_cities = len(context.instance)

    while context.remaining_time() > timedelta():
        # Generate a random solution (permutation of city indices)
        solution = random.sample(range(num_cities), num_cities)
        context.report_new_best_solution(solution)
        break  # since it's pointless to continue in this example
