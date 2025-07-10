import copy
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple


class Context:
    def __init__(self, instance: List[Tuple[float, float]], end_time: datetime) -> None:
        self._instance = instance
        self._end_time = end_time
        self._best_solution: Optional[List[int]] = None

    @property
    def instance(self) -> List[Tuple[float, float]]:
        """
        Returns:
            A list of (x, y) coordinates for each city in the TSP instance.
            City indices correspond to positions in this list.
        """
        return self._instance

    def remaining_time(self) -> timedelta:
        """
        Returns:
            The remaining time available for computation as a timedelta object.
        """
        now = datetime.now(tz=timezone.utc)
        remaining = self._end_time - now
        if remaining < timedelta():
            remaining = timedelta()

        return remaining

    def report_new_best_solution(self, solution: List[int]) -> None:
        """
        Reports a new best TSP tour.

        Args:
            solution: A list of city indices defining the visiting order.

        Notes:
            - Call this method only when you have found a strictly better solution than any
              previously reported.
            - The last reported solution before time expires will be taken as the final answer.
            - Solutions reported after the time runs out will be ignored.
        """
        if self.remaining_time() <= timedelta():
            return

        self._best_solution = copy.copy(solution)
