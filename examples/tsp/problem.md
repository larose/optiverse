Implement a heuristic solver for the Traveling Salesman Problem (TSP) in Python by completing the `solve` function below. Your goal is to find the shortest possible tour within the given time limit.

## Requirements

You MUST define a `solve` function with the signature shown below. Any helper functions MUST also be defined at the top level (not nested).

```
def solve(context: Context) -> None:
    ...
```

You MUST NOT modify the `Context` class or its interface.

You MUST submit only valid tours covering all cities exactly once, forming a loop.

You MUST ensure solutions are reported before the time limit expires. Solutions reported after the time limit will be ignored.

You MUST use only the Python Standard Library; external packages are not allowed.

You SHOULD call `context.report_new_best_solution(solution)` only when a better solution is found to minimise overhead.

## `Context` Interface

```python
class Context:
    @property
    def instance(self) -> List[Tuple[float, float]]:
        """
        Returns a list of (x, y) coordinates for each city.
        The index of a city in this list is its unique integer ID.
        """
        ...

    def remaining_time(self) -> timedelta:
        """
        Returns the time remaining for computation.
        """
        ...

    def report_new_best_solution(self, solution: List[int]) -> None:
        """
        Reports a new best solution found. The 'solution' is a list of city IDs
        representing the order of cities in the tour. The first and last city
        in the tour are implicitly connected to form a loop.
        """
        ...
```

## Evaluation Criteria

The score is the total length of the last valid tour submitted before the time limit.
