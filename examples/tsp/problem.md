Implement a heuristic solver for the Traveling Salesman Problem (TSP) in Python by completing the `solve` function below. Your goal is to find the shortest possible tour within the given time limit.

## Requirements

Define a `solve` function with the following signature:

```
def solve(context: Context) -> None:
    ...
```

Define any helper functions at the top level (do not nest functions).

Do not modify the `Context` class or its interface.

Submit only valid tours that visit each city exactly once and form a closed loop.

Ensure that solutions are reported before the time limit expires. Solutions reported after the time limit will be ignored.

Use only the Python Standard Library; external packages are not allowed.

Call `context.report_new_best_solution(solution)` only when a better solution is found to minimise overhead.

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

Your score is the total length of the last valid tour submitted before the time limit.
