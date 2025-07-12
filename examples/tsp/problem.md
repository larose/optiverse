Implement a heuristic solver for the Traveling Salesman Problem (TSP) in Python by completing the `solve` function below. Your goal is to find the shortest possible tour within the given time limit.

```
def solve(context: Context) -> None:
    pass
```

Each city in the problem instance is identified by a unique integer ID, which corresponds to its index (position) in the `context.instance` list. A solution should be represented as a list of these city IDs, specifying the order in which the cities are visited to form a complete tour.

## Constraints

- You may only modify the `solve` function and any helper functions you introduce within the same file.
- You're permitted to import any modules from the Python Standard Library, but no external packages are allowed.
- Do not alter the `Context` class or its interface.
- Solutions reported after the time limit will be ignored.
- All submitted tours must be valid (cover all cities exactly once, form a loop).
- Do not use nested functions unless they are strictly necessary.

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

## Guidelines

- You should periodically check `context.remaining_time()` to ensure there is sufficient time remaining.
- You should call `context.report_new_best_solution(solution)` only when a new better solution is found.

## Evaluation Criteria

The score is the total length of the last valid tour submitted before the time limit.
