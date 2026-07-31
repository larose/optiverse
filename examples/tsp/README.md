# Traveling Salesman Problem

## Problem Summary

The Traveling Salesman Problem (TSP) can be described as follows: given a list of cities and the distances between them, the goal is to find the shortest route that visits each city exactly once and returns to the starting city.

See [problem.md](problem.md) for the complete problem description and requirements.

## Layout

- [optimize.py](optimize.py) — starts an optimization run. `make run.tsp` runs it.
- [initial/](initial/) — the seed codebase: a `solver.py` that returns one random tour. The first solution is a copy of this, and every later one descends from it.
- [harness/](harness/) — everything the example owns for measuring. Never copied into a codebase and never visible to an agent.
  - `evaluate.py` — the evaluator command, and the only thing here anyone else calls. `score` averages three 30-second runs; `validate` does one 3-second run plus the no-nested-functions check that [problem.md](problem.md) requires, which is enough to answer "does this run and produce a legal tour" without spending the full budget.
  - `run.py` — runs one solver against one instance, once.
  - `context.py` — what `solve` is handed.
  - `instance.py` — parses a TSPLIB file, and measures a tour.
  - `a280.tsp` — the scoring instance. It lives here and nowhere else, so a candidate never sees the coordinates while it is being written.

The agent has no way to run any of it. It writes code and calls `validate`, which
answers valid or invalid with diagnostics and nothing more.

## How a candidate is measured

`evaluate.py` builds a workspace per run and deletes it afterwards:

```
workspace/
  run.py  context.py  instance.py  a280.tsp    <- from harness/
  candidate/                                   <- the codebase, verbatim
```

The candidate sits one level down, and `run.py` **appends** it to `sys.path`.
Appended, it loses every name collision: `import datetime` finds the standard
library, `import context` finds the harness's, and only `import solver` resolves
in the candidate. So a codebase can hold arbitrary files without any of them
shadowing the harness, and nothing has to be overwritten or forbidden.

`run.py` writes out the tour and computes nothing. `evaluate.py` checks it is a
legal tour and measures its length itself, in a process the candidate cannot
reach — a candidate chooses which tour it submits, but not what that tour is
worth.

## Heuristic Discovered by Optiverse

After approximately 300 iterations, using Qwen3-235B-A22B as the LLM, Optiverse produced a heuristic based on Iterated Local Search (ILS). It includes:

- Nearest neighbor construction from multiple diverse starting cities
- 2-opt local search with delta evaluation for efficient local improvement
- Perturbation operators to escape local minima, including:
  - Ruin-and-recreate
  - Block reversals and reinsertions
  - Random city swaps
  - Double-bridge moves
- Precomputed distance matrix for fast lookup

## Evaluation

On a 280-city instance (3 runs of 30 seconds each), the best solution achieved an average tour length of 2593, within ~0.5% of the known optimal (2579).

## Full Solution Code

```python
import random
import math
from datetime import timedelta
from typing import List, Tuple
from context import Context

def compute_tour_length(tour: List[int], dist_matrix: List[List[float]]) -> float:
    length = 0.0
    n = len(tour)
    for i in range(n):
        a, b = tour[i], tour[(i+1)%n]
        length += dist_matrix[a][b]
    return length

def nearest_neighbor(instance: List[Tuple[float, float]], start: int, dist_matrix: List[List[float]], nearest_neighbors: List[List[int]]) -> List[int]:
    n = len(instance)
    visited = [False]*n
    tour, current = [start], start
    visited[start] = True
    while len(tour) < n:
        for city in nearest_neighbors[current]:
            if not visited[city]:
                next_city = city
                break
        tour.append(next_city)
        visited[next_city] = True
        current = next_city
    return tour

def two_opt(tour: List[int], dist_matrix: List[List[float]]) -> List[int]:
    tour = tour.copy()
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n-1):
            for j in range(i+2, n):
                a, b = tour[i], tour[i+1]
                c, d = tour[j-1], tour[j]
                if dist_matrix[a][b] + dist_matrix[c][d] > dist_matrix[a][c] + dist_matrix[b][d]:
                    tour[i+1:j] = tour[i+1:j][::-1]
                    improved = True
    return tour

def solve(context: Context) -> None:
    instance = context.instance
    n = len(instance)
    if n <= 1:
        context.report_new_best_solution(list(range(n)))
        return

    dist_matrix = [[math.hypot(x1-x2, y1-y2) for x2, y2 in instance] for x1, y1 in instance]

    nearest_neighbors = []
    for u in range(n):
        sorted_cities = sorted(range(n), key=lambda c: dist_matrix[u][c])
        nearest_neighbors.append(sorted_cities)

    best_solution, best_length = None, float('inf')
    start_points = [0, n//4, n//2, (3*n)//4, n-1]
    if n >= 8:
        start_points += random.sample(range(n), 3)

    for start in start_points:
        if context.remaining_time() <= timedelta(seconds=1):
            break
        tour = nearest_neighbor(instance, start, dist_matrix, nearest_neighbors)
        optimized = two_opt(tour, dist_matrix)
        current_length = compute_tour_length(optimized, dist_matrix)
        if current_length < best_length:
            best_solution, best_length = optimized, current_length
            context.report_new_best_solution(best_solution)

    while context.remaining_time() > timedelta(seconds=0.1) and best_solution:
        new_solution = best_solution.copy()
        move = random.random()
        n_cities = len(best_solution)

        if move < 0.2:  # Ruin and Recreate
            remove_count = max(2, int(n_cities * 0.2))
            removed = random.sample(best_solution, remove_count)
            current_tour = [city for city in best_solution if city not in removed]
            for city in removed:
                best_pos, best_delta = 0, float('inf')
                for i in range(len(current_tour)):
                    a = current_tour[i]
                    b = current_tour[(i+1) % len(current_tour)]
                    delta = dist_matrix[a][city] + dist_matrix[city][b] - dist_matrix[a][b]
                    if delta < best_delta:
                        best_delta, best_pos = delta, i
                current_tour.insert(best_pos + 1, city)
            new_solution = current_tour

        elif move < 0.4:  # Block move
            i = random.randint(0, n_cities-3)
            j = random.randint(i+1, n_cities-1)
            block = new_solution[i+1:j+1]
            if random.random() < 0.5:
                block = block[::-1]
            new_solution = new_solution[:i+1] + new_solution[j+1:]
            k = random.randint(0, len(new_solution)-1)
            new_solution = new_solution[:k+1] + block + new_solution[k+1:]

        elif move < 0.6:  # Swap
            i, j = random.sample(range(n_cities), 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        else:  # Double bridge
            if n_cities < 4:
                i, j = random.sample(range(n_cities), 2)
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            else:
                a, b, c, d = sorted(random.sample(range(n_cities), 4))
                new_solution = (new_solution[:a+1] +
                               new_solution[c+1:d+1] +
                               new_solution[a+1:b+1][::-1] +
                               new_solution[b+1:c+1] +
                               new_solution[d+1:])

        optimized = two_opt(new_solution, dist_matrix)
        new_length = compute_tour_length(optimized, dist_matrix)
        if new_length < best_length:
            best_solution, best_length = optimized, new_length
            context.report_new_best_solution(best_solution)

    if best_solution is None:
        solution = list(range(n))
        random.shuffle(solution)
        context.report_new_best_solution(solution)
    else:
        final_tour = two_opt(best_solution, dist_matrix)
        context.report_new_best_solution(final_tour)
```
