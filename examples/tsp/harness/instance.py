"""TSPLIB instances, and what a tour costs.

Shared by `run.py`, which parses an instance to build a `Context`, and by
`evaluate.py`, which parses the same file in its own process to measure the tour
that came back. Measuring outside the candidate's process is the point: a
candidate can choose which tour it submits, but not what that tour is worth.
"""

import math
from pathlib import Path
from typing import List, Tuple

Coordinates = Tuple[float, float]

COORDINATE_SECTION = "NODE_COORD_SECTION"


def parse(path: Path) -> List[Coordinates]:
    """The coordinates in a TSPLIB file, in file order.

    A city's id is its position in the list. Only `NODE_COORD_SECTION` is read;
    everything above it is header we do not need, and `EOF` ends it.
    """
    lines = path.read_text().splitlines()

    try:
        start = lines.index(COORDINATE_SECTION) + 1
    except ValueError:
        raise ValueError(f"{path} has no {COORDINATE_SECTION}") from None

    coordinates: List[Coordinates] = []

    for line in lines[start:]:
        parts = line.split()

        if not parts or parts[0] == "EOF":
            break

        if len(parts) < 3:
            raise ValueError(f"{path}: cannot read a coordinate from {line!r}")

        coordinates.append((float(parts[1]), float(parts[2])))

    if not coordinates:
        raise ValueError(f"{path} has no coordinates")

    return coordinates


def distance(first: Coordinates, second: Coordinates) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def tour_length(tour: List[int], coordinates: List[Coordinates]) -> float:
    """The length of a closed tour. The caller checks that it is a legal one."""
    return sum(
        distance(coordinates[tour[index]], coordinates[tour[(index + 1) % len(tour)]])
        for index in range(len(tour))
    )


def illegal_reason(tour: List[int], city_count: int) -> str:
    """Why `tour` is not a tour of `city_count` cities, or "" if it is one."""
    if len(tour) != city_count:
        return f"the tour visits {len(tour)} cities, not {city_count}"

    missing = set(range(city_count)) - set(tour)

    if missing:
        return f"the tour misses {len(missing)} cities, including {min(missing)}"

    return ""
