import math
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import List, Tuple
from context import Context
from solver import solve


def parse_coordinate_line(line: str) -> Tuple[float, float]:
    """Parse a coordinate line and extract x, y coordinates.

    Args:
        line: A line like "1 981036 508139"

    Returns:
        Tuple of (x, y) coordinates as floats
    """
    parts = line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Invalid coordinate line: {line}")

    # Extract x and y coordinates (ignoring node_id)
    x = float(parts[1])
    y = float(parts[2])

    return (x, y)


def parse_coordinates_section(lines: List[str]) -> List[Tuple[float, float]]:
    """Parse all coordinate lines from the NODE_COORD_SECTION.

    Args:
        lines: List of lines from the coordinates section

    Returns:
        List of (x, y) coordinate tuples
    """
    coordinates: List[Tuple[float, float]] = []

    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            break

        coord = parse_coordinate_line(line)
        coordinates.append(coord)

    return coordinates


def parse_tsplib_file(filepath: Path) -> List[Tuple[float, float]]:
    """Parse a TSPLIB file and extract coordinates.

    Args:
        filepath: Path to the TSPLIB file

    Returns:
        List of (x, y) coordinate tuples

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Find the NODE_COORD_SECTION
    coord_section_start = None
    for i, line in enumerate(lines):
        if line.strip() == "NODE_COORD_SECTION":
            coord_section_start = i + 1
            break

    if coord_section_start is None:
        raise ValueError("NODE_COORD_SECTION not found in file")

    # Parse coordinates from the section
    coord_lines = lines[coord_section_start:]
    coordinates = parse_coordinates_section(coord_lines)

    if not coordinates:
        raise ValueError("No valid coordinates found in file")

    return coordinates


def calculate_euclidean_distance(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        point1: First point as (x, y) tuple
        point2: Second point as (x, y) tuple

    Returns:
        Euclidean distance between the two points
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_tour_distance(
    solution: List[int], instance: List[Tuple[float, float]]
) -> float:
    # Check that every city appears exactly once
    solution_set = set(solution)
    expected_set = set(range(len(instance)))

    # Cities in solution but not in instance (invalid indices)
    invalid_cities = solution_set - expected_set
    if invalid_cities:
        raise ValueError(f"Solution contains invalid city indices: {invalid_cities}")

    # Cities in instance but not in solution (missing cities)
    missing_cities = expected_set - solution_set
    if missing_cities:
        raise ValueError(f"Solution is missing cities: {missing_cities}")

    total_distance = 0.0

    # Create pairs of consecutive cities, including wraparound from last to first
    for i in range(len(solution)):
        city1 = solution[i]
        city2 = solution[(i + 1) % len(solution)]
        total_distance += calculate_euclidean_distance(instance[city1], instance[city2])

    return total_distance


def main():
    timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "30"))

    instance = parse_tsplib_file(Path(__file__).parent / "a280.tsp")

    now = datetime.now(tz=timezone.utc)
    end_time = now + timedelta(seconds=timeout_seconds)

    context = Context(instance=instance, end_time=end_time)
    solve(context)

    # Calculate and output the tour distance
    if context.best_solution is None:
        raise Exception("No solution found")

    tour_distance = calculate_tour_distance(context.best_solution, instance)
    print(f">>> {tour_distance}")


if __name__ == "__main__":
    main()
