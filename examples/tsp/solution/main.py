import copy
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from solver import Context as _Context, solve


class Context(_Context):
    def __init__(self, instance: List[Tuple[float, float]], end_time: datetime) -> None:
        self._instance = instance
        self._end_time = end_time
        self._best_solution: Optional[List[int]] = None

    @property
    def instance(self) -> List[Tuple[float, float]]:
        return self._instance

    def remaining_time(self) -> timedelta:
        now = datetime.now(tz=timezone.utc)
        remaining = self._end_time - now
        if remaining < timedelta():
            remaining = timedelta()

        return remaining

    def report_new_best_solution(self, solution: List[int]) -> None:
        if self.remaining_time() <= timedelta():
            return

        self._best_solution = copy.copy(solution)


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
    coordinates = []

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
    """Calculate total distance of a TSP tour.

    Args:
        solution: List of city indices representing the tour order
        instance: List of (x, y) coordinate tuples for all cities

    Returns:
        Total distance of the tour
    """
    if not solution or len(solution) < 2:
        return float("inf")

    total_distance = 0.0

    # Create pairs of consecutive cities, including wraparound from last to first
    for i in range(len(solution)):
        city1 = solution[i]
        city2 = solution[(i + 1) % len(solution)]
        total_distance += calculate_euclidean_distance(instance[city1], instance[city2])

    return total_distance


def main():
    instance = parse_tsplib_file(Path(__file__).parent / "a280.tsp")

    now = datetime.now(tz=timezone.utc)
    end_time = now + timedelta(seconds=30)

    context = Context(instance=instance, end_time=end_time)
    solve(context)

    # Calculate and output the tour distance
    if context._best_solution is not None:
        tour_distance = calculate_tour_distance(context._best_solution, instance)
        print(f">>> {tour_distance}")
    else:
        print(f">>> {float('inf')}")


if __name__ == "__main__":
    main()
