import json
from dataclasses import dataclass
from typing import Dict, List, TypeAlias
from pathlib import Path

# Type alias for grid representation
Grid: TypeAlias = List[List[int]]


@dataclass
class Pair:
    input: Grid
    output: Grid


@dataclass
class Task:
    train: List[Pair]
    test: List[Grid]


@dataclass
class ArcData:
    evaluation_challenges: Dict[str, Task]
    evaluation_solutions: Dict[str, List[Grid]]

    test_challenges: Dict[str, Task]

    training_challenges: Dict[str, Task]
    training_solutions: Dict[str, List[Grid]]


@dataclass
class Solution:
    attempt1: Grid
    attempt2: Grid


Solutions: TypeAlias = Dict[str, List[Solution]]


def load_challenge_data(file_path: Path) -> Dict[str, Task]:
    with file_path.open("r") as f:
        challenges: Dict[str, Dict[str, List[Dict[str, List[List[int]]]]]] = json.load(f)

    data = {}
    for challenge_id, challenge_raw_data in challenges.items():
        train = [
            Pair(
                input=ex["input"],
                output=ex["output"],
            )
            for ex in challenge_raw_data["train"]
        ]

        # Convert test inputs to Grid format
        test = [ex["input"] for ex in challenge_raw_data["test"]]

        data[challenge_id] = Task(train=train, test=test)

    return data


def load_solutions_data(file_path: Path) -> Dict[str, List[Grid]]:
    with file_path.open("r") as f:
        solutions = json.load(f)

    data = {}
    for challenge_id, raw_solutions in solutions.items():
        data[challenge_id] = raw_solutions

    return data


def load_arc_data(directory: Path) -> ArcData:
    return ArcData(
        evaluation_challenges=load_challenge_data(
            directory / "arc-agi_evaluation_challenges.json"
        ),
        evaluation_solutions=load_solutions_data(
            directory / "arc-agi_evaluation_solutions.json"
        ),
        test_challenges=load_challenge_data(directory / "arc-agi_test_challenges.json"),
        training_challenges=load_challenge_data(
            directory / "arc-agi_training_challenges.json"
        ),
        training_solutions=load_solutions_data(
            directory / "arc-agi_training_solutions.json"
        ),
    )


def find_challenge_by_id(arc_data: ArcData, challenge_id: str) -> tuple[Task, List[Grid] | None]:
    """
    Find a challenge by ID across all datasets.

    Returns:
        tuple: (task, solutions) where solutions is None if not available
    """
    # Check training challenges
    if challenge_id in arc_data.training_challenges:
        task = arc_data.training_challenges[challenge_id]
        solutions = arc_data.training_solutions.get(challenge_id)
        return task, solutions

    # Check evaluation challenges
    if challenge_id in arc_data.evaluation_challenges:
        task = arc_data.evaluation_challenges[challenge_id]
        solutions = arc_data.evaluation_solutions.get(challenge_id)
        return task, solutions

    # Check test challenges
    if challenge_id in arc_data.test_challenges:
        task = arc_data.test_challenges[challenge_id]
        # Test challenges don't have solutions
        return task, None

    raise ValueError(f"Challenge ID '{challenge_id}' not found in any dataset")
