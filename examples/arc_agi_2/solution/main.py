import os
import sys
from pathlib import Path
from data_loader import ArcData, find_challenge_by_id, load_arc_data
from solver import transform


def main():
    # Get challenge ID from environment
    challenge_id = os.getenv("CHALLENGE_ID")
    if not challenge_id:
        print("Error: CHALLENGE_ID environment variable not set", file=sys.stderr)
        sys.exit(1)

    # Load ARC data
    data_dir = Path(__file__).parent.parent / "data"
    arc_data = load_arc_data(data_dir)

    # Find the specific challenge
    task, solutions = find_challenge_by_id(arc_data, challenge_id)

    # Run solver on test cases
    results = []
    for i, test_input in enumerate(task.test):
        try:
            output = transform(test_input)
            results.append(output)
        except Exception as e:
            print(f"Error solving test case {i}: {e}", file=sys.stderr)
            results.append(None)

    # Calculate accuracy if solutions are available
    if solutions:
        correct = 0
        for result, expected in zip(results, solutions):
            if result is not None and result == expected:
                correct += 1
        accuracy = correct / len(solutions)
        print(f">>>ACCURACY: {accuracy}")
    else:
        # For test challenges without solutions, report completion rate
        completed = sum(1 for r in results if r is not None)
        completion_rate = completed / len(results)
        print(f">>>COMPLETION: {completion_rate}")

    # Output individual results for debugging
    print("Results:")
    for i, result in enumerate(results):
        if result is not None:
            print(f"Test {i}: {len(result)}x{len(result[0]) if result else 0} grid")
        else:
            print(f"Test {i}: Failed")


if __name__ == "__main__":
    main()
