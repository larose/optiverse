# ARC-AGI 2 Example

This example demonstrates visual pattern recognition and grid transformation optimization using optiverse.

## Overview

The challenge involves learning transformation patterns from training examples and applying them to test cases. Each task consists of input-output grid pairs that demonstrate a visual transformation rule.

## Setup

1. **Download ARC-AGI dataset** and place JSON files in the `data/` directory:
   - `arc-agi_evaluation_challenges.json`
   - `arc-agi_evaluation_solutions.json`
   - `arc-agi_test_challenges.json`
   - `arc-agi_training_challenges.json`
   - `arc-agi_training_solutions.json`

2. **Set environment variables**:
   ```bash
   export CHALLENGE_ID="007bbfb7"  # Specify which challenge to optimize
   ```

## Usage

Run the optimization:

```bash
cd examples/arc-agi-2
python main.py
```

## How it Works

1. **Data Loading**: Loads the specified challenge from the ARC dataset
2. **Evaluation**: Tests solutions by running the `transform` function on test inputs
3. **Scoring**: Measures accuracy by exact match comparison with expected outputs
4. **Optimization**: Uses optiverse to iteratively improve the transformation function

## Grid Format

Grids are represented as `List[List[int]]`:
- Each row is a list of integers
- Values typically range 0-9 representing colors
- Sizes vary from 1x1 to 30x30

## Challenge Types

The dataset includes three types of challenges:
- **Training**: Has both challenges and solutions (for learning patterns)
- **Evaluation**: Has both challenges and solutions (for validation)
- **Test**: Has challenges only (solutions withheld)

## Example Challenge IDs

Try these challenge IDs to get started:
- `007bbfb7` - Simple color inversion
- `025d127b` - Geometric transformation
- `045e512c` - Shape completion

## Implementation Tips

Your `transform` function should:
1. Analyze the training examples to identify the pattern
2. Implement the transformation logic
3. Handle edge cases and different grid sizes
4. Return the exact expected output format

## Files Structure

- `main.py` - Optiverse integration and entry point
- `evaluator.py` - Grid transformation evaluation logic
- `problem.md` - LLM instructions for implementing transforms
- `solution/` - Template files for evaluation
  - `solver.py` - Baseline transform function
  - `main.py` - Test runner for individual challenges
  - `data_loader.py` - ARC dataset loading utilities
- `data/` - ARC-AGI dataset JSON files (not included)
