Implement a grid transformation function in Python by completing the `transform` function below. Your goal is to learn visual patterns from training examples and apply them to test cases.

## Requirements

Define a `transform` function with the following signature:

```python
def transform(input_grid: Grid) -> Grid:
    ...
```

Where `Grid = List[List[int]]` represents a 2D grid of integers.

Define any helper functions at the top level (do not nest functions).

Use only the Python Standard Library; external packages are not allowed.

## Task Structure

Each challenge consists of:
- **Training examples**: Input-output grid pairs that demonstrate the transformation rule
- **Test cases**: Input grids for which you must predict the correct output by applying the learned transformation

## Grid Format

Grids are represented as `List[List[int]]` where:
- Each inner list represents a row
- Each integer represents a color/value (typically 0-9)
- Grid sizes vary but are usually small (1x1 to 30x30)

Example grid:
```python
[
    [0, 1, 0],
    [1, 2, 1],
    [0, 1, 0]
]
```

## Implementation Strategy

1. **Analyze training examples** to identify the transformation pattern
2. **Extract the rule** that converts input to output
3. **Implement the transformation** that can be applied to any input grid
4. **Handle edge cases** and validate your solution

## Evaluation Criteria

Your score is based on exact match accuracy:
- Each test case output must match the expected solution exactly
- Score = (number of correct outputs) / (total number of test cases)
- Perfect score is 1.0 when all test outputs are correct

## Example Transformation

Training example:
```
Input:  [[1, 0, 1],     Output: [[0, 1, 0],
         [0, 1, 0],              [1, 0, 1],
         [1, 0, 1]]              [0, 1, 0]]
```

Pattern: Invert colors (0↔1)

Your `transform` function should learn this pattern and apply it to test inputs.
