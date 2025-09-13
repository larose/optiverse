from data_loader import Grid


def transform(input_grid: Grid) -> Grid:
    """
    Baseline grid transformer that simply returns the input grid unchanged.

    Args:
        input_grid: The input grid as a list of lists of integers

    Returns:
        The same input grid (baseline solution)
    """
    return input_grid
