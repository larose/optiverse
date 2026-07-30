from datetime import timedelta
import random
from context import Context


def solve(context: Context) -> None:
    num_cities = len(context.instance)

    while context.remaining_time() > timedelta():
        # Generate a random solution (permutation of city indices)
        solution = random.sample(range(num_cities), num_cities)
        context.report_new_best_solution(solution)
        break  # since it's pointless to continue in this example
