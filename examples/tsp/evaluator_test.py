import unittest
import ast
from examples.tsp.evaluator import NestedFunctionDetector


class TestNestedFunctionDetector(unittest.TestCase):
    def test_has_nested_functions(self):
        code = """
def outer_function():
    def inner_function():
        return "nested"
    return inner_function()
"""
        tree = ast.parse(code)
        detector = NestedFunctionDetector()
        detector.visit(tree)
        self.assertTrue(detector.has_nested)

    def test_no_nested_functions(self):
        code = """
class MyClass:
    def method1(self):
        pass

    def method2(self):
        return "hello"

def standalone_function():
    return "not nested"
"""
        tree = ast.parse(code)
        detector = NestedFunctionDetector()
        detector.visit(tree)
        self.assertFalse(detector.has_nested)


if __name__ == "__main__":
    unittest.main()
