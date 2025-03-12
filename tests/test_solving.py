import unittest
from unittest.mock import patch
import argparse
from Rhapso.solver.solver import main

class TestSolving(unittest.TestCase):
    @patch("builtins.print")
    def test_solve_transformation(self, mock_print):
        args = argparse.Namespace(
            method="Affine",
            iterations=50,
            tolerance=0.0005
        )
        main(args)
        mock_print.assert_any_call("Solving Transformations with the following arguments:")
        mock_print.assert_any_call("Method: Affine")
        mock_print.assert_any_call("Iterations: 50")
        mock_print.assert_any_call("Tolerance: 0.0005")

if __name__ == "__main__":
    unittest.main()
