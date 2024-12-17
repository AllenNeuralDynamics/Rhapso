import unittest
from unittest.mock import patch
import argparse
from Rhapso.detection.interest_points import main

class TestDetection(unittest.TestCase):
    @patch("builtins.print")
    def test_detect_interest_points(self, mock_print):
        args = argparse.Namespace(
            medianFilter=10,
            sigma=1.8,
            threshold=0.05
        )
        main(args)
        mock_print.assert_any_call("Interest Point Detection Running with the following arguments:")
        mock_print.assert_any_call("Median Filter Radius: 10")
        mock_print.assert_any_call("Sigma: 1.8")
        mock_print.assert_any_call("Threshold: 0.05")

if __name__ == "__main__":
    unittest.main()
