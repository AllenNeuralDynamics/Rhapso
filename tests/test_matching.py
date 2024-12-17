import unittest
from unittest.mock import patch
import argparse
from Rhapso.matching.feature_matching import main

class TestMatching(unittest.TestCase):
    @patch("builtins.print")
    def test_match_features(self, mock_print):
        args = argparse.Namespace(
            method="ORB",
            distance=0.7,
            verbose=True
        )
        main(args)
        mock_print.assert_any_call("Feature Matching Running with the following arguments:")
        mock_print.assert_any_call("Method: ORB")
        mock_print.assert_any_call("Distance Threshold: 0.7")
        mock_print.assert_any_call("Verbose: True")

if __name__ == "__main__":
    unittest.main()
