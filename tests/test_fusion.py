import unittest
from unittest.mock import patch
import argparse
from Rhapso.fusion.affine_fusion import main

class TestFusion(unittest.TestCase):
    @patch("builtins.print")
    def test_affine_fusion(self, mock_print):
        args = argparse.Namespace(
            scale=2.0,
            output="./output/fused.tif",
            blend=True
        )
        main(args)
        mock_print.assert_any_call("Affine Fusion Running with the following arguments:")
        mock_print.assert_any_call("Scale: 2.0")
        mock_print.assert_any_call("Output Path: ./output/fused.tif")
        mock_print.assert_any_call("Blend Overlaps: True")

if __name__ == "__main__":
    unittest.main()
