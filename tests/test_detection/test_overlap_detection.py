import unittest
<<<<<<< HEAD
import numpy as np

import pandas as pd
import bioio_tifffile
from Rhapso.detection.overlap_detection import OverlapDetection


class TestOverlapDetection(unittest.TestCase):
    def setUp(self):
        self.transform_models = {}
        self.dataframes = {"image_loader": pd.DataFrame()}
        self.dsxy = 2
        self.dsz = 2
        self.prefix = "test_prefix"
        self.file_type = "tiff"
        self.od = OverlapDetection(
            self.transform_models,
            self.dataframes,
            self.dsxy,
            self.dsz,
            self.prefix,
            self.file_type,
        )
        self.image_shape_cache = {}

    def test_load_image_metadata_invalid_file_path(self):
        with self.assertRaises(ValueError) as context:
            self.od.load_image_metadata(None)
        self.assertEqual(
            str(context.exception), "The file path does not exist or is not valid."
        )

    def test_load_image_metadata_unsupported_file_type(self):
        self.od.file_type = "unsupported"
        with self.assertRaises(ValueError) as context:
            self.od.load_image_metadata("some_path")
        self.assertEqual(
            str(context.exception),
            "This file type is not tiff or zarr and is not supported.",
        )

    def test_get_inverse_mipmap_transform_non_invertible_matrix(self):
        non_invertible_matrix = np.array([[1, 2], [2, 4]])
        result = self.od.get_inverse_mipmap_transform(non_invertible_matrix)
        self.assertIsNone(result)

    def test_estimate_bounds_invalid_interval_length(self):
        with self.assertRaises(AssertionError) as context:
            self.od.estimate_bounds(np.eye(4), [1, 2, 3])
        self.assertEqual(str(context.exception), "Interval dimensions do not match.")

    def test_find_overlapping_area_empty_dataframe(self):
        with self.assertRaises(ValueError) as context:
            self.od.find_overlapping_area()
        self.assertEqual(str(context.exception), "Image Loader dataframe is empty.")
=======


class TestOverlapDetecttion(unittest.TestCase):

    def setUp(self):
        pass
>>>>>>> main


if __name__ == "__main__":
    unittest.main()
