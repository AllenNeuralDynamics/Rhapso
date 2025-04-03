import unittest
import pandas as pd

from tests.test_data.test_view_transform_model_data import test_df
from Rhapso.detection.view_transform_models import ViewTransformModels


class TestViewTransformModels(unittest.TestCase):
    def setUp(self):
        self.model = ViewTransformModels(test_df)

    def test_create_transform_matrices(self):
        self.model.create_transform_matrices()
        self.assertIn("timepoint: 18, setup: 0", self.model.calibration_matrices)
        self.assertIn("timepoint: 18, setup: 2", self.model.calibration_matrices)
        self.assertIn("timepoint: 18, setup: 0", self.model.rotation_matrices)
        self.assertIn("timepoint: 18, setup: 4", self.model.rotation_matrices)

    def test_concatenate_matrices_by_view_id(self):
        self.model.create_transform_matrices()
        self.model.concatenate_matrices_by_view_id()
        self.assertIn("timepoint: 18, setup: 0", self.model.concatenated_matrices)
        self.assertIn("timepoint: 18, setup: 1", self.model.concatenated_matrices)

    def test_run(self):
        concatenated_matrices = self.model.run()
        self.assertIn("timepoint: 18, setup: 0", concatenated_matrices)
        self.assertIn("timepoint: 18, setup: 4", concatenated_matrices)

    def test_empty_view_registrations_df(self):
        empty_df = {"view_registrations": pd.DataFrame([])}
        model = ViewTransformModels(empty_df)
        with self.assertRaises(ValueError):
            model.create_transform_matrices()

    def test_no_matrices_to_concatenate(self):
        empty_df = {"view_registrations": []}
        model = ViewTransformModels(empty_df)
        with self.assertRaises(ValueError):
            model.concatenate_matrices_by_view_id()


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
