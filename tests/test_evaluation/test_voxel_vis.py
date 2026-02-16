import unittest
import numpy as np
from Rhapso.accuracy_metrics.matching_voxel_vis import VoxelVis

class TestVoxelVis(unittest.TestCase):

    def setUp(self):
        self.matches = {
            0: [
                {"p1": np.array([0, 0, 0]), "p2": np.array([10, 10, 10])},
                {"p1": np.array([20, 20, 20]), "p2": np.array([30, 30, 30])}
            ]
        }
        self.voxel_vis = VoxelVis(view_id=0, matches=self.matches)
        self.voxel_vis.get_matches()

    def test_get_matches(self):
        expected_points = [
            np.array([0, 0, 0]), np.array([10, 10, 10]),
            np.array([20, 20, 20]), np.array([30, 30, 30])
        ]
        
        for i, point in enumerate(expected_points):
            np.testing.assert_array_equal(self.voxel_vis.just_points[i], point)

    def test_validate_input_invalid(self):
        self.voxel_vis.just_points = [np.array([1, 2])]
        with self.assertRaises(ValueError):
            self.voxel_vis.validate_input()

    def test_compute_bounding_box(self):
        self.voxel_vis.just_points = np.array(self.voxel_vis.just_points)
        self.voxel_vis.compute_bounding_box()
        np.testing.assert_array_equal(self.voxel_vis.min_coords, [0, 0, 0])
        np.testing.assert_array_equal(self.voxel_vis.max_coords, [30, 30, 30])

    def test_create_voxel_grid(self):
        self.voxel_vis.just_points = np.array(self.voxel_vis.just_points)
        self.voxel_vis.compute_bounding_box()
        self.voxel_vis.create_voxel_grid()
        self.assertEqual(list(self.voxel_vis.voxel_grid.shape), [4, 4, 4])
        self.assertTrue(np.sum(self.voxel_vis.voxel_grid) > 0)

if __name__ == '__main__':
    unittest.main()
