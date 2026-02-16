import unittest
import numpy as np
from Rhapso.accuracy_metrics.matching_voxelization import Voxelizer
from unittest.mock import patch
import io


class TestVoxelizer(unittest.TestCase):

    def setUp(self):
        self.points = np.array([
            [0, 0, 0],
            [10, 10, 10],
            [20, 20, 20],
            [30, 30, 30],
            [40, 40, 40]
        ])
        self.voxelizer = Voxelizer(self.points, voxel_size=20)

    def test_validate_normal_input_valid(self):
        self.voxelizer.validate_input() 

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            Voxelizer(np.array([])).validate_input()

    def test_empty_inner_input(self):
        with self.assertRaises(ValueError):
            Voxelizer(np.array([[],[],[]])).validate_input()

    def test_single_point(self):
        with self.assertRaises(ValueError):
            Voxelizer(np.array([1, 2, 3])).validate_input()

    def test_all_points_in_one_voxel(self):
        points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        voxelizer = Voxelizer(points, voxel_size=100)
        stats = voxelizer.compute_statistics()
        self.assertEqual(stats["Non-empty voxels"], 1)
        self.assertEqual(stats["Max points in a voxel"], 3)
        self.assertEqual(stats["Min points in a voxel"], 3)
        self.assertEqual(stats["Entropy of voxel distribution"], 0.0)

    def test_points_on_voxel_boundaries(self):
        points = np.array([
            [0, 0, 0],
            [20, 0, 0],
            [0, 20, 0],
            [0, 0, 20],
            [20, 20, 20]
        ])
        voxelizer = Voxelizer(points, voxel_size=20)
        stats = voxelizer.compute_statistics()
        self.assertEqual(stats["Non-empty voxels"], 5)

    def test_negative_coordinates(self):
        points = np.array([
            [-10, -10, -10],
            [-20, -20, -20],
            [0, 0, 0]
        ])
        voxelizer = Voxelizer(points, voxel_size=15)
        stats = voxelizer.compute_statistics()
        self.assertGreaterEqual(stats["Non-empty voxels"], 1)

    def test_compute_bounding_box(self):
        self.voxelizer.compute_bounding_box()
        np.testing.assert_array_equal(self.voxelizer.min_coords, [0, 0, 0])
        np.testing.assert_array_equal(self.voxelizer.max_coords, [40, 40, 40])

    def test_create_voxel_grid(self):
        self.voxelizer.compute_bounding_box()
        self.voxelizer.create_voxel_grid()
        expected_shape = np.array([3, 3, 3])  # (40-0)/20 = 2, +1 = 3
        np.testing.assert_array_equal(self.voxelizer.grid_shape, expected_shape)
        self.assertEqual(self.voxelizer.voxel_grid.shape, tuple(expected_shape))

    def test_count_points_per_voxel(self):
        self.voxelizer.compute_bounding_box()
        self.voxelizer.create_voxel_grid()
        self.voxelizer.count_points_per_voxel()
        total_points = sum(self.voxelizer.voxel_counts.values())
        self.assertEqual(total_points, len(self.points))

    def test_compute_statistics(self):
        stats = self.voxelizer.compute_statistics()
        self.assertIn("Total voxels in grid", stats)
        self.assertIn("Non-empty voxels", stats)
        self.assertGreater(stats["Non-empty voxels"], 0)
        self.assertLessEqual(stats["Non-empty voxels"], stats["Total voxels in grid"])

    def test_compute_statistics_print_output(self):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            self.voxelizer.compute_statistics()
            output = fake_out.getvalue()
            self.assertIn("Total voxels in grid", output)
            self.assertIn("Non-empty voxels", output)
            self.assertIn("Average points per non-empty voxel", output)
            self.assertIn("Standard deviation of voxel counts", output)
            self.assertIn("Coefficient of Variation (CV)", output)
            self.assertIn("Occupancy percentage", output)
            self.assertIn("Entropy of voxel distribution", output)
            self.assertIn("Max points in a voxel", output)
            self.assertIn("Min points in a voxel", output)
            self.assertIn("Skewness of voxel counts", output)
            self.assertIn("Kurtosis of voxel counts", output)

if __name__ == '__main__':
    unittest.main()
