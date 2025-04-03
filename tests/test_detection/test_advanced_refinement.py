import unittest
import numpy as np
from collections import OrderedDict
from scipy.spatial import KDTree

from Rhapso.detection.advanced_refinement import AdvancedRefinement


class TestAdvancedRefinement(unittest.TestCase):
    def setUp(self):
        interest_points = [
            {
                "view_id": "timepoint=18, setup=4",
                "interval_key": ((0, 92, 0), (347, 168, 25), (348, 77, 26)),
                "interest_points": [(320, 636, 64), (356, 700, 64)],
                "intensities": [30.77, 16.92],
            },
            {
                "view_id": "timepoint=18, setup=4",
                "interval_key": ((0, 0, 2), (347, 260, 23), (348, 261, 22)),
                "interest_points": [
                    (320, 700, 56),
                    (356, 764, 56),
                ],
                "intensities": [128.34, 121.28],
            },
        ]
        self.ar = AdvancedRefinement(interest_points)

    def test_consolidate_interest_points(self):
        self.ar.consolidate_interest_points()
        expected_data = OrderedDict(
            {
                "timepoint=18, setup=4": [
                    ((320, 636, 64), 30.77),
                    ((356, 700, 64), 16.92),
                    ((320, 700, 56), 128.34),
                    ((356, 764, 56), 121.28),
                ],
            }
        )
        self.assertEqual(self.ar.consolidated_data, expected_data)

    def test_kd_tree(self):
        self.ar.consolidate_interest_points()
        self.ar.kd_tree()
        filtered_data = self.ar.consolidated_data
        for view_id, points in filtered_data.items():
            pts = np.array([p[0] for p in points])
            tree = KDTree(pts)
            for i, point in enumerate(pts):
                distances, _ = tree.query(point, k=3)
                self.assertTrue(distances[2] > self.ar.combine_distance)

    def test_filter_points(self):
        interest_points = [(320, 636, 64), (356, 700, 64), (320, 700, 56)]
        intensities = [30.77, 0.9, 0.7]
        max_spots = 2
        filtered_points, filtered_intensities = self.ar.filter_points(
            interest_points, intensities, max_spots
        )
        self.assertEqual(len(filtered_points), max_spots)
        self.assertEqual(len(filtered_intensities), max_spots)

    # def test_max_spots(self):
    #     self.ar.max_spots_per_overlap = True
    #     self.ar._max_spots = 2
    #     self.ar.consolidate_interest_points()
    #     self.ar.max_spots()
    #     for view_id, points in self.ar.interest_points_per_view_id.items():
    #         self.assertLessEqual(len(points), self.ar._max_spots)

    def test_run(self):
        result = self.ar.run()
        print(result)
        self.assertGreaterEqual(len(result), 1)
        self.assertIn(((320, 636, 64), 30.77), result["timepoint=18, setup=4"])
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
