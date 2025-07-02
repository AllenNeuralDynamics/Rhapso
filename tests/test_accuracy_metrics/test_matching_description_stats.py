import unittest
from unittest.mock import patch
import numpy as np
from Rhapso.accuracy_metrics.matching_descriptors import DescriptiveStatsMatching

class TestDescriptiveStatsMatching(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.stats = DescriptiveStatsMatching({('30', '5'):[ {'p1': np.array([1288.26358115,  822.34935865,   59.33322283]), 'p2': np.array([1291.87676123,  612.10384713,   15.05991703])},
                                                             {'p1': np.array([676.14930295, 930.68295147,  85.59932219]), 'p2': np.array([680.98046159, 739.46465793,   4.11655491])}]}, 2)
        self.stats.get_matches()

    def test_get_matches(self):
        result = DescriptiveStatsMatching({('30', '5'):[ {'p1': np.array([1288.26358115,  822.34935865,   59.33322283]), 'p2': np.array([1291.87676123,  612.10384713,   15.05991703])},
                                                         {'p1': np.array([676.14930295, 930.68295147,  85.59932219]), 'p2': np.array([680.98046159, 739.46465793,   4.11655491])}]}, 2).get_matches()

        self.assertEqual(len(result), 4)

    def test_get_plane_coordinates(self): 
        x,y,z = self.stats.get_plane_coordinates()

        self.assertEqual(x, [1288.26358115,1291.87676123, 676.14930295, 680.98046159])
        self.assertEqual(y, [822.34935865, 612.10384713,930.68295147, 739.46465793])
        self.assertEqual(z, [59.33322283, 15.05991703, 85.59932219, 4.11655491])

    def test_get_bounding_box(self):
        min_bb, max_bb = self.stats.get_bounding_box()
        self.assertEqual(min_bb, {"x": 676.14930295, "y": 612.10384713, "z": 4.11655491})
        self.assertEqual(max_bb, {"x": 1291.87676123, 'y': 930.68295147, 'z': 85.59932219})

    def test_get_standard_deviation(self):
        sd_x, sd_y, sd_z = self.stats.get_standard_deviation()
        self.assertAlmostEqual(sd_x, 353.06, places=2)
        self.assertAlmostEqual(sd_y, 134.50, places=2)
        self.assertAlmostEqual(sd_z, 38.12, places=2)

    def test_average_standard_deviation(self):
        result = self.stats.average_standard_deviation()
        self.assertAlmostEqual(result,175.23, places=2)

    def test_bounding_box_volume(self):
        result = self.stats.bounding_box_volume()
        self.assertAlmostEqual(result, 15983488.69, places=2)

    def test_result(self):

        result = self.stats.results()
        print(result)
        expected_output = {'Number of matches': 2,
                            'Number of interest points in matches': 4, 
                            'Average  matches per tile': 2.0, 
                            'Actual matches points per tile': {'30, 5': 2}, 
                            'Bounding Box Min': {'x': 676.14930295, 'y': 612.10384713, 'z': 4.11655491}, 
                            'Bounding Box Max': {'x': 1291.87676123, 'y': 930.68295147, 'z': 85.59932219},
                            'std_Dev_x': 353.06, 
                            'std_Dev_y': 134.5, 
                            'Std_Dev_z': 38.12, ''
                            'Average Std Dev': 175.23, 
                            'Bounding Box Volume': 15983488.69}

        self.assertEqual(result, expected_output)

    def test_get_matches_raises_error_when_empty(self):
        no_matches = {}
        stats = DescriptiveStatsMatching(no_matches, 0)
        with self.assertRaises(ValueError) as context:
            stats.get_matches()
        self.assertEqual(str(context.exception), "There are no matches to be evaluated")

    def test_get_plane_coordinates_raises_error_on_invalid_point(self):
        matches_with_invalid_point = {("5", "30"): [{"p1":np.array([1,2]),"p2":np.array([1,2,4])}]}
        stats = DescriptiveStatsMatching(matches_with_invalid_point, 1)
        stats.get_matches()

        with self.assertRaises(ValueError) as context:
            stats.get_plane_coordinates()
        self.assertEqual(str(context.exception), "Coordinates missing, an error has occurred.")

if __name__ == '__main__':
    unittest.main()
