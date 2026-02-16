import unittest
import numpy as np
from Rhapso.evaluation.matching_KDE import MatchingKDE


class TestMatchingKDE(unittest.TestCase):
    def setUp(self):
        self.sample_pair = [[13, 14, 15], [16, 17, 18]]
        self.sample_data = {
            (30, 0): [
                {"p1": np.array([1, 2, 3]), "p2": np.array([4, 5, 6])},
                {"p1": np.array([7, 8, 9]), "p2": np.array([10, 11, 12])},
            ],
            (30, 1): [{"p1": np.array([13, 14, 15]), "p2": np.array([16, 17, 18])}],
        }


    def test_get_data_with_pair(self):
        kde = MatchingKDE(None, "pair", None, None, [[[13, 14, 15], [16, 17, 18]]], False)
        result = kde.get_data()
        expected = {
            'minimum KDE': 0.44046798835233947,
            'maximum KDE': 0.44046798835233947,
            'mean KDE': 0.44046798835233947,
            'median KDE': 0.44046798835233947,
            'std': 0.0
        }
        self.assertEqual(result, expected)

    def test_get_matches_from_df(self):
        kde = MatchingKDE(self.sample_data, None, 1, None, None, False)
        matches = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(matches[0], [1, 2, 3])

    def test_kde_full(self):
        kde = MatchingKDE(self.sample_data, "all", None, None, None, False)
        results = kde.get_data()
        expected = {
            'minimum KDE': 0.44046798835233947,
            'maximum KDE': 0.44046798835233947,
            'mean KDE': 0.44046798835233947,
            'median KDE': 0.44046798835233947,
            'std': 0.0
        }
        self.assertEqual(results, expected)

    def test_get_matches_from_view(self):
        kde = MatchingKDE(self.sample_data, "tile", 1, (30, 0), None, False)
        matches = kde.get_matches_from_view()
        self.assertEqual(matches[0][0], [1, 2, 3])

if __name__ == '__main__':
    unittest.main()



