import unittest
import numpy as np
from Rhapso.accuracy_metrics.matching_KDE import MatchingKDE


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
        kde = MatchingKDE(None, "pair", None, None, [[13, 14, 15], [16, 17, 18]], False)
        result = kde.get_data()
        self.assertAlmostEqual(
            result,
            {
                "min": 0.04275650925236316,
                "max": 0.04275650925236316,
                "mean": 0.04275650925236316,
                "median": 0.04275650925236316,
                "std": 0.0,
            },
        )

    def test_get_matches_from_df(self):
        kde = MatchingKDE(self.sample_data, None, 1, None, None, False)
        matches = kde.get_matches_from_df()
        self.assertEqual(len(matches), 6)
        self.assertEqual(matches[0], [1, 2, 3])

    def test_kde_full(self):
        kde = MatchingKDE(self.sample_data, "all", None, None, None, False)
        results = kde.get_data()
        self.assertEqual(
            results,
            {
                "max": np.float64(0.04363032005297173),
                "mean": np.float64(0.039915865071559496),
                "median": np.float64(0.04289343321438295),
                "min": np.float64(0.03322384194732379),
                "std": np.float64(0.004741527930579641),
            },
        )

    def test_get_matches_from_view(self):
        kde = MatchingKDE(self.sample_data, "tile", 1, (30, 0), None, False)
        matches = kde.get_matches_from_view()
        self.assertEqual(len(matches), 4)
        self.assertEqual(matches[0], [1, 2, 3])
        self.assertEqual(matches[1], [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
