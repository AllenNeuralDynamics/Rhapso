
import unittest
import numpy as np
from Rhapso.Validation.matching_KDE import MatchingKDE

class TestMatchingKDE(unittest.TestCase):
    def setUp(self):
        self.sample_pair = [[13, 14, 15],[16, 17, 18]]
        self.sample_data = {
            (30,0): [
                {'p1': np.array([1, 2, 3]), 'p2': np.array([4, 5, 6])},
                {'p1': np.array([7, 8, 9]), 'p2': np.array([10, 11, 12])}
            ],
            (30, 1): [
            {'p1': np.array([13, 14, 15]), 'p2': np.array([16, 17, 18])}
            ]
        }

    def test_get_data_with_pair(self):
        kde = MatchingKDE( None,"pair", None, None, [[13, 14, 15],[16, 17, 18]])
        result = kde.get_data()
        self.assertAlmostEqual(0.04272805146296269,result[0])

    def test_get_matches_from_df(self):
        kde = MatchingKDE(self.sample_data,None, 1, None, None)
        matches = kde.get_matches_from_df()  
        self.assertEqual(len(matches), 6)
        self.assertEqual(matches[0], [1, 2, 3])

    def test_kde_full(self):
        kde = MatchingKDE(self.sample_data, "all", 1, None, None)
        results = kde.get_data()
        self.assertEqual(results, {'min': np.float64(0.010582287163569687), 'max': np.float64(0.010582301671432535), 'mean': np.float64(0.010582296835478252), 'std': np.float64(6.839072133830884e-09)} )

    def test_get_matches_from_view(self):
        kde = MatchingKDE(self.sample_data,"tile", 1, (30,0), None)
        matches = kde.get_matches_from_view()
        self.assertEqual(len(matches), 4)
        self.assertEqual(matches[0], [1, 2, 3])
        self.assertEqual(matches[1], [4, 5, 6])

if __name__ == '__main__':
    unittest.main()