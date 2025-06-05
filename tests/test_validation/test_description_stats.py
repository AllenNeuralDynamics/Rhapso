import unittest
from unittest.mock import patch
from Rhapso.Validation.alignment_displacement import DescriptiveStatsAlignment

class TestDescriptiveStats(unittest.TestCase):
    def test_typical_case(self):
        tiles = {
        "timepoint: 18, setup: 0":{"distance": 10},
        "timepoint: 18, setup: 1":{"distance": 20},
        "timepoint: 18, setup: 2":{"distance": 30}
        }

        stats = DescriptiveStatsAlignment(tiles)
        stats.min_max_mean()
        self.assertEqual(stats.min_error, 10)
        self.assertEqual(stats.max_error, 30)
        self.assertEqual(stats.mean_error, 20)

    def test_large_set(self):
        tiles = {"timepoint: 18, setup: 0": {"distance": 4},"timepoint: 18, setup: 1": {"distance": 14},
                "timepoint: 18, setup: 2": {"distance": 16}, "timepoint: 18, setup: 3": {"distance": 6},
                "timepoint: 18, setup: 4": {"distance": 16}, "timepoint: 18, setup: 5": {"distance": 7}, 
                "timepoint: 18, setup: 6": {"distance": 20}, "timepoint: 18, setup: 7": {"distance": 13}, 
                "timepoint: 18, setup: 8": {"distance": 19}, "timepoint: 18, setup: 9": {"distance": 6}, 
                "timepoint: 18, setup: 10": {"distance": 19}, "timepoint: 18, setup: 11": {"distance": 9}, 
                "timepoint: 18, setup: 12": {"distance": 7}, "timepoint: 18, setup: 13": {"distance": 19}, 
                "timepoint: 18, setup: 14": {"distance": 2}, "timepoint: 18, setup: 15": {"distance": 7}, 
                "timepoint: 18, setup: 16": {"distance": 20}, "timepoint: 18, setup: 17": {"distance": 2}, 
                "timepoint: 18, setup: 18": {"distance": 20}, "timepoint: 18, setup: 19": {"distance": 11}, 
                "timepoint: 18, setup: 20": {"distance": 12}, "timepoint: 18, setup: 21": {"distance": 16}, 
                "timepoint: 18, setup: 22": {"distance": 6}, "timepoint: 18, setup: 23": {"distance": 1}, 
                "timepoint: 18, setup: 24": {"distance": 3}, "timepoint: 18, setup: 25": {"distance": 2}, 
                "timepoint: 18, setup: 26": {"distance": 1}, "timepoint: 18, setup: 27": {"distance": 19}, 
                "timepoint: 18, setup: 28": {"distance": 1}, "timepoint: 18, setup: 29": {"distance": 1}, 
                "timepoint: 18, setup: 30": {"distance": 15}, "timepoint: 18, setup: 31": {"distance": 19}, 
                "timepoint: 18, setup: 32": {"distance": 20}, "timepoint: 18, setup: 33": {"distance": 16}, 
                "timepoint: 18, setup: 34": {"distance": 20}, "timepoint: 18, setup: 35": {"distance": 11}, 
                "timepoint: 18, setup: 36": {"distance": 8}, "timepoint: 18, setup: 37": {"distance": 4}, 
                "timepoint: 18, setup: 38": {"distance": 16}, "timepoint: 18, setup: 39": {"distance": 3}, 
                "timepoint: 18, setup: 40": {"distance": 19}, "timepoint: 18, setup: 41": {"distance": 15}, 
                "timepoint: 18, setup: 42": {"distance": 18}, "timepoint: 18, setup: 43": {"distance": 8}, 
                "timepoint: 18, setup: 44": {"distance": 7}, "timepoint: 18, setup: 45": {"distance": 20}, 
                "timepoint: 18, setup: 46": {"distance": 8}, "timepoint: 18, setup: 47": {"distance": 14}, 
                "timepoint: 18, setup: 48": {"distance": 11}, "timepoint: 18, setup: 49": {"distance": 12}, 
                "timepoint: 18, setup: 50": {"distance": 9}, "timepoint: 18, setup: 51": {"distance": 3}, 
                "timepoint: 18, setup: 52": {"distance": 19}, "timepoint: 18, setup: 53": {"distance": 15}, 
                "timepoint: 18, setup: 54": {"distance": 3}, "timepoint: 18, setup: 55": {"distance": 12}, 
                "timepoint: 18, setup: 56": {"distance": 11}, "timepoint: 18, setup: 57": {"distance": 14}, 
                "timepoint: 18, setup: 58": {"distance": 17}, "timepoint: 18, setup: 59": {"distance": 7}, 
                "timepoint: 18, setup: 60": {"distance": 1}, "timepoint: 18, setup: 61": {"distance": 15}, 
                "timepoint: 18, setup: 62": {"distance": 12}, "timepoint: 18, setup: 63": {"distance": 7},
                "timepoint: 18, setup: 64": {"distance": 20}, "timepoint: 18, setup: 65": {"distance": 2}, 
                "timepoint: 18, setup: 66": {"distance": 7}, "timepoint: 18, setup: 67": {"distance": 11}, 
                "timepoint: 18, setup: 68": {"distance": 14}, "timepoint: 18, setup: 69": {"distance": 1}, 
                "timepoint: 18, setup: 70": {"distance": 20}, "timepoint: 18, setup: 71": {"distance": 13}, 
                "timepoint: 18, setup: 72": {"distance": 4}, "timepoint: 18, setup: 73": {"distance": 12}, 
                "timepoint: 18, setup: 74": {"distance": 7}, "timepoint: 18, setup: 75": {"distance": 7}, 
                "timepoint: 18, setup: 76": {"distance": 5}, "timepoint: 18, setup: 77": {"distance": 8}, 
                "timepoint: 18, setup: 78": {"distance": 19}, "timepoint: 18, setup: 79": {"distance": 14}, 
                "timepoint: 18, setup: 80": {"distance": 12}, "timepoint: 18, setup: 81": {"distance": 18}, 
                "timepoint: 18, setup: 82": {"distance": 12}, "timepoint: 18, setup: 83": {"distance": 5}, 
                "timepoint: 18, setup: 84": {"distance": 12}, "timepoint: 18, setup: 85": {"distance": 9}, 
                "timepoint: 18, setup: 86": {"distance": 15}, "timepoint: 18, setup: 87": {"distance": 3}, 
                "timepoint: 18, setup: 88": {"distance": 13}, "timepoint: 18, setup: 89": {"distance": 4}, 
                "timepoint: 18, setup: 90": {"distance": 1}, "timepoint: 18, setup: 91": {"distance": 9}, 
                "timepoint: 18, setup: 92": {"distance": 8}, "timepoint: 18, setup: 93": {"distance": 15}, 
                "timepoint: 18, setup: 94": {"distance": 1}, "timepoint: 18, setup: 95": {"distance": 19}, 
                "timepoint: 18, setup: 96": {"distance": 17}, "timepoint: 18, setup: 97": {"distance": 17}, 
                "timepoint: 18, setup: 98": {"distance": 15}, "timepoint: 18, setup: 99": {"distance": 7}}

        stats = DescriptiveStatsAlignment(tiles)
        stats.min_max_mean()
        self.assertEqual(stats.min_error, 1)
        self.assertEqual(stats.max_error, 20)
        self.assertEqual(stats.mean_error, 10.84)

    def test_uniform_distances(self):
        tiles = {
        "timepoint: 18, setup: 0":{"distance": 5},
        "timepoint: 18, setup: 1":{"distance": 5},
        "timepoint: 18, setup: 2":{"distance": 5}
        }
        stats = DescriptiveStatsAlignment(tiles)
        stats.min_max_mean()
        self.assertEqual(stats.min_error, 5)
        self.assertEqual(stats.max_error, 5)
        self.assertEqual(stats.mean_error, 5)

    @patch("builtins.print")
    def test_empty_input(self, mock_print):
        tiles = []
        stats = DescriptiveStatsAlignment(tiles).min_max_mean()
        mock_print.assert_any_call("There are no tiles to be evaluated")
    
    @patch("builtins.print")
    def test_tile_distance_never_updated(self, mock_print):
        tiles = {
        "timepoint: 18, setup: 0":{"distance": 0.0},
        "timepoint: 18, setup: 1":{"distance": 0.0},
        "timepoint: 18, setup: 2":{"distance": 0.0}
        }
        stats = DescriptiveStatsAlignment(tiles)
        stats.min_max_mean()
        mock_print.assert_any_call("All the distances between the tiles are 0.0 and is likely due to the distance of each tile not being updated.")


if __name__ == '__main__':
    unittest.main()
