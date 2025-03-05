import unittest
from Rhapso.detection import advanced_refinement

# from Rhapso.detection.advanced_refinement import (
#     AdvancedRefinement,
#     filter_points,
# )


class TestAdvancedRefinement(unittest.TestCase):
    def setUp(self):
        self.data = [
            [
                [18, 4],
                {
                    "lower": [0, 92, 0],
                    "upper": [346, 168, 49],
                    "dimensions": [347, 77, 50],
                },
                [
                    [468.345027587921, 488.36223951244153, 2.970786928172377],
                    [856.6701082186948, 416.01488311517676, 3.4227515981883694],
                    [791.9558945564568, 433.6806488102801, 3.249267582240008],
                ],
                [
                    30.77733612060547,
                    16.91626739501953,
                    24.641128540039062,
                ],
            ],
            [
                [30, 4],
                {
                    "lower": [0, 16, 0],
                    "upper": [346, 244, 49],
                    "dimensions": [347, 229, 50],
                },
                [
                    [1297.5348073773343, 142.58036182599938, 2.679636510894805],
                    [1244.4709777440216, 90.09388496845614, 4.249346326216867],
                    [1311.1575370642881, 197.3828616640413, 4.98718680191121],
                    [931.5577670275395, 816.0184848703304, 24.588659490589006],
                ],
                [
                    12.750273704528809,
                    17.272457122802734,
                    14.833109855651855,
                ],
            ],
            [
                [18, 5],
                {
                    "lower": [0, 18, 0],
                    "upper": [346, 242, 44],
                    "dimensions": [347, 225, 45],
                },
                [
                    [904.1266256525571, 702.9668186793718, 2.865155744767481],
                    [356.5371573393498, 783.7573389573807, 1.7635092849906973],
                    [374.9872039679151, 503.22745832789155, 4.41255838223876],
                ],
                [
                    17.8055419921875,
                    23.071125030517578,
                    22.90916633605957,
                ],
            ],
        ]

        self.to_process = [
            [
                (18, 4),
                {
                    "lower": [0, 92, 0],
                    "upper": [346, 168, 49],
                    "dimensions": [347, 77, 50],
                },
            ],
            [
                (30, 4),
                {
                    "lower": [0, 16, 0],
                    "upper": [346, 244, 49],
                    "dimensions": [347, 229, 50],
                },
            ],
        ]

        self.advanced_refinement = advanced_refinement.AdvancedRefinement(
            self.data, to_process=self.to_process, max_spots=2
        )

    def test_filter_points(self):
        interest_points = [
            [468.345027587921, 488.36223951244153, 2.970786928172377],
            [856.6701082186948, 416.01488311517676, 3.4227515981883694],
            [791.9558945564568, 433.6806488102801, 3.249267582240008],
        ]
        intensities = [10, 20, 30]
        max_spots = 2
        filtered_points, filtered_intensities = advanced_refinement.filter_points(
            interest_points, intensities, max_spots
        )
        self.assertEqual(len(filtered_points), max_spots)

        self.assertEqual(len(filtered_intensities), max_spots)


if __name__ == "__main__":
    unittest.main()
