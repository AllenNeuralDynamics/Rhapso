import unittest
import numpy as np

from Rhapso.solver.model_and_tile_setup import ModelAndTileSetup


class TestModelAndTileSetup(unittest.TestCase):

    def setUp(self):
        self.connected_views = {
            "timepoint: 18, setup: 0": {
                "18,1,beads": 0,
            },
            "timepoint: 18, setup: 1": {
                "18,0,beads": 0,
            },
        }
        self.corresponding_interest_points = {
            "timepoint: 18, setup: 0": [
                {
                    "detection_id": 0,
                    "detection_p1": np.array([1.0, 2.0, 3.0]),
                    "corresponding_detection_id": 0,
                    "corresponding_detection_p2": np.array([1.0, 2.0, 3.0]),
                    "corresponding_view_id": "timepoint: 18, setup: 1",
                    "label": "beads",
                }
            ],
            "timepoint: 18, setup: 1": [],
        }
        self.interest_points = {
            "timepoint: 18, setup: 0": [
                np.array([1.0, 2.0, 3.0]),
            ],
            "timepoint: 18, setup: 1": [
                np.array([1.0, 2.0, 3.0]),
            ],
        }

        self.view_transform_matrices = {
            "timepoint: 18, setup: 0": np.array(
                [
                    [1.0, 0.0, 0.0, -1],
                    [0.0, 1.0, 0.0, -1],
                    [0.0, 0.0, 2, -1],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "timepoint: 18, setup: 1": np.array(
                [
                    [1.0, 0.0, 0.0, -1],
                    [0.0, 1.0, 0.0, -1],
                    [0.0, 0.0, 2, -1],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        }

        self.view_id_set = {("18", "0"), ("18", "1")}

        self.label_map = {
            "timepoint: 18, setup: 0": {"beads": 1.0},
            "timepoint: 18, setup: 1": {"beads": 1.0},
        }

    def test_create_tiles(self):
        self.model_and_tile_setup = ModelAndTileSetup(
            self.connected_views,
            self.corresponding_interest_points,
            self.interest_points,
            self.view_transform_matrices,
            self.view_id_set,
            self.label_map,
        )

        self.model_and_tile_setup.create_models()
        self.model_and_tile_setup.create_tiles()
        self.assertIn("timepoint: 18, setup: 0", self.model_and_tile_setup.tiles)
        self.assertIn("timepoint: 18, setup: 1", self.model_and_tile_setup.tiles)
        self.assertEqual(len(self.model_and_tile_setup.tiles), 2)

    def test_empty_view_ids(self):
        empty_view_id_set = set()
        model_and_tile_setup_empty = ModelAndTileSetup(
            self.connected_views,
            self.corresponding_interest_points,
            self.interest_points,
            self.view_transform_matrices,
            empty_view_id_set,
            self.label_map,
        )
        model_and_tile_setup_empty.create_models()
        with self.assertRaises(KeyError) as context:
            model_and_tile_setup_empty.create_tiles()
            self.assertEqual(
                str(context.exception), "There are no viewIds to create tiles from."
            )

    def test_no_corresponding_interest_points(self):
        self.model_and_tile_setup = ModelAndTileSetup(
            self.connected_views,
            self.corresponding_interest_points,
            self.interest_points,
            self.view_transform_matrices,
            self.view_id_set,
            self.label_map,
        )
        self.model_and_tile_setup.corresponding_interest_points = {
            "timepoint: 18, setup: 0": [],
            "timepoint: 18, setup: 1": [],
        }

        self.model_and_tile_setup.create_models()
        self.model_and_tile_setup.create_tiles()
        for key in self.model_and_tile_setup.tiles:
            self.assertEqual(self.model_and_tile_setup.tiles[key]["matches"], [])

    # This test is flaky
    def test_no_connected_views(self):
        self.view_id_set = {(1, 2)}
        self.corresponding_interest_points = {
            "timepoint: 1, setup: 2": [
                {"detection_p1": "A", "corresponding_detection_p2": "B"}
            ]
        }
        self.connected_views = {}
        self.model_and_tile_setup = ModelAndTileSetup(
            self.connected_views,
            self.corresponding_interest_points,
            self.interest_points,
            self.view_transform_matrices,
            self.view_id_set,
            self.label_map,
        )
        with self.assertRaises(KeyError) as context:
            self.model_and_tile_setup.create_tiles()
            self.assertEqual(
                str(context.exception), "There are no viewIds to create tiles from."
            )
        TestModelAndTileSetup.tearDown(self)

    def test_setup_point_matches_from_interest_points(self):
        self.instance = ModelAndTileSetup(
            self.connected_views,
            self.corresponding_interest_points,
            self.interest_points,
            self.view_transform_matrices,
            self.view_id_set,
            self.label_map,
        )
        self.instance.setup_point_matches_from_interest_points()

        expected_pairs = [
            (
                ("timepoint: 18, setup: 0", "timepoint: 18, setup: 1"),
                [
                    (
                        {
                            "l": np.array([0.0, 1.0, 5.0]),
                            "w": np.array([0.0, 1.0, 5.0]),
                        },
                        {
                            "l": np.array([0.0, 1.0, 5.0]),
                            "w": np.array([0.0, 1.0, 5.0]),
                        },
                    )
                ],
            )
        ]
        self.assertEqual(len(self.instance.pairs), 1)

        for (expected_key, expected_inliers), (actual_key, actual_inliers) in zip(
            expected_pairs, self.instance.pairs
        ):
            self.assertEqual(expected_key, actual_key)
            for expected_inlier, actual_inlier in zip(expected_inliers, actual_inliers):
                np.testing.assert_array_equal(
                    expected_inlier[0]["l"], actual_inlier[0]["l"]
                )
                np.testing.assert_array_equal(
                    expected_inlier[0]["w"], actual_inlier[0]["w"]
                )
                np.testing.assert_array_equal(
                    expected_inlier[1]["l"], actual_inlier[1]["l"]
                )
                np.testing.assert_array_equal(
                    expected_inlier[1]["w"], actual_inlier[1]["w"]
                )


if __name__ == "__main__":
    unittest.main()
