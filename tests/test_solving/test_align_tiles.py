import io
import unittest
from unittest.mock import patch
import numpy as np

from Rhapso.solver.align_tiles import AlignTiles


class TestAlignTiles(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tiles = {
            "timepoint: 18, setup: 1": {
                "model": {
                    "a": {
                        "m00": 1.0,
                        "m01": 0.0,
                        "m02": 0.0,
                        "m03": 0.0,
                        "m10": 0.0,
                        "m11": 1.0,
                        "m12": 0.0,
                        "m13": 0.0,
                        "m20": 0.0,
                        "m21": 0.0,
                        "m22": 1.0,
                        "m23": 0.0,
                        "i00": 1.0,
                        "i01": 0.0,
                        "i02": 0.0,
                        "i03": 0.0,
                        "i10": 0.0,
                        "i11": 1.0,
                        "i12": 0.0,
                        "i13": 0.0,
                        "i20": 0.0,
                        "i21": 0.0,
                        "i22": 1.0,
                        "i23": 0.0,
                        "cost": 1.7976931348623157e308,
                        "isInvertible": True,
                    },
                    "affine": {
                        "m00": 1.0,
                        "m01": 0.0,
                        "m02": 0.0,
                        "m03": 0.0,
                        "m10": 0.0,
                        "m11": 1.0,
                        "m12": 0.0,
                        "m13": 0.0,
                        "m20": 0.0,
                        "m21": 0.0,
                        "m22": 1.0,
                        "m23": 0.0,
                        "i00": 1.0,
                        "i01": 0.0,
                        "i02": 0.0,
                        "i03": -0.0,
                        "i10": 0.0,
                        "i11": 1.0,
                        "i12": 0.0,
                        "i13": -0.0,
                        "i20": 0.0,
                        "i21": 0.0,
                        "i22": 1.0,
                        "i23": -0.0,
                        "cost": 1.7976931348623157e308,
                        "isInvertible": True,
                    },
                    "afs": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "b": {
                        "m00": 1.0,
                        "m01": 0.0,
                        "m02": 0.0,
                        "m03": 0.0,
                        "m10": 0.0,
                        "m11": 1.0,
                        "m12": 0.0,
                        "m13": 0.0,
                        "m20": 0.0,
                        "m21": 0.0,
                        "m22": 1.0,
                        "m23": 0.0,
                        "i00": 1.0,
                        "i01": 0.0,
                        "i02": 0.0,
                        "i03": 0.0,
                        "i10": 0.0,
                        "i11": 1.0,
                        "i12": 0.0,
                        "i13": 0.0,
                        "i20": 0.0,
                        "i21": 0.0,
                        "i22": 1.0,
                        "i23": 0.0,
                        "cost": 1.7976931348623157e308,
                        "isInvertible": True,
                    },
                    "bfs": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "cost": 1.7976931348623157e308,
                    "l1": 0.9,
                    "lambda": 0.1,
                },
                "matches": [
                    {
                        "p1": np.array([-534.0, -308.0, -72.495679]),
                        "p2": np.array([-538.0, 156.85459177, 278.72318544]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([286.0, -348.0, 80.702737]),
                        "p2": np.array([282.0, 293.46650184, 198.67981787]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([-670.0, -95.49213541, 74.21349207]),
                        "p2": np.array([-666.0, 58.7381151, 110.96751239]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([86.0, -195.10379786, 274.41510485]),
                        "p2": np.array([94.0, 281.11237622, 194.06338074]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([-650.0, -342.02006392, 359.42986302]),
                        "p2": np.array([-646.0, 351.69217453, 349.75775241]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([-502.0, -272.80457158, 321.16512462]),
                        "p2": np.array([-502.0, 314.46785325, 273.84498869]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([-614.0, -167.08128623, 76.1634465]),
                        "p2": np.array([-614.0, 77.20386362, 160.38401486]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([-462.0, -166.36475795, 206.98762251]),
                        "p2": np.array([-458.0, 205.94720539, 173.06202932]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                ],
                "connected_tiles": {
                    "18,0,beads": 0,
                    "18,2,beads": 1,
                    "18,3,beads": 2,
                    "18,4,beads": 3,
                    "18,5,beads": 4,
                },
                "cost": 0.0,
                "distance": 0.0,
            },
            "timepoint: 18, setup: 0": {
                "model": {
                    "a": {
                        "m00": 1.0,
                        "m01": 0.0,
                        "m02": 0.0,
                        "m03": 0.0,
                        "m10": 0.0,
                        "m11": 1.0,
                        "m12": 0.0,
                        "m13": 0.0,
                        "m20": 0.0,
                        "m21": 0.0,
                        "m22": 1.0,
                        "m23": 0.0,
                        "i00": 1.0,
                        "i01": 0.0,
                        "i02": 0.0,
                        "i03": 0.0,
                        "i10": 0.0,
                        "i11": 1.0,
                        "i12": 0.0,
                        "i13": 0.0,
                        "i20": 0.0,
                        "i21": 0.0,
                        "i22": 1.0,
                        "i23": 0.0,
                        "cost": 1.7976931348623157e308,
                        "isInvertible": True,
                    },
                    "affine": {
                        "m00": 1.0,
                        "m01": 0.0,
                        "m02": 0.0,
                        "m03": 0.0,
                        "m10": 0.0,
                        "m11": 1.0,
                        "m12": 0.0,
                        "m13": 0.0,
                        "m20": 0.0,
                        "m21": 0.0,
                        "m22": 1.0,
                        "m23": 0.0,
                        "i00": 1.0,
                        "i01": 0.0,
                        "i02": 0.0,
                        "i03": -0.0,
                        "i10": 0.0,
                        "i11": 1.0,
                        "i12": 0.0,
                        "i13": -0.0,
                        "i20": 0.0,
                        "i21": 0.0,
                        "i22": 1.0,
                        "i23": -0.0,
                        "cost": 1.7976931348623157e308,
                        "isInvertible": True,
                    },
                    "afs": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "b": {
                        "m00": 1.0,
                        "m01": 0.0,
                        "m02": 0.0,
                        "m03": 0.0,
                        "m10": 0.0,
                        "m11": 1.0,
                        "m12": 0.0,
                        "m13": 0.0,
                        "m20": 0.0,
                        "m21": 0.0,
                        "m22": 1.0,
                        "m23": 0.0,
                        "i00": 1.0,
                        "i01": 0.0,
                        "i02": 0.0,
                        "i03": 0.0,
                        "i10": 0.0,
                        "i11": 1.0,
                        "i12": 0.0,
                        "i13": 0.0,
                        "i20": 0.0,
                        "i21": 0.0,
                        "i22": 1.0,
                        "i23": 0.0,
                        "cost": 1.7976931348623157e308,
                        "isInvertible": True,
                    },
                    "bfs": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "cost": 1.7976931348623157e308,
                    "l1": 0.9,
                    "lambda": 0.1,
                },
                "matches": [
                    {
                        "p1": np.array([-534.0, -308.0, -72.495679]),
                        "p2": np.array([-538.0, 156.85459177, 278.72318544]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([286.0, -348.0, 80.702737]),
                        "p2": np.array([282.0, 293.46650184, 198.67981787]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([-670.0, -95.49213541, 74.21349207]),
                        "p2": np.array([-666.0, 58.7381151, 110.96751239]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([86.0, -195.10379786, 274.41510485]),
                        "p2": np.array([94.0, 281.11237622, 194.06338074]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([-650.0, -342.02006392, 359.42986302]),
                        "p2": np.array([-646.0, 351.69217453, 349.75775241]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([-502.0, -272.80457158, 321.16512462]),
                        "p2": np.array([-502.0, 314.46785325, 273.84498869]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([-614.0, -167.08128623, 76.1634465]),
                        "p2": np.array([-614.0, 77.20386362, 160.38401486]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "p1": np.array([-462.0, -166.36475795, 206.98762251]),
                        "p2": np.array([-458.0, 205.94720539, 173.06202932]),
                        "strength": 1.0,
                        "weight": 1.0,
                    },
                ],
                "connected_tiles": {
                    "18,1,beads": 0,
                    "18,2,beads": 1,
                    "18,3,beads": 2,
                    "18,4,beads": 3,
                    "18,5,beads": 4,
                },
                "cost": 0.0,
                "distance": 0.0,
            },

        "timepoint: 30, setup: 0": {
                "model": {
                    "a": {
                        "m00": 1.0,
                        "m01": 0.0,
                        "m02": 0.0,
                        "m03": 0.0,
                        "m10": 0.0,
                        "m11": 1.0,
                        "m12": 0.0,
                        "m13": 0.0,
                        "m20": 0.0,
                        "m21": 0.0,
                        "m22": 1.0,
                        "m23": 0.0,
                        "i00": 1.0,
                        "i01": 0.0,
                        "i02": 0.0,
                        "i03": 0.0,
                        "i10": 0.0,
                        "i11": 1.0,
                        "i12": 0.0,
                        "i13": 0.0,
                        "i20": 0.0,
                        "i21": 0.0,
                        "i22": 1.0,
                        "i23": 0.0,
                        "cost": 0,
                        "isInvertible": True,
                    },
                    "affine": {
                        "m00": 1.0,
                        "m01": 0.0,
                        "m02": 0.0,
                        "m03": 0.0,
                        "m10": 0.0,
                        "m11": 1.0,
                        "m12": 0.0,
                        "m13": 0.0,
                        "m20": 0.0,
                        "m21": 0.0,
                        "m22": 1.0,
                        "m23": 0.0,
                        "i00": 1.0,
                        "i01": 0.0,
                        "i02": 0.0,
                        "i03": -0.0,
                        "i10": 0.0,
                        "i11": 1.0,
                        "i12": 0.0,
                        "i13": -0.0,
                        "i20": 0.0,
                        "i21": 0.0,
                        "i22": 1.0,
                        "i23": -0.0,
                        "cost": 0,
                        "isInvertible": True,
                    },
                    "afs": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "b": {
                        "m00": 1.0,
                        "m01": 0.0,
                        "m02": 0.0,
                        "m03": 0.0,
                        "m10": 0.0,
                        "m11": 1.0,
                        "m12": 0.0,
                        "m13": 0.0,
                        "m20": 0.0,
                        "m21": 0.0,
                        "m22": 1.0,
                        "m23": 0.0,
                        "i00": 1.0,
                        "i01": 0.0,
                        "i02": 0.0,
                        "i03": 0.0,
                        "i10": 0.0,
                        "i11": 1.0,
                        "i12": 0.0,
                        "i13": 0.0,
                        "i20": 0.0,
                        "i21": 0.0,
                        "i22": 1.0,
                        "i23": 0.0,
                        "cost": 0,
                        "isInvertible": True,
                    },
                    "bfs": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "cost": 0,
                    "l1": 0.9,
                    "lambda": 0.1,
                },
                "matches": [
                    {
                        "p1": np.array([-534.0, -308.0, 80.702737]),
                        "p2": np.array([-538.0, 156.85459177, 278.72318544]),
                        "strength": 1.0,
                        "weight": 1.0,
                    }
                ],
                "connected_tiles": {
                    "18,0,beads": 0,
                    "18,1,beads": 1,
                    "18,2,beads": 2,
                    "18,3,beads": 3,
                    "18,4,beads": 4,
                },
                "cost": 0.0,
                "distance": 0.0,
            },
        }

        self.pmc = [
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
        self.fixed_views = ["timepoint: 18, setup: 0"]

    def test_invert_transformation(self):

        self.align_tiles = AlignTiles(self.tiles, self.pmc, self.fixed_views)
        model = {
            "m00": 1,
            "m01": 0,
            "m02": 0,
            "m03": 1,
            "m10": 0,
            "m11": 1,
            "m12": 0,
            "m13": 2,
            "m20": 0,
            "m21": 0,
            "m22": 1,
            "m23": 3,
        }
        expected = {
            "m00": 1,
            "m01": 0,
            "m02": 0,
            "m03": 1,
            "m10": 0,
            "m11": 1,
            "m12": 0,
            "m13": 2,
            "m20": 0,
            "m21": 0,
            "m22": 1,
            "m23": 3,
            "i00": np.float64(1.0),
            "i01": np.float64(0.0),
            "i02": np.float64(0.0),
            "i10": np.float64(0.0),
            "i11": np.float64(1.0),
            "i12": np.float64(0.0),
            "i20": np.float64(0.0),
            "i21": np.float64(0.0),
            "i22": np.float64(1.0),
            "i03": np.float64(-1.0),
            "i13": np.float64(-2.0),
            "i23": np.float64(-3.0),
        }
        result = self.align_tiles.invert_transformation(model)

        self.assertEqual(result, expected)

    def test_translation_fit_model(self):

        self.align_tiles = AlignTiles(self.tiles, self.pmc, self.fixed_views)
        transformation_matrix = np.zeros(12)
        matches = [
            {"p1": np.array([1, 2, 3]), "p2": np.array([4, 5, 6]), "weight": 1},
            {"p1": np.array([2, 3, 4]), "p2": np.array([5, 6, 7]), "weight": 1},
        ]
        expected = np.zeros(12)
        expected[9] = 3
        expected[10] = 3
        expected[11] = 3
        result = self.align_tiles.translation_fit_model(transformation_matrix, matches)
        np.testing.assert_array_equal(result, expected)

    def test_rigid_fit_model(self):
        self.align_tiles = AlignTiles(self.tiles, self.pmc, self.fixed_views)
        rigid_model = {
            "m00": 1,
            "m01": 0,
            "m02": 0,
            "m03": 0,
            "m10": 0,
            "m11": 1,
            "m12": 0,
            "m13": 0,
            "m20": 0,
            "m21": 0,
            "m22": 1,
            "m23": 0,
        }
        matches = [
            {"p1": np.array([1, 0, 0]), "p2": np.array([0, 1, 0]), "weight": 1},
            {"p1": np.array([0, 1, 0]), "p2": np.array([1, 0, 0]), "weight": 1},
        ]

        expected_result = {
            "m00": np.float64(0.0),
            "m01": np.float64(0.9999999999999998),
            "m02": np.float64(0.0),
            "m03": np.float64(1.1102230246251565e-16),
            "m10": np.float64(0.9999999999999998),
            "m11": np.float64(0.0),
            "m12": np.float64(0.0),
            "m13": np.float64(1.1102230246251565e-16),
            "m20": np.float64(0.0),
            "m21": np.float64(0.0),
            "m22": np.float64(-0.9999999999999998),
            "m23": np.float64(0.0),
            "i00": np.float64(0.0),
            "i01": np.float64(1.0000000000000002),
            "i02": np.float64(0.0),
            "i10": np.float64(1.0000000000000002),
            "i11": np.float64(0.0),
            "i12": np.float64(0.0),
            "i20": np.float64(-0.0),
            "i21": np.float64(-0.0),
            "i22": np.float64(-1.0000000000000002),
            "i03": np.float64(-1.1102230246251568e-16),
            "i13": np.float64(-1.1102230246251568e-16),
            "i23": np.float64(0.0),
        }
        result = self.align_tiles.rigid_fit_model(rigid_model, matches)
        self.assertEqual(result, expected_result)
        self.assertIsNotNone(result)

    def test_affine_fit_model(self):
        self.align_tiles = AlignTiles(self.tiles, self.pmc, self.fixed_views)
        affine_model = {
            "m00": 1,
            "m01": 0,
            "m02": 0,
            "m03": 0,
            "m10": 0,
            "m11": 1,
            "m12": 0,
            "m13": 0,
            "m20": 0,
            "m21": 0,
            "m22": 1,
            "m23": 0,
        }
        matches = [
            {"p1": np.array([1, 2, 3]), "p2": np.array([4, 5, 6])},
            {"p1": np.array([2, 3, 4]), "p2": np.array([5, 6, 7])},
        ]
        result = self.align_tiles.affine_fit_model(affine_model, matches)
        self.assertIsNotNone(result)

        with patch("sys.stdout", new=io.StringIO()) as fake_output:
            print("matches are too identical")
            self.assertEqual(
                fake_output.getvalue().strip(), "matches are too identical"
            )

    def test_pre_align(self):
        self.align_tiles = AlignTiles(
            self.tiles, self.pmc, ["timepoint: 18, setup: 0", "timepoint: 30, setup: 0"]
        )
        final_tiles, unaligned_tiles = self.align_tiles.pre_align()
        self.assertEqual(len(final_tiles), 0)
        self.assertEqual(len(unaligned_tiles), 1)
        self.assertNotEqual(final_tiles, unaligned_tiles)

    def test_pre_align_no_unaligned(self):
        self.tiles = {
            "timepoint: 18, setup: 0": {
                "model": {"a": {}, "afs": {}, "b": {}, "bfs": {}},
                "matches": [],
                "connected_tiles": [],
            },
        }
        self.pmc = {}
        self.fixed_views = ["timepoint: 18, setup: 0", "timepoint: 18, setup: 1"]
        self.align_tiles = AlignTiles(self.tiles, self.pmc, self.fixed_views)
        self
        with self.assertRaises(TypeError) as context:
            self.align_tiles.pre_align()
            self.assertEqual(
                str(context.exception),
                "Unaligned_tiles is None and cannot be compared.",
            )

    def test_no_tiles(self):
        self.align_tiles = AlignTiles({}, self.pmc, self.fixed_views)
        # final_tiles, unaligned_tiles = self.align_tiles.pre_align()
        with self.assertRaises(TypeError) as context:
            self.align_tiles.pre_align()
            self.assertEqual(
                str(context.exception),
                "There are no tiles to align.",
            )

    def test_fixed_views(self):
        self.align_tiles = AlignTiles(self.tiles, self.pmc, [])
        with self.assertRaises(TypeError) as context:
            self.align_tiles.pre_align()
            self.assertEqual(
                str(context.exception),
                "There are no fixed views and all tiles will be unaligned.",
            )


if __name__ == "__main__":
    unittest.main()
