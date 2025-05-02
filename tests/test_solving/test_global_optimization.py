import unittest
import numpy as np
from Rhapso.solver.align_tiles import AlignTiles
from Rhapso.solver.global_optimization import GlobalOptimization

# Adjust the import according to your module structure


class TestGlobalOptimization(unittest.TestCase):
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
        self.data_prefix = "test_prefix"
        self.alignment_option = 1
        self.relative_threshold = 0.1
        self.absolute_threshold = 0.1
        self.min_matches = 1
        self.damp = 0.5
        self.max_iterations = 10
        self.max_allowed_error = 0.01
        self.max_plateauwidth = 5
        self.model = {"a": {}, "b": {}}

        self.optimizer = GlobalOptimization(
            self.tiles,
            self.pmc,
            self.fixed_views,
            self.data_prefix,
            self.alignment_option,
            self.relative_threshold,
            self.absolute_threshold,
            self.min_matches,
            self.damp,
            self.max_iterations,
            self.max_allowed_error,
            self.max_plateauwidth,
            self.model,
        )

    def test_update_cost(self):
        self.optimizer.update_cost(
            "timepoint: 18, setup: 1", self.tiles["timepoint: 18, setup: 1"]
        )
        self.assertIn("distance", self.tiles["timepoint: 18, setup: 1"])
        self.assertIn("cost", self.tiles["timepoint: 18, setup: 1"])
        self.assertGreaterEqual(self.tiles["timepoint: 18, setup: 1"]["distance"], 0)
        self.assertGreaterEqual(self.tiles["timepoint: 18, setup: 1"]["cost"], 0)

    def test_update_errors(self):
        average_error = self.optimizer.update_errors()
        self.assertGreaterEqual(average_error, 0)

    def test_apply_damp(self):
        self.optimizer.apply_damp(
            "timepoint: 18, setup: 1", self.tiles["timepoint: 18, setup: 1"]
        )
        updated_p1 = self.tiles["timepoint: 18, setup: 1"]["matches"][0]["p1"]
        self.assertNotEqual(updated_p1, [-534.0, -308.0, -72.495679])

    def test_fit_model(self):
        self.optimizer.fit_model(
            "timepoint: 18, setup: 1", self.tiles["timepoint: 18, setup: 1"]
        )
        self.assertIn("a", self.tiles["timepoint: 18, setup: 1"]["model"])
        self.assertIn("b", self.tiles["timepoint: 18, setup: 1"]["model"])

    def test_apply(self):
        self.optimizer.apply(
            "timepoint: 18, setup: 1", self.tiles["timepoint: 18, setup: 1"]
        )
        updated_p1 = self.tiles["timepoint: 18, setup: 1"]["matches"][0]["p1"]
        self.assertEqual(updated_p1, [-534.0, -308.0, -72.495679])

    # Optimize silently has TODO's left.
    # def test_optimize_silently(self):
    #     self.optimizer.optimize_silently()


if __name__ == "__main__":
    unittest.main()
