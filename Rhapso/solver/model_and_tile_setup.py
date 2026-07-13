import numpy as np

"""
Model and Tile Setup initializes models, tiles, and view-pair matches.
"""

class ModelAndTileSetup:
    def __init__(self, connected_views, corresponding_interest_points, interest_points, view_transform_matrices,
                 view_id_set, label_map):
        self.connected_views = connected_views
        self.corresponding_interest_points = corresponding_interest_points
        self.interest_points = interest_points
        self.view_transform_matrices = view_transform_matrices
        self.view_id_set = view_id_set
        self.label_map = label_map
        self.pairs = []
        self.tiles = {}

    def apply_transform(self, point, matrix):
        """
        Apply a 3D affine transformation to a point using homogeneous coordinates.
        """
        point_homogeneous = np.append(point, 1.0)
        return matrix.dot(point_homogeneous)[:3]

    def setup_point_matches_from_interest_points(self):
        """
        Generate transformed point matches between views.
        """
        view_id_list = sorted(self.view_id_set)

        for i in range(len(view_id_list)):
            for j in range(i + 1, len(view_id_list)):
                key_i = (
                    f"timepoint: {view_id_list[i][0]}, "
                    f"setup: {view_id_list[i][1]}"
                )
                key_j = (
                    f"timepoint: {view_id_list[j][0]}, "
                    f"setup: {view_id_list[j][1]}"
                )

                matrix_a = self.view_transform_matrices.get(key_i)
                matrix_b = self.view_transform_matrices.get(key_j)

                if matrix_a is None or matrix_b is None:
                    continue

                labels_a = self.label_map.get(key_i, [])
                labels_b = self.label_map.get(key_j, [])
                shared_labels = set(labels_a).intersection(labels_b)

                for raw_label in sorted(shared_labels):
                    label = str(raw_label)
                    is_synthetic = label.startswith("splitPoints")
                    base_weight = 0.03 if is_synthetic else 1.0

                    edge_key = (key_i, key_j, label)

                    correspondences = [
                        point
                        for point in self.corresponding_interest_points.get(
                            key_i,
                            [],
                        )
                        if str(point.get("label", "")) == label
                        and point.get("corresponding_view_id") == key_j
                    ]

                    if not correspondences:
                        continue

                    points_by_label_a = self.interest_points.get(key_i, {})
                    points_by_label_b = self.interest_points.get(key_j, {})

                    interest_points_a = points_by_label_a.get(
                        raw_label,
                        points_by_label_a.get(label, []),
                    )
                    interest_points_b = points_by_label_b.get(
                        raw_label,
                        points_by_label_b.get(label, []),
                    )

                    inliers = []

                    for correspondence in correspondences:
                        detection_id_a = int(correspondence["detection_id"])
                        detection_id_b = int(
                            correspondence["corresponding_detection_id"]
                        )

                        if (
                            detection_id_a < 0
                            or detection_id_a >= len(interest_points_a)
                            or detection_id_b < 0
                            or detection_id_b >= len(interest_points_b)
                        ):
                            continue

                        point_a = np.asarray(
                            interest_points_a[detection_id_a],
                            dtype=np.float64,
                        )
                        point_b = np.asarray(
                            interest_points_b[detection_id_b],
                            dtype=np.float64,
                        )

                        transformed_a = self.apply_transform(point_a, matrix_a)
                        transformed_b = self.apply_transform(point_b, matrix_b)

                        interest_point_a = {
                            "l": transformed_a.copy(),
                            "w": transformed_a.copy(),
                            "index": detection_id_a,
                            "weight": base_weight,
                            "strength": 1.0,
                        }

                        interest_point_b = {
                            "l": transformed_b.copy(),
                            "w": transformed_b.copy(),
                            "index": detection_id_b,
                            "weight": base_weight,
                            "strength": 1.0,
                        }

                        match_key = (
                            key_i,
                            label,
                            detection_id_a,
                            key_j,
                            label,
                            detection_id_b,
                        )

                        inliers.append(
                            {
                                "p1": interest_point_a,
                                "p2": interest_point_b,
                                "base_weight": base_weight,
                                "cleanup_weight": 1.0,
                                "weight": base_weight,
                                "strength": 1.0,
                                "is_synthetic": is_synthetic,
                                "label": label,
                                "edge_key": edge_key,
                                "match_key": match_key,
                                "source_view": key_i,
                                "target_view": key_j,
                                "source_detection_id": detection_id_a,
                                "target_detection_id": detection_id_b,
                            }
                        )

                    if not inliers:
                        continue

                    self.pairs.append(
                        {
                            "view": (key_i, key_j),
                            "label": label,
                            "base_weight": base_weight,
                            "weight": base_weight,
                            "is_synthetic": is_synthetic,
                            "edge_key": edge_key,
                            "inliers": inliers,
                            "flipped": None,
                        }
                    )

    def run(self):
        """
        Execute model-and-tile match setup.
        """
        self.setup_point_matches_from_interest_points()
        print("Tile Pairs Set Up")

        return self.pairs