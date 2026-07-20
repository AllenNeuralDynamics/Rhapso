import json
import os
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
from urllib.parse import urlparse
import fsspec
import numpy as np
import pandas as pd

class MatchingMetrics:
    """Evaluate rigid, affine, or split-affine saved-match progression."""

    def __init__(self, pre_xml_path, post_xml_path, alignment_base, downsample_xyz, match_type,
             split_xml_path=None, solver_history_path=None):
        self.pre_xml_path = str(pre_xml_path)
        self.post_xml_path = str(post_xml_path)
        self.alignment_base = str(alignment_base)
        self.split_xml_path = None if split_xml_path is None else str(split_xml_path)
        self.solver_history_path = None if solver_history_path is None else str(solver_history_path)
        self.metric_downsample_xyz = tuple(float(value) for value in downsample_xyz)
        self.match_type = str(match_type).strip().lower()

        self.timepoint = 0
        self.match_label = "beads"
        self.split_real_label = "beads_split"
        self.actual_match_dedupe_decimals = 6
        self.target_pair: Optional[Tuple[int, int]] = None
        self.parent_rigid_match_count: Optional[int] = None
        self.grid_bins_zyx = (4, 8, 8)
        self.min_matches_per_grid_cell = 5
        self.print_match_distance_rows_count = 10
        self.percent_baseline_epsilon = 1e-6

        self.loop_max_path_edges = 6
        self.loop_max_paths_per_edge = 4
        self.loop_translation_warning_px = max(self.metric_downsample_xyz)
        self.loop_rotation_warning_degrees = 0.10

        self.affine_distance_targets_scaled = {
            "median": 1.5,
            "p90": 3.0,
            "p95": 4.0,
        }
        self.affine_geometry_linear_error_scale = 0.05
        self.affine_geometry_translation_scaled_scale = 1.0
        self.affine_sanity_scale_deviation_scale = 0.05
        self.affine_sanity_shear_scale = 0.05
        self.affine_sanity_condition_excess_scale = 0.20

        self.rejected_edge_keys = set()

    # ==============================================================================
    # IO
    # ==============================================================================

    def has_uri_scheme(self, path):
        """
        Return True for paths such as s3://bucket/key or file:///tmp/file.
        """
        return bool(urlparse(str(path)).scheme)

    def join_uri(self, base, *parts):
        """
        Join local paths or URI paths without corrupting protocol separators.

        Absolute URI children replace the current base. Absolute local children
        replace a local base, but a leading slash on a child of an S3/URI base is
        treated as a path inside that URI.
        """
        result = str(base).rstrip("/")
        for part in parts:
            if part is None:
                continue
            part = str(part)
            if self.has_uri_scheme(part):
                result = part.rstrip("/")
                continue
            if not self.has_uri_scheme(result) and os.path.isabs(part):
                result = part.rstrip("/")
                continue
            result = f"{result.rstrip('/')}/{part.lstrip('/')}"
        return result

    def resolve_uri(self, base, path):
        """
        Preserve absolute S3/URI paths stored in manifests and indexes.
        Resolve relative paths against ALIGNMENT_BASE.
        """
        path = str(path)
        if self.has_uri_scheme(path):
            return path
        if not self.has_uri_scheme(base) and os.path.isabs(path):
            return path
        return self.join_uri(base, path)

    def open_uri(self, uri, mode):
        """
        Open local files, S3 objects, or any other fsspec-supported URI.
        """
        fs, path = fsspec.core.url_to_fs(str(uri))
        return fs.open(path, mode)

    def read_json(self, uri):
        with self.open_uri(uri, "rt") as f:
            return json.load(f)

    def read_parquet(self, uri):
        with self.open_uri(uri, "rb") as f:
            return pd.read_parquet(f, engine="pyarrow")

    def load_xml(self, uri):
        with self.open_uri(uri, "rb") as f:
            return ET.parse(f).getroot()

    # ==============================================================================
    # TRANSFORMS
    # ==============================================================================

    def affine_matrix(self, text):
        values = [float(v) for v in text.replace(",", " ").split()]
        if len(values) != 12:
            raise RuntimeError(f"Expected 12 affine values, got {len(values)}")
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, :4] = values[0:4]
        matrix[1, :4] = values[4:8]
        matrix[2, :4] = values[8:12]
        return matrix

    def xml_transforms(self, root):
        """
        Compose ViewTransforms using the same convention as matching:
            final = final @ next_transform
        """
        transforms = {}
        for registration in root.findall(".//ViewRegistration"):
            setup = int(registration.get("setup"))
            if int(registration.get("timepoint", 0)) != self.timepoint:
                continue
            composed = np.eye(4, dtype=np.float64)
            for view_transform in registration.findall("ViewTransform"):
                affine = view_transform.findtext("affine")
                if not affine:
                    continue
                composed = composed @ self.affine_matrix(affine)
            transforms[setup] = composed
        return transforms

    def apply_affine(self, points, matrix):
        homogeneous = np.column_stack([points, np.ones(len(points))])
        return (homogeneous @ matrix.T)[:, :3]

    def parse_split_setup_definitions(self, root):
        """
        Map active split setup ID -> original setup ID and source bounds.

        Split point coordinates are local to the chunk. Adding source_min_xyz
        restores the point to the coordinate system of the original, non-split tile.
        """
        definitions = {}
        setup_ids = root.find(".//SetupIds")
        if setup_ids is None:
            raise RuntimeError(
                "No <SetupIds> table found in split XML; cannot map split setups back to their original affine setups"
            )
        for definition in setup_ids.findall("SetupIdDefinition"):
            new_id_text = definition.findtext("NewId")
            old_id_text = definition.findtext("OldId")
            min_text = definition.findtext("min")
            max_text = definition.findtext("max")
            if new_id_text is None or old_id_text is None or min_text is None or (max_text is None):
                continue
            new_id = int(new_id_text)
            definitions[new_id] = {
                "old_setup": int(old_id_text),
                "source_min_xyz": np.asarray(
                    [float(value) for value in min_text.split()], dtype=np.float64
                ),
                "source_max_xyz": np.asarray(
                    [float(value) for value in max_text.split()], dtype=np.float64
                ),
            }
        if not definitions:
            raise RuntimeError("The split XML contains no usable SetupIdDefinition entries")
        return definitions

    # ==============================================================================
    # SAVED MATCHES
    # ==============================================================================

    def point_manifest(self):
        return self.read_json(self.join_uri(self.alignment_base, "manifest.json"))["points"]

    def match_index(self):
        return self.read_parquet(
            self.join_uri(self.alignment_base, "matches", "match_index.parquet")
        )

    def read_points(self, manifest, setup, label):
        key = f"{self.timepoint}/{setup}/{label}"
        if key not in manifest:
            raise KeyError(f"Missing point manifest entry: {key}")
        df = self.read_parquet(self.resolve_uri(self.alignment_base, manifest[key]))
        return df[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=False)

    def read_correspondences(self, index, setup, label=None):
        """
        Read every correspondence table registered for one setup and label.
        """
        label = self.match_label if label is None else str(label)
        rows = index[
            (index["timepoint"].astype(int) == self.timepoint)
            & (index["setup"].astype(int) == setup)
            & (index["label"].astype(str) == label)
        ]
        if len(rows) == 0:
            return pd.DataFrame()
        chunks = []
        for path in rows["correspondences_path"].dropna().astype(str).unique():
            chunks.append(self.read_parquet(self.resolve_uri(self.alignment_base, path)))
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)

    def read_direction(self, manifest, index, source_setup, target_setup, label=None):
        label = self.match_label if label is None else str(label)
        corr = self.read_correspondences(index, source_setup, label=label)
        if len(corr) == 0:
            return (np.empty((0, 3)), np.empty((0, 3)))
        corr = corr[
            (corr["target_timepoint"].astype(int) == self.timepoint)
            & (corr["target_setup"].astype(int) == target_setup)
        ]
        if len(corr) == 0:
            return (np.empty((0, 3)), np.empty((0, 3)))
        source_points = self.read_points(manifest, source_setup, label)
        source_chunks = []
        target_chunks = []
        for target_label, rows in corr.groupby("target_label"):
            target_points = self.read_points(manifest, target_setup, str(target_label))
            source_ids = rows["source_point_id"].to_numpy(dtype=np.int64)
            target_ids = rows["target_point_id"].to_numpy(dtype=np.int64)
            if len(source_ids) and (source_ids.min() < 0 or source_ids.max() >= len(source_points)):
                raise IndexError(f"Source index out of bounds for setup {source_setup}")
            if len(target_ids) and (target_ids.min() < 0 or target_ids.max() >= len(target_points)):
                raise IndexError(f"Target index out of bounds for setup {target_setup}")
            source_chunks.append(source_points[source_ids])
            target_chunks.append(target_points[target_ids])
        return (np.vstack(source_chunks), np.vstack(target_chunks))

    def read_pair(self, manifest, index, setup_a, setup_b, label=None):
        label = self.match_label if label is None else str(label)
        a1, b1 = self.read_direction(manifest, index, setup_a, setup_b, label=label)
        b2, a2 = self.read_direction(manifest, index, setup_b, setup_a, label=label)
        a_parts = [x for x in (a1, a2) if len(x)]
        b_parts = [x for x in (b1, b2) if len(x)]
        if not a_parts:
            return (np.empty((0, 3)), np.empty((0, 3)))
        return (np.vstack(a_parts), np.vstack(b_parts))
    
    def edge_key(self, setup_a, setup_b):
        return tuple(sorted((int(setup_a), int(setup_b))))

    def load_rejected_edge_keys(self):
        if self.solver_history_path is None:
            return set()

        history_path = self.resolve_uri(
            self.alignment_base,
            self.solver_history_path,
        )

        history = self.read_json(history_path)
        history = history.get("validation_stats", history)

        weak_edge_history = (
            history
            .get("solve_metrics_per_tile", {})
            .get("weak_edge_rejection", {})
        )

        rejected_edges = set()

        for edge in weak_edge_history.get("rejected_edges", []):
            edge_value = edge.get("edge_key")

            if edge_value is None or len(edge_value) != 2:
                continue

            rejected_edges.add(
                self.edge_key(edge_value[0], edge_value[1])
            )

        print(
            f"Loaded {len(rejected_edges)} solver-rejected edges "
            f"from {history_path}"
        )

        for edge in sorted(rejected_edges):
            print(f"  Excluding solver-rejected edge {edge}")

        return rejected_edges
    
    def saved_pairs(self, index, label=None):
        label = self.match_label if label is None else str(label)
        pairs = set()

        rows = index[
            (index["timepoint"].astype(int) == self.timepoint)
            & (index["label"].astype(str) == label)
        ]

        for _, row in rows.iterrows():
            setup = int(row["setup"])

            corr = self.read_parquet(
                self.resolve_uri(
                    self.alignment_base,
                    row["correspondences_path"],
                )
            )

            if len(corr) == 0:
                continue

            corr = corr[
                corr["target_timepoint"].astype(int) == self.timepoint
            ]

            for target in corr["target_setup"].astype(int).unique():
                if target == setup:
                    continue

                edge = self.edge_key(setup, target)

                if edge in self.rejected_edge_keys:
                    continue

                pairs.add(edge)

        return sorted(pairs)

    # def saved_pairs(self, index, label=None):
    #     label = self.match_label if label is None else str(label)
    #     pairs = set()
    #     rows = index[
    #         (index["timepoint"].astype(int) == self.timepoint)
    #         & (index["label"].astype(str) == label)
    #     ]
    #     for _, row in rows.iterrows():
    #         setup = int(row["setup"])
    #         corr = self.read_parquet(
    #             self.resolve_uri(self.alignment_base, row["correspondences_path"])
    #         )
    #         if len(corr) == 0:
    #             continue
    #         corr = corr[corr["target_timepoint"].astype(int) == self.timepoint]
    #         for target in corr["target_setup"].astype(int).unique():
    #             if target != setup:
    #                 pairs.add(tuple(sorted((setup, int(target)))))
    #     return sorted(pairs)

    def matching_labels(self, index, prefix):
        """
        Return saved-match labels beginning with the requested prefix.
        """
        labels = index.loc[index["timepoint"].astype(int) == self.timepoint, "label"].astype(str)
        return sorted((label for label in labels.unique() if label.startswith(prefix)))

    def dedupe_pairs(self, a, b, decimals=3):
        if len(a) != len(b):
            raise RuntimeError(f"Unpaired matches: A={len(a)}, B={len(b)}")
        if len(a) == 0:
            return (a, b)
        keys = np.round(np.column_stack([a, b]), decimals)
        _, keep = np.unique(keys, axis=0, return_index=True)
        keep = np.sort(keep)
        return (a[keep], b[keep])

    # ==============================================================================
    # GENERAL METRICS
    # ==============================================================================

    def summary(self, values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    def percent_reduction(self, before, after):
        if before <= self.percent_baseline_epsilon:
            return np.nan
        return 100.0 * (before - after) / before

    def unit_interval_exp(self, value, scale):
        if not np.isfinite(value):
            return 0.0
        return float(np.exp(-max(0.0, value) / max(scale, 1e-12)))
    
    def fit_rigid_transform(self, source_points, target_points):
        """
        Fit the least-squares rigid transform:

            target ~= R @ source + t

        using the Kabsch algorithm.
        """
        source_points = np.asarray(source_points, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)

        if source_points.shape != target_points.shape:
            raise ValueError(
                "source_points and target_points must have matching shapes"
            )

        if len(source_points) < 3:
            raise ValueError(
                "At least three point pairs are required for a rigid fit"
            )

        source_center = source_points.mean(axis=0)
        target_center = target_points.mean(axis=0)

        source_centered = source_points - source_center
        target_centered = target_points - target_center

        covariance = source_centered.T @ target_centered

        u, _, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T

        # Prevent a reflected solution.
        if np.linalg.det(rotation) < 0.0:
            vt[-1, :] *= -1.0
            rotation = vt.T @ u.T

        translation = target_center - rotation @ source_center

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation

        return matrix

    def collect_rigid_edge_diagnostics(
        self,
        manifest,
        index,
        pre,
        post,
    ):
        """
        Diagnose every saved rigid edge.

        Returns:
            edge_table:
                Per-edge residual, coverage, and geometry metrics.

            edge_models:
                Independently fitted rigid transform and anchor points for
                every edge. These are used for alternate-path loop closure.
        """
        rows = []
        edge_models = {}

        for setup_a, setup_b in self.saved_pairs(index):
            if (
                setup_a not in pre
                or setup_b not in pre
                or setup_a not in post
                or setup_b not in post
            ):
                continue

            a_raw, b_raw = self.read_pair(
                manifest,
                index,
                setup_a,
                setup_b,
            )

            raw_count = len(a_raw)

            a, b = self.dedupe_pairs(
                a_raw,
                b_raw,
            )

            match_count = len(a)

            if match_count < 3:
                continue

            # Original PRE placement.
            a_pre = self.apply_affine(
                a,
                pre[setup_a],
            )
            b_pre = self.apply_affine(
                b,
                pre[setup_b],
            )

            # Independent rigid fit for this edge.
            direct_matrix = self.fit_rigid_transform(
                a_pre,
                b_pre,
            )

            edge_models[(setup_a, setup_b)] = {
                "matrix": direct_matrix,
                "source_anchor": a_pre.mean(axis=0),
                "target_anchor": b_pre.mean(axis=0),
            }

            a_direct = self.apply_affine(
                a_pre,
                direct_matrix,
            )

            direct_delta = b_pre - a_direct
            direct_distance = np.linalg.norm(
                direct_delta,
                axis=1,
            )
            direct_summary = self.summary(
                direct_distance,
            )

            # Residual produced by the global rigid solution.
            a_post = self.apply_affine(
                a,
                post[setup_a],
            )
            b_post = self.apply_affine(
                b,
                post[setup_b],
            )

            global_delta = b_post - a_post
            global_distance = np.linalg.norm(
                global_delta,
                axis=1,
            )
            global_summary = self.summary(
                global_distance,
            )

            midpoint_post = 0.5 * (
                a_post + b_post
            )

            spatial = self.grid_metrics(
                midpoint_post,
                global_delta,
                global_distance,
            )

            centered_midpoints = (
                midpoint_post
                - midpoint_post.mean(axis=0)
            )

            geometry_singular_values = np.linalg.svd(
                centered_midpoints,
                compute_uv=False,
            )

            largest_geometry_scale = max(
                float(geometry_singular_values[0]),
                1e-12,
            )

            geometry_middle_ratio = float(
                geometry_singular_values[1]
                / largest_geometry_scale
            )
            geometry_smallest_ratio = float(
                geometry_singular_values[2]
                / largest_geometry_scale
            )

            rows.append(
                {
                    "setup_a": setup_a,
                    "setup_b": setup_b,
                    "raw_rows": raw_count,
                    "match_count": match_count,

                    "direct_mean": direct_summary["mean"],
                    "direct_median": direct_summary["median"],
                    "direct_p90": direct_summary["p90"],
                    "direct_p95": direct_summary["p95"],
                    "direct_max": direct_summary["max"],

                    "global_mean": global_summary["mean"],
                    "global_median": global_summary["median"],
                    "global_p90": global_summary["p90"],
                    "global_p95": global_summary["p95"],
                    "global_max": global_summary["max"],

                    "mean_global_degradation": (
                        global_summary["mean"]
                        - direct_summary["mean"]
                    ),
                    "p95_global_degradation": (
                        global_summary["p95"]
                        - direct_summary["p95"]
                    ),

                    "span_x": spatial["span"][0],
                    "span_y": spatial["span"][1],
                    "span_z": spatial["span"][2],
                    "occupancy": spatial["occupancy"],
                    "occupied_cells": spatial["occupied_cells"],
                    "supported_cells": spatial["supported_cells"],
                    "max_cell_fraction": spatial["max_cell_fraction"],

                    "cell_vector_dispersion_mean": spatial[
                        "cell_vector_dispersion_mean"
                    ],
                    "cell_vector_dispersion_p90": spatial[
                        "cell_vector_dispersion_p90"
                    ],
                    "cell_vector_dispersion_max": spatial[
                        "cell_vector_dispersion_max"
                    ],

                    "geometry_middle_ratio": geometry_middle_ratio,
                    "geometry_smallest_ratio": geometry_smallest_ratio,
                }
            )

        return pd.DataFrame(rows), edge_models


    def rigid_edge_key(self, setup_a, setup_b):
        return tuple(
            sorted(
                (
                    int(setup_a),
                    int(setup_b),
                )
            )
        )


    def directed_rigid_edge_model(
        self,
        edge_models,
        source_setup,
        target_setup,
    ):
        """
        Return an independently fitted rigid edge in the requested direction.

        The stored model always maps the lower setup ID to the higher setup ID.
        Reverse traversal returns the inverse transform.
        """
        source_setup = int(source_setup)
        target_setup = int(target_setup)

        key = self.rigid_edge_key(
            source_setup,
            target_setup,
        )

        if key not in edge_models:
            raise KeyError(
                f"No fitted rigid model exists for edge {key}"
            )

        model = edge_models[key]

        if source_setup == key[0]:
            return {
                "matrix": model["matrix"],
                "source_anchor": model["source_anchor"],
                "target_anchor": model["target_anchor"],
            }

        return {
            "matrix": np.linalg.inv(model["matrix"]),
            "source_anchor": model["target_anchor"],
            "target_anchor": model["source_anchor"],
        }

    def compose_rigid_path(
        self,
        edge_models,
        path,
    ):
        """
        Compose independently fitted edge transforms along a directed path.

        For:

            A -> B -> C

        the result is:

            T_A_to_C = T_B_to_C @ T_A_to_B
        """
        composed = np.eye(
            4,
            dtype=np.float64,
        )

        for source_setup, target_setup in zip(
            path[:-1],
            path[1:],
        ):
            edge_model = self.directed_rigid_edge_model(
                edge_models=edge_models,
                source_setup=source_setup,
                target_setup=target_setup,
            )

            composed = (
                edge_model["matrix"]
                @ composed
            )

        return composed


    def rigid_rotation_disagreement(
        self,
        direct_matrix,
        alternate_matrix,
    ):
        """
        Return rotational disagreement between two rigid transforms in degrees.
        """
        relative_linear = (
            direct_matrix[:3, :3].T
            @ alternate_matrix[:3, :3]
        )

        u, _, vt = np.linalg.svd(
            relative_linear
        )
        rotation = u @ vt

        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt

        trace_value = np.clip(
            (np.trace(rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        )

        return float(
            np.degrees(
                np.arccos(trace_value)
            )
        )


    def rigid_anchor_disagreement(
        self,
        direct_matrix,
        alternate_matrix,
        source_anchor,
    ):
        """
        Measure translation disagreement at the direct edge's source-overlap
        centroid.

        Comparing the transformed anchor is more meaningful than comparing raw
        matrix translations because the transforms rotate around the global origin.
        """
        source_anchor = np.asarray(
            source_anchor,
            dtype=np.float64,
        ).reshape(1, 3)

        direct_position = self.apply_affine(
            source_anchor,
            direct_matrix,
        )[0]

        alternate_position = self.apply_affine(
            source_anchor,
            alternate_matrix,
        )[0]

        return float(
            np.linalg.norm(
                alternate_position
                - direct_position
            )
        )


    def find_alternate_rigid_paths(
        self,
        edge_models,
        source_setup,
        target_setup,
        max_path_edges=6,
        max_paths_per_edge=4,
    ):
        """
        Find short edge-disjoint alternate paths while excluding the direct edge.
        """
        source_setup = int(source_setup)
        target_setup = int(target_setup)

        excluded_edge = self.rigid_edge_key(
            source_setup,
            target_setup,
        )

        adjacency = {}

        for setup_a, setup_b in edge_models:
            edge = self.rigid_edge_key(
                setup_a,
                setup_b,
            )

            if edge == excluded_edge:
                continue

            adjacency.setdefault(
                setup_a,
                set(),
            ).add(setup_b)

            adjacency.setdefault(
                setup_b,
                set(),
            ).add(setup_a)

        candidate_paths = []

        def search(current_setup, path):
            edge_count = len(path) - 1

            if edge_count > max_path_edges:
                return

            if current_setup == target_setup:
                candidate_paths.append(
                    path.copy()
                )
                return

            for neighbor in sorted(
                adjacency.get(
                    current_setup,
                    set(),
                )
            ):
                if neighbor in path:
                    continue

                search(
                    neighbor,
                    path + [neighbor],
                )

        search(
            source_setup,
            [source_setup],
        )

        candidate_paths.sort(
            key=lambda path: (
                len(path),
                path,
            )
        )

        # Prefer alternate paths that do not reuse one another's edges.
        selected_paths = []
        used_edges = set()

        for path in candidate_paths:
            path_edges = {
                self.rigid_edge_key(
                    path[index],
                    path[index + 1],
                )
                for index in range(
                    len(path) - 1
                )
            }

            if not path_edges.isdisjoint(
                used_edges
            ):
                continue

            selected_paths.append(
                path
            )
            used_edges.update(
                path_edges
            )

            if len(selected_paths) >= max_paths_per_edge:
                break

        return selected_paths


    def collect_rigid_loop_closure_diagnostics(
        self,
        edge_table,
        edge_models,
        max_path_edges=6,
        max_paths_per_edge=4,
        rotation_warning_degrees=0.10,
        translation_warning_px=None,
    ):
        """
        Compare every direct fitted rigid edge against independent alternate paths.

        The alternate path and direct edge both map setup_a -> setup_b. Their
        rotational difference and source-anchor displacement measure loop closure.
        """
        if translation_warning_px is None:
            translation_warning_px = max(
                self.metric_downsample_xyz
            )

        loop_rows = []
        edge_rows = []

        edge_metrics = {
            self.rigid_edge_key(
                row.setup_a,
                row.setup_b,
            ): row
            for row in edge_table.itertuples(
                index=False
            )
        }

        for setup_a, setup_b in sorted(
            edge_models
        ):
            direct_model = self.directed_rigid_edge_model(
                edge_models=edge_models,
                source_setup=setup_a,
                target_setup=setup_b,
            )

            direct_matrix = direct_model["matrix"]
            source_anchor = direct_model["source_anchor"]

            alternate_paths = self.find_alternate_rigid_paths(
                edge_models=edge_models,
                source_setup=setup_a,
                target_setup=setup_b,
                max_path_edges=max_path_edges,
                max_paths_per_edge=max_paths_per_edge,
            )

            rotation_errors = []
            translation_errors = []
            disagreement_count = 0

            for path_index, path in enumerate(
                alternate_paths,
                start=1,
            ):
                alternate_matrix = self.compose_rigid_path(
                    edge_models=edge_models,
                    path=path,
                )

                rotation_error_degrees = (
                    self.rigid_rotation_disagreement(
                        direct_matrix=direct_matrix,
                        alternate_matrix=alternate_matrix,
                    )
                )

                translation_error_px = (
                    self.rigid_anchor_disagreement(
                        direct_matrix=direct_matrix,
                        alternate_matrix=alternate_matrix,
                        source_anchor=source_anchor,
                    )
                )

                disagrees = bool(
                    rotation_error_degrees
                    > rotation_warning_degrees
                    or translation_error_px
                    > translation_warning_px
                )

                if disagrees:
                    disagreement_count += 1

                rotation_errors.append(
                    rotation_error_degrees
                )
                translation_errors.append(
                    translation_error_px
                )

                loop_rows.append(
                    {
                        "setup_a": setup_a,
                        "setup_b": setup_b,
                        "path_index": path_index,
                        "alternate_path": "->".join(
                            str(setup)
                            for setup in path
                        ),
                        "path_edge_count": len(path) - 1,
                        "rotation_closure_degrees": (
                            rotation_error_degrees
                        ),
                        "translation_closure_px": (
                            translation_error_px
                        ),
                        "disagrees": disagrees,
                    }
                )

            metrics_row = edge_metrics.get(
                self.rigid_edge_key(
                    setup_a,
                    setup_b,
                )
            )

            edge_rows.append(
                {
                    "setup_a": setup_a,
                    "setup_b": setup_b,
                    "match_count": (
                        int(metrics_row.match_count)
                        if metrics_row is not None
                        else 0
                    ),
                    "direct_mean": (
                        float(metrics_row.direct_mean)
                        if metrics_row is not None
                        else np.nan
                    ),
                    "global_mean": (
                        float(metrics_row.global_mean)
                        if metrics_row is not None
                        else np.nan
                    ),
                    "global_p95": (
                        float(metrics_row.global_p95)
                        if metrics_row is not None
                        else np.nan
                    ),
                    "alternate_path_count": len(
                        alternate_paths
                    ),
                    "disagreeing_path_count": (
                        disagreement_count
                    ),
                    "disagreement_fraction": (
                        disagreement_count
                        / len(alternate_paths)
                        if alternate_paths
                        else np.nan
                    ),
                    "rotation_closure_median_degrees": (
                        float(np.median(rotation_errors))
                        if rotation_errors
                        else np.nan
                    ),
                    "rotation_closure_max_degrees": (
                        float(np.max(rotation_errors))
                        if rotation_errors
                        else np.nan
                    ),
                    "translation_closure_median_px": (
                        float(np.median(translation_errors))
                        if translation_errors
                        else np.nan
                    ),
                    "translation_closure_max_px": (
                        float(np.max(translation_errors))
                        if translation_errors
                        else np.nan
                    ),
                    "rotation_warning_degrees": (
                        rotation_warning_degrees
                    ),
                    "translation_warning_px": (
                        translation_warning_px
                    ),
                }
            )

        return (
            pd.DataFrame(loop_rows),
            pd.DataFrame(edge_rows),
        )


    def print_rigid_loop_closure_diagnostics(
        self,
        loop_table,
        edge_consistency_table,
    ):
        print("\n" + "#" * 120)
        print("RIGID LOOP-CLOSURE / ALTERNATE-PATH DIAGNOSTICS")
        print("#" * 120)

        if len(edge_consistency_table) == 0:
            print("\nNo rigid edge models were available.")
            return

        first_row = edge_consistency_table.iloc[0]

        print("\nDiagnostic warning thresholds:")
        print(
            "  rotation:    "
            f"{first_row['rotation_warning_degrees']:.4f} degrees"
        )
        print(
            "  translation: "
            f"{first_row['translation_warning_px']:.3f} full-res px"
        )

        edge_columns = [
            "setup_a",
            "setup_b",
            "match_count",
            "direct_mean",
            "global_mean",
            "global_p95",
            "alternate_path_count",
            "disagreeing_path_count",
            "disagreement_fraction",
            "rotation_closure_max_degrees",
            "translation_closure_max_px",
        ]

        ordered_edges = edge_consistency_table.sort_values(
            [
                "disagreeing_path_count",
                "translation_closure_max_px",
                "rotation_closure_max_degrees",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            na_position="last",
        )

        print("\nPer-edge agreement with independent alternate paths:")
        print(
            ordered_edges[
                edge_columns
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

        if len(loop_table) == 0:
            print(
                "\nNo alternate paths were available. "
                "All positive edges are bridges or outside cycles."
            )
            return

        loop_columns = [
            "setup_a",
            "setup_b",
            "alternate_path",
            "path_edge_count",
            "rotation_closure_degrees",
            "translation_closure_px",
            "disagrees",
        ]

        ordered_loops = loop_table.sort_values(
            [
                "disagrees",
                "translation_closure_px",
                "rotation_closure_degrees",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )

        print("\nIndividual alternate-path comparisons, worst first:")
        print(
            ordered_loops[
                loop_columns
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )
    
    def rigid_graph_topology(
        self,
        edge_table,
        setups,
    ):
        """
        Build graph diagnostics from saved positive-match edges.

        Returns connected components, bridges, articulation setups,
        and per-setup degree/support.
        """
        nodes = set(int(setup) for setup in setups)
        adjacency = {
            node: set()
            for node in nodes
        }
        edge_match_counts = {}

        for row in edge_table.itertuples(index=False):
            setup_a = int(row.setup_a)
            setup_b = int(row.setup_b)

            nodes.add(setup_a)
            nodes.add(setup_b)

            adjacency.setdefault(
                setup_a,
                set(),
            ).add(setup_b)

            adjacency.setdefault(
                setup_b,
                set(),
            ).add(setup_a)

            edge_match_counts[
                tuple(sorted((setup_a, setup_b)))
            ] = int(row.match_count)

        for node in nodes:
            adjacency.setdefault(
                node,
                set(),
            )

        # Connected components.
        components = []
        visited = set()

        for start in sorted(nodes):
            if start in visited:
                continue

            component = []
            stack = [start]
            visited.add(start)

            while stack:
                current = stack.pop()
                component.append(current)

                for neighbor in sorted(
                    adjacency[current]
                ):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            components.append(
                sorted(component)
            )

        # Tarjan articulation-point and bridge search.
        discovery = {}
        low = {}
        parent = {}
        articulation_points = set()
        bridges = []
        timer = [0]

        def visit(node):
            discovery[node] = timer[0]
            low[node] = timer[0]
            timer[0] += 1

            child_count = 0

            for neighbor in sorted(
                adjacency[node]
            ):
                if neighbor not in discovery:
                    parent[neighbor] = node
                    child_count += 1

                    visit(neighbor)

                    low[node] = min(
                        low[node],
                        low[neighbor],
                    )

                    is_root = node not in parent

                    if (
                        is_root
                        and child_count > 1
                    ):
                        articulation_points.add(
                            node
                        )

                    if (
                        not is_root
                        and low[neighbor]
                        >= discovery[node]
                    ):
                        articulation_points.add(
                            node
                        )

                    if (
                        low[neighbor]
                        > discovery[node]
                    ):
                        bridges.append(
                            tuple(
                                sorted(
                                    (node, neighbor)
                                )
                            )
                        )

                elif parent.get(node) != neighbor:
                    low[node] = min(
                        low[node],
                        discovery[neighbor],
                    )

        for node in sorted(nodes):
            if node not in discovery:
                visit(node)

        setup_rows = []

        for setup in sorted(nodes):
            neighbors = sorted(
                adjacency[setup]
            )

            total_matches = sum(
                edge_match_counts.get(
                    tuple(
                        sorted(
                            (setup, neighbor)
                        )
                    ),
                    0,
                )
                for neighbor in neighbors
            )

            setup_rows.append(
                {
                    "setup": setup,
                    "degree": len(neighbors),
                    "total_edge_matches": total_matches,
                    "neighbors": ",".join(
                        str(neighbor)
                        for neighbor in neighbors
                    ),
                    "articulation": (
                        setup
                        in articulation_points
                    ),
                    "isolated": (
                        len(neighbors) == 0
                    ),
                }
            )

        return {
            "components": sorted(
                components,
                key=lambda component: (
                    -len(component),
                    component,
                ),
            ),
            "bridges": sorted(
                set(bridges)
            ),
            "articulation_points": sorted(
                articulation_points
            ),
            "setup_table": pd.DataFrame(
                setup_rows
            ),
        }
    
    def print_rigid_edge_diagnostics(
        self,
        edge_table,
        graph,
    ):
        print("\n" + "#" * 120)
        print("ALL-EDGE RIGID DIAGNOSTICS")
        print("#" * 120)

        if len(edge_table) == 0:
            print(
                "No saved rigid edges were available."
            )
            return

        columns = [
            "setup_a",
            "setup_b",
            "match_count",
            "direct_mean",
            "direct_p95",
            "global_mean",
            "global_p95",
            "mean_global_degradation",
            "p95_global_degradation",
            "occupancy",
            "max_cell_fraction",
            "cell_vector_dispersion_p90",
            "geometry_smallest_ratio",
        ]

        worst_global = edge_table.sort_values(
            [
                "global_p95",
                "global_mean",
            ],
            ascending=False,
        )

        print(
            "\nEdges ordered by worst POST-global p95:"
        )
        print(
            worst_global[columns].to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.4f}"
                ),
            )
        )

        worst_degradation = edge_table.sort_values(
            [
                "p95_global_degradation",
                "mean_global_degradation",
            ],
            ascending=False,
        )

        print(
            "\nEdges most degraded by the global graph "
            "relative to their independent rigid fit:"
        )
        print(
            worst_degradation[columns]
            .head(10)
            .to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.4f}"
                ),
            )
        )

        if len(edge_table) >= 3:
            correlation_columns = [
                "match_count",
                "global_mean",
                "global_p95",
                "mean_global_degradation",
                "p95_global_degradation",
                "occupancy",
                "cell_vector_dispersion_p90",
                "geometry_smallest_ratio",
            ]

            correlations = edge_table[
                correlation_columns
            ].corr(
                method="spearman"
            )

            print(
                "\nSpearman correlation against match count:"
            )

            for column in correlation_columns[1:]:
                correlation = correlations.loc[
                    "match_count",
                    column,
                ]

                print(
                    f"  match_count vs {column:<30} "
                    f"{correlation: .4f}"
                )

        print("\nPositive-edge graph:")

        print(
            f"  connected components: "
            f"{len(graph['components'])}"
        )

        for component_index, component in enumerate(
            graph["components"],
            start=1,
        ):
            print(
                f"    component {component_index}: "
                f"{component}"
            )

        print(
            f"  bridge edges: "
            f"{graph['bridges']}"
        )
        print(
            f"  articulation setups: "
            f"{graph['articulation_points']}"
        )

        print(
            "\nPer-setup connectivity:"
        )
        print(
            graph["setup_table"].to_string(
                index=False,
            )
        )
    
    def print_affine_all_edge_diagnostics(
        self,
        edge_table,
        tile_table,
        graph,
    ):
        print("\n" + "#" * 140)
        print("ALL-EDGE AFFINE DIAGNOSTICS")
        print("#" * 140)

        if len(edge_table) == 0:
            print(
                "No saved affine edges were available."
            )
            return

        main_columns = [
            "setup_a",
            "setup_b",
            "match_count",
            "parent_rigid_mean",
            "global_affine_mean",
            "global_affine_p95",
            "mean_reduction",
            "p95_reduction",
            "improved_fraction",
            "worsened_fraction",
            "direct_affine_mean",
            "direct_affine_p95",
            "mean_global_degradation",
            "p95_global_degradation",
            "geometry_smallest_ratio",
        ]

        worst_global = edge_table.sort_values(
            [
                "global_affine_p95",
                "global_affine_mean",
            ],
            ascending=False,
        )

        print(
            "\nEdges ordered by worst global AFFINE p95:"
        )
        print(
            worst_global[
                main_columns
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

        most_improved = edge_table.sort_values(
            [
                "p95_reduction",
                "mean_reduction",
            ],
            ascending=False,
        )

        print(
            "\nEdges most improved from parent RIGID to global AFFINE:"
        )
        print(
            most_improved[
                main_columns
            ].head(10).to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

        worsened = edge_table[
            (
                edge_table["mean_reduction"] < 0.0
            )
            | (
                edge_table["p95_reduction"] < 0.0
            )
        ].sort_values(
            [
                "p95_reduction",
                "mean_reduction",
            ],
            ascending=True,
        )

        print(
            "\nEdges worsened from parent RIGID to global AFFINE:"
        )

        if len(worsened):
            print(
                worsened[
                    main_columns
                ].to_string(
                    index=False,
                    float_format=lambda value: f"{value:.4f}",
                )
            )
        else:
            print(
                "  None."
            )

        compromised_columns = [
            "setup_a",
            "setup_b",
            "match_count",
            "direct_affine_mean",
            "direct_affine_p95",
            "global_affine_mean",
            "global_affine_p95",
            "mean_global_degradation",
            "p95_global_degradation",
            "geometry_smallest_ratio",
            "direct_affine_condition",
            "direct_affine_scale_deviation",
            "direct_affine_shear",
        ]

        most_compromised = edge_table.sort_values(
            [
                "p95_global_degradation",
                "mean_global_degradation",
            ],
            ascending=False,
        )

        print(
            "\nEdges most degraded by the global graph relative "
            "to their independent affine training fit:"
        )
        print(
            most_compromised[
                compromised_columns
            ].head(10).to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

        headroom_columns = [
            "setup_a",
            "setup_b",
            "match_count",
            "global_affine_mean",
            "global_affine_p95",
            "residual_affine_mean",
            "residual_affine_p95",
            "extra_affine_mean_reduction",
            "extra_affine_p95_reduction",
            "residual_affine_condition",
            "residual_affine_scale_deviation",
            "residual_affine_shear",
            "geometry_smallest_ratio",
        ]

        remaining_headroom = edge_table.sort_values(
            [
                "extra_affine_p95_reduction",
                "extra_affine_mean_reduction",
            ],
            ascending=False,
        )

        print(
            "\nLargest remaining affine training-fit headroom "
            "after the global affine solve:"
        )
        print(
            remaining_headroom[
                headroom_columns
            ].head(10).to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

        print(
            "\nAffine positive-edge graph:"
        )
        print(
            f"  connected components: "
            f"{len(graph['components'])}"
        )

        for component_index, component in enumerate(
            graph["components"],
            start=1,
        ):
            print(
                f"    component {component_index}: "
                f"{component}"
            )

        print(
            f"  bridge edges: "
            f"{graph['bridges']}"
        )
        print(
            f"  articulation setups: "
            f"{graph['articulation_points']}"
        )

        print("\nPer-tile affine updates:")

        if len(tile_table) == 0:
            print(
                "  No tile transforms were available."
            )
            return

        tile_columns = [
            "setup",
            "rotation_degrees",
            "scale_1",
            "scale_2",
            "scale_3",
            "max_scale_deviation",
            "shear_magnitude",
            "condition_number",
            "translation_norm",
            "determinant",
            "reflection",
        ]

        ordered_tiles = tile_table.sort_values(
            [
                "max_scale_deviation",
                "shear_magnitude",
                "condition_number",
            ],
            ascending=False,
        )

        print(
            ordered_tiles[
                tile_columns
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            )
        )
    
    def collect_affine_edge_diagnostics(
        self,
        manifest,
        index,
        pre,
        post,
    ):
        """
        Diagnose every saved affine edge.

        For each edge, compare:

            parent rigid placement
            global affine placement
            best independent affine fit
            remaining affine fit after the global affine solve

        The independent and residual affine fits are evaluated on their fitting
        matches, so they represent training floors rather than held-out validation.
        """
        rows = []

        for setup_a, setup_b in self.saved_pairs(index):
            if (
                setup_a not in pre
                or setup_b not in pre
                or setup_a not in post
                or setup_b not in post
            ):
                continue

            a_raw, b_raw = self.read_pair(
                manifest,
                index,
                setup_a,
                setup_b,
            )

            raw_count = len(a_raw)

            a, b = self.dedupe_pairs(
                a_raw,
                b_raw,
            )

            match_count = len(a)

            if match_count < 4:
                continue

            # Parent rigid placement.
            a_pre = self.apply_affine(
                a,
                pre[setup_a],
            )
            b_pre = self.apply_affine(
                b,
                pre[setup_b],
            )

            # Solved global affine placement.
            a_post = self.apply_affine(
                a,
                post[setup_a],
            )
            b_post = self.apply_affine(
                b,
                post[setup_b],
            )

            progression = self.calculate_match_metrics_from_transformed(
                a_pre,
                b_pre,
                a_post,
                b_post,
            )

            # Best independent affine fit for this edge at the parent rigid stage.
            direct_matrix, direct_rank = self.fit_affine(
                a_pre,
                b_pre,
            )

            a_direct = self.apply_affine(
                a_pre,
                direct_matrix,
            )

            direct_distance = np.linalg.norm(
                b_pre - a_direct,
                axis=1,
            )

            direct_summary = self.summary(
                direct_distance
            )

            direct_geometry = self.linear_geometry(
                direct_matrix
            )

            # Remaining independent affine fit after the global affine solve.
            residual_matrix, residual_rank = self.fit_affine(
                a_post,
                b_post,
            )

            a_residual = self.apply_affine(
                a_post,
                residual_matrix,
            )

            residual_distance = np.linalg.norm(
                b_post - a_residual,
                axis=1,
            )

            residual_summary = self.summary(
                residual_distance
            )

            residual_geometry = self.linear_geometry(
                residual_matrix
            )

            # Spatial conditioning of the actual overlap.
            midpoint_post = 0.5 * (
                a_post + b_post
            )

            centered_midpoints = (
                midpoint_post
                - midpoint_post.mean(axis=0)
            )

            geometry_singular_values = np.linalg.svd(
                centered_midpoints,
                compute_uv=False,
            )

            largest_geometry_scale = max(
                float(geometry_singular_values[0]),
                1e-12,
            )

            geometry_middle_ratio = float(
                geometry_singular_values[1]
                / largest_geometry_scale
            )

            geometry_smallest_ratio = float(
                geometry_singular_values[2]
                / largest_geometry_scale
            )

            parent_mean = progression["pre"]["mean"]
            parent_p95 = progression["pre"]["p95"]

            affine_mean = progression["post"]["mean"]
            affine_p95 = progression["post"]["p95"]

            rows.append(
                {
                    "setup_a": setup_a,
                    "setup_b": setup_b,
                    "raw_rows": raw_count,
                    "match_count": match_count,

                    "parent_rigid_mean": parent_mean,
                    "parent_rigid_median": progression["pre"]["median"],
                    "parent_rigid_p90": progression["pre"]["p90"],
                    "parent_rigid_p95": parent_p95,

                    "global_affine_mean": affine_mean,
                    "global_affine_median": progression["post"]["median"],
                    "global_affine_p90": progression["post"]["p90"],
                    "global_affine_p95": affine_p95,

                    "mean_reduction": (
                        parent_mean - affine_mean
                    ),
                    "p95_reduction": (
                        parent_p95 - affine_p95
                    ),
                    "mean_reduction_pct": self.percent_reduction(
                        parent_mean,
                        affine_mean,
                    ),
                    "p95_reduction_pct": self.percent_reduction(
                        parent_p95,
                        affine_p95,
                    ),

                    "improved_fraction": (
                        progression["improved"]
                        / match_count
                    ),
                    "worsened_fraction": (
                        progression["worsened"]
                        / match_count
                    ),

                    # Best independent affine training fit.
                    "direct_affine_mean": direct_summary["mean"],
                    "direct_affine_median": direct_summary["median"],
                    "direct_affine_p90": direct_summary["p90"],
                    "direct_affine_p95": direct_summary["p95"],

                    "mean_global_degradation": (
                        affine_mean
                        - direct_summary["mean"]
                    ),
                    "p95_global_degradation": (
                        affine_p95
                        - direct_summary["p95"]
                    ),

                    # Remaining affine training headroom after global solve.
                    "residual_affine_mean": residual_summary["mean"],
                    "residual_affine_median": residual_summary["median"],
                    "residual_affine_p90": residual_summary["p90"],
                    "residual_affine_p95": residual_summary["p95"],

                    "extra_affine_mean_reduction": (
                        affine_mean
                        - residual_summary["mean"]
                    ),
                    "extra_affine_p95_reduction": (
                        affine_p95
                        - residual_summary["p95"]
                    ),
                    "extra_affine_mean_reduction_pct": self.percent_reduction(
                        affine_mean,
                        residual_summary["mean"],
                    ),
                    "extra_affine_p95_reduction_pct": self.percent_reduction(
                        affine_p95,
                        residual_summary["p95"],
                    ),

                    "direct_affine_rank": direct_rank,
                    "direct_affine_condition": direct_geometry[
                        "condition_number"
                    ],
                    "direct_affine_scale_deviation": direct_geometry[
                        "max_scale_deviation"
                    ],
                    "direct_affine_shear": direct_geometry[
                        "shear_magnitude"
                    ],

                    "residual_affine_rank": residual_rank,
                    "residual_affine_condition": residual_geometry[
                        "condition_number"
                    ],
                    "residual_affine_scale_deviation": residual_geometry[
                        "max_scale_deviation"
                    ],
                    "residual_affine_shear": residual_geometry[
                        "shear_magnitude"
                    ],
                    "residual_affine_translation": residual_geometry[
                        "translation_norm"
                    ],

                    "occupancy": progression["occupancy"],
                    "max_cell_fraction": progression[
                        "max_cell_fraction"
                    ],
                    "cell_vector_dispersion_p90": progression[
                        "cell_vector_dispersion_p90"
                    ],

                    "geometry_middle_ratio": geometry_middle_ratio,
                    "geometry_smallest_ratio": geometry_smallest_ratio,
                }
            )

        return pd.DataFrame(rows)


    def collect_affine_tile_transform_diagnostics(
        self,
        pre,
        post,
    ):
        """
        Diagnose each tile's global affine update relative to its parent rigid
        transform.

            update = post @ inverse(pre)
        """
        rows = []

        for setup in sorted(
            set(pre) & set(post)
        ):
            update_matrix = (
                post[setup]
                @ np.linalg.inv(pre[setup])
            )

            geometry = self.linear_geometry(
                update_matrix
            )

            singular_values = geometry[
                "singular_values"
            ]

            rows.append(
                {
                    "setup": setup,
                    "rotation_degrees": geometry[
                        "rotation_degrees"
                    ],
                    "scale_1": singular_values[0],
                    "scale_2": singular_values[1],
                    "scale_3": singular_values[2],
                    "max_scale_deviation": geometry[
                        "max_scale_deviation"
                    ],
                    "shear_magnitude": geometry[
                        "shear_magnitude"
                    ],
                    "condition_number": geometry[
                        "condition_number"
                    ],
                    "linear_identity_error": geometry[
                        "linear_identity_error"
                    ],
                    "translation_norm": geometry[
                        "translation_norm"
                    ],
                    "translation_scaled_norm": geometry[
                        "translation_scaled_norm"
                    ],
                    "determinant": geometry[
                        "determinant"
                    ],
                    "reflection": geometry[
                        "reflection"
                    ],
                    "finite": geometry[
                        "finite"
                    ],
                }
            )

        return pd.DataFrame(rows)

    def grid_metrics(self, points, deltas, distances):
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        spans = maxs - mins
        safe_spans = np.where(spans > 0.0, spans, 1.0)
        bins_z, bins_y, bins_x = self.grid_bins_zyx
        bins_xyz = np.array([bins_x, bins_y, bins_z], dtype=np.int64)
        normalized = (points - mins) / safe_spans
        cell_xyz = np.floor(normalized * bins_xyz).astype(np.int64)
        cell_xyz = np.clip(cell_xyz, 0, bins_xyz - 1)
        cell_ids = cell_xyz[:, 2] * bins_y * bins_x + cell_xyz[:, 1] * bins_x + cell_xyz[:, 0]
        unique_ids, counts = np.unique(cell_ids, return_counts=True)
        total_cells = bins_x * bins_y * bins_z
        supported_cell_median_distances = []
        supported_cell_median_vectors = []
        supported_cell_counts = []
        for cell_id, cell_count in zip(unique_ids, counts):
            mask = cell_ids == cell_id
            if cell_count < self.min_matches_per_grid_cell:
                continue
            supported_cell_median_distances.append(float(np.median(distances[mask])))
            supported_cell_median_vectors.append(np.median(deltas[mask], axis=0))
            supported_cell_counts.append(int(cell_count))
        if supported_cell_median_distances:
            cell_distances = np.asarray(supported_cell_median_distances, dtype=np.float64)
            cell_vectors = np.asarray(supported_cell_median_vectors, dtype=np.float64)
            cell_counts = np.asarray(supported_cell_counts, dtype=np.float64)
            global_median_vector = np.median(deltas, axis=0)
            vector_offsets = np.linalg.norm(cell_vectors - global_median_vector, axis=1)
            cell_distance_summary = self.summary(cell_distances)
            vector_dispersion_mean = float(np.average(vector_offsets, weights=cell_counts))
            vector_dispersion_p90 = float(np.percentile(vector_offsets, 90))
            vector_dispersion_max = float(np.max(vector_offsets))
        else:
            cell_distance_summary = {
                "mean": np.nan,
                "median": np.nan,
                "p90": np.nan,
                "p95": np.nan,
                "max": np.nan,
            }
            vector_dispersion_mean = np.nan
            vector_dispersion_p90 = np.nan
            vector_dispersion_max = np.nan
        return {
            "span": spans,
            "occupancy": float(len(unique_ids) / total_cells),
            "max_cell_fraction": float(np.max(counts) / len(points)),
            "occupied_cells": int(len(unique_ids)),
            "supported_cells": int(len(supported_cell_median_distances)),
            "cell_median_distance": cell_distance_summary,
            "cell_vector_dispersion_mean": vector_dispersion_mean,
            "cell_vector_dispersion_p90": vector_dispersion_p90,
            "cell_vector_dispersion_max": vector_dispersion_max,
        }

    def calculate_match_metrics_from_transformed(self, a_pre, b_pre, a_post, b_post):
        """
        Calculate the same metrics from already-transformed world points.

        This is used by split-affine so many split setup pairs can be
        combined without pretending they all share one transform.
        """
        pre_delta = b_pre - a_pre
        post_delta = b_post - a_post
        pre_distance = np.linalg.norm(pre_delta, axis=1)
        post_distance = np.linalg.norm(post_delta, axis=1)
        scale = np.asarray(self.metric_downsample_xyz, dtype=np.float64)
        pre_scaled_distance = np.linalg.norm(pre_delta / scale, axis=1)
        post_scaled_distance = np.linalg.norm(post_delta / scale, axis=1)
        reduction = pre_distance - post_distance
        reduction_pct = np.divide(
            100.0 * reduction,
            pre_distance,
            out=np.full_like(reduction, np.nan),
            where=pre_distance > self.percent_baseline_epsilon,
        )
        midpoints_post = 0.5 * (a_post + b_post)
        distance_rows = pd.DataFrame(
            {
                "match_i": np.arange(len(a_pre), dtype=np.int64),
                "distance_pre_fullres_px": pre_distance,
                "distance_post_fullres_px": post_distance,
                "distance_reduction_fullres_px": reduction,
                "distance_reduction_percent": reduction_pct,
                "distance_pre_scaled_px": pre_scaled_distance,
                "distance_post_scaled_px": post_scaled_distance,
                "dx_post_full": post_delta[:, 0],
                "dy_post_full": post_delta[:, 1],
                "dz_post_full": post_delta[:, 2],
            }
        )
        return {
            "a_pre": a_pre,
            "b_pre": b_pre,
            "a_post": a_post,
            "b_post": b_post,
            "pre_delta": pre_delta,
            "post_delta": post_delta,
            "pre_distance": pre_distance,
            "post_distance": post_distance,
            "pre_scaled_distance": pre_scaled_distance,
            "post_scaled_distance": post_scaled_distance,
            "pre": self.summary(pre_distance),
            "post": self.summary(post_distance),
            "pre_scaled": self.summary(pre_scaled_distance),
            "post_scaled": self.summary(post_scaled_distance),
            "reduction": self.summary(reduction),
            "median_reduction_pct": float(np.nanmedian(reduction_pct)),
            "improved": int(np.count_nonzero(reduction > 1e-06)),
            "worsened": int(np.count_nonzero(reduction < -1e-06)),
            "unchanged": int(np.count_nonzero(np.abs(reduction) <= 1e-06)),
            "pre_signed": pre_delta.mean(axis=0),
            "post_signed": post_delta.mean(axis=0),
            "pre_abs": np.abs(pre_delta).mean(axis=0),
            "post_abs": np.abs(post_delta).mean(axis=0),
            "distance_rows": distance_rows,
            **self.grid_metrics(midpoints_post, post_delta, post_distance),
        }

    def validate_split_start(self, manifest, affine, split_start, split_definitions, label):
        errors = []
        rows = []
        for split_setup, definition in sorted(split_definitions.items()):
            old_setup = int(definition["old_setup"])
            source_min = np.asarray(definition["source_min_xyz"], dtype=np.float64)
            if old_setup not in affine:
                raise RuntimeError(f"Original setup {old_setup} missing from affine XML")
            if split_setup not in split_start:
                raise RuntimeError(f"Split setup {split_setup} missing from split XML transforms")
            split_local = self.read_points(manifest, split_setup, label)
            if len(split_local) == 0:
                continue
            original_local = split_local + source_min
            affine_world = self.apply_affine(original_local, affine[old_setup])
            split_world = self.apply_affine(split_local, split_start[split_setup])
            setup_errors = np.linalg.norm(split_world - affine_world, axis=1)
            errors.append(setup_errors)
            rows.append(
                {
                    "split_setup": split_setup,
                    "old_setup": old_setup,
                    "points": len(split_local),
                    "mean_error": float(np.mean(setup_errors)),
                    "median_error": float(np.median(setup_errors)),
                    "max_error": float(np.max(setup_errors)),
                }
            )
        if not errors:
            raise RuntimeError(f"No {label!r} points were available for validation")
        all_errors = np.concatenate(errors)
        print("\n" + "=" * 96)
        print("SPLIT START VALIDATION: PARENT AFFINE VS UNSOLVED SPLIT XML")
        print("=" * 96)
        print(f"points:  {len(all_errors):,}")
        print(f"mean:    {np.mean(all_errors):.12f}")
        print(f"median:  {np.median(all_errors):.12f}")
        print(f"p90:     {np.percentile(all_errors, 90):.12f}")
        print(f"p95:     {np.percentile(all_errors, 95):.12f}")
        print(f"max:     {np.max(all_errors):.12f}")
        worst = pd.DataFrame(rows).sort_values("max_error", ascending=False)
        print("\nWorst split setups:")
        print(worst.head(10).to_string(index=False))
        return (all_errors, worst)

    def calculate_match_metrics(self, a, b, pre_a, pre_b, post_a, post_b):
        return self.calculate_match_metrics_from_transformed(
            self.apply_affine(a, pre_a),
            self.apply_affine(b, pre_b),
            self.apply_affine(a, post_a),
            self.apply_affine(b, post_b),
        )

    def collect_split_real_progression_metrics(
        self, manifest, index, affine, split_post, split_definitions, label
    ):
        """
        Measure real cross-tile bead progression from parent AFFINE to SPLIT AFFINE.

        For the parent affine baseline:
            original_local = split_local + source_min_xyz
            world_affine   = affine[old_setup](original_local)

        For the split-affine result:
            world_split = split_post[split_setup](split_local)

        Same-parent split pairs are excluded because they are duplicate chunk copies
        from one original tile. Their affine distance is normally exactly zero, so
        they measure chunk continuity, not cross-tile alignment progression.

        Matches are globally deduplicated in original-tile coordinates so overlap
        copies do not count the same physical correspondence multiple times.
        """
        label = str(label)
        rows = []
        raw_count = 0
        pair_deduped_count = 0
        same_parent_excluded = 0
        split_pairs_used = set()
        original_pairs_used = set()
        for split_setup_a, split_setup_b in self.saved_pairs(index, label=label):
            if split_setup_a not in split_definitions:
                raise RuntimeError(f"Split setup {split_setup_a} is missing from SetupIds")
            if split_setup_b not in split_definitions:
                raise RuntimeError(f"Split setup {split_setup_b} is missing from SetupIds")
            if split_setup_a not in split_post or split_setup_b not in split_post:
                continue
            definition_a = split_definitions[split_setup_a]
            definition_b = split_definitions[split_setup_b]
            old_setup_a = int(definition_a["old_setup"])
            old_setup_b = int(definition_b["old_setup"])
            if old_setup_a not in affine or old_setup_b not in affine:
                raise RuntimeError(
                    f"Original setup missing from parent affine XML: split pair=({split_setup_a}, {split_setup_b}), original pair=({old_setup_a}, {old_setup_b})"
                )
            a_raw, b_raw = self.read_pair(
                manifest, index, split_setup_a, split_setup_b, label=label
            )
            raw_count += len(a_raw)
            a_local, b_local = self.dedupe_pairs(a_raw, b_raw)
            pair_deduped_count += len(a_local)
            if len(a_local) == 0:
                continue
            if old_setup_a == old_setup_b:
                same_parent_excluded += len(a_local)
                continue
            a_original = a_local + definition_a["source_min_xyz"]
            b_original = b_local + definition_b["source_min_xyz"]
            a_affine = self.apply_affine(a_original, affine[old_setup_a])
            b_affine = self.apply_affine(b_original, affine[old_setup_b])
            a_split = self.apply_affine(a_local, split_post[split_setup_a])
            b_split = self.apply_affine(b_local, split_post[split_setup_b])
            if old_setup_a > old_setup_b:
                old_setup_a, old_setup_b = (old_setup_b, old_setup_a)
                split_setup_a, split_setup_b = (split_setup_b, split_setup_a)
                a_original, b_original = (b_original, a_original)
                a_affine, b_affine = (b_affine, a_affine)
                a_split, b_split = (b_split, a_split)
            split_pairs_used.add((split_setup_a, split_setup_b))
            original_pairs_used.add((old_setup_a, old_setup_b))
            for i in range(len(a_local)):
                rows.append(
                    {
                        "label": label,
                        "split_setup_a": split_setup_a,
                        "split_setup_b": split_setup_b,
                        "original_setup_a": old_setup_a,
                        "original_setup_b": old_setup_b,
                        "a_original_x": a_original[i, 0],
                        "a_original_y": a_original[i, 1],
                        "a_original_z": a_original[i, 2],
                        "b_original_x": b_original[i, 0],
                        "b_original_y": b_original[i, 1],
                        "b_original_z": b_original[i, 2],
                        "a_affine_x": a_affine[i, 0],
                        "a_affine_y": a_affine[i, 1],
                        "a_affine_z": a_affine[i, 2],
                        "b_affine_x": b_affine[i, 0],
                        "b_affine_y": b_affine[i, 1],
                        "b_affine_z": b_affine[i, 2],
                        "a_split_x": a_split[i, 0],
                        "a_split_y": a_split[i, 1],
                        "a_split_z": a_split[i, 2],
                        "b_split_x": b_split[i, 0],
                        "b_split_y": b_split[i, 1],
                        "b_split_z": b_split[i, 2],
                    }
                )
        if not rows:
            return None
        table = pd.DataFrame(rows)
        dedupe_columns = [
            "original_setup_a",
            "original_setup_b",
            "a_original_x",
            "a_original_y",
            "a_original_z",
            "b_original_x",
            "b_original_y",
            "b_original_z",
        ]
        dedupe_keys = table[dedupe_columns].copy()
        coordinate_columns = dedupe_columns[2:]
        dedupe_keys[coordinate_columns] = dedupe_keys[coordinate_columns].round(
            self.actual_match_dedupe_decimals
        )
        keep = ~dedupe_keys.duplicated(keep="first")
        duplicate_actual_matches_removed = int((~keep).sum())
        table = table.loc[keep].reset_index(drop=True)
        a_affine = table[["a_affine_x", "a_affine_y", "a_affine_z"]].to_numpy()
        b_affine = table[["b_affine_x", "b_affine_y", "b_affine_z"]].to_numpy()
        a_split = table[["a_split_x", "a_split_y", "a_split_z"]].to_numpy()
        b_split = table[["b_split_x", "b_split_y", "b_split_z"]].to_numpy()
        metrics = self.calculate_match_metrics_from_transformed(
            a_affine, b_affine, a_split, b_split
        )
        metadata_columns = [
            "label",
            "split_setup_a",
            "split_setup_b",
            "original_setup_a",
            "original_setup_b",
        ]
        for insert_at, column in enumerate(metadata_columns, start=1):
            metrics["distance_rows"].insert(insert_at, column, table[column].to_numpy())
        return {
            "raw_count": raw_count,
            "pair_deduped_count": pair_deduped_count,
            "same_parent_excluded": same_parent_excluded,
            "duplicate_actual_matches_removed": duplicate_actual_matches_removed,
            "count": len(table),
            "pair_count": len(split_pairs_used),
            "original_pair_count": len(original_pairs_used),
            "label": label,
            "metrics": metrics,
        }

    # ==============================================================================
    # AFFINE-ONLY GEOMETRY
    # ==============================================================================

    def fit_affine(self, source_points, target_points):
        design = np.column_stack([source_points, np.ones(len(source_points), dtype=np.float64)])
        coefficients, _, rank, _ = np.linalg.lstsq(design, target_points, rcond=None)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :4] = coefficients.T
        return (matrix, int(rank))

    def linear_geometry(self, matrix):
        linear = np.asarray(matrix[:3, :3], dtype=np.float64)
        translation = np.asarray(matrix[:3, 3], dtype=np.float64)
        finite = bool(np.all(np.isfinite(linear)) and np.all(np.isfinite(translation)))
        if not finite:
            return {
                "finite": False,
                "determinant": np.nan,
                "condition_number": np.inf,
                "singular_values": np.full(3, np.nan),
                "max_scale_deviation": np.inf,
                "linear_identity_error": np.inf,
                "translation_norm": np.inf,
                "translation_scaled_norm": np.inf,
                "rotation_degrees": np.nan,
                "shear_magnitude": np.inf,
                "reflection": True,
            }
        determinant = float(np.linalg.det(linear))
        singular_values = np.linalg.svd(linear, compute_uv=False)
        condition_number = float(
            np.inf if singular_values[-1] <= 1e-12 else singular_values[0] / singular_values[-1]
        )
        u, _, vt = np.linalg.svd(linear)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        trace_value = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_degrees = float(np.degrees(np.arccos(trace_value)))
        stretch = rotation.T @ linear
        stretch_diagonal = np.diag(np.diag(stretch))
        shear_magnitude = float(np.linalg.norm(stretch - stretch_diagonal, ord="fro"))
        scale = np.asarray(self.metric_downsample_xyz, dtype=np.float64)
        return {
            "finite": True,
            "determinant": determinant,
            "condition_number": condition_number,
            "singular_values": singular_values,
            "max_scale_deviation": float(np.max(np.abs(singular_values - 1.0))),
            "linear_identity_error": float(np.linalg.norm(linear - np.eye(3), ord="fro")),
            "translation_norm": float(np.linalg.norm(translation)),
            "translation_scaled_norm": float(np.linalg.norm(translation / scale)),
            "rotation_degrees": rotation_degrees,
            "shear_magnitude": shear_magnitude,
            "reflection": bool(determinant <= 0.0),
        }

    def affine_geometry_metrics(self, match_metrics, pre_a, pre_b, post_a, post_b):
        a_post = match_metrics["a_post"]
        b_post = match_metrics["b_post"]
        residual_fit_matrix, fit_rank = self.fit_affine(a_post, b_post)
        fitted_b = self.apply_affine(a_post, residual_fit_matrix)
        residual_fit_delta = b_post - fitted_b
        residual_fit_distance = np.linalg.norm(residual_fit_delta, axis=1)
        scale = np.asarray(self.metric_downsample_xyz, dtype=np.float64)
        residual_fit_scaled_distance = np.linalg.norm(residual_fit_delta / scale, axis=1)
        residual_geometry = self.linear_geometry(residual_fit_matrix)
        update_a = post_a @ np.linalg.inv(pre_a)
        update_b = post_b @ np.linalg.inv(pre_b)
        differential_update = np.linalg.inv(update_a) @ update_b
        differential_geometry = self.linear_geometry(differential_update)
        post_median = match_metrics["post"]["median"]
        residual_fit_median = float(np.median(residual_fit_distance))
        unresolved_affine_fraction = (
            0.0
            if post_median <= 1e-12
            else float(np.clip((post_median - residual_fit_median) / post_median, 0.0, 1.0))
        )
        return {
            "residual_fit_matrix": residual_fit_matrix,
            "residual_fit_rank": fit_rank,
            "residual_fit": self.summary(residual_fit_distance),
            "residual_fit_scaled": self.summary(residual_fit_scaled_distance),
            "residual_geometry": residual_geometry,
            "differential_update_matrix": differential_update,
            "differential_geometry": differential_geometry,
            "extra_affine_median_reduction_pct": self.percent_reduction(
                match_metrics["post"]["median"], residual_fit_median
            ),
            "extra_affine_p90_reduction_pct": self.percent_reduction(
                match_metrics["post"]["p90"], float(np.percentile(residual_fit_distance, 90))
            ),
            "unresolved_affine_fraction": unresolved_affine_fraction,
        }

    def affine_quality_score(self, match_metrics, affine_metrics, count):
        post_scaled = match_metrics["post_scaled"]
        distance_score = (
            0.45
            * self.unit_interval_exp(
                post_scaled["median"], self.affine_distance_targets_scaled["median"]
            )
            + 0.35
            * self.unit_interval_exp(post_scaled["p90"], self.affine_distance_targets_scaled["p90"])
            + 0.2
            * self.unit_interval_exp(post_scaled["p95"], self.affine_distance_targets_scaled["p95"])
        )
        median_gain = self.percent_reduction(
            match_metrics["pre"]["median"], match_metrics["post"]["median"]
        )
        p90_gain = self.percent_reduction(match_metrics["pre"]["p90"], match_metrics["post"]["p90"])
        improved_fraction = match_metrics["improved"] / count
        worsened_fraction = match_metrics["worsened"] / count
        gain_score = float(
            np.clip(
                0.4 * (0.5 + 0.5 * median_gain / 100.0)
                + 0.3 * (0.5 + 0.5 * p90_gain / 100.0)
                + 0.2 * improved_fraction
                + 0.1 * (1.0 - worsened_fraction),
                0.0,
                1.0,
            )
        )
        residual_geometry = affine_metrics["residual_geometry"]
        geometry_score = (
            self.unit_interval_exp(
                residual_geometry["linear_identity_error"], self.affine_geometry_linear_error_scale
            )
            * self.unit_interval_exp(
                residual_geometry["translation_scaled_norm"],
                self.affine_geometry_translation_scaled_scale,
            )
            * (1.0 - affine_metrics["unresolved_affine_fraction"])
        )
        occupancy = match_metrics["occupancy"]
        concentration = match_metrics["max_cell_fraction"]
        coverage_score = float(
            np.clip(np.sqrt(max(0.0, occupancy)) * (1.0 - min(1.0, concentration)), 0.0, 1.0)
        )
        differential = affine_metrics["differential_geometry"]
        if (
            not differential["finite"]
            or differential["reflection"]
            or (not np.isfinite(differential["condition_number"]))
        ):
            sanity_score = 0.0
        else:
            sanity_score = (
                self.unit_interval_exp(
                    differential["max_scale_deviation"], self.affine_sanity_scale_deviation_scale
                )
                * self.unit_interval_exp(
                    differential["shear_magnitude"], self.affine_sanity_shear_scale
                )
                * self.unit_interval_exp(
                    max(0.0, differential["condition_number"] - 1.0),
                    self.affine_sanity_condition_excess_scale,
                )
            )
        combined_distance_score = 0.7 * distance_score + 0.3 * gain_score
        total = 100.0 * (
            0.45 * combined_distance_score
            + 0.3 * geometry_score
            + 0.15 * coverage_score
            + 0.1 * sanity_score
        )
        return {
            "affine_score": float(total),
            "distance_score": float(100.0 * combined_distance_score),
            "absolute_distance_score": float(100.0 * distance_score),
            "gain_score": float(100.0 * gain_score),
            "residual_geometry_score": float(100.0 * geometry_score),
            "coverage_score": float(100.0 * coverage_score),
            "transform_sanity_score": float(100.0 * sanity_score),
        }
    
    def print_overall_and_tile_metrics(
        self,
        run_type,
        result,
    ):
        """
        Print one row per tile followed by a match-weighted overall total.

        The final best/worst saved-match tables are selected globally from every
        deduplicated physical match across every saved edge.
        """
        before_label = "PRE"
        after_label = "RIGID"

        if run_type == "affine":
            before_label = "PARENT RIGID"
            after_label = "AFFINE"

        metrics = result["metrics"]
        tile_table = result["tile_table"]

        print("\n" + "=" * 140)
        print(
            f"{before_label} / {after_label} "
            f"ALL-TILE SAVED-MATCH METRICS — "
            f"NO IMAGE LOADING"
        )
        print("=" * 140)

        print("Counts:")

        print(
            f"  setups present in both XML files:             "
            f"{result['tile_count']:,}"
        )

        print(
            f"  setups with saved cross-tile matches:         "
            f"{result['matched_tile_count']:,}"
        )

        print(
            f"  saved matched edges:                          "
            f"{result['edge_count']:,}"
        )

        print(
            f"  raw loaded bidirectional correspondence rows: "
            f"{result['raw_count']:,}"
        )

        print(
            f"  physical saved matches after edge dedupe:     "
            f"{result['match_count']:,}"
        )

        print(
            f"  metric downsample xyz:                        "
            f"{self.metric_downsample_xyz}"
        )

        print(
            "\nPer-tile fixed-match progression:"
        )

        print(
            "  Each physical match is counted for both endpoint "
            "tiles in this table."
        )

        tile_columns = [
            "setup",
            "degree",
            "neighbors",
            "match_count",
            "pre_mean",
            "post_mean",
            "mean_reduction_pct",
            "pre_median",
            "post_median",
            "pre_p90",
            "post_p90",
            "pre_p95",
            "post_p95",
            "post_max",
            "improved_fraction",
            "worsened_fraction",
        ]

        printable_tiles = tile_table[
            tile_columns
        ].copy()

        printable_tiles[
            "improved_fraction"
        ] *= 100.0

        printable_tiles[
            "worsened_fraction"
        ] *= 100.0

        printable_tiles = printable_tiles.rename(
            columns={
                "match_count": "matches",
                "pre_mean": (
                    f"{before_label.lower()}_mean"
                ),
                "post_mean": (
                    f"{after_label.lower()}_mean"
                ),
                "mean_reduction_pct": (
                    "mean_gain_pct"
                ),
                "pre_median": (
                    f"{before_label.lower()}_median"
                ),
                "post_median": (
                    f"{after_label.lower()}_median"
                ),
                "pre_p90": (
                    f"{before_label.lower()}_p90"
                ),
                "post_p90": (
                    f"{after_label.lower()}_p90"
                ),
                "pre_p95": (
                    f"{before_label.lower()}_p95"
                ),
                "post_p95": (
                    f"{after_label.lower()}_p95"
                ),
                "post_max": (
                    f"{after_label.lower()}_max"
                ),
                "improved_fraction": (
                    "improved_pct"
                ),
                "worsened_fraction": (
                    "worsened_pct"
                ),
            }
        )

        print(
            printable_tiles.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.3f}"
                ),
            )
        )

        print("\n" + "-" * 140)
        print("OVERALL FIXED-MATCH TOTAL")
        print("-" * 140)

        print(
            "Each physical correspondence is counted exactly once "
            "in the overall total."
        )

        print(
            f"\nFull-res distance across all "
            f"{result['edge_count']:,} saved edges:"
        )

        print(
            f"  {'statistic':<12}"
            f"{before_label:>16}"
            f"{after_label:>16}"
            f"{'reduction':>16}"
            f"{'reduction %':>16}"
        )

        print(
            "  " + "-" * 74
        )

        for statistic in (
            "mean",
            "median",
            "p90",
            "p95",
            "max",
        ):
            before = metrics[
                "pre"
            ][statistic]

            after = metrics[
                "post"
            ][statistic]

            reduction = (
                before
                - after
            )

            reduction_percent = (
                self.percent_reduction(
                    before,
                    after,
                )
            )

            print(
                f"  {statistic:<12}"
                f"{before:>16.3f}"
                f"{after:>16.3f}"
                f"{reduction:>16.3f}"
                f"{reduction_percent:>15.2f}%"
            )

        print(
            "\nMetric-scale distance "
            "(mean / median / p90 / p95 / max):"
        )

        order = (
            "mean",
            "median",
            "p90",
            "p95",
            "max",
        )

        print(
            f"  {before_label:<14}: "
            + " ".join(
                f"{metrics['pre_scaled'][key]:9.3f}"
                for key in order
            )
        )

        print(
            f"  {after_label:<14}: "
            + " ".join(
                f"{metrics['post_scaled'][key]:9.3f}"
                for key in order
            )
        )

        total_count = result[
            "match_count"
        ]

        improved = metrics[
            "improved"
        ]

        worsened = metrics[
            "worsened"
        ]

        unchanged = metrics[
            "unchanged"
        ]

        print("\nOverall solver effect:")

        print(
            f"  mean reduction full-res px:       "
            f"{metrics['reduction']['mean']:.3f}"
        )

        print(
            f"  median reduction full-res px:     "
            f"{metrics['reduction']['median']:.3f}"
        )

        print(
            f"  median reduction percent:         "
            f"{metrics['median_reduction_pct']:.2f}%"
        )

        print(
            f"  improved: "
            f"{improved:,} "
            f"({100.0 * improved / total_count:.2f}%)"
        )

        print(
            f"  worsened: "
            f"{worsened:,} "
            f"({100.0 * worsened / total_count:.2f}%)"
        )

        print(
            f"  unchanged: "
            f"{unchanged:,} "
            f"({100.0 * unchanged / total_count:.2f}%)"
        )

        print(
            "\nGlobally best and worst saved matches "
            "across all tile pairs:"
        )

        self.print_match_distance_rows(
            result["distance_rows"]
        )

        print("=" * 140)

    def collect_overall_and_tile_metrics(
        self,
        manifest,
        index,
        pre,
        post,
    ):
        """
        Collect fixed-match progression across every saved edge.

        Overall metrics:
            Each deduplicated physical correspondence contributes exactly once.

        Per-tile metrics:
            Each physical correspondence contributes to both endpoint tiles because
            the residual describes the placement quality of both tiles.

        Distance rows:
            Contains every deduplicated physical correspondence across every edge.
            This is used to print the globally best and worst saved matches.
        """
        all_pre_distances = []
        all_post_distances = []

        all_pre_scaled_distances = []
        all_post_scaled_distances = []

        all_distance_rows = []

        tile_data = {}

        raw_count = 0
        edge_count = 0

        common_setups = sorted(
            set(pre) & set(post)
        )

        for setup in common_setups:
            tile_data[setup] = {
                "neighbors": set(),
                "pre_distances": [],
                "post_distances": [],
            }

        for setup_a, setup_b in self.saved_pairs(
            index
        ):
            if (
                setup_a not in pre
                or setup_b not in pre
                or setup_a not in post
                or setup_b not in post
            ):
                continue

            a_raw, b_raw = self.read_pair(
                manifest=manifest,
                index=index,
                setup_a=setup_a,
                setup_b=setup_b,
            )

            raw_count += len(a_raw)

            a, b = self.dedupe_pairs(
                a_raw,
                b_raw,
            )

            if len(a) == 0:
                continue

            a_pre = self.apply_affine(
                a,
                pre[setup_a],
            )

            b_pre = self.apply_affine(
                b,
                pre[setup_b],
            )

            a_post = self.apply_affine(
                a,
                post[setup_a],
            )

            b_post = self.apply_affine(
                b,
                post[setup_b],
            )

            pre_delta = (
                b_pre
                - a_pre
            )

            post_delta = (
                b_post
                - a_post
            )

            pre_distances = np.linalg.norm(
                pre_delta,
                axis=1,
            )

            post_distances = np.linalg.norm(
                post_delta,
                axis=1,
            )

            scale = np.asarray(
                self.metric_downsample_xyz,
                dtype=np.float64,
            )

            pre_scaled_distances = np.linalg.norm(
                pre_delta / scale,
                axis=1,
            )

            post_scaled_distances = np.linalg.norm(
                post_delta / scale,
                axis=1,
            )

            reduction = (
                pre_distances
                - post_distances
            )

            reduction_percent = np.divide(
                100.0 * reduction,
                pre_distances,
                out=np.full_like(
                    reduction,
                    np.nan,
                ),
                where=(
                    pre_distances
                    > self.percent_baseline_epsilon
                ),
            )

            all_pre_distances.append(
                pre_distances
            )

            all_post_distances.append(
                post_distances
            )

            all_pre_scaled_distances.append(
                pre_scaled_distances
            )

            all_post_scaled_distances.append(
                post_scaled_distances
            )

            all_distance_rows.append(
                pd.DataFrame(
                    {
                        "setup_a": np.full(
                            len(a),
                            setup_a,
                            dtype=np.int64,
                        ),
                        "setup_b": np.full(
                            len(a),
                            setup_b,
                            dtype=np.int64,
                        ),
                        "distance_pre_fullres_px": (
                            pre_distances
                        ),
                        "distance_post_fullres_px": (
                            post_distances
                        ),
                        "distance_reduction_fullres_px": (
                            reduction
                        ),
                        "distance_reduction_percent": (
                            reduction_percent
                        ),
                        "distance_pre_scaled_px": (
                            pre_scaled_distances
                        ),
                        "distance_post_scaled_px": (
                            post_scaled_distances
                        ),
                        "dx_post_full": (
                            post_delta[:, 0]
                        ),
                        "dy_post_full": (
                            post_delta[:, 1]
                        ),
                        "dz_post_full": (
                            post_delta[:, 2]
                        ),
                    }
                )
            )

            tile_data[setup_a]["neighbors"].add(
                setup_b
            )

            tile_data[setup_b]["neighbors"].add(
                setup_a
            )

            for setup in (
                setup_a,
                setup_b,
            ):
                tile_data[setup][
                    "pre_distances"
                ].append(
                    pre_distances
                )

                tile_data[setup][
                    "post_distances"
                ].append(
                    post_distances
                )

            edge_count += 1

        if not all_pre_distances:
            raise RuntimeError(
                "No saved correspondences were available across "
                "the common PRE and POST tile set"
            )

        overall_pre = np.concatenate(
            all_pre_distances
        )

        overall_post = np.concatenate(
            all_post_distances
        )

        overall_pre_scaled = np.concatenate(
            all_pre_scaled_distances
        )

        overall_post_scaled = np.concatenate(
            all_post_scaled_distances
        )

        overall_reduction = (
            overall_pre
            - overall_post
        )

        overall_reduction_percent = np.divide(
            100.0 * overall_reduction,
            overall_pre,
            out=np.full_like(
                overall_reduction,
                np.nan,
            ),
            where=(
                overall_pre
                > self.percent_baseline_epsilon
            ),
        )

        overall_metrics = {
            "pre": self.summary(
                overall_pre
            ),
            "post": self.summary(
                overall_post
            ),
            "pre_scaled": self.summary(
                overall_pre_scaled
            ),
            "post_scaled": self.summary(
                overall_post_scaled
            ),
            "reduction": self.summary(
                overall_reduction
            ),
            "median_reduction_pct": float(
                np.nanmedian(
                    overall_reduction_percent
                )
            ),
            "improved": int(
                np.count_nonzero(
                    overall_reduction > 1e-6
                )
            ),
            "worsened": int(
                np.count_nonzero(
                    overall_reduction < -1e-6
                )
            ),
            "unchanged": int(
                np.count_nonzero(
                    np.abs(
                        overall_reduction
                    ) <= 1e-6
                )
            ),
        }

        distance_rows = pd.concat(
            all_distance_rows,
            ignore_index=True,
        )

        distance_rows.insert(
            0,
            "match_i",
            np.arange(
                len(distance_rows),
                dtype=np.int64,
            ),
        )

        tile_rows = []

        for setup in common_setups:
            data = tile_data[setup]

            neighbors = sorted(
                data["neighbors"]
            )

            if not data["pre_distances"]:
                tile_rows.append(
                    {
                        "setup": setup,
                        "degree": 0,
                        "neighbors": "",
                        "match_count": 0,
                        "pre_mean": np.nan,
                        "post_mean": np.nan,
                        "mean_reduction": np.nan,
                        "mean_reduction_pct": np.nan,
                        "pre_median": np.nan,
                        "post_median": np.nan,
                        "pre_p90": np.nan,
                        "post_p90": np.nan,
                        "pre_p95": np.nan,
                        "post_p95": np.nan,
                        "post_max": np.nan,
                        "improved_fraction": np.nan,
                        "worsened_fraction": np.nan,
                    }
                )

                continue

            pre_distances = np.concatenate(
                data["pre_distances"]
            )

            post_distances = np.concatenate(
                data["post_distances"]
            )

            reduction = (
                pre_distances
                - post_distances
            )

            pre_summary = self.summary(
                pre_distances
            )

            post_summary = self.summary(
                post_distances
            )

            match_count = len(
                pre_distances
            )

            improved_count = int(
                np.count_nonzero(
                    reduction > 1e-6
                )
            )

            worsened_count = int(
                np.count_nonzero(
                    reduction < -1e-6
                )
            )

            mean_reduction = (
                pre_summary["mean"]
                - post_summary["mean"]
            )

            tile_rows.append(
                {
                    "setup": setup,
                    "degree": len(neighbors),
                    "neighbors": ",".join(
                        str(neighbor)
                        for neighbor in neighbors
                    ),
                    "match_count": match_count,
                    "pre_mean": pre_summary["mean"],
                    "post_mean": post_summary["mean"],
                    "mean_reduction": mean_reduction,
                    "mean_reduction_pct": (
                        self.percent_reduction(
                            pre_summary["mean"],
                            post_summary["mean"],
                        )
                    ),
                    "pre_median": pre_summary["median"],
                    "post_median": post_summary["median"],
                    "pre_p90": pre_summary["p90"],
                    "post_p90": post_summary["p90"],
                    "pre_p95": pre_summary["p95"],
                    "post_p95": post_summary["p95"],
                    "post_max": post_summary["max"],
                    "improved_fraction": (
                        improved_count
                        / match_count
                    ),
                    "worsened_fraction": (
                        worsened_count
                        / match_count
                    ),
                }
            )

        tile_table = pd.DataFrame(
            tile_rows
        )

        matched_tile_count = int(
            np.count_nonzero(
                tile_table[
                    "match_count"
                ].to_numpy() > 0
            )
        )

        return {
            "raw_count": raw_count,
            "match_count": len(
                overall_pre
            ),
            "edge_count": edge_count,
            "tile_count": len(
                common_setups
            ),
            "matched_tile_count": (
                matched_tile_count
            ),
            "metrics": overall_metrics,
            "tile_table": tile_table,
            "distance_rows": distance_rows,
        }

    # ==============================================================================
    # PRINTING
    # ==============================================================================

    def fmt(self, value, width=10, precision=3):
        return f"{value:{width}.{precision}f}"

    def print_matrix(self, name, matrix):
        print(f"\n{name}:")
        for row in matrix[:3, :4]:
            print("  " + " ".join((f"{value:12.6f}" for value in row)))

    def print_match_distance_rows(self, distance_rows):
        if self.print_match_distance_rows_count <= 0 or len(distance_rows) == 0:
            return
        columns = [
            "match_i",
            *[
                column
                for column in (
                    "label",
                    "split_setup_a",
                    "split_setup_b",
                    "original_setup_a",
                    "original_setup_b",
                    "setup_a",
                    "setup_b",
                )
                if column in distance_rows.columns
            ],
            "distance_pre_fullres_px",
            "distance_post_fullres_px",
            "distance_reduction_fullres_px",
            "distance_reduction_percent",
            "distance_post_scaled_px",
            "dx_post_full",
            "dy_post_full",
            "dz_post_full",
        ]
        best = distance_rows.sort_values("distance_post_scaled_px", ascending=True).head(
            self.print_match_distance_rows_count
        )
        worst = distance_rows.sort_values("distance_post_scaled_px", ascending=False).head(
            self.print_match_distance_rows_count
        )
        print(f"\nBest {len(best):,} saved matches by POST metric-scale distance:")
        print(best[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\nWorst {len(worst):,} saved matches by POST metric-scale distance:")
        print(worst[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    def print_common_metrics(self, run_type, pair, raw_count, count, metrics):
        before_label = "PRE"
        after_label = "RIGID"
        if run_type == "affine":
            before_label = "PARENT RIGID"
            after_label = "AFFINE"
        elif run_type == "split-affine":
            before_label = "AFFINE"
            after_label = "SPLIT AFFINE"
        print(f"Run type: {run_type.upper()}")
        print(f"Pair:     {pair}\n")
        print("=" * 96)
        print(f"{before_label} / {after_label} SAVED-MATCH METRICS — NO IMAGE LOADING")
        print("=" * 96)
        print("Counts:")
        print(f"  raw loaded bidirectional correspondence rows: {raw_count:,}")
        print(f"  saved/RANSAC inlier matches after dedupe:      {count:,}")
        print(f"  metric downsample xyz:                         {self.metric_downsample_xyz}")
        if run_type == "affine" and self.parent_rigid_match_count:
            print(
                f"  linked rigid saved match count:                {self.parent_rigid_match_count:,}"
            )
            print(
                f"  affine / linked rigid count ratio:             {count / self.parent_rigid_match_count:.4f}"
            )
        print("\nFull-res global A-B distance across all saved/RANSAC matches:")
        print(f"  statistic      {before_label:>14} {after_label:>14}    reduction    reduction %")
        print("  " + "-" * 75)
        for stat in ("mean", "median", "p90", "p95", "max"):
            before = metrics["pre"][stat]
            after = metrics["post"][stat]
            reduction = before - after
            percent = self.percent_reduction(before, after)
            print(
                f"  {stat:<12}{self.fmt(before, 14)}{self.fmt(after, 15)}{self.fmt(reduction, 13)}{self.fmt(percent, 15, 2)}"
            )
        order = ("mean", "median", "p90", "p95", "max")
        print("\nMetric-scale A-B distance (mean / median / p90 / p95 / max):")
        print(
            f"  {before_label:<13}: "
            + " ".join((self.fmt(metrics["pre_scaled"][key], 9) for key in order))
        )
        print(
            f"  {after_label:<13}: "
            + " ".join((self.fmt(metrics["post_scaled"][key], 9) for key in order))
        )
        print("\nSolver effect:")
        print(
            f"  mean / median reduction full-res px: {metrics['reduction']['mean']:.3f} / {metrics['reduction']['median']:.3f}"
        )
        print(f"  median reduction percent:            {metrics['median_reduction_pct']:.2f}%")
        print(
            f"  improved: {metrics['improved']:,} ({100.0 * metrics['improved'] / count:.2f}%) | worsened: {metrics['worsened']:,} ({100.0 * metrics['worsened'] / count:.2f}%) | unchanged: {metrics['unchanged']:,} ({100.0 * metrics['unchanged'] / count:.2f}%)"
        )
        print("\nMean full-res delta dx / dy / dz:")
        print(
            f"  {before_label:<13} signed: "
            + " ".join((self.fmt(v) for v in metrics["pre_signed"]))
        )
        print(
            f"  {after_label:<13} signed: "
            + " ".join((self.fmt(v) for v in metrics["post_signed"]))
        )
        print(
            f"  {before_label:<13} abs:    " + " ".join((self.fmt(v) for v in metrics["pre_abs"]))
        )
        print(
            f"  {after_label:<13} abs:    " + " ".join((self.fmt(v) for v in metrics["post_abs"]))
        )
        print(f"\n{after_label} match-midpoint spatial spread:")
        print("  span xyz full-res px: " + ", ".join((f"{v:.3f}" for v in metrics["span"])))
        print(
            f"  grid occupancy {self.grid_bins_zyx} zyx: {metrics['occupancy']:.4f} ({metrics['occupied_cells']} occupied cells)"
        )
        print(f"  max cell fraction:                  {metrics['max_cell_fraction']:.4f}")
        print(
            f"  supported cells (>= {self.min_matches_per_grid_cell}):       {metrics['supported_cells']}"
        )
        print("  supported-cell median distance median / p90 / max:")
        print(
            f"    {metrics['cell_median_distance']['median']:.3f} / {metrics['cell_median_distance']['p90']:.3f} / {metrics['cell_median_distance']['max']:.3f}"
        )
        print("  cell residual-vector dispersion mean / p90 / max:")
        print(
            f"    {metrics['cell_vector_dispersion_mean']:.3f} / {metrics['cell_vector_dispersion_p90']:.3f} / {metrics['cell_vector_dispersion_max']:.3f}"
        )

    def print_split_result(self, title, result):
        """
        Print real cross-parent bead progression from AFFINE to SPLIT AFFINE.
        """
        if result is None:
            print("\n" + "=" * 96)
            print(title)
            print("=" * 96)
            print("No real cross-parent saved correspondences were found.")
            return
        print("\n" + "#" * 96)
        print(title)
        print("#" * 96)
        print(f"Cross-parent split setup-pairs represented: {result['pair_count']:,}")
        print(f"Original affine setup-pairs represented:    {result['original_pair_count']:,}")
        print(f"Rows after per-pair bidirectional dedupe:    {result['pair_deduped_count']:,}")
        print(f"Same-parent chunk rows excluded:             {result['same_parent_excluded']:,}")
        print(
            f"Repeated overlap copies removed globally:   {result['duplicate_actual_matches_removed']:,}"
        )
        self.print_common_metrics(
            "split-affine",
            "ALL REAL CROSS-PARENT MATCHES",
            result["raw_count"],
            result["count"],
            result["metrics"],
        )
        self.print_match_distance_rows(result["metrics"]["distance_rows"])

    def print_geometry_block(self, title, geometry):
        singular = geometry["singular_values"]
        print(f"\n{title}:")
        print(f"  finite:                       {geometry['finite']}")
        print(f"  determinant:                  {geometry['determinant']:.6f}")
        print(f"  reflection:                   {geometry['reflection']}")
        print("  singular values / scales:     " + ", ".join((f"{v:.6f}" for v in singular)))
        print(f"  max scale deviation from 1:   {geometry['max_scale_deviation']:.6f}")
        print(f"  condition number:             {geometry['condition_number']:.6f}")
        print(f"  linear identity error:        {geometry['linear_identity_error']:.6f}")
        print(f"  shear magnitude:              {geometry['shear_magnitude']:.6f}")
        print(f"  rotation magnitude degrees:   {geometry['rotation_degrees']:.6f}")
        print(f"  translation norm full-res px: {geometry['translation_norm']:.6f}")
        print(f"  translation norm metric-scale: {geometry['translation_scaled_norm']:.6f}")

    def print_affine_metrics(self, metrics, affine_metrics, score):
        print("\n" + "-" * 96)
        print("AFFINE-ONLY GEOMETRY")
        print("-" * 96)
        print("\nResidual affine fit after the saved affine result (A_post -> B_post):")
        print(f"  least-squares design rank:              {affine_metrics['residual_fit_rank']}")
        print("  residual fit full-res mean / median / p90 / p95 / max:")
        print(
            "    "
            + " ".join(
                (
                    self.fmt(affine_metrics["residual_fit"][key], 9)
                    for key in ("mean", "median", "p90", "p95", "max")
                )
            )
        )
        print("  residual fit metric-scale mean / median / p90 / p95 / max:")
        print(
            "    "
            + " ".join(
                (
                    self.fmt(affine_metrics["residual_fit_scaled"][key], 9)
                    for key in ("mean", "median", "p90", "p95", "max")
                )
            )
        )
        print(
            f"  extra affine median reduction available: {affine_metrics['extra_affine_median_reduction_pct']:.2f}%"
        )
        print(
            f"  extra affine p90 reduction available:    {affine_metrics['extra_affine_p90_reduction_pct']:.2f}%"
        )
        print(
            f"  unresolved affine fraction:              {affine_metrics['unresolved_affine_fraction']:.4f}"
        )
        self.print_geometry_block(
            "Residual fitted affine geometry", affine_metrics["residual_geometry"]
        )
        self.print_geometry_block(
            "Differential affine update between setup B and setup A",
            affine_metrics["differential_geometry"],
        )
        self.print_matrix("Residual fitted affine matrix", affine_metrics["residual_fit_matrix"])
        self.print_matrix(
            "Differential update matrix", affine_metrics["differential_update_matrix"]
        )
        print("\nAffine quality score:")
        print(f"  TOTAL:                     {score['affine_score']:.2f} / 100")
        print(f"  distance:                  {score['distance_score']:.2f}")
        print(f"    absolute distance:       {score['absolute_distance_score']:.2f}")
        print(f"    gain vs parent rigid:    {score['gain_score']:.2f}")
        print(f"  residual geometry:         {score['residual_geometry_score']:.2f}")
        print(f"  spatial coverage:          {score['coverage_score']:.2f}")
        print(f"  transform sanity:          {score['transform_sanity_score']:.2f}")
        print("\nRecommended affine tracking fields:")
        print(
            "  match_count | post_median | post_p90 | post_p95 | median_gain_vs_rigid_pct | p90_gain_vs_rigid_pct"
        )
        print(
            "  worsened_pct | occupancy | max_cell_fraction | cell_vector_dispersion_p90 | residual_linear_error"
        )
        print(
            "  residual_translation_scaled | extra_affine_median_reduction_pct | max_scale_deviation | shear | condition | affine_score"
        )

    # ==============================================================================
    # RUN
    # ==============================================================================

    def run(self):
        run_type = str(
            self.match_type
        ).strip().lower()

        if run_type not in {
            "rigid",
            "affine",
            "split-affine",
        }:
            raise ValueError(
                "match_type must be 'rigid', 'affine', "
                "or 'split-affine', "
                f"got {self.match_type!r}"
            )

        self.rejected_edge_keys = self.load_rejected_edge_keys()

        pre_root = self.load_xml(
            self.pre_xml_path
        )

        post_root = self.load_xml(
            self.post_xml_path
        )

        pre = self.xml_transforms(
            pre_root
        )

        post = self.xml_transforms(
            post_root
        )

        manifest = self.point_manifest()
        index = self.match_index()

        # ==========================================================================
        # SPLIT-AFFINE
        # ==========================================================================

        if run_type == "split-affine":
            split_root = self.load_xml(
                self.split_xml_path
            )

            split_definitions = (
                self.parse_split_setup_definitions(
                    split_root
                )
            )

            split_start = self.xml_transforms(
                split_root
            )

            self.validate_split_start(
                manifest=manifest,
                affine=pre,
                split_start=split_start,
                split_definitions=split_definitions,
                label=self.split_real_label,
            )

            real_result = (
                self.collect_split_real_progression_metrics(
                    manifest=manifest,
                    index=index,
                    affine=pre,
                    split_post=post,
                    split_definitions=split_definitions,
                    label=self.split_real_label,
                )
            )

            self.print_split_result(
                (
                    "REAL CROSS-TILE POINT PROGRESSION: "
                    f"AFFINE -> SPLIT AFFINE "
                    f"({self.split_real_label})"
                ),
                real_result,
            )

            print("=" * 96)
            return

        # ==========================================================================
        # RIGID / AFFINE OVERALL + PER-TILE METRICS
        # ==========================================================================

        overall_result = (
            self.collect_overall_and_tile_metrics(
                manifest=manifest,
                index=index,
                pre=pre,
                post=post,
            )
        )

        self.print_overall_and_tile_metrics(
            run_type=run_type,
            result=overall_result,
        )

        # ==========================================================================
        # RIGID ALL-EDGE + GRAPH DIAGNOSTICS
        # ==========================================================================

        if run_type == "rigid":
            edge_table, edge_models = (
                self.collect_rigid_edge_diagnostics(
                    manifest=manifest,
                    index=index,
                    pre=pre,
                    post=post,
                )
            )

            common_setups = sorted(
                set(pre) & set(post)
            )

            graph = self.rigid_graph_topology(
                edge_table=edge_table,
                setups=common_setups,
            )

            loop_table, edge_consistency_table = (
                self.collect_rigid_loop_closure_diagnostics(
                    edge_table=edge_table,
                    edge_models=edge_models,
                )
            )

            self.print_rigid_edge_diagnostics(
                edge_table=edge_table,
                graph=graph,
            )

            self.print_rigid_loop_closure_diagnostics(
                loop_table=loop_table,
                edge_consistency_table=(
                    edge_consistency_table
                ),
            )

        # ==========================================================================
        # AFFINE ALL-EDGE + PER-TILE TRANSFORM DIAGNOSTICS
        # ==========================================================================

        if run_type == "affine":
            affine_edge_table = (
                self.collect_affine_edge_diagnostics(
                    manifest=manifest,
                    index=index,
                    pre=pre,
                    post=post,
                )
            )

            affine_tile_table = (
                self.collect_affine_tile_transform_diagnostics(
                    pre=pre,
                    post=post,
                )
            )

            common_setups = sorted(
                set(pre) & set(post)
            )

            affine_graph = self.rigid_graph_topology(
                edge_table=affine_edge_table,
                setups=common_setups,
            )

            self.print_affine_all_edge_diagnostics(
                edge_table=affine_edge_table,
                tile_table=affine_tile_table,
                graph=affine_graph,
            )
        