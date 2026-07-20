import numpy as np
import math
import copy

"""
GlobalOptimization iteratively refines per-tile transforms to achieve sub-pixel alignment
using matched point correspondences.
"""

class GlobalOptimization:
    def __init__(self, tiles, relative_threshold, absolute_threshold, min_matches, damp, regularization_weight,
                 max_iterations, max_allowed_error, max_cleanup_rounds, max_plateauwidth, run_type, metrics_output_path, 
                 initial_pos):
        self.tiles = tiles
        self.relative_threshold = relative_threshold
        self.absolute_threshold = absolute_threshold
        self.min_matches = min_matches
        self.damp = damp
        self.regularization_weight = regularization_weight
        self.max_iterations = max_iterations
        self.max_allowed_error = max_allowed_error
        self.max_cleanup_rounds = max_cleanup_rounds
        self.max_plateauwidth = max_plateauwidth
        self.run_type = run_type
        self.metrics_output_path = metrics_output_path
        self.initial_pos = initial_pos
        self.validation_stats = {
            "solve_metrics_per_tile": {
                "i": 0,
                "stats": [],
            },
            "solver_metrics_per_tile": {
                "stats": [],
            },
        }
        self.observer = {
            "max": 0,
            "mean": 0,
            "median": 0,
            "min": float("inf"),
            "slope": [],
            "values": [],
            "square_differences": 0,
            "squares": 0,
            "std": 0,
            "std_0": 0,
            "var": 0,
            "var_0": 0,
        }
    
    def is_synthetic_match(self, match):
        """
        Identify synthetic constraints
        """
        if "is_synthetic" in match:
            return bool(match["is_synthetic"])

        label = str(match.get("label", ""))

        return (
            label.startswith("splitPoints")
            or label == "weakLink"
        )

    def reset_observer(self):
        """
        Reset convergence tracking before a new cleanup solve pass.
        """
        self.observer = {
            "max": 0,
            "mean": 0,
            "median": 0,
            "min": float("inf"),
            "slope": [],
            "values": [],
            "square_differences": 0,
            "squares": 0,
            "std": 0,
            "std_0": 0,
            "var": 0,
            "var_0": 0,
        }

    def refresh_weight_arrays_from_matches(self):
        """
        Refresh cached solver weights from the live match dictionaries
        """
        for tile in self.tiles:
            matches = tile.get("matches", [])

            if not matches:
                continue

            weights = []
            synthetic_flags = []

            for match in matches:
                base_weight = float(
                    match.get(
                        "base_weight",
                        match.get("weight", 1.0),
                    )
                )

                cleanup_weight = float(
                    match.get("cleanup_weight", 1.0)
                )

                effective_weight = (
                    base_weight * cleanup_weight
                )

                match["base_weight"] = base_weight
                match["cleanup_weight"] = cleanup_weight
                match["weight"] = effective_weight

                weights.append(effective_weight)
                synthetic_flags.append(
                    self.is_synthetic_match(match)
                )

            weights = np.asarray(
                weights,
                dtype=np.float64,
            )

            synthetic_flags = np.asarray(
                synthetic_flags,
                dtype=bool,
            )

            if tile["_weights"].shape != weights.shape:
                raise RuntimeError(
                    "Match count changed after tile arrays were prepared"
                )

            tile["_weights"][:] = weights
            tile["_is_synthetic"][:] = synthetic_flags

    def get_final_match_stats(self):
        """
        Calculate final active bead and synthetic residual distributions
        """
        self.refresh_target_arrays_from_matches()
        self.refresh_weight_arrays_from_matches()

        beads = []
        synthetic = []

        for tile in self.tiles:
            if len(tile["_p1_w"]) == 0:
                continue

            distances = np.linalg.norm(
                tile["_p1_w"] - tile["_p2_w"],
                axis=1,
            )

            active_mask = tile["_weights"] > 0.0

            beads_mask = (
                ~tile["_is_synthetic"]
                & active_mask
            )

            synthetic_mask = (
                tile["_is_synthetic"]
                & active_mask
            )

            if np.any(beads_mask):
                beads.append(
                    distances[beads_mask]
                )

            if np.any(synthetic_mask):
                synthetic.append(
                    distances[synthetic_mask]
                )

        def summarize(chunks):
            if not chunks:
                return {
                    "count": 0,
                    "mean": 0.0,
                    "median": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                    "max": 0.0,
                }

            values = np.concatenate(chunks)

            return {
                "count": int(len(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }

        return {
            "beads": summarize(beads),
            "synthetic": summarize(synthetic),
        }
    
    def prepare_tile_arrays(self):
        """
        Convert match dictionaries into NumPy arrays.
        """
        for tile in self.tiles:
            matches = tile.get("matches", [])

            if len(matches) == 0:
                tile["_p1_l"] = np.empty(
                    (0, 3),
                    dtype=np.float64,
                )

                tile["_p1_w"] = np.empty(
                    (0, 3),
                    dtype=np.float64,
                )

                tile["_p2_w"] = np.empty(
                    (0, 3),
                    dtype=np.float64,
                )

                tile["_weights"] = np.empty(
                    (0,),
                    dtype=np.float64,
                )

                tile["_is_synthetic"] = np.empty(
                    (0,),
                    dtype=bool,
                )

                continue

            tile["_p1_l"] = np.asarray(
                [
                    match["p1"]["l"]
                    for match in matches
                ],
                dtype=np.float64,
            ).reshape(-1, 3)

            tile["_p1_w"] = np.asarray(
                [
                    match["p1"]["w"]
                    for match in matches
                ],
                dtype=np.float64,
            ).reshape(-1, 3)

            tile["_p2_w"] = np.asarray(
                [
                    match["p2"]["w"]
                    for match in matches
                ],
                dtype=np.float64,
            ).reshape(-1, 3)

            tile["_weights"] = np.asarray(
                [
                    match.get(
                        "weight",
                        match.get("base_weight", 1.0),
                    )
                    for match in matches
                ],
                dtype=np.float64,
            )

            tile["_is_synthetic"] = np.asarray(
                [
                    self.is_synthetic_match(match)
                    for match in matches
                ],
                dtype=bool,
            )

        self.refresh_weight_arrays_from_matches()

    def refresh_target_arrays_from_matches(self):
        """
        Refresh cached target-side world coordinates from the live match graph.
        """
        for tile in self.tiles:
            matches = tile.get("matches", [])

            if len(matches) == 0:
                continue

            tile["_p2_w"][:] = np.asarray(
                [m["p2"]["w"] for m in matches],
                dtype=np.float64,
            ).reshape(-1, 3)

    def sync_tile_array_to_matches(self, tile):
        """
        Sync one tile's optimized source-side world coordinates back into matches.
        """
        if "_p1_w" not in tile:
            return

        p1_w = tile["_p1_w"]
        matches = tile.get("matches", [])

        for match, w in zip(matches, p1_w):
            match["p1"]["w"][:] = w.tolist()

    def sync_arrays_to_matches(self):
        """
        Final full sync for downstream code that reads match["p1"]["w"].
        """
        for tile in self.tiles:
            self.sync_tile_array_to_matches(tile)
    
    def capture_solver_state(self):
        """
        Capture the mutable solver state for the current iteration.
        """
        state = []

        for tile in self.tiles:
            state.append(
                {
                    "model": copy.deepcopy(tile["model"]),
                    "p1_w": tile["_p1_w"].copy(),
                }
            )

        return state

    def restore_solver_state(self, state):
        """
        Restore a previously captured solver state
        """
        if len(state) != len(self.tiles):
            raise RuntimeError(
                "Saved solver-state tile count does not match "
                "the current tile count"
            )

        for tile, saved_tile in zip(self.tiles, state):
            if tile["_p1_w"].shape != saved_tile["p1_w"].shape:
                raise RuntimeError(
                    "Saved solver-state point shape does not match "
                    "the current tile"
                )

            tile["_p1_w"][:] = saved_tile["p1_w"]

            # Preserve the existing model dictionary object in case other
            # objects hold references to it.
            tile["model"].clear()
            tile["model"].update(
                copy.deepcopy(saved_tile["model"])
            )

        # Push every restored p1 world coordinate into the shared matches.
        self.sync_arrays_to_matches()

        # Rebuild target coordinates and final diagnostics from the restored graph.
        self.update_errors()

    def update_observer(self, new_value):
        obs = self.observer
        obs["values"].append(new_value)

        n = len(obs["values"])

        if n == 1:
            obs["slope"].append(0.0)
            obs["mean"] = new_value
            obs["var"] = 0.0
            obs["var_0"] = 0.0
            obs["squares"] = new_value * new_value
        else:
            obs["slope"].append(new_value - obs["values"][-2])

            delta = new_value - obs["mean"]
            obs["mean"] += delta / n

            obs["square_differences"] += delta * (new_value - obs["mean"])
            obs["var"] = obs["square_differences"] / (n - 1)

            obs["squares"] += new_value * new_value
            obs["var_0"] = obs["squares"] / n

        obs["std"] = math.sqrt(max(obs["var"], 0.0))
        obs["std_0"] = math.sqrt(max(obs["var_0"], 0.0))

        obs["min"] = min(obs["min"], new_value)
        obs["max"] = max(obs["max"], new_value)

        # This is only metrics. It is not part of the solve.
        obs["median"] = float(np.median(obs["values"])) if n > 0 else 0.0

    def get_wide_slope(self, values, width):
        width = int(width)
        return (values[-1] - values[-1 - width]) / width
    
    def append_iteration_metrics(self, i, error):
        """
        Store compact metrics for each completed solver iteration.
        """
        breakdown = getattr(self, "error_breakdown", {})

        self.validation_stats.setdefault(
            "solver_metrics_per_tile",
            {},
        ).setdefault(
            "stats",
            [],
        ).append(
            {
                "iteration": i,
                "error": float(error),
                "convergence_error": float(error),
                "mean_tile_error": float(
                    breakdown.get("mean_tile_error", np.nan)
                ),
                "all_match_mean": float(
                    breakdown.get("all_match_mean", np.nan)
                ),
                "beads_mean": float(
                    breakdown.get("beads_mean", np.nan)
                ),
                "synthetic_mean": float(
                    breakdown.get("synthetic_mean", np.nan)
                ),
                "weighted_rms": float(
                    breakdown.get("weighted_rms", np.nan)
                ),
                "observer": {
                    "mean": self.observer["mean"],
                    "median": self.observer["median"],
                    "min": self.observer["min"],
                    "max": self.observer["max"],
                    "std": self.observer["std"],
                    "slope": (
                        self.observer["slope"][-1]
                        if self.observer["slope"]
                        else 0.0
                    ),
                },
            }
        )

    def model_to_matrix_translation(self, model):
        M = np.array(
            [
                [model["m00"], model["m01"], model["m02"]],
                [model["m10"], model["m11"], model["m12"]],
                [model["m20"], model["m21"], model["m22"]],
            ],
            dtype=np.float64,
        )

        t = np.array(
            [model["m03"], model["m13"], model["m23"]],
            dtype=np.float64,
        )

        return M, t

    def apply_model_array(self, points, model):
        M, t = self.model_to_matrix_translation(model)
        return points @ M.T + t

    def get_active_model(self, tile):
        if self.run_type in ("affine", "split-affine"):
            return tile["model"]["regularized"]

        if self.run_type == "rigid":
            return tile["model"]["b"]

        raise ValueError(f"Unknown run_type: {self.run_type}")

    def regularize_models(self, affine, rigid):
        l1 = 1.0 - self.regularization_weight

        keys = [
            "m00", "m01", "m02", "m03",
            "m10", "m11", "m12", "m13",
            "m20", "m21", "m22", "m23",
        ]

        return {
            key: l1 * affine[key] + self.regularization_weight * rigid[key]
            for key in keys
        }

    def update_cost(self, tile):
        """
        Compute average distance and weighted cost using active matches only.
        """
        p1_w = tile["_p1_w"]
        p2_w = tile["_p2_w"]
        weights = tile["_weights"]

        active_mask = weights > 0.0

        if not np.any(active_mask):
            tile["model"]["cost"] = 0.0
            tile["cost"] = 0.0
            tile["distance"] = 0.0
            return

        active_p1 = p1_w[active_mask]
        active_p2 = p2_w[active_mask]
        active_weights = weights[active_mask]

        distances = np.linalg.norm(
            active_p1 - active_p2,
            axis=1,
        )

        distance = float(
            np.mean(distances)
        )

        sum_weight = float(
            np.sum(active_weights)
        )

        cost = float(
            np.sum(
                distances
                * distances
                * active_weights
            )
            / sum_weight
        )

        tile["model"]["cost"] = cost
        tile["cost"] = cost
        tile["distance"] = distance

    def update_errors(self):
        """
        Refresh target arrays and score the current global state.
        """
        if not self.tiles:
            self.error_breakdown = {
                "mean_tile_error": 0.0,
                "all_match_mean": 0.0,
                "beads_mean": 0.0,
                "synthetic_mean": 0.0,
                "weighted_rms": 0.0,
                "all_count": 0,
                "beads_count": 0,
                "synthetic_count": 0,
            }

            return 0.0

        self.refresh_target_arrays_from_matches()

        total_tile_distance = 0.0

        all_distance_sum = 0.0
        all_count = 0

        beads_distance_sum = 0.0
        beads_count = 0

        synthetic_distance_sum = 0.0
        synthetic_count = 0

        weighted_squared_sum = 0.0
        total_weight = 0.0

        for tile in self.tiles:
            self.update_cost(tile)

            total_tile_distance += tile["distance"]

            p1_w = tile["_p1_w"]
            p2_w = tile["_p2_w"]
            weights = tile["_weights"]

            if len(p1_w) == 0:
                continue

            distances = np.linalg.norm(
                p1_w - p2_w,
                axis=1,
            )

            active_mask = weights > 0.0

            beads_mask = (
                ~tile["_is_synthetic"]
                & active_mask
            )

            synthetic_mask = (
                tile["_is_synthetic"]
                & active_mask
            )

            if np.any(active_mask):
                active_distances = distances[active_mask]

                all_distance_sum += float(
                    np.sum(active_distances)
                )

                all_count += len(active_distances)

            if np.any(beads_mask):
                bead_distances = distances[beads_mask]

                beads_distance_sum += float(
                    np.sum(bead_distances)
                )

                beads_count += len(bead_distances)

            if np.any(synthetic_mask):
                synthetic_distances = distances[
                    synthetic_mask
                ]

                synthetic_distance_sum += float(
                    np.sum(synthetic_distances)
                )

                synthetic_count += len(
                    synthetic_distances
                )

            weighted_squared_sum += float(
                np.sum(
                    distances
                    * distances
                    * weights
                )
            )

            total_weight += float(
                np.sum(weights)
            )

        mean_tile_error = (
            total_tile_distance / len(self.tiles)
        )

        all_match_mean = (
            all_distance_sum / all_count
            if all_count > 0
            else 0.0
        )

        beads_mean = (
            beads_distance_sum / beads_count
            if beads_count > 0
            else 0.0
        )

        synthetic_mean = (
            synthetic_distance_sum / synthetic_count
            if synthetic_count > 0
            else 0.0
        )

        weighted_rms = (
            math.sqrt(
                weighted_squared_sum / total_weight
            )
            if total_weight > 0.0
            else 0.0
        )

        self.error_breakdown = {
            "mean_tile_error": mean_tile_error,
            "all_match_mean": all_match_mean,
            "beads_mean": beads_mean,
            "synthetic_mean": synthetic_mean,
            "weighted_rms": weighted_rms,
            "all_count": all_count,
            "beads_count": beads_count,
            "synthetic_count": synthetic_count,
        }

        return weighted_rms

    def rigid_fit_model(self, rigid_model, tile):
        """
        Compute a weighted best-fit rigid transform using active matches only.
        """
        all_weights = np.asarray(
            tile["_weights"],
            dtype=np.float64,
        )

        active_mask = all_weights > 0.0

        P = tile["_p1_l"][active_mask]
        Q = tile["_p2_w"][active_mask]
        weights = all_weights[active_mask]

        if len(P) == 0:
            return rigid_model

        if len(P) != len(Q) or len(P) != len(weights):
            raise ValueError(
                "Rigid fit received mismatched point and weight counts"
            )

        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
        ):
            raise ValueError(
                "Rigid fit weights must be finite and non-negative"
            )

        weight_sum = float(
            np.sum(weights)
        )

        if weight_sum <= 0.0:
            return rigid_model

        pc = np.sum(
            P * weights[:, None],
            axis=0,
        ) / weight_sum

        qc = np.sum(
            Q * weights[:, None],
            axis=0,
        ) / weight_sum

        X = P - pc
        Y = Q - qc

        S = (
            X * weights[:, None]
        ).T @ Y

        Sxx, Sxy, Sxz = S[0, :]
        Syx, Syy, Syz = S[1, :]
        Szx, Szy, Szz = S[2, :]

        N = np.array(
            [
                [
                    Sxx + Syy + Szz,
                    Syz - Szy,
                    Szx - Sxz,
                    Sxy - Syx,
                ],
                [
                    Syz - Szy,
                    Sxx - Syy - Szz,
                    Sxy + Syx,
                    Szx + Sxz,
                ],
                [
                    Szx - Sxz,
                    Sxy + Syx,
                    -Sxx + Syy - Szz,
                    Syz + Szy,
                ],
                [
                    Sxy - Syx,
                    Szx + Sxz,
                    Syz + Szy,
                    -Sxx - Syy + Szz,
                ],
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(N)):
            raise ValueError(
                "Matrix N contains NaNs or Infs"
            )

        eigenvalues, eigenvectors = np.linalg.eigh(N)

        q = eigenvectors[
            :,
            np.argmax(eigenvalues),
        ]

        q_norm = float(
            np.linalg.norm(q)
        )

        if q_norm == 0.0 or not np.isfinite(q_norm):
            raise ValueError(
                "Invalid quaternion norm during rigid fit"
            )

        q /= q_norm

        q0, qx, qy, qz = q

        R = np.array(
            [
                [
                    q0 * q0 + qx * qx - qy * qy - qz * qz,
                    2.0 * (qx * qy - q0 * qz),
                    2.0 * (qx * qz + q0 * qy),
                ],
                [
                    2.0 * (qy * qx + q0 * qz),
                    q0 * q0 - qx * qx + qy * qy - qz * qz,
                    2.0 * (qy * qz - q0 * qx),
                ],
                [
                    2.0 * (qz * qx - q0 * qy),
                    2.0 * (qz * qy + q0 * qx),
                    q0 * q0 - qx * qx - qy * qy + qz * qz,
                ],
            ],
            dtype=np.float64,
        )

        t = qc - R @ pc

        rigid_model["m00"], rigid_model["m01"], rigid_model["m02"] = R[0]
        rigid_model["m10"], rigid_model["m11"], rigid_model["m12"] = R[1]
        rigid_model["m20"], rigid_model["m21"], rigid_model["m22"] = R[2]

        rigid_model["m03"] = float(t[0])
        rigid_model["m13"] = float(t[1])
        rigid_model["m23"] = float(t[2])

        return rigid_model

    def optimize_one_pass(self, iteration_offset):
        """
        Run one complete optimization pass using the current match weights.
        """
        self.reset_observer()

        i = 0
        proceed = i < self.max_iterations
        stop_reason = "maximum iterations reached"

        while proceed:
            self.refresh_target_arrays_from_matches()

            for tile in self.tiles:
                if tile.get("fixed", False):
                    continue

                self.fit(tile)
                self.apply_damp(tile)
                self.sync_tile_array_to_matches(tile)

            error = self.update_errors()

            self.update_observer(error)

            self.append_iteration_metrics(
                iteration_offset + i,
                error,
            )

            if i > self.max_plateauwidth:
                proceed = error > self.max_allowed_error
                d = self.max_plateauwidth

                while not proceed and d >= 1:
                    slope = abs(
                        self.get_wide_slope(
                            self.observer["values"],
                            d,
                        )
                    )

                    proceed = (
                        proceed
                        or slope > 0.0001
                    )

                    d /= 2

                if not proceed:
                    stop_reason = (
                        f"weighted RMS reached target "
                        f"{self.max_allowed_error:.6f} "
                        f"and stabilized"
                    )

            i += 1

            if i >= self.max_iterations:
                proceed = False
                stop_reason = "maximum iterations reached"

        self.update_errors()

        return i, stop_reason

    def reject_bad_real_matches(self):
        """
        Reject bad individual real correspondences after a completed solve.
        Residual thresholds are computed independently for every edge. Synthetic
        constraints are never rejected
        """
        self.sync_arrays_to_matches()

        copies_by_match_key = {}
        unique_matches = {}

        for tile in self.tiles:
            for match in tile.get("matches", []):
                if self.is_synthetic_match(match):
                    continue

                match_key = match.get("match_key")
                edge_key = match.get("edge_key")

                if match_key is None or edge_key is None:
                    continue

                copies_by_match_key.setdefault(
                    match_key,
                    [],
                ).append(match)

                if match_key in unique_matches:
                    continue

                current_weight = float(
                    match.get(
                        "weight",
                        match.get("base_weight", 1.0),
                    )
                )

                p1 = np.asarray(
                    match["p1"]["w"],
                    dtype=np.float64,
                )

                p2 = np.asarray(
                    match["p2"]["w"],
                    dtype=np.float64,
                )

                unique_matches[match_key] = {
                    "edge_key": edge_key,
                    "weight": current_weight,
                    "residual": float(
                        np.linalg.norm(p1 - p2)
                    ),
                }

        matches_by_edge = {}

        for match_key, record in unique_matches.items():
            if record["weight"] <= 0.0:
                continue

            matches_by_edge.setdefault(
                record["edge_key"],
                [],
            ).append(
                (
                    match_key,
                    record["residual"],
                )
            )

        rejected_match_keys = set()
        candidate_count = 0
        changed_edges = 0

        for edge_key, edge_matches in matches_by_edge.items():
            active_count = len(edge_matches)

            if active_count <= self.min_matches:
                continue

            residuals = np.asarray(
                [
                    residual
                    for _, residual in edge_matches
                ],
                dtype=np.float64,
            )

            median = float(
                np.median(residuals)
            )

            mad = float(
                np.median(
                    np.abs(residuals - median)
                )
            )

            cleanup_sigma = 1.4826 * mad

            threshold = max(
                float(self.absolute_threshold),
                median
                + float(self.relative_threshold)
                * cleanup_sigma,
            )

            candidates = [
                (match_key, residual)
                for match_key, residual in edge_matches
                if residual > threshold
            ]

            if not candidates:
                continue

            candidates.sort(
                key=lambda item: item[1],
                reverse=True,
            )

            candidate_count += len(candidates)

            maximum_drop_count = (
                active_count - self.min_matches
            )

            drop_count = min(
                len(candidates),
                maximum_drop_count,
            )

            if drop_count <= 0:
                continue

            changed_edges += 1

            for match_key, _ in candidates[:drop_count]:
                rejected_match_keys.add(match_key)

        for match_key in rejected_match_keys:
            for match in copies_by_match_key.get(
                match_key,
                [],
            ):
                match["cleanup_weight"] = 0.0
                match["weight"] = 0.0

        self.refresh_weight_arrays_from_matches()

        return {
            "rejected": len(rejected_match_keys),
            "candidate_count": candidate_count,
            "changed_edges": changed_edges,
            "active_edges": len(matches_by_edge),
        }

    def affine_fit_model(self, affine_model, tile):
        """
        Compute a weighted full 3D affine fit using active matches only.
        """
        all_weights = np.asarray(
            tile["_weights"],
            dtype=np.float64,
        )

        active_mask = all_weights > 0.0

        P = tile["_p1_l"][active_mask]
        Q = tile["_p2_w"][active_mask]
        weights = all_weights[active_mask]

        if len(P) < 4:
            return affine_model

        if len(P) != len(Q) or len(P) != len(weights):
            raise ValueError(
                "Affine fit received mismatched point and weight counts"
            )

        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
        ):
            raise ValueError(
                "Affine fit weights must be finite and non-negative"
            )

        weight_sum = float(
            np.sum(weights)
        )

        if weight_sum <= 0.0:
            return affine_model

        pc = np.sum(
            P * weights[:, None],
            axis=0,
        ) / weight_sum

        qc = np.sum(
            Q * weights[:, None],
            axis=0,
        ) / weight_sum

        X = P - pc
        Y = Q - qc

        sqrt_weights = np.sqrt(
            weights
        )[:, None]

        weighted_X = X * sqrt_weights
        weighted_Y = Y * sqrt_weights

        M_t, _, rank, _ = np.linalg.lstsq(
            weighted_X,
            weighted_Y,
            rcond=None,
        )

        if rank < 3:
            raise ValueError(
                "Weighted affine fit is rank deficient"
            )

        M = M_t.T
        t = qc - M @ pc

        affine_model["m00"], affine_model["m01"], affine_model["m02"] = M[0]
        affine_model["m10"], affine_model["m11"], affine_model["m12"] = M[1]
        affine_model["m20"], affine_model["m21"], affine_model["m22"] = M[2]

        affine_model["m03"] = float(t[0])
        affine_model["m13"] = float(t[1])
        affine_model["m23"] = float(t[2])

        return affine_model

    def fit(self, tile):
        """Fit the active transform model against current target world points."""
        if len(tile["_p1_l"]) == 0:
            return

        rigid = self.rigid_fit_model(tile["model"]["b"], tile)
        tile["model"]["b"] = rigid

        if self.run_type == "rigid":
            return

        elif self.run_type == "split-affine" or self.run_type == "affine":
            affine = self.affine_fit_model(tile["model"]["a"], tile)
        else:
            raise ValueError(f"Unknown run_type: {self.run_type}")

        tile["model"]["a"] = affine
        tile["model"]["regularized"] = self.regularize_models(affine, rigid)

    def apply_damp(self, tile):
        """
        Damp current p1 world positions toward the tile's model-applied local points.
        """
        if tile["_p1_l"].shape[0] == 0:
            return

        model = self.get_active_model(tile)
        target = self.apply_model_array(tile["_p1_l"], model)

        tile["_p1_w"] += self.damp * (target - tile["_p1_w"])
    
    def apply(self):
        for tile in self.tiles:
            if tile["_p1_l"].shape[0] == 0:
                continue

            if not tile.get("fixed", False):
                model = self.get_active_model(tile)
                tile["_p1_w"][:] = self.apply_model_array(
                    tile["_p1_l"],
                    model,
                )

            self.sync_tile_array_to_matches(tile)

    def optimize_silently(self):
        """
        Solve, reject bad real correspondences within each edge, and re-solve.
        """
        if not self.tiles:
            return

        self.prepare_tile_arrays()
        self.apply()

        total_iterations = 0
        stop_reason = "maximum iterations reached"
        cleanup_history = []

        for cleanup_round in range(self.max_cleanup_rounds + 1):
            pass_iterations, stop_reason = self.optimize_one_pass(
                iteration_offset=total_iterations,
            )

            total_iterations += pass_iterations

            # The last pass is solve-only.
            if cleanup_round >= self.max_cleanup_rounds:
                break

            cleanup = self.reject_bad_real_matches()
            cleanup["round"] = cleanup_round + 1
            cleanup_history.append(cleanup)

            if cleanup["rejected"] == 0:
                break

        final_error = self.update_errors()
        final_breakdown = dict(self.error_breakdown)
        final_stats = self.get_final_match_stats()

        solve_stats = self.validation_stats["solve_metrics_per_tile"]
        solve_stats["i"] = total_iterations
        solve_stats["stop_reason"] = stop_reason
        solve_stats["initial"] = self.initial_pos
        solve_stats["final"] = final_breakdown
        solve_stats["final_error"] = final_error
        solve_stats["final_match_stats"] = final_stats
        solve_stats["cleanup_history"] = cleanup_history

        total_dropped = sum(
            cleanup["rejected"]
            for cleanup in cleanup_history
        )

        print("\n" + "=" * 72)
        print(f"{self.run_type.upper()} SOLVER")
        print("=" * 72)
        print(f"Iterations:         {total_iterations:,}")
        print(f"Final weighted RMS: {final_error:.6f}")
        print(f"Final bead mean:    {final_breakdown['beads_mean']:.6f}")
        print(f"Cleanup rounds:     {len(cleanup_history)}")

        for cleanup in cleanup_history:
            print(
                f"  Round {cleanup['round']}: "
                f"dropped {cleanup['rejected']:,} matches"
            )

        print(f"Total dropped:      {total_dropped:,}")
        print(f"Stop:               {stop_reason}")
        print("=" * 72)

        self.sync_arrays_to_matches()
    
    def run(self):
        """
        Executes the entry point of the solver.
        """
        self.optimize_silently()
        print("Global Optimization Complete")

        return self.tiles, self.validation_stats