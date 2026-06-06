import numpy as np
import math

"""
GlobalOptimization iteratively refines per-tile transforms to achieve sub-pixel alignment
using matched point correspondences.

This optimized version keeps the NumPy hot path, but preserves the original global
optimization behavior by syncing source-side world points back into the match graph
and refreshing target-side world points from the live match graph each iteration.
"""


class GlobalOptimization:
    def __init__(
        self,
        tiles,
        relative_threshold,
        absolute_threshold,
        min_matches,
        damp,
        regularization_weight,
        max_iterations,
        max_allowed_error,
        max_plateauwidth,
        run_type,
        metrics_output_path,
    ):
        self.tiles = tiles
        self.relative_threshold = relative_threshold
        self.absolute_threshold = absolute_threshold
        self.min_matches = min_matches
        self.damp = damp
        self.regularization_weight = regularization_weight
        self.max_iterations = max_iterations
        self.max_allowed_error = max_allowed_error
        self.max_plateauwidth = max_plateauwidth
        self.run_type = run_type
        self.metrics_output_path = metrics_output_path

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

    # --------------------------------------------------
    # Data preparation / synchronization
    # --------------------------------------------------

    def prepare_tile_arrays(self):
        """
        Convert match dictionaries into NumPy arrays once.

        The important part:
          _p1_l is static local source points.
          _p1_w is this tile's current optimized world source points.
          _p2_w is only a cache and must be refreshed from live matches.
          _weights is static match weights.
        """
        for tile in self.tiles:
            matches = tile.get("matches", [])

            if len(matches) == 0:
                tile["_p1_l"] = np.empty((0, 3), dtype=np.float64)
                tile["_p1_w"] = np.empty((0, 3), dtype=np.float64)
                tile["_p2_w"] = np.empty((0, 3), dtype=np.float64)
                tile["_weights"] = np.empty((0,), dtype=np.float64)
                continue

            tile["_p1_l"] = np.asarray(
                [m["p1"]["l"] for m in matches],
                dtype=np.float64,
            ).reshape(-1, 3)

            tile["_p1_w"] = np.asarray(
                [m["p1"]["w"] for m in matches],
                dtype=np.float64,
            ).reshape(-1, 3)

            # This starts as a snapshot, but it is refreshed every iteration.
            tile["_p2_w"] = np.asarray(
                [m["p2"]["w"] for m in matches],
                dtype=np.float64,
            ).reshape(-1, 3)

            tile["_weights"] = np.asarray(
                [m.get("weight", 1.0) for m in matches],
                dtype=np.float64,
            )

    def refresh_target_arrays_from_matches(self):
        """
        Refresh cached target-side world coordinates from the live match graph.

        This is the critical fix. The old optimized version froze _p2_w once.
        That changed the global optimization because tiles no longer fit against
        currently updated neighbor positions.
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

        This preserves the original solver behavior where match["p1"]["w"] is
        updated during optimization and can be seen by neighboring tiles through
        their match["p2"]["w"] references.
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

    # --------------------------------------------------
    # Observer / convergence
    # --------------------------------------------------

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
        Store compact metrics instead of deepcopying the whole growing observer.
        """
        self.validation_stats.setdefault("solver_metrics_per_tile", {}).setdefault("stats", []).append(
            {
                "iteration": i,
                "error": error,
                "observer": {
                    "mean": self.observer["mean"],
                    "median": self.observer["median"],
                    "min": self.observer["min"],
                    "max": self.observer["max"],
                    "std": self.observer["std"],
                    "slope": self.observer["slope"][-1] if self.observer["slope"] else 0.0,
                },
            }
        )

    # --------------------------------------------------
    # Model helpers
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Cost / error scoring
    # --------------------------------------------------

    def update_cost(self, tile):
        """
        Computes and stores average distance and weighted cost for one tile.

        Assumes _p2_w has already been refreshed from the live match graph.
        """
        p1_w = tile["_p1_w"]
        p2_w = tile["_p2_w"]
        weights = tile["_weights"]

        if p1_w.shape[0] == 0:
            tile["model"]["cost"] = 0.0
            tile["cost"] = 0.0
            tile["distance"] = 0.0
            return

        diff = p1_w - p2_w
        distances = np.linalg.norm(diff, axis=1)

        distance = float(distances.mean())

        sum_weight = float(weights.sum())
        if sum_weight > 0:
            cost = float(np.sum(distances * distances * weights) / sum_weight)
        else:
            cost = 0.0

        tile["model"]["cost"] = cost
        tile["cost"] = cost
        tile["distance"] = distance

    def update_errors(self):
        """
        Refresh target arrays and score the current global state.
        """
        if not self.tiles:
            return 0.0

        self.refresh_target_arrays_from_matches()

        total_distance = 0.0
        min_error = float("inf")
        max_error = 0.0

        for tile in self.tiles:
            self.update_cost(tile)

            distance = tile["distance"]
            min_error = min(min_error, distance)
            max_error = max(max_error, distance)
            total_distance += distance

        return total_distance / len(self.tiles)

    # --------------------------------------------------
    # Model fitting
    # --------------------------------------------------

    def rigid_fit_model(self, rigid_model, tile):
        """
        Computes best-fit rigid transform using quaternion-based estimation.
        Vectorized over all matches in the tile.
        """
        P = tile["_p1_l"]
        Q = tile["_p2_w"]

        if P.shape[0] == 0:
            return rigid_model

        pc = P.mean(axis=0)
        qc = Q.mean(axis=0)

        X = P - pc
        Y = Q - qc

        S = X.T @ Y

        Sxx, Sxy, Sxz = S[0, :]
        Syx, Syy, Syz = S[1, :]
        Szx, Szy, Szz = S[2, :]

        N = np.array(
            [
                [Sxx + Syy + Szz, Syz - Szy,       Szx - Sxz,       Sxy - Syx],
                [Syz - Szy,       Sxx - Syy - Szz, Sxy + Syx,       Szx + Sxz],
                [Szx - Sxz,       Sxy + Syx,      -Sxx + Syy - Szz, Syz + Szy],
                [Sxy - Syx,       Szx + Sxz,       Syz + Szy,      -Sxx - Syy + Szz],
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(N)):
            raise ValueError("Matrix N contains NaNs or Infs")

        eigenvalues, eigenvectors = np.linalg.eigh(N)
        q = eigenvectors[:, np.argmax(eigenvalues)]

        q_norm = np.linalg.norm(q)
        if q_norm == 0 or not np.isfinite(q_norm):
            raise ValueError("Invalid quaternion norm during rigid fit")

        q /= q_norm
        q0, qx, qy, qz = q

        R = np.array(
            [
                [
                    q0 * q0 + qx * qx - qy * qy - qz * qz,
                    2 * (qx * qy - q0 * qz),
                    2 * (qx * qz + q0 * qy),
                ],
                [
                    2 * (qy * qx + q0 * qz),
                    q0 * q0 - qx * qx + qy * qy - qz * qz,
                    2 * (qy * qz - q0 * qx),
                ],
                [
                    2 * (qz * qx - q0 * qy),
                    2 * (qz * qy + q0 * qx),
                    q0 * q0 - qx * qx - qy * qy + qz * qz,
                ],
            ],
            dtype=np.float64,
        )

        t = qc - R @ pc

        rigid_model["m00"], rigid_model["m01"], rigid_model["m02"] = R[0, :]
        rigid_model["m10"], rigid_model["m11"], rigid_model["m12"] = R[1, :]
        rigid_model["m20"], rigid_model["m21"], rigid_model["m22"] = R[2, :]
        rigid_model["m03"], rigid_model["m13"], rigid_model["m23"] = t

        return rigid_model

    def affine_fit_model(self, affine_model, tile):
        """
        Affine transformation model update.
        Vectorized over all matches in the tile.
        """
        P = tile["_p1_l"]
        Q = tile["_p2_w"]

        if P.shape[0] < 3:
            raise ValueError("Not enough matches for affine fit")

        pc = P.mean(axis=0)
        qc = Q.mean(axis=0)

        X = P - pc
        Y = Q - qc

        A = X.T @ X
        B = X.T @ Y

        try:
            # Solves A @ M_t = B.
            # M_t is the transpose of the final affine matrix.
            M_t = np.linalg.solve(A, B)
        except np.linalg.LinAlgError as e:
            raise ValueError("Affine matrix is singular") from e

        M = M_t.T
        t = qc - M @ pc

        affine_model["m00"], affine_model["m01"], affine_model["m02"] = M[0, :]
        affine_model["m10"], affine_model["m11"], affine_model["m12"] = M[1, :]
        affine_model["m20"], affine_model["m21"], affine_model["m22"] = M[2, :]
        affine_model["m03"], affine_model["m13"], affine_model["m23"] = t

        return affine_model

    def fit(self, tile):
        """
        Fits transformation models to a tile using current target-side world points.
        """
        if tile["_p1_l"].shape[0] == 0:
            return

        affine = self.affine_fit_model(tile["model"]["a"], tile)
        rigid = self.rigid_fit_model(tile["model"]["b"], tile)
        regularized = self.regularize_models(affine, rigid)

        tile["model"]["a"] = affine
        tile["model"]["b"] = rigid
        tile["model"]["regularized"] = regularized

    # --------------------------------------------------
    # Application / dampening
    # --------------------------------------------------

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
        """
        Apply current model to local points to initialize p1 world positions.
        Also syncs these initialized positions into the live match graph.
        """
        for tile in self.tiles:
            if tile["_p1_l"].shape[0] == 0:
                continue

            model = self.get_active_model(tile)
            tile["_p1_w"][:] = self.apply_model_array(tile["_p1_l"], model)

            # Important: make initialized world points visible globally.
            self.sync_tile_array_to_matches(tile)

    # --------------------------------------------------
    # Optimization loop
    # --------------------------------------------------

    def optimize_silently(self):
        """
        Iteratively refines tile alignments until convergence or max iterations.

        The critical global behavior is:
          1. refresh target arrays from the live match graph,
          2. fit each tile against current target positions,
          3. damp source positions,
          4. sync source positions back into the live match graph,
          5. average current global error,
          6. repeat.
        """
        if not self.tiles:
            return

        self.prepare_tile_arrays()
        self.apply()

        i = 0
        proceed = i < self.max_iterations

        while proceed:
            # Pull current neighbor/global positions into _p2_w.
            self.refresh_target_arrays_from_matches()

            for tile in self.tiles:
                # Fit against the current live global state.
                self.fit(tile)

                # Update this tile's private optimized world points.
                self.apply_damp(tile)

                # Make this tile's update visible to the rest of the graph.
                self.sync_tile_array_to_matches(tile)

            # Score after all tile updates using current global positions.
            error = self.update_errors()
            self.update_observer(error)
            self.append_iteration_metrics(i, error)

            if i > self.max_plateauwidth:
                proceed = error > self.max_allowed_error
                d = self.max_plateauwidth

                while not proceed and d >= 1:
                    proceed = proceed or abs(self.get_wide_slope(self.observer["values"], d)) > 0.0001
                    d /= 2

            i += 1

            if i >= self.max_iterations:
                proceed = False

        self.validation_stats["solve_metrics_per_tile"]["i"] = i

        # Final downstream sync.
        self.sync_arrays_to_matches()

    def run(self):
        """
        Executes the entry point of the solver.
        """
        self.optimize_silently()
        return self.tiles, self.validation_stats