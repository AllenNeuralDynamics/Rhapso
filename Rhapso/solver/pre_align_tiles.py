import numpy as np
import random

"""
Pre Align Tiles roughly align p1 with p2 to speed up global optimization rounds
"""

class PreAlignTiles:
    def __init__(self, min_matches, run_type, fixed_tile, regularization_weight):
        self.min_matches = min_matches
        self.run_type = run_type
        self.fixed_tile = fixed_tile
        self.regularization_weight = regularization_weight
    
    def compute_error_breakdown(self, tiles):
        """
        Compute the same starting error metrics used by GlobalOptimization,
        before pre-alignment changes any tile models.
        """
        if not tiles:
            return {
                "mean_tile_error": 0.0,
                "all_match_mean": 0.0,
                "beads_mean": 0.0,
                "synthetic_mean": 0.0,
                "weighted_rms": 0.0,
                "all_count": 0,
                "beads_count": 0,
                "synthetic_count": 0,
            }

        tile_mean_sum = 0.0
        all_distance_sum = 0.0
        all_count = 0
        beads_distance_sum = 0.0
        beads_count = 0
        synthetic_distance_sum = 0.0
        synthetic_count = 0
        weighted_squared_sum = 0.0
        total_weight = 0.0

        for tile in tiles:
            matches = tile.get("matches", [])

            if not matches:
                tile_mean_sum += 0.0
                continue

            p1_w = np.asarray(
                [match["p1"]["w"] for match in matches],
                dtype=np.float64,
            )

            p2_w = np.asarray(
                [match["p2"]["w"] for match in matches],
                dtype=np.float64,
            )

            # weights = np.asarray(
            #     [match.get("weight", 1.0) for match in matches],
            #     dtype=np.float64,
            # )

            weights = np.asarray(
                [
                    match.get(
                        "base_weight",
                        match.get("weight", 1.0),
                    )
                    for match in matches
                ],
                dtype=np.float64,
            )

            distances = np.linalg.norm(
                p1_w - p2_w,
                axis=1,
            )

            tile_mean_sum += float(np.mean(distances))

            all_distance_sum += float(np.sum(distances))
            all_count += len(distances)

            synthetic_mask = np.asarray(
                [
                    match.get(
                        "is_synthetic",
                        match.get(
                            "base_weight",
                            match.get("weight", 1.0),
                        ) < 1.0,
                    )
                    for match in matches
                ],
                dtype=bool,
            )

            beads_mask = ~synthetic_mask

            if np.any(beads_mask):
                bead_distances = distances[beads_mask]
                beads_distance_sum += float(np.sum(bead_distances))
                beads_count += len(bead_distances)

            if np.any(synthetic_mask):
                synthetic_distances = distances[synthetic_mask]
                synthetic_distance_sum += float(
                    np.sum(synthetic_distances)
                )
                synthetic_count += len(synthetic_distances)

            weighted_squared_sum += float(
                np.sum(distances * distances * weights)
            )
            total_weight += float(np.sum(weights))

        return {
            "mean_tile_error": tile_mean_sum / len(tiles),
            "all_match_mean": (
                all_distance_sum / all_count
                if all_count > 0
                else 0.0
            ),
            "beads_mean": (
                beads_distance_sum / beads_count
                if beads_count > 0
                else 0.0
            ),
            "synthetic_mean": (
                synthetic_distance_sum / synthetic_count
                if synthetic_count > 0
                else 0.0
            ),
            "weighted_rms": (
                np.sqrt(weighted_squared_sum / total_weight)
                if total_weight > 0.0
                else 0.0
            ),
            "all_count": all_count,
            "beads_count": beads_count,
            "synthetic_count": synthetic_count,
        }
    
    def rigid_fit_model(self, rigid_model, matches):
        """
        Compute a weighted best-fit rigid transform.
        """
        if len(matches) == 0:
            return rigid_model

        P = np.asarray(
            [match["p1"]["l"] for match in matches],
            dtype=np.float64,
        )

        Q = np.asarray(
            [match["p2"]["w"] for match in matches],
            dtype=np.float64,
        )

        weights = np.asarray(
            [
                match.get(
                    "base_weight",
                    match.get(
                        "weight",
                        match.get("p1", {}).get("weight", 1.0),
                    ),
                )
                for match in matches
            ],
            dtype=np.float64,
        )

        # weights = np.asarray(
        #     [
        #         match.get(
        #             "weight",
        #             match.get("p1", {}).get("weight", 1.0),
        #         )
        #         for match in matches
        #     ],
        #     dtype=np.float64,
        # )

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

        weight_sum = float(np.sum(weights))

        if weight_sum <= 0.0:
            raise ValueError(
                "Rigid fit requires positive total weight"
            )

        # Weighted centroids.
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

        # Weighted cross-covariance.
        S = (X * weights[:, None]).T @ Y

        Sxx, Sxy, Sxz = S[0]
        Syx, Syy, Syz = S[1]
        Szx, Szy, Szz = S[2]

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
        q = eigenvectors[:, np.argmax(eigenvalues)]

        q_norm = float(np.linalg.norm(q))

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

    def affine_fit_model(self, affine_model, matches):
        """
        Compute a weighted full 3D affine transform.

        This function is selected by fit() for split-affine.
        """
        if len(matches) < 4:
            raise ValueError(
                "Not enough matches for affine fit"
            )

        P = np.asarray(
            [match["p1"]["l"] for match in matches],
            dtype=np.float64,
        )

        Q = np.asarray(
            [match["p2"]["w"] for match in matches],
            dtype=np.float64,
        )

        weights = np.asarray(
            [
                match.get(
                    "base_weight",
                    match.get(
                        "weight",
                        match.get("p1", {}).get("weight", 1.0),
                    ),
                )
                for match in matches
            ],
            dtype=np.float64,
        )

        # weights = np.asarray(
        #     [
        #         match.get(
        #             "weight",
        #             match.get("p1", {}).get("weight", 1.0),
        #         )
        #         for match in matches
        #     ],
        #     dtype=np.float64,
        # )

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

        weight_sum = float(np.sum(weights))

        if weight_sum <= 0.0:
            raise ValueError(
                "Affine fit requires positive total weight"
            )

        # Weighted centroids.
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

        # Convert weighted least squares into ordinary least squares
        # by multiplying each row by sqrt(weight).
        sqrt_weights = np.sqrt(weights)[:, None]

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
    
    def regularize_models(self, affine, rigid):
        """
        Blend affine and rigid models into a single "regularized" 3x4 affine by convex combination 
        (90% affine, 10% rigid)
        """
        alpha=self.regularization_weight
        l1 = 1.0 - alpha

        def to_array(model):
            return [
                model['m00'], model['m01'], model['m02'], model['m03'], 
                model['m10'], model['m11'], model['m12'], model['m13'],  
                model['m20'], model['m21'], model['m22'], model['m23'], 
            ]

        afs = to_array(affine)
        bfs = to_array(rigid)
        rfs = [l1 * a + alpha * b for a, b in zip(afs, bfs)]

        keys = [
            'm00', 'm01', 'm02', 'm03',
            'm10', 'm11', 'm12', 'm13',
            'm20', 'm21', 'm22', 'm23',
        ]
        regularized = dict(zip(keys, rfs))

        return regularized
    
    def fit(self, tile, pm):
        """
        Fit the same model family used by global optimization.
        """
        rigid = self.rigid_fit_model(
            tile["model"]["b"],
            pm,
        )
        tile["model"]["b"] = rigid

        if self.run_type == "rigid":
            return tile

        elif self.run_type == "split-affine" or self.run_type == "affine":
            affine = self.affine_fit_model(
                tile["model"]["a"],
                pm,
            )
        else:
            raise ValueError(
                f"Unknown run_type: {self.run_type}"
            )

        tile["model"]["a"] = affine
        tile["model"]["regularized"] = self.regularize_models(
            affine,
            rigid,
        )

        return tile
    
    def get_connected_point_matches(self, target_tile, reference_tile):
        """
        Finds point matches in the target tile that connect to the reference tile.
        """
        reference_point_ids = {id(match['p1']) for match in reference_tile['matches']}

        # Collect matches in the target tile that connect to any reference point by object identity
        connected_point_matches = [
            match for match in target_tile['matches']
            if id(match['p2']) in reference_point_ids
        ]

        return connected_point_matches

    def apply_model_in_place(self, point, model):
        x, y, z = point[0], point[1], point[2]
        point[0] = model['m00'] * x + model['m01'] * y + model['m02'] * z + model['m03']
        point[1] = model['m10'] * x + model['m11'] * y + model['m12'] * z + model['m13']
        point[2] = model['m20'] * x + model['m21'] * y + model['m22'] * z + model['m23']

        return point
    
    def apply_transform_to_tile(self, tile):  
        if self.run_type == "affine" or self.run_type == "split-affine":
            model = tile['model']['regularized'] 
        elif self.run_type == "rigid":
            model = tile['model']['b'] 
        
        for match in tile['matches']:
            match['p1']['w'][:] = match['p1']['l']
            self.apply_model_in_place(match['p1']['w'], model) 

    def pre_align(self, tiles):
        """
        Greedily seed an initial alignment.
        """
        random.shuffle(tiles["tiles"])

        unaligned_tiles = []
        aligned_tiles = []

        if not tiles["tiles"]:
            return unaligned_tiles

        for tile in tiles["tiles"]:
            tile["fixed"] = False

        if self.fixed_tile is None:
            # Use the first tile in the current list.
            seed = tiles["tiles"][0]
        else:
            seed = next(
                (
                    tile
                    for tile in tiles["tiles"]
                    if tile.get("view") == self.fixed_tile
                ),
                None,
            )

            if seed is None:
                raise ValueError(
                    f"Fixed tile '{self.fixed_tile}' not found in tiles."
                )

        seed["fixed"] = True
        tiles["fixed_tiles"] = [seed]

        for tile in tiles["tiles"]:
            if tile.get("fixed", False):
                aligned_tiles.append(tile)
            else:
                unaligned_tiles.append(tile)

        ref_index = 0

        while ref_index < len(aligned_tiles):
            if len(unaligned_tiles) == 0:
                break

            reference_tile = aligned_tiles[ref_index]
            self.apply_transform_to_tile(reference_tile)

            tiles_added = 0
            target_index = 0

            while target_index < len(unaligned_tiles):
                target_tile = unaligned_tiles[target_index]

                if any(
                    conn["view"] == target_tile["view"]
                    for conn in reference_tile["connected_tiles"]
                ):
                    pm = self.get_connected_point_matches(
                        target_tile,
                        reference_tile,
                    )

                    if len(pm) >= self.min_matches:
                        target_tile = self.fit(target_tile, pm)
                        unaligned_tiles.pop(target_index)
                        aligned_tiles.append(target_tile)
                        tiles_added += 1
                        continue

                target_index += 1

            # Always move to the next reference tile
            ref_index += 1

        return unaligned_tiles
    
    def run(self, tiles):
        """
        Executes the entry point of the script.
        """
        initial_pos = self.compute_error_breakdown(
            tiles["tiles"],
        )

        unaligned_tiles = self.pre_align(tiles)

        if len(unaligned_tiles) > 0:
            print(
                f"Unable to Pre-Align "
                f"{len(unaligned_tiles)} Tiles"
            )

        print("Tiles Pre-Aligned")
        return tiles["tiles"], initial_pos
    