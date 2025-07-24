import datetime
from Rhapso.accuracy_metrics.save_metrics import JSONFileHandler
from Rhapso.solver.align_tiles import AlignTiles
import numpy as np

# This class implements a refinement process for aligning point matches

class GlobalOptimization:
    def __init__(self, tiles, pmc, fixed_views, data_prefix, alignment_option, relative_threshold,
                absolute_threshold, min_matches, damp, max_iterations, max_allowed_error, max_plateauwidth, model, metrics_output_path):
        self.tiles = tiles
        self.fixed_views = fixed_views
        self.data_prefix = data_prefix
        self.alignment_option = alignment_option
        self.relative_threshold = relative_threshold
        self.absolute_threshold = absolute_threshold
        self.min_matches = min_matches
        self.damp = damp
        self.max_iterations = max_iterations
        self.max_allowed_error = max_allowed_error
        self.max_plateauwidth = max_plateauwidth
        self.model = model
        self.metrics_output_path = metrics_output_path

        self.align_tile = AlignTiles(pmc, tiles, fixed_views)

        # self.save_metrics = JSONFileHandler(self.metrics_output_path)

    def update_cost(self, view, tile):
        """
        Computes and stores the average distance and weighted cost (fit quality) of point matches for a tile.
        """
        total_distance = 0.0
        total_cost = 0.0

        if len(tile["matches"]) > 0:
            sum_weight = 0.0

            for match in tile["matches"]:
                p1 = match["p1"]
                p2 = match["p2"]
                distance = np.linalg.norm(np.array(p1) - np.array(p2))

                squared_dist = np.sum((p1 - p2) ** 2)
                distance = np.sqrt(squared_dist)

                weight = match["weight"]
                total_distance += distance
                total_cost += distance * distance * weight
                sum_weight += weight

            average_distance = total_distance / len(tile["matches"])
            average_cost = total_cost / sum_weight if sum_weight != 0 else 0

            # Update tile's attributes
            self.tiles[view]["distance"] = average_distance
            self.tiles[view]["cost"] = average_cost

    def update_errors(self):
        """
        Monitor convergence by updating cost metrics for all tiles and returns the average alignment error.
        """
        total_distance = 0.0
        min_error = float("inf")
        max_error = 0.0
        tile_count = len(self.tiles)

        for view, tile in self.tiles.items():
            self.update_cost(view, tile)
            if len(self.tiles) == 0:
                print("There are no tiles to get information from.")
                return
            distance = tile["distance"]

            if distance < min_error:
                min_error = distance
            if distance > max_error:
                max_error = distance
            total_distance += distance

        if tile_count > 0:
            average_error = total_distance / tile_count
        else:
            average_error = 0

        print( f"({datetime.datetime.now()}): Min Error: {min_error}px")
        print( f"({datetime.datetime.now()}): Max Error: {max_error}px")
        print( f"({datetime.datetime.now()}): Mean Error: {average_error}px")    

        # self.save_metrics.update(
        #     "alignment errors",
        #     {
        #         "min_error": min_error,
        #         "max_error": max_error,
        #         "mean_error": average_error,
        #     },
        # )

        return average_error, min_error, max_error

    def apply_damp(self, view, tile):
        """
        Applies dampening to point positions by blending transformed and target coordinates.
        """
        model = tile["model"]["a"]
        matches = tile["matches"]

        for index, match in enumerate(matches):
            p1 = np.array(match["p1"])
            w = np.array(match["p2"])

            # Calculate transformed position
            transformed_p1 = np.array(
                [
                    p1[0] * model["m00"]
                    + p1[1] * model["m01"]
                    + p1[2] * model["m02"]
                    + model["m03"],
                    p1[0] * model["m10"]
                    + p1[1] * model["m11"]
                    + p1[2] * model["m12"]
                    + model["m13"],
                    p1[0] * model["m20"]
                    + p1[1] * model["m21"]
                    + p1[2] * model["m22"]
                    + model["m23"],
                ]
            )

            # Apply dampening to blend between current position and transformed position
            new_position = w + self.damp * (transformed_p1 - w)

            # Update tile with the new position
            self.tiles[view]["matches"][index]["p1"] = new_position.tolist()

    def fit_model(self, view, tile):
        """
        Fits and updates affine and rigid models for a tile using its point matches.
        """
        a = self.align_tile.affine_fit_model(tile["model"]["a"], tile["matches"])
        b = self.align_tile.rigid_fit_model(tile["model"]["b"], tile["matches"])
        self.tiles[view]["model"]["a"] = a
        self.tiles[view]["model"]["b"] = b

    def apply(self, view, tile):
        """
        Applies the affine transformation to all point matches in a tile.
        """
        model = "affine"

        if model == "affine":
            model = tile["model"]["a"]
        elif model == "rigid":
            model = tile["model"]["b"]

        matches = tile["matches"]
        for index, match in enumerate(matches):
            p1 = match["p1"]
            transformed_p1 = np.zeros(3)
            transformed_p1[0] = (
                p1[0] * model["m00"]
                + p1[1] * model["m01"]
                + p1[2] * model["m02"]
                + model["m03"]
            )
            transformed_p1[1] = (
                p1[0] * model["m10"]
                + p1[1] * model["m11"]
                + p1[2] * model["m12"]
                + model["m13"]
            )
            transformed_p1[2] = (
                p1[0] * model["m20"]
                + p1[1] * model["m21"]
                + p1[2] * model["m22"]
                + model["m23"]
            )
            self.tiles[view]["matches"][index]["p1"] = transformed_p1.tolist()

    def optimize_silently(self):
        """
        Iteratively refines tile alignments using model fitting and dampening until convergence or max iterations.
        """
        iteration = 0
        errors = []

        # Apply transformation matrix to point matches to position P1 very close to P2
        for view, tile in self.tiles.items():
            self.apply(view, tile)

        while iteration < self.max_iterations:
            for view, tile in self.tiles.items():
                if view in self.fixed_views:
                    continue
                self.fit_model(view, tile)
                self.apply_damp(view, tile)

            current_error, min_error, max_error = self.update_errors()
            errors.append(current_error)

            if current_error < self.max_allowed_error:
                print(f"Convergence reached at iteration {iteration} with error {current_error}")
                break
                
            # TODO - Finish implementing this rare use case
            if iteration > self.max_plateauwidth:
                recent_errors = errors[-self.max_plateauwidth:]
                error_change = recent_errors[-1] - recent_errors[0]
                slope = abs(error_change / self.max_plateauwidth)

            iteration += 1
            print(f"Iteration {iteration}: Average Error = {current_error}")  
        
    def compute_tiles(self):
        """
        Interface the types of optimization set with user params
        """
        # 1 round simple
        if(self.alignment_option == 1):
            self.optimize_silently()
        
        # 1 round interative
        if(self.alignment_option == 2):
            pass
        
        # 2 round simple
        if(self.alignment_option == 3):
            pass

    def run(self):
        """
        Executes the entry point of the script.
        """
        self.compute_tiles()
        return self.tiles