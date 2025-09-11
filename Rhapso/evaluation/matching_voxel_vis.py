from matplotlib import pyplot as plt
import numpy as np

class VoxelVis:
    def __init__(self, view_id, matches):
        self.view_id = view_id
        self.matches = matches
        self.just_points = []
        self.voxel_size = 10

    def get_matches(self):
        for match in self.matches[self.view_id]:
            self.just_points.append(match["p1"])
            self.just_points.append(match["p2"])

    def validate_input(self):
        points = np.array(self.just_points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("match_points must be a 2D array with shape (n_points, 3)")

        if len(points) <= 1:
            raise ValueError("Input must have more than one coordinate")

    def compute_bounding_box(self):
        self.min_coords = np.min(self.just_points, axis=0)
        self.max_coords = np.max(self.just_points, axis=0)
        self.bounding_box = {
            "min": dict(zip(["x", "y", "z"], self.min_coords)),
            "max": dict(zip(["x", "y", "z"], self.max_coords)),
        }

    def create_voxel_grid(self):
        # Creates the grid and voxels by mapping the points to the voxel indices in which they lay
        self.grid_shape = (
            np.ceil((self.max_coords - self.min_coords) / self.voxel_size).astype(int)
            + 1
        )
        self.voxel_grid = np.zeros(self.grid_shape, dtype=int)
        for point in self.just_points:
            voxel_index = np.floor((point - self.min_coords) / self.voxel_size).astype(
                int
            )
            self.voxel_grid[tuple(voxel_index)] += 1

    def visualize_voxel_grid(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        non_empty_voxels = np.argwhere(self.voxel_grid > 0)
        counts = self.voxel_grid[self.voxel_grid > 0]

        # Viridis coloring shows density of points within voxels. Purple is low density, yellow is high density.
        ax.scatter(
            non_empty_voxels[:, 0],
            non_empty_voxels[:, 1],
            non_empty_voxels[:, 2],
            c=counts,
            cmap="viridis",
            marker="s",
        )
        ax.set(xlabel="X Voxel Index", ylabel="Y Voxel Index", zlabel="Z Voxel Index")
        ax.set_title(f"3D Voxel Grid of view_id:{self.view_id}")
        plt.show()

    def run_voxel_vis(self):
        self.get_matches()
        self.validate_input()
        self.compute_bounding_box()
        self.create_voxel_grid()
        self.visualize_voxel_grid()
