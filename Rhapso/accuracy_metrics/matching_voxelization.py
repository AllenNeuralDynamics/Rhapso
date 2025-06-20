import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from scipy.stats import entropy, skew, kurtosis


class Voxelizer:
    def __init__(self, match_points, voxel_size=20):
        self.match_points = np.array(match_points)
        self.voxel_size = voxel_size
        self.voxel_grid = None
        self.voxel_counts = defaultdict(int)
        self.bounding_box = None

    def validate_input(self):
        points = np.array(self.match_points)
        print(self.match_points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("match_points must be a 2D array with shape (n_points, 3)")
        if len(points) <= 1:
            raise ValueError("Input must have more than 1 coordinate.")

    def compute_bounding_box(self):
        self.min_coords = np.min(self.match_points, axis=0)
        self.max_coords = np.max(self.match_points, axis=0)
        self.bounding_box = {
            "min": dict(zip(["x", "y", "z"], self.min_coords)),
            "max": dict(zip(["x", "y", "z"], self.max_coords)),
        }

    def create_voxel_grid(self):
        self.grid_shape = (
            np.ceil((self.max_coords - self.min_coords) / self.voxel_size).astype(int)
            + 1
        )
        self.voxel_grid = np.zeros(self.grid_shape, dtype=int)

    def count_points_per_voxel(self):
        voxel_indices = np.floor(
            (self.match_points - self.min_coords) / self.voxel_size
        ).astype(int)
        for idx in map(tuple, voxel_indices):
            self.voxel_grid[idx] += 1
            self.voxel_counts[idx] += 1

    def compute_statistics(self):
        self.validate_input()
        self.compute_bounding_box()
        self.create_voxel_grid()
        self.count_points_per_voxel()
        total_voxels = np.prod(self.grid_shape)

        counts = np.array(list(self.voxel_counts.values()))
        non_empty_voxels = len(counts)
        average_points_per_voxel = np.mean(counts)
        std_dev = np.std(counts)
        cv = std_dev / average_points_per_voxel if average_points_per_voxel > 0 else 0
        occupancy_ratio = (non_empty_voxels / total_voxels) * 100
        probabilities = counts / np.sum(counts)
        entropy_value = entropy(probabilities)
        max_points = np.max(counts)
        min_points = np.min(counts)
        skewness = skew(counts)
        kurt = kurtosis(counts)

        stats = {
            "Total voxels in grid": int(total_voxels),
            "Non-empty voxels": (len(counts)),
            "Average points per non-empty voxel": round(average_points_per_voxel, 2),
            "bounding_box minimum": self.min_coords,
            "bounding_box maximum": self.max_coords,
            "Standard deviation of voxel counts": round(std_dev, 2),
            "Coefficient of Variation (CV)": round(cv, 2),
            "Occupancy ratio": round(occupancy_ratio, 4),
            "Entropy of voxel distribution": round(entropy_value, 4),
            "Max points in a voxel": int(max_points),
            "Min points in a voxel": int(min_points),
            "Skewness of voxel counts": round(skewness, 4),
            "Kurtosis of voxel counts": round(kurt, 4),
        }
        print(
            f"""
        "Total voxels in grid": {int(total_voxels)}
        "Non-empty voxels":{len(counts)}
        "Average points per non-empty voxel": {round(average_points_per_voxel, 2)}
        "Standard deviation of voxel counts": {round(std_dev, 2)}
        "Coefficient of Variation (CV)": {round(cv, 2)}
        "Occupancy percentage": {round(occupancy_ratio, 4)}
        "Entropy of voxel distribution": {round(entropy_value, 4)}
        "Max points in a voxel": {int(max_points)}
        "Min points in a voxel": {int(min_points)}
        "Skewness of voxel counts": {round(skewness, 4)}
        "Kurtosis of voxel counts": {round(kurt, 4)}  
        """
        )
        self.understanding()

        return stats

    def understanding(self):

        print(
            """
        Low CV (≈ 0): Points are evenly distributed across voxels — very uniform.
        Moderate CV (≈ 0.3–0.6): Some variability — mild clustering or gaps.
        High CV (> 0.6): Strong clustering — many voxels are empty or sparse, while a few are dense.

        High Entropy: Points are spread across many voxels — high diversity in distribution.
        Low Entropy: Points are concentrated in fewer voxels — low diversity, more localized.

        Skewness ≈ 0: Symmetric distribution — voxel counts are balanced.
        Skewness > 0: Right-skewed — most voxels have few points, but a few have many.
        Skewness < 0: Left-skewed — most voxels have many points, few have few.

        Kurtosis ≈ 0: Normal distribution of voxel counts.
        Kurtosis > 0: Heavy tails — presence of outlier voxels with very high counts.
        Kurtosis < 0: Light tails — fewer extreme values, more uniform.
    """
        )
