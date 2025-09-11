import numpy as np

"""
Points Validation prints out the metrics for the results of interest point detection
"""

class PointsValidation:
    def __init__(self, consolidated_data):
        self.consolidated_data = consolidated_data

    def validation_suite(self):
        total_points = sum(len(points) for points in self.consolidated_data.values())
        print(f"\nTotal Interest Points Found: {total_points}")

        print("\nInterest Points by View ID:")
        for view_id, points in self.consolidated_data.items():

            # Sort points by index
            sorted_points = sorted(points, key=lambda x: x[0][2])  # x[1] is the (x,y,z) index to sort by

            if len(sorted_points) == 0:
                print(f"\nView ID: {view_id} | Num points: 0")
                print("\n--- Detection Stats ---")
                print("No points found for this view.\n")
                continue

            coords = np.array([p[0] for p in sorted_points])
            intensities = np.array([p[1] for p in sorted_points])

            # Print metrics on interest points
            print("\n--- Detection Stats ---")
            print(f"Total Points: {len(coords)}")
            print(f"Intensity: min={intensities.min():.2f}, max={intensities.max():.2f}, mean={intensities.mean():.2f}, std={intensities.std():.2f}")

            for dim, name in zip(range(3), ['X', 'Y', 'Z']):
                values = coords[:, dim]
                print(f"{name} Range: {values.min():.2f} – {values.max():.2f} | Spread (std): {values.std():.2f}")

            # Density per 1000x1000x1000 space
            volume = np.ptp(coords[:, 0]) * np.ptp(coords[:, 1]) * np.ptp(coords[:, 2])
            density = len(coords) / (volume / 1e9) if volume > 0 else 0
            print(f"Estimated Density: {density:.2f} points per 1000³ volume")
            print("-----------------------\n")
    
    def run(self):
        """
        Executes the entry point of the script.
        """
        self.validation_suite()
