from collections import OrderedDict
from scipy.spatial import KDTree
import numpy as np
from collections import defaultdict, OrderedDict

# This class groups interest points by view_id and if overlapping only, uses kd tree algorithm to remove duplicate points,
# otherwise checks for max spots

class AdvancedRefinement:
    def __init__(self, interest_points, combine_distance):
        self.interest_points = interest_points
        self.consolidated_data = {}
        self.combine_distance = combine_distance
        self.overlapping_only = True
        self.sorted_view_ids = None
        self.result = interest_points  
        self.store_intensities = False
        self._max_spots = 0
        self.max_spots_per_overlap = False
        self.to_process = interest_points
        self.max_interval_size = 0
        self.interest_points_per_view_id = {}
        self.intensities_per_view_id = {}
        self.intervals_per_view_id = {}
    
    def kd_tree(self):
        """
        Constructs a KD-Tree for each view's points and retains points that are distanced from the nearest 
        two neighbours by at least `combine_distance`.
        """
        filtered_data = {}

        for view_id, points in self.consolidated_data.items():
            if not points:
                continue

            pts = np.array([p[0] for p in points], dtype=np.float32)
            intensities = np.array([p[1] for p in points], dtype=np.float32)

            # Safety check for NaNs or infs
            if not np.all(np.isfinite(pts)):
                print(f"[Warning] Skipping view {view_id} due to non-finite coordinates.")
                continue

            tree = KDTree(pts)
            visited = np.zeros(len(pts), dtype=bool)
            kept_indices = []

            for i in range(len(pts)):
                if visited[i]:
                    continue

                indices = tree.query_ball_point(pts[i], self.combine_distance)
                visited[indices] = True

                best_idx = indices[np.argmax(intensities[indices])]
                kept_indices.append(best_idx)

            filtered_data[view_id] = [points[i] for i in kept_indices]

        self.consolidated_data = filtered_data

    def consolidate_interest_points(self):
        """
        Aggregates and sorts interest points from multiple entries into a consolidated dictionary, 
        organized by view_id.
        """
        temp_data = defaultdict(list)

        for entry in self.interest_points:
            view_id = entry["view_id"]
            points = entry["interest_points"]
            intensities = entry["intensities"]
            temp_data[view_id].extend(zip(points, intensities))

        self.consolidated_data = OrderedDict(sorted(temp_data.items()))

    def filter_points(self, interest_points, intensities, max_spots):
        """
        Filters and returns the top `max_spots` interest points based on their intensities.
        """
        combined_list = []
        for i in range(len(interest_points)):
            combined_list.append((intensities[i], interest_points[i]))
            print((intensities[i], interest_points[i]))

        combined_list.sort(reverse=True)
        intensities.clear()
        interest_points.clear()

        # Add back the top max_spots elements
        for i in range(max_spots):
            intensity, ip = combined_list[i]
            intensities.append(intensity)
            interest_points.append((ip))

        return interest_points, intensities

    def run(self):
        """
        Executes the entry point of the script.
        """
        self.consolidate_interest_points()
        self.kd_tree()
        return self.consolidated_data