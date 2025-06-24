from collections import OrderedDict
from scipy.spatial import KDTree
import numpy as np

# This class groups interest points by view_id and if overlapping only, uses kd tree algorithm to remove duplicate points,
# otherwise checks for max spots

class AdvancedRefinement:
    def __init__(self, interest_points):
        self.interest_points = interest_points
        self.consolidated_data = {}
        self.combine_distance = 0.5
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
            if not points: continue
            
            # Prepare a numpy array from the first coordinate of each point for KDTree
            pts = np.array([p[0] for p in points])
            tree = KDTree(pts)
            unique_indices = set()

            # Identify unique points based on distance criteria
            for i, point in enumerate(pts):
                distances, _ = tree.query(point, k=3)

                # Include point if the third closest point is beyond the combine distance
                if distances[2] > self.combine_distance:
                    unique_indices.add(i)

            filtered_data[view_id] = [points[i] for i in unique_indices]

        self.consolidated_data = filtered_data

    def consolidate_interest_points(self):
        """
        Aggregates and sorts interest points from multiple entries into a consolidated dictionary, 
        organized by view_id.
        """
        for entry in self.interest_points:
            view_id = entry["view_id"]
            points = entry["interest_points"]
            intensities = entry["intensities"]

            if view_id not in self.consolidated_data:
                self.consolidated_data[view_id] = []

            for point, intensity in zip(points, intensities):
                self.consolidated_data[view_id].append((point, intensity))

        self.consolidated_data = OrderedDict(sorted(self.consolidated_data.items()))

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

    # TODO - test integration when Rhapso is ran without --overlappingonly param
    def max_spots(self):
        # max_spots_overlap is a set param
        if self.max_spots_per_overlap and self._max_spots > 0:
            ips_list = []
            intensities_list = []
            intervals_list = []
            # This is trying to access a list like a dictionary.
            for id in self.interest_points["view_id"]:
                # we never add anything to interest_points_per_view_id, self.intensities_per_view_id, intervals_per_view_id so it will always be empty and will always have a type error.
                ips_list.append(self.interest_points_per_view_id[id])
                intensities_list.append(self.intensities_per_view_id[id])
                intervals_list.append(self.intervals_per_view_id[id])
                interval_data = []

                for pair in self.to_process:
                    if pair[0] == list(id):
                        to_process_interval = pair[1]
                        ips_block = []
                        intensities_block = []

                        for l in range(len(ips_list) - 1):

                            block_interval = intervals_list[l]

                            if (
                                block_interval in interval_data
                                and to_process_interval in interval_data
                            ):

                                ips_block.extend(ips_list[l])
                                intensities_block.extend(intensities_list[l])
                        interval_data.append(
                            (to_process_interval, ips_block, intensities_block)
                        )

            # To later put back into interest_points_per_view_id and intensities_per_view_id
            self.interest_points_per_view_id[self.interest_points["view_id"]].clear()
            self.intensities_per_view_id[self.interest_points["view_id"]].clear()

            for interval in interval_data:
                intervals = interval[0]
                ips = interval[1]
                intensity_list = interval[2]

                my_max_spots = round(
                    self.max_spots
                    * (sum(intervals["dimensions"]) / self.max_interval_size)
                )
                if my_max_spots > 0 and my_max_spots < len(ips):
                    old_size = len(ips)
                    # filter points from ips, intensity_list, mymaxspots
                    intervals, intensities_list = self.filter_points(
                        ips,
                        intensity_list,
                    )
                    print(
                        f"filtered interval: limit "
                        + my_max_spots
                        + " old Size:"
                        + old_size
                        + "interval: "
                        + intervals
                    )
                else:
                    print("NOT filtered interval")
                self.interest_points_per_view_id[self.interest_points["view_id"]] += ips
                self.intensities_per_view_id[
                    self.interest_points["view_id"]
                ] += intensities_list
        return (
            self.interest_points_per_view_id,
            self.intensities_per_view_id,
            self.intervals_per_view_id,
        )

    def run(self):
        """
        Executes the entry point of the script.
        """
        self.consolidate_interest_points()
        if self.overlapping_only:
            self.kd_tree()
        else:
            self.max_spots()
        return self.consolidated_data
