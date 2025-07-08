import statistics
import numpy as np

"""
Number of matches 
    -Number of all matches, average number of matches/tile.  

Bounding box (Min  and Max of matches) 	
    -Smallest box that contains all match points. 

Standard Deviation 
	-How dispersed the match points are along each axis. 

Volume Covered (ΔX × ΔY × ΔZ) 
    - Volume of the bounding box that encloses all match points. """


class DescriptiveStatsMatching:
    def __init__(self, total_matches, total_match_length):
        self.total_matches = total_matches
        self.total_match_length = total_match_length
        self.just_points = []
        self.matches_per_view = {}

    def get_matches(self):
        if len(self.total_matches) == 0:
            raise ValueError("There are no matches to be evaluated")

        for view, matches in self.total_matches.items():
            self.matches_per_view[f"{view[0]}, {view[1]}"] = len(matches)
            for match in matches:
                self.just_points.append(match["p1"].tolist())
                self.just_points.append(match["p2"].tolist())

        return self.just_points

    def get_plane_coordinates(self):
        points_x = []
        points_y = []
        points_z = []

        for match in self.just_points:
            if len(match) != 3:
                raise ValueError("Coordinates missing, an error has occurred.")
            points_x.append(match[0])
            points_y.append(match[1])
            points_z.append(match[2])

        return points_x, points_y, points_z

    def get_bounding_box(self):
        points_x, points_y, points_z = self.get_plane_coordinates()

        return {"x": min(points_x), "y": min(points_y), "z": min(points_z)}, {
            "x": max(points_x),
            "y": max(points_y),
            "z": max(points_z),
        }

    def get_standard_deviation(self):
        points_x, points_y, points_z = self.get_plane_coordinates()
        return [
            statistics.stdev(points_x),
            statistics.stdev(points_y),
            statistics.stdev(points_z),
        ]

    def average_standard_deviation(self):
        return sum(self.get_standard_deviation()) / 3

    def bounding_box_volume(self):
        min_coordinates, max_coordinates = self.get_bounding_box()

        delta_x = max_coordinates["x"] - min_coordinates["x"]
        delta_y = max_coordinates["y"] - min_coordinates["y"]
        delta_z = max_coordinates["z"] - min_coordinates["z"]

        return delta_x * delta_y * delta_z

    def results(self):

        bounding_box_minimum, bounding_box_maximum = self.get_bounding_box()
        sd_x, sd_y, sd_z = self.get_standard_deviation()
        print(f"""
        Number of matches: {self.total_match_length}
        Number of interest points contained in matches: {len(self.just_points)}
        Average number of matches per tile: {self.total_match_length/len(self.total_matches)}
        Actual matches per tile: {self.matches_per_view},
        Bounding Box Min: {bounding_box_minimum}
        Bounding Box Max: {bounding_box_maximum}
        Std Dev (x, y, z): ({sd_x:.2f}, {sd_y:.2f}, {sd_z:.2f})
        Average Std Dev: {self.average_standard_deviation():.2f}
        Bounding Box Volume: {self.bounding_box_volume():.2f}
        """
        )

        data_summary = {
            "Number of matches": self.total_match_length,
            "Number of interest points in matches": len(self.just_points),
            "Average  matches per tile": self.total_match_length
            / len(self.total_matches),
            "Actual matches points per tile": self.matches_per_view,
            "Bounding Box Min": bounding_box_minimum,
            "Bounding Box Max": bounding_box_maximum,
            "std_Dev_x": round(sd_x, 2),
            "std_Dev_y": round(sd_y, 2),
            "Std_Dev_z": round(sd_z, 2),
            "Average Std Dev": round(self.average_standard_deviation(), 2),
            "Bounding Box Volume": round(self.bounding_box_volume(), 2),
        }

        return data_summary
