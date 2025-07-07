import json
import os
import sys

class Threshold:
    def __init__(self, min_alignment, max_alignment, minimum_points, maximum_points,
                 minimum_total_matches, maximum_total_matches, max_kde, min_kde,
                 max_cv, min_cv, metric_path):
        self.min_alignment = min_alignment
        self.max_alignment = max_alignment
        self.minimum_points = minimum_points
        self.maximum_points = maximum_points
        self.minimum_total_matches = minimum_total_matches
        self.maximum_total_matches = maximum_total_matches
        self.max_kde = max_kde
        self.min_kde = min_kde
        self.max_cv = max_cv
        self.min_cv = min_cv
        self.metric_path = metric_path
        self.data = None

    def get_metric_json(self):
        if not os.path.exists(self.metric_path):
            print(f"File not found: {self.metric_path}")
            return

        with open(self.metric_path, 'r') as f:
            self.data = json.load(f)

    def check_alignment(self):
        value = self.data["alignment errors"]
        minimum = value["min_error"]
        maximum = value["max_error"]

        if self.min_alignment is not None and minimum < self.min_alignment:
            print(f"Aborting: minimum alignment error value {minimum} is out of acceptable range.")
            sys.exit(1)
        if self.max_alignment is not None and maximum > self.max_alignment:
            print(f"Aborting: maximum alignment error value {maximum} is out of acceptable range.")
            sys.exit(1)

    def check_points(self):
        value = self.data["total_ips"]
        if self.minimum_points is not None and value < self.minimum_points:
            print(f"Aborting: total number of interest points value {value} is below the minimum threshold.")
            sys.exit(1)
        if self.maximum_points is not None and value > self.maximum_points:
            print(f"Aborting: total number of interest points value {value} is above the maximum threshold.")
            sys.exit(1)

    def check_matches(self):
        value = self.data["descriptive_stats"]["Number of matches"]
        if self.minimum_total_matches is not None and value < self.minimum_total_matches:
            print(f"Aborting: total matches value {value} is below the minimum threshold.")
            sys.exit(1)
        if self.maximum_total_matches is not None and value > self.maximum_total_matches:
            print(f"Aborting: total matches value {value} is above the maximum threshold.")
            sys.exit(1)

    def check_kde(self):
        value = self.data["KDE"]
        minimum = value["min"]
        maximum = value["max"]

        if self.min_kde is not None and minimum < self.min_kde:
            print(f"Aborting: KDE minimum value {minimum} is out of acceptable range.")
            sys.exit(1)
        if self.max_kde is not None and maximum > self.max_kde:
            print(f"Aborting: KDE maximum value {maximum} is out of acceptable range.")
            sys.exit(1)

    def check_cv(self):
        value = self.data["voxelization stats"]["Coefficient of Variation (CV)"]
        if self.min_cv is not None and value < self.min_cv:
            print(f"Aborting: Coefficient of Variation value {value} is below the minimum threshold.")
            sys.exit(1)
        if self.max_cv is not None and value > self.max_cv:
            print(f"Aborting: Coefficient of Variation value {value} is above the maximum threshold.")
            sys.exit(1)
