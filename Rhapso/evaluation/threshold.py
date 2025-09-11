import json
import os
import sys

class Threshold:
    def __init__(
        self,
        minimum_points,
        maximum_points,
        minimum_total_matches,
        maximum_total_matches,
        max_kde,
        min_kde,
        max_cv,
        min_cv,
        metric_path,
    ):

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

        with open(self.metric_path, "r") as f:
            self.data = json.load(f)

    def check_points(self):
        value = self.data["Total IPS"]

        if not value:
            return
        if self.minimum_points is not None and value < self.minimum_points:
            print(
                f"Aborting: total number of interest points value {value} is below the minimum threshold."
            )
            sys.exit(1)
        if self.maximum_points is not None and value > self.maximum_points:
            print(
                f"Aborting: total number of interest points value {value} is above the maximum threshold."
            )
            sys.exit(1)

    def check_matches(self):
        value = self.data["Descriptive stats"]["Number of matches"]
        if (
            self.minimum_total_matches is not None
            and value < self.minimum_total_matches
        ):
            print(
                f"Aborting: total matches value {value} is below the minimum threshold."
            )
            sys.exit(1)
        if (
            self.maximum_total_matches is not None
            and value > self.maximum_total_matches
        ):
            print(
                f"Aborting: total matches value {value} is above the maximum threshold."
            )
            sys.exit(1)

    def check_kde(self):
        value = self.data["KDE"]
        minimum = value["minimum KDE"]
        maximum = value["maximum KDE"]

        if self.min_kde is not None and minimum < self.min_kde:
            print(f"Aborting: KDE minimum value {minimum} is out of acceptable range.")
            sys.exit(1)
        if self.max_kde is not None and maximum > self.max_kde:
            print(f"Aborting: KDE maximum value {maximum} is out of acceptable range.")
            sys.exit(1)

    def check_cv(self):
        value = self.data["Voxelization stats"]["Coefficient of Variation (CV)"]
        if self.min_cv is not None and value < self.min_cv:
            print(
                f"Aborting: Coefficient of Variation value {value} is below the minimum threshold."
            )
            sys.exit(1)
        if self.max_cv is not None and value > self.max_cv:
            print(
                f"Aborting: Coefficient of Variation value {value} is above the maximum threshold."
            )
            sys.exit(1)

    def run_threshold_checks(self):
        if self.data is None:
            print("No data loaded. Exiting.")
            return

        self.check_points()
        self.check_matches()
        self.check_kde()
        self.check_cv()
