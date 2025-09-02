
import json
import os
import sys

class AlignmentThreshold():
    def __init__(self,min_alignment, max_alignment, metric_path):
        self.min_alignment = min_alignment
        self.max_alignment = max_alignment
        self.metric_path = metric_path
        self.data = None
        self.get_metric_json()

    def get_metric_json(self):
        if not os.path.exists(self.metric_path):
            print(f"File not found: {self.metric_path}")
            return

        with open(self.metric_path, 'r') as f:
            self.data = json.load(f)

    def check_alignment(self):
        try: 
            value = self.data["alignment errors"]
        except:
            raise KeyError("No alignment errors have been saved.")
        minimum = value["minimum error"]
        maximum = value["maximum error"]

        if self.min_alignment is not None and minimum < self.min_alignment:
            print(f"Aborting: minimum alignment error value {minimum} is out of acceptable range.")
            sys.exit(1)
        if self.max_alignment is not None and maximum > self.max_alignment:
            print(f"Aborting: maximum alignment error value {maximum} is out of acceptable range.")
            sys.exit(1)