import json


class JSONFileHandler:
    def __init__(
        self,
        file_path,
    ):
        self.file_path = str(file_path)
        self.metrics = {}

    def update_json(self, field, data):
        self.metrics[field] = data

    def update(self, field, new_data):
        """Update existing JSON data in the file with new data."""
        try:
            with open(self.file_path, "r") as f:
                self.metrics = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.metrics = {}

        self.update_json(field, new_data)

        with open(self.file_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Data updated in {self.file_path}")
