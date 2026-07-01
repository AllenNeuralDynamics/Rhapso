import json
import xml.etree.ElementTree as ET
from io import BytesIO
import fsspec
import numpy as np
import pandas as pd

"""
Save Interest Points saves interest points as simple Parquet tables w/ a JSON manifest.
"""

class SaveInterestPoints:
    def __init__(self, dataframes, consolidated_data, xml_file_path, xml_output_file_path, n5_output_file_prefix,
                 downsample_xy, downsample_z, min_intensity, max_intensity, sigma, threshold):
        self.consolidated_data = consolidated_data
        self.image_loader_df = dataframes["image_loader"]
        self.xml_file_path = xml_file_path
        self.xml_output_file_path = xml_output_file_path
        self.alignment_output_prefix = n5_output_file_prefix.rstrip("/")
        self.downsample_xy = downsample_xy
        self.downsample_z = downsample_z
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.sigma = sigma
        self.threshold = threshold
        self.label = "beads"
        self.storage_options = {}
        self.overlappingOnly = "true"
        self.findMin = "true"
        self.findMax = "true"
        self.point_index_rows = []
        self.manifest_points = {}

    def join_uri(self, *parts):
        cleaned = []

        for i, part in enumerate(parts):
            if part is None:
                continue

            part = str(part)

            if i == 0:
                cleaned.append(part.rstrip("/"))
            else:
                cleaned.append(part.strip("/"))

        return "/".join(cleaned)

    def get_fs_and_path(self, uri):
        return fsspec.core.url_to_fs(uri, **self.storage_options)

    def ensure_parent_dir(self, uri):
        fs, path = self.get_fs_and_path(uri)
        parent = path.rsplit("/", 1)[0] if "/" in path else ""

        if parent:
            fs.makedirs(parent, exist_ok=True)

        return fs, path

    def write_bytes(self, uri, data):
        fs, path = self.ensure_parent_dir(uri)

        with fs.open(path, "wb") as f:
            f.write(data)

    def write_text(self, uri, text):
        self.write_bytes(uri, text.encode("utf-8"))

    def read_bytes(self, uri):
        fs, path = self.get_fs_and_path(uri)

        with fs.open(path, "rb") as f:
            return f.read()

    def write_json(self, uri, obj):
        text = json.dumps(obj, indent=2, sort_keys=True)
        self.write_text(uri, text + "\n")

    def write_parquet(self, uri, df):
        fs, path = self.ensure_parent_dir(uri)

        with fs.open(path, "wb") as f:
            df.to_parquet(
                f,
                engine="pyarrow",
                index=False,
            )

    def make_view_id(self, timepoint, setup):
        return f"timepoint: {int(timepoint)}, setup: {int(setup)}"

    def parse_view_id(self, view_id):
        parts = str(view_id).split(",")

        if len(parts) != 2:
            raise ValueError(f"Could not parse view_id: {view_id}")

        timepoint_part = parts[0].strip()
        setup_part = parts[1].strip()

        timepoint = int(timepoint_part.split(":")[1].strip())
        setup = int(setup_part.split(":")[1].strip())

        return timepoint, setup

    def points_for_view_to_dataframe(self, view_id):
        """
        Convert consolidated_data[view_id] into a flat point table
        """
        points_for_view = self.consolidated_data.get(view_id, [])

        if len(points_for_view) == 0:
            return pd.DataFrame(
                {
                    "point_id": pd.Series(dtype="uint64"),
                    "x": pd.Series(dtype="float64"),
                    "y": pd.Series(dtype="float64"),
                    "z": pd.Series(dtype="float64"),
                    "intensity": pd.Series(dtype="float32"),
                }
            )

        coords = np.asarray(
            [point[0] for point in points_for_view],
            dtype=np.float64,
        ).reshape(-1, 3)

        intensities = np.asarray(
            [point[1] for point in points_for_view],
            dtype=np.float32,
        )

        num_points = coords.shape[0]

        return pd.DataFrame(
            {
                "point_id": np.arange(num_points, dtype=np.uint64),
                "x": coords[:, 0].astype(np.float64, copy=False),
                "y": coords[:, 1].astype(np.float64, copy=False),
                "z": coords[:, 2].astype(np.float64, copy=False),
                "intensity": intensities.astype(np.float32, copy=False),
            }
        )

    def point_relative_path(self, timepoint, setup, label):
        return (
            f"points/"
            f"timepoint={int(timepoint)}/"
            f"setup={int(setup)}/"
            f"label={label}/"
            f"points.parquet"
        )

    def point_output_uri(self, timepoint, setup, label):
        return self.join_uri(
            self.alignment_output_prefix,
            self.point_relative_path(timepoint, setup, label),
        )

    def save_points_for_view(self, timepoint, setup, label):
        view_id = self.make_view_id(timepoint, setup)

        df = self.points_for_view_to_dataframe(view_id)
        rel_path = self.point_relative_path(timepoint, setup, label)
        out_uri = self.join_uri(self.alignment_output_prefix, rel_path)

        self.write_parquet(out_uri, df)

        num_points = int(len(df))
        key = f"{int(timepoint)}/{int(setup)}/{label}"

        self.manifest_points[key] = rel_path

        self.point_index_rows.append(
            {
                "timepoint": int(timepoint),
                "setup": int(setup),
                "label": str(label),
                "path": rel_path,
                "num_points": num_points,
            }
        )

    def save_points(self):
        """
        Save every view listed in image_loader_df.
        """
        self.point_index_rows = []
        self.manifest_points = {}

        for _, row in self.image_loader_df.iterrows():
            timepoint = int(row["timepoint"])
            setup = int(row["view_setup"])

            self.save_points_for_view(timepoint=timepoint, setup=setup, label=self.label,
            )

        index_df = pd.DataFrame(
            self.point_index_rows,
            columns=[
                "timepoint",
                "setup",
                "label",
                "path",
                "num_points",
            ],
        )

        index_uri = self.join_uri(
            self.alignment_output_prefix,
            "point_index.parquet",
        )

        self.write_parquet(index_uri, index_df)

    def build_manifest(self):
        return {
            "format": "rhapso-interest-points",
            "format_version": 1,
            "coordinate_order": "x,y,z",
            "storage": {
                "point_index": "point_index.parquet",
                "points_root": "points/",
            },
            "points": self.manifest_points,
            "detection": {
                "label": self.label,
                "method": "DOG",
                "sigma": self.sigma,
                "threshold": self.threshold,
                "overlappingOnly": self.overlappingOnly,
                "findMin": self.findMin,
                "findMax": self.findMax,
                "downsample_xy": self.downsample_xy,
                "downsample_z": self.downsample_z,
                "min_intensity": self.min_intensity,
                "max_intensity": self.max_intensity,
            },
        }

    def save_manifest(self):
        manifest = self.build_manifest()

        manifest_uri = self.join_uri(
            self.alignment_output_prefix,
            "manifest.json",
        )

        self.write_json(manifest_uri, manifest)

    def load_xml_tree(self, file_path):
        xml_bytes = self.read_bytes(file_path)
        tree = ET.parse(BytesIO(xml_bytes))
        root = tree.getroot()
        return tree, root

    def save_to_xml(self):
        tree, root = self.load_xml_tree(self.xml_file_path)
        interest_points_section = root.find(".//ViewInterestPoints")

        if interest_points_section is None:
            interest_points_section = ET.SubElement(root, "ViewInterestPoints")
            interest_points_section.text = "\n    "
        else:
            interest_points_section.clear()
            interest_points_section.text = "\n    "

        params = (
            "DOG (Spark) s={} t={} overlappingOnly={} min={} max={} "
            "downsampleXY={} downsampleZ={} minIntensity={} maxIntensity={}"
        ).format(
            self.sigma,
            self.threshold,
            self.overlappingOnly,
            self.findMin,
            self.findMax,
            self.downsample_xy,
            self.downsample_z,
            self.min_intensity,
            self.max_intensity,
        )

        for row in self.point_index_rows:
            timepoint = int(row["timepoint"])
            setup = int(row["setup"])
            label = str(row["label"])

            xml_path = f"tpId_{timepoint}_viewSetupId_{setup}/{label}"

            new_interest_point = ET.SubElement(
                interest_points_section,
                "ViewInterestPointsFile",
                {
                    "timepoint": str(timepoint),
                    "setup": str(setup),
                    "label": label,
                    "params": params,
                },
            )

            new_interest_point.text = xml_path
            new_interest_point.tail = "\n    "

        interest_points_section.tail = "\n  "

        xml_bytes = BytesIO()
        tree.write(xml_bytes, encoding="utf-8", xml_declaration=True)
        xml_bytes.seek(0)
        self.write_bytes(self.xml_output_file_path, xml_bytes.read())

    def run(self):
        """
        Execute save step.
        """
        self.save_points()
        self.save_manifest()
        self.save_to_xml()
        print("Interest Points Saved")
