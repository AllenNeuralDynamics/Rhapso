import json
import fsspec
import numpy as np
import pandas as pd

class SavePoints:
    """
    Save split interest points into the Parquet/JSON alignment store.
    """

    def __init__(self, split_interest_points, n5_prefix):
        self.split_interest_points = split_interest_points
        self.alignment_prefix = str(n5_prefix).rstrip("/")

    def join_uri(self, *parts):
        return "/".join(
            str(part).strip("/") if i > 0 else str(part).rstrip("/")
            for i, part in enumerate(parts)
            if part is not None
        )

    def get_fs_and_path(self, uri):
        return fsspec.core.url_to_fs(uri)

    def ensure_parent_dir(self, uri):
        fs, path = self.get_fs_and_path(uri)
        parent = path.rsplit("/", 1)[0] if "/" in path else ""

        if parent:
            fs.makedirs(parent, exist_ok=True)

        return fs, path

    def exists(self, uri):
        fs, path = self.get_fs_and_path(uri)
        return fs.exists(path)

    def read_json(self, uri):
        fs, path = self.get_fs_and_path(uri)

        with fs.open(path, "r") as f:
            return json.load(f)

    def write_json(self, uri, obj):
        fs, path = self.ensure_parent_dir(uri)

        with fs.open(path, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")

    def read_parquet(self, uri):
        fs, path = self.get_fs_and_path(uri)

        with fs.open(path, "rb") as f:
            return pd.read_parquet(f, engine="pyarrow")

    def write_parquet(self, uri, df):
        fs, path = self.ensure_parent_dir(uri)

        with fs.open(path, "wb") as f:
            df.to_parquet(f, engine="pyarrow", index=False)

    def parse_view_id(self, view_id):
        parts = str(view_id).split(",")

        timepoint = int(parts[0].split(":")[1].strip())
        setup = int(parts[1].split(":")[1].strip())

        return timepoint, setup

    def point_relative_path(self, timepoint, setup, label):
        return (
            f"points/"
            f"timepoint={int(timepoint)}/"
            f"setup={int(setup)}/"
            f"label={label}/"
            f"points.parquet"
        )

    def points_to_dataframe(self, raw_points):
        if raw_points is None or len(raw_points) == 0:
            return pd.DataFrame(
                {
                    "point_id": pd.Series(dtype="uint64"),
                    "x": pd.Series(dtype="float64"),
                    "y": pd.Series(dtype="float64"),
                    "z": pd.Series(dtype="float64"),
                    "intensity": pd.Series(dtype="float32"),
                }
            )

        point_ids = []
        coords = []

        for i, point in enumerate(raw_points):
            point_ids.append(int(point[0]) if isinstance(point, (list, tuple)) else i)
            coords.append(point[1] if isinstance(point, (list, tuple)) else point)

        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)

        return pd.DataFrame(
            {
                "point_id": np.asarray(point_ids, dtype=np.uint64),
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
                "intensity": np.full(len(coords), np.nan, dtype=np.float32),
            }
        )

    def manifest_uri(self):
        return self.join_uri(self.alignment_prefix, "manifest.json")

    def point_index_uri(self):
        return self.join_uri(self.alignment_prefix, "point_index.parquet")

    def load_manifest(self):
        uri = self.manifest_uri()

        if self.exists(uri):
            return self.read_json(uri)

        return {
            "format": "rhapso-interest-points",
            "format_version": 1,
            "coordinate_order": "x,y,z",
            "storage": {
                "point_index": "point_index.parquet",
                "points_root": "points/",
            },
            "points": {},
            "detection": {},
        }

    def load_point_index(self):
        uri = self.point_index_uri()

        if self.exists(uri):
            return self.read_parquet(uri)

        return pd.DataFrame(
            columns=[
                "timepoint",
                "setup",
                "label",
                "path",
                "num_points",
            ]
        )

    def save_interest_points_to_parquet(self):
        manifest = self.load_manifest()
        manifest.setdefault("points", {})
        manifest.setdefault("storage", {})
        manifest.setdefault("detection", {})

        manifest["format"] = "rhapso-interest-points"
        manifest["format_version"] = 1
        manifest["coordinate_order"] = "x,y,z"
        manifest["storage"]["point_index"] = "point_index.parquet"
        manifest["storage"]["points_root"] = "points/"

        old_index_df = self.load_point_index()
        new_index_rows = []

        for view_id, label_entries in self.split_interest_points.items():
            timepoint, setup = self.parse_view_id(view_id)

            for label_entry in label_entries:
                label = str(label_entry["label"])
                raw_points = label_entry["ip_list"].get("interest_points", [])

                points_df = self.points_to_dataframe(raw_points)
                rel_path = self.point_relative_path(timepoint, setup, label)
                output_uri = self.join_uri(self.alignment_prefix, rel_path)

                self.write_parquet(output_uri, points_df)

                key = f"{int(timepoint)}/{int(setup)}/{label}"
                manifest["points"][key] = rel_path

                new_index_rows.append(
                    {
                        "timepoint": int(timepoint),
                        "setup": int(setup),
                        "label": label,
                        "path": rel_path,
                        "num_points": int(len(points_df)),
                    }
                )

        new_index_df = pd.DataFrame(new_index_rows)

        index_df = pd.concat(
            [old_index_df, new_index_df],
            ignore_index=True,
        )

        index_df = (
            index_df
            .drop_duplicates(["timepoint", "setup", "label"], keep="last")
            .sort_values(["timepoint", "setup", "label"])
            .reset_index(drop=True)
        )

        index_df["timepoint"] = index_df["timepoint"].astype("int32")
        index_df["setup"] = index_df["setup"].astype("int32")
        index_df["label"] = index_df["label"].astype(str)
        index_df["path"] = index_df["path"].astype(str)
        index_df["num_points"] = index_df["num_points"].astype("int64")

        self.write_json(self.manifest_uri(), manifest)
        self.write_parquet(self.point_index_uri(), index_df)

    def run(self):
        self.save_interest_points_to_parquet()
        print("Split Points Saved")