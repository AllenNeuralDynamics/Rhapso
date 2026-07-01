# import zarr
# import s3fs
# import numpy as np
# import boto3
# import json

# class SavePoints:
#     def __init__(self, label_entries, n5_prefix):
#         self.label_entries = label_entries
#         self.n5_prefix = n5_prefix
#         self.s3_filesystem = s3fs.S3FileSystem()

#     def open_n5_group(self, path, mode="a"):
#         """
#         Open the interest-points store.
#         """
#         path = path.rstrip("/")

#         if path.startswith("s3://"):
#             s3 = s3fs.S3FileSystem(
#                 anon=False,
#                 skip_instance_cache=True,
#             )

#             mapper = s3fs.S3Map(
#                 root=path,
#                 s3=s3,
#                 check=False,
#             )

#             if hasattr(zarr.storage, "FsspecStore"):
#                 store = zarr.storage.FsspecStore.from_mapper(mapper)
#             else:
#                 store = mapper

#         else:
#             if hasattr(zarr.storage, "LocalStore"):
#                 store = zarr.storage.LocalStore(path)
#             else:
#                 store = zarr.DirectoryStore(path)

#         return zarr.open_group(store=store, mode=mode)

#     def write_json_to_s3(self, id_dataset_path, loc_dataset_path, attributes):
#         """
#         Write attributes file into both the ID and LOC dataset directories on S3.
#         """
#         bucket, key = id_dataset_path.replace("s3://", "", 1).split("/", 1)
#         json_path = key + "/attributes.json"
#         json_bytes = json.dumps(attributes).encode("utf-8")
#         s3 = boto3.client("s3")
#         s3.put_object(Bucket=bucket, Key=json_path, Body=json_bytes)

#         bucket, key = loc_dataset_path.replace("s3://", "", 1).split("/", 1)
#         json_path = key + "/attributes.json"
#         json_bytes = json.dumps(attributes).encode("utf-8")
#         s3.put_object(Bucket=bucket, Key=json_path, Body=json_bytes)

#     def write_one_block_dataset(self, root, name, data, dtype, attrs):
#         """
#         Write points as one block/chunk.
#         """
#         data = np.asarray(data, dtype=dtype)
#         chunks = tuple(max(1, dim) for dim in data.shape)

#         if name in root:
#             del root[name]

#         if hasattr(root, "create_array"):
#             # Zarr v3 path.
#             arr = root.create_array(
#                 name=name,
#                 shape=data.shape,
#                 chunks=chunks,
#                 dtype=dtype,
#             )
#         else:
#             # Zarr v2 path.
#             arr = root.create_dataset(
#                 name=name,
#                 shape=data.shape,
#                 chunks=chunks,
#                 dtype=dtype,
#             )

#         if data.size > 0:
#             arr[...] = data

#         for k, v in attrs.items():
#             arr.attrs[k] = v

#         return arr

#     def save_interest_points_to_n5(self):
#         for label_entry in self.label_entries:
#             n5_path = label_entry["ip_list"]["n5_path"]
#             output_path = self.n5_prefix + n5_path + "/interestpoints"

#             root = self.open_n5_group(output_path, mode="a")

#             root.attrs["pointcloud"] = "1.0.0"
#             root.attrs["type"] = "list"
#             root.attrs["list version"] = "1.0.0"

#             id_dataset = "id"
#             loc_dataset = "loc"

#             if self.n5_prefix.startswith("s3://"):
#                 id_path = f"{output_path}/id"
#                 loc_path = f"{output_path}/loc"
#                 attrs_dict = dict(root.attrs)
#                 self.write_json_to_s3(id_path, loc_path, attrs_dict)

#             raw_points = label_entry["ip_list"].get("interest_points", [])

#             if len(raw_points) > 0:
#                 interest_points = np.asarray(
#                     [point[1] for point in raw_points],
#                     dtype=np.float64,
#                 ).reshape(-1, 3)
#             else:
#                 interest_points = np.empty((0, 3), dtype=np.float64)

#             num_points = interest_points.shape[0]

#             interest_point_ids = np.arange(
#                 num_points,
#                 dtype=np.uint64,
#             ).reshape(-1, 1)

#             self.write_one_block_dataset(
#                 root=root,
#                 name=id_dataset,
#                 data=interest_point_ids,
#                 dtype="u8",
#                 attrs={
#                     "dimensions": [num_points, 1],
#                     "blockSize": [max(num_points, 1), 1],
#                 },
#             )

#             self.write_one_block_dataset(
#                 root=root,
#                 name=loc_dataset,
#                 data=interest_points,
#                 dtype="f8",
#                 attrs={
#                     "dimensions": [num_points, 3],
#                     "blockSize": [max(num_points, 1), 3],
#                 },
#             )

#     def run(self):
#         self.save_interest_points_to_n5()

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