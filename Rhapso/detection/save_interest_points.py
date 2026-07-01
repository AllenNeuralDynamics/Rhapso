# import zarr
# import numpy as np
# import xml.etree.ElementTree as ET
# import s3fs
# import boto3
# from botocore import UNSIGNED
# from botocore.config import Config
# from io import BytesIO
# import io
# import json
# import asyncio
# import gc
# import aiohttp

# """
# Save Interest Points saves interest points into a Zarr-backed store
# using the existing interestpoints.n5 path convention and updates the XML.
# """

# class SaveInterestPoints:
#     def __init__(self, dataframes, consolidated_data, xml_file_path, xml_output_file_path, n5_output_file_prefix, downsample_xy,
#                  downsample_z, min_intensity, max_intensity, sigma, threshold):
#         self.consolidated_data = consolidated_data
#         self.image_loader_df = dataframes["image_loader"]
#         self.xml_file_path = xml_file_path
#         self.xml_output_file_path = xml_output_file_path
#         self.n5_output_file_prefix = n5_output_file_prefix
#         self.downsample_xy = downsample_xy
#         self.downsample_z = downsample_z
#         self.min_intensity = min_intensity
#         self.max_intensity = max_intensity
#         self.sigma = sigma
#         self.threshold = threshold
#         self.s3_filesystem = s3fs.S3FileSystem()
#         self.overlappingOnly = "true"
#         self.findMin = "true"
#         self.findMax = "true"

#     def load_xml_file(self, file_path):
#         tree = ET.parse(file_path)
#         root = tree.getroot()
#         return tree, root

#     def fetch_from_s3(self, s3, bucket_name, input_file):
#         try:
#             response = s3.get_object(Bucket=bucket_name, Key=input_file)
#         except Exception:
#             s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
#             response = s3.get_object(Bucket=bucket_name, Key=input_file)

#         return response["Body"].read().decode("utf-8")

#     def open_n5_group(self, path, mode="a"):
#         """
#         Open the interest-points store.
#         """
#         path = path.rstrip("/")

#         if path.startswith("s3://"):
#             mapper = s3fs.S3Map(
#                 root=path,
#                 s3=self.s3_filesystem,
#                 check=False,
#             )

#             if hasattr(zarr.storage, "FsspecStore"):
#                 store = zarr.storage.FsspecStore.from_mapper(mapper)
#             else:
#                 store = mapper

#         else:
#             if hasattr(zarr.storage, "LocalStore"):
#                 # Zarr v3
#                 store = zarr.storage.LocalStore(path)
#             else:
#                 # Zarr v2
#                 store = zarr.DirectoryStore(path)

#         return zarr.open_group(store=store, mode=mode)

#     def save_to_xml(self):
#         """
#         Rebuild the <ViewInterestPoints> section and write the updated XML back.
#         """
#         if self.xml_file_path.startswith("s3://"):
#             bucket, key = self.xml_file_path.replace("s3://", "", 1).split("/", 1)
#             s3 = boto3.client("s3")
#             xml_string = self.fetch_from_s3(s3, bucket, key)
#             tree = ET.parse(io.BytesIO(xml_string.encode("utf-8")))
#             root = tree.getroot()
#         else:
#             tree, root = self.load_xml_file(self.xml_file_path)

#         interest_points_section = root.find(".//ViewInterestPoints")

#         if interest_points_section is None:
#             interest_points_section = ET.SubElement(root, "ViewInterestPoints")
#             interest_points_section.text = "\n    "
#         else:
#             interest_points_section.clear()
#             interest_points_section.text = "\n    "

#         for view_id, _ in self.consolidated_data.items():
#             parts = view_id.split(",")
#             timepoint_part = parts[0].strip()
#             setup_part = parts[1].strip()

#             timepoint = int(timepoint_part.split(":")[1].strip())
#             setup = int(setup_part.split(":")[1].strip())
#             label = "beads"

#             params = (
#                 "DOG (Spark) s={} t={} overlappingOnly={} min={} max={} "
#                 "downsampleXY={} downsampleZ={} minIntensity={} maxIntensity={}"
#             ).format(
#                 self.sigma,
#                 self.threshold,
#                 self.overlappingOnly,
#                 self.findMin,
#                 self.findMax,
#                 self.downsample_xy,
#                 self.downsample_z,
#                 self.min_intensity,
#                 self.max_intensity,
#             )

#             value = f"tpId_{timepoint}_viewSetupId_{setup}/beads"

#             new_interest_point = ET.SubElement(
#                 interest_points_section,
#                 "ViewInterestPointsFile",
#                 {
#                     "timepoint": str(timepoint),
#                     "setup": str(setup),
#                     "label": label,
#                     "params": params,
#                 },
#             )
#             new_interest_point.text = value
#             new_interest_point.tail = "\n    "

#         interest_points_section.tail = "\n  "

#         if self.xml_output_file_path.startswith("s3://"):
#             bucket, key = self.xml_output_file_path.replace("s3://", "", 1).split("/", 1)
#             xml_bytes = BytesIO()
#             tree.write(xml_bytes, encoding="utf-8", xml_declaration=True)
#             xml_bytes.seek(0)
#             s3 = boto3.client("s3")
#             s3.upload_fileobj(xml_bytes, bucket, key)
#         else:
#             tree.write(self.xml_output_file_path, encoding="utf-8", xml_declaration=True)

#     def write_json_to_s3(self, id_dataset_path, loc_dataset_path, attributes):
#         """
#         Preserve existing S3-side attributes.json behavior.
#         """
#         bucket, key = id_dataset_path.replace("s3://", "", 1).split("/", 1)
#         json_path = key + "/attributes.json"
#         json_bytes = json.dumps(attributes).encode("utf-8")
#         s3 = boto3.client("s3")
#         s3.put_object(Bucket=bucket, Key=json_path, Body=json_bytes)

#         bucket, key = loc_dataset_path.replace("s3://", "", 1).split("/", 1)
#         json_path = key + "/attributes.json"
#         json_bytes = json.dumps(attributes).encode("utf-8")
#         s3 = boto3.client("s3")
#         s3.put_object(Bucket=bucket, Key=json_path, Body=json_bytes)

#     def write_one_block_dataset(self, root, name, data, dtype, attrs):
#         """
#         Write a dataset as exactly one block/chunk.
#         """
#         data = np.asarray(data, dtype=dtype)

#         # Empty datasets can have shape 0, but chunk dims cannot be 0.
#         chunks = tuple(max(1, dim) for dim in data.shape)

#         if hasattr(root, "create_array"):
#             # Zarr v3 path.
#             arr = root.create_array(
#                 name=name,
#                 shape=data.shape,
#                 chunks=chunks,
#                 dtype=dtype,
#                 overwrite=True,
#             )
#         else:
#             # Zarr v2 path.
#             if name in root:
#                 del root[name]

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

#     def save_intensities_to_n5(self, view_id, n5_path):
#         """
#         Write intensities into the interestpoints group.
#         """
#         output_path = self.n5_output_file_prefix + n5_path + "/interestpoints"
#         root = self.open_n5_group(output_path, mode="a")

#         # Kept for compatibility with existing metadata conventions.
#         root.attrs["n5"] = "4.0.0"

#         intensities_path = "intensities"

#         try:
#             points_for_view = self.consolidated_data.get(view_id, [])

#             if len(points_for_view) > 0:
#                 intensities = np.asarray(
#                     [point[1] for point in points_for_view],
#                     dtype=np.float32,
#                 )
#             else:
#                 intensities = np.empty((0,), dtype=np.float32)

#             num_intensities = intensities.shape[0]

#             self.write_one_block_dataset(
#                 root=root,
#                 name=intensities_path,
#                 data=intensities,
#                 dtype="f4",
#                 attrs={
#                     "dimensions": [num_intensities],
#                     "blockSize": [max(num_intensities, 1)],
#                 },
#             )

#         except Exception as e:
#             print(f"Error writing intensities dataset at {intensities_path}: {e}")
#             raise

#     def save_interest_points_to_n5(self, view_id, n5_path):
#         """
#         Write interest point IDs and 3D locations into the interestpoints group.
#         """
#         output_path = self.n5_output_file_prefix + n5_path + "/interestpoints"
#         root = self.open_n5_group(output_path, mode="a")

#         root.attrs["pointcloud"] = "1.0.0"
#         root.attrs["type"] = "list"
#         root.attrs["list version"] = "1.0.0"

#         id_dataset = "id"
#         loc_dataset = "loc"

#         if self.n5_output_file_prefix.startswith("s3://"):
#             id_path = f"{output_path}/id"
#             loc_path = f"{output_path}/loc"
#             attrs_dict = dict(root.attrs)
#             self.write_json_to_s3(id_path, loc_path, attrs_dict)

#         points_for_view = self.consolidated_data.get(view_id, [])

#         if len(points_for_view) > 0:
#             interest_points = np.asarray(
#                 [point[0] for point in points_for_view],
#                 dtype=np.float64,
#             ).reshape(-1, 3)

#             num_points = interest_points.shape[0]

#             interest_point_ids = np.arange(
#                 num_points,
#                 dtype=np.uint64,
#             ).reshape(-1, 1)

#         else:
#             interest_points = np.empty((0, 3), dtype=np.float64)
#             interest_point_ids = np.empty((0, 1), dtype=np.uint64)
#             num_points = 0

#         self.write_one_block_dataset(
#             root=root,
#             name=id_dataset,
#             data=interest_point_ids,
#             dtype="u8",
#             attrs={
#                 "dimensions": [num_points, 1],
#                 "blockSize": [max(num_points, 1), 1],
#             },
#         )

#         self.write_one_block_dataset(
#             root=root,
#             name=loc_dataset,
#             data=interest_points,
#             dtype="f8",
#             attrs={
#                 "dimensions": [num_points, 3],
#                 "blockSize": [max(num_points, 1), 3],
#             },
#         )

#     def save_points(self):
#         """
#         Write interest points and intensities into the existing
#         interestpoints.n5 path convention
#         """
#         for _, row in self.image_loader_df.iterrows():
#             view_id = f"timepoint: {row['timepoint']}, setup: {row['view_setup']}"
#             n5_path = (
#                 f"interestpoints.n5/"
#                 f"tpId_{row['timepoint']}_viewSetupId_{row['view_setup']}/"
#                 f"beads"
#             )

#             self.save_interest_points_to_n5(view_id, n5_path)
#             self.save_intensities_to_n5(view_id, n5_path)

#         path = self.n5_output_file_prefix + "interestpoints.n5"
#         root = self.open_n5_group(path, mode="a")

#         # Kept for compatibility with existing metadata conventions.
#         root.attrs["n5"] = "4.0.0"

#     def run_async_cleanup(self, coro, loop=None):
#         if loop is not None and loop.is_running():
#             return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=30)

#         if loop is not None and not loop.is_closed():
#             return loop.run_until_complete(coro)

#         return asyncio.run(coro)

#     def close_aiohttp_sessions(self):
#         gc.collect()

#         for obj in gc.get_objects():
#             if isinstance(obj, aiohttp.ClientSession) and not obj.closed:
#                 self.run_async_cleanup(obj.close(), getattr(obj, "_loop", None))

#     def run(self):
#         """
#         Executes the entry point of the script.
#         """
#         self.save_points()
#         self.save_to_xml()
#         self.close_aiohttp_sessions()


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
