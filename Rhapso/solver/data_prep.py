# import zarr
# import json
# import os
# import s3fs
# import asyncio
# import gc
# import aiohttp

# """
# Data Prep fetches and preps interest points data.
# """

# class DataPrep():
#     def __init__(self, interest_points_df, view_transform_matrices, xml_file_path, n5_input_path):
#         self.interest_points_df = interest_points_df
#         self.view_transform_matrices = view_transform_matrices
#         self.xml_file_path = xml_file_path
#         self.n5_input_path = n5_input_path

#         self.connected_views = {} 
#         self.corresponding_interest_points = {}
#         self.interest_points = {}
#         self.label_map_global = {}
#         self.s3_filesystem = s3fs.S3FileSystem(anon=False)

#     def get_n5_root_path(self):
#         """
#         Return the full path to interestpoints.n5.
#         """
#         if self.n5_input_path.startswith("s3://"):
#             return self.n5_input_path.rstrip("/") + "/interestpoints.n5"

#         return os.path.join(self.n5_input_path, "interestpoints.n5")
    
#     def open_n5_root(self, mode="r"):
#         """
#         Open the interestpoints.n5 root.
#         """
#         path = self.get_n5_root_path().rstrip("/")

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

#             return zarr.open(store, mode=mode)

#         if hasattr(zarr.storage, "LocalStore"):
#             store = zarr.storage.LocalStore(path)
#         else:
#             store = zarr.DirectoryStore(path)

#         return zarr.open(store, mode=mode)

#     # def open_n5_root(self, mode="r"):
#     #     """
#     #     Open the interestpoints.n5 root.
#     #     """
#     #     path = self.get_n5_root_path().rstrip("/")

#     #     if path.startswith("s3://"):
#     #         anon_options = [True, False] if mode == "r" else [False]
#     #         last_error = None

#     #         for anon in anon_options:
#     #             try:
#     #                 s3 = s3fs.S3FileSystem(
#     #                     anon=anon,
#     #                     skip_instance_cache=True,
#     #                 )

#     #                 mapper = s3fs.S3Map(
#     #                     root=path,
#     #                     s3=s3,
#     #                     check=False,
#     #                 )

#     #                 if hasattr(zarr.storage, "FsspecStore"):
#     #                     store = zarr.storage.FsspecStore.from_mapper(mapper)
#     #                 else:
#     #                     store = mapper

#     #                 return zarr.open(store, mode=mode)

#     #             except Exception as e:
#     #                 last_error = e

#     #         raise last_error

#     #     if hasattr(zarr.storage, "LocalStore"):
#     #         store = zarr.storage.LocalStore(path)
#     #     else:
#     #         store = zarr.DirectoryStore(path)

#     #     return zarr.open(store, mode=mode)

#     def get_connected_views_from_n5(self):
#         """
#         Loads connected view mappings from metadata, supporting both S3 and local sources.
#         """
#         root = self.open_n5_root(mode="r")

#         for _, row in self.interest_points_df.iterrows():
#             view_id = f"timepoint: {row['timepoint']}, setup: {row['setup']}"
#             correspondences_key = f"{row['path']}/correspondences"

#             try:
#                 self.connected_views[view_id] = root[correspondences_key].attrs["idMap"]
#             except Exception:
#                 print(f"No connected views for tile {view_id}")

#     def load_json_data(self, json_path):
#         """
#         Legacy local fallback for reading attributes.json directly.
#         Kept here only as a fallback in case older data does not expose attrs cleanly.
#         """
#         try:
#             if os.path.isabs(json_path):
#                 path = json_path
#             else:
#                 path = os.path.join(self.n5_input_path, json_path)

#             if not os.path.exists(path):
#                 return {}

#             with open(path, "r") as f:
#                 obj = json.load(f)

#             id_map = obj.get("idMap", {})
#             return id_map if isinstance(id_map, dict) else {}

#         except Exception:
#             return {}
        
#     def get_corresponding_data_from_n5(self):
#         """
#         Parses and transforms corresponding interest point data into world space coordinates.
#         """
#         root = self.open_n5_root(mode="r")

#         for _, row in self.interest_points_df.iterrows():
#             view_id = f"timepoint: {row['timepoint']}, setup: {row['setup']}"  
#             correspondences_prefix = f"{row['path']}/correspondences"

#             # Load attrs data for idMap.
#             try:
#                 id_map = root[correspondences_prefix].attrs["idMap"]
#             except Exception:
#                 # Fallback for older local data if attrs are not exposed through the store.
#                 if self.n5_input_path.startswith("s3://"):
#                     continue

#                 attributes_path = os.path.join(
#                     "interestpoints.n5",
#                     row["path"],
#                     "correspondences",
#                     "attributes.json",
#                 )
#                 id_map = self.load_json_data(attributes_path)

#                 if not id_map:
#                     continue
            
#             try:
#                 interest_points_index_map = root[correspondences_prefix + "/data"][:]
#             except (KeyError, FileNotFoundError, AttributeError, TypeError):
#                 print(f"⚠️ Skipping {view_id}: missing correspondences.")
#                 continue

#             # Load corresponding interest points data.
#             for ip_index, corr_index, corr_group_id in interest_points_index_map:
#                 if corr_group_id == view_id:
#                     continue

#                 corresponding_view_id = next(
#                     (k for k, v in id_map.items() if v == int(corr_group_id)),
#                     None,
#                 )

#                 if corresponding_view_id is None:
#                     continue

#                 parts = corresponding_view_id.split(",")
#                 timepoint, setup, label = parts[0], parts[1], parts[2]
#                 corresponding_view_id = f"timepoint: {timepoint}, setup: {setup}"

#                 ip = self.interest_points[view_id][label][int(ip_index)]
#                 corr_ip = self.interest_points[corresponding_view_id][label][int(corr_index)]

#                 if view_id not in self.corresponding_interest_points:
#                     self.corresponding_interest_points[view_id] = [] 
                
#                 self.corresponding_interest_points[view_id].append({
#                     "detection_id": ip_index,
#                     "detection_p1": ip,
#                     "corresponding_detection_id": corr_index,
#                     "corresponding_detection_p2": corr_ip,
#                     "corresponding_view_id": corresponding_view_id,
#                     "label": label,
#                 })
    
#     def get_all_interest_points_from_n5(self):
#         """
#         Loads raw interest point coordinates from storage into memory, keyed by view ID.
#         """
#         root = self.open_n5_root(mode="r")

#         for _, row in self.interest_points_df.iterrows():
#             view_id = f"timepoint: {row['timepoint']}, setup: {row['setup']}"  
#             interestpoints_prefix = f"{row['path']}/interestpoints/loc"
#             interest_points = root[interestpoints_prefix][:]

#             label = str(row["path"]).replace("\\", "/").lstrip("/").split("/", 2)[1]
#             self.interest_points.setdefault(view_id, {})[label] = interest_points
                            
#     def build_label_map(self):
#         """
#         Constructs a mapping of labels for each view ID from the interest points dataframe.
#         """
#         for _, row in self.interest_points_df.iterrows():
#             view_id_key = f"timepoint: {row['timepoint']}, setup: {row['setup']}"
            
#             if view_id_key not in self.label_map_global:
#                 self.label_map_global[view_id_key] = {}

#             self.label_map_global[view_id_key][row["label"]] = 1.0  

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
#         self.build_label_map()
#         self.get_all_interest_points_from_n5()
#         self.get_corresponding_data_from_n5()
#         self.get_connected_views_from_n5()

#         view_id_set = set()
#         for k in self.corresponding_interest_points.keys():
#             try:
#                 parts = [p.strip() for p in k.split(",")]
#                 tp = parts[0].split(":")[-1].strip()
#                 su = parts[1].split(":")[-1].strip()
#                 view_id_set.add((str(tp), str(su)))
#             except Exception:
#                 continue

#         self.view_id_set = sorted(view_id_set, key=lambda x: (int(x[0]), int(x[1])))

#         self.close_aiohttp_sessions()

#         return (
#             self.connected_views,
#             self.corresponding_interest_points,
#             self.interest_points,
#             self.label_map_global,
#             self.view_id_set,
#         )

import json
import fsspec
import numpy as np
import pandas as pd

"""
Load interest points and correspondences from the Parquet/JSON alignment store.
"""

class DataPrep:
    def __init__(self, interest_points_df, view_transform_matrices, xml_file_path, n5_input_path):
        self.interest_points_df = interest_points_df
        self.view_transform_matrices = view_transform_matrices
        self.xml_file_path = xml_file_path
        self.alignment_input_path = str(n5_input_path).rstrip("/")
        self.storage_options = {}
        self.connected_views = {}
        self.corresponding_interest_points = {}
        self.interest_points = {}
        self.label_map_global = {}
        self.view_id_set = []
        self.point_manifest = {}
        self.match_manifest = {}
        self.point_index_df = None
        self.match_index_df = None

    def join_uri(self, *parts):
        cleaned = []

        for i, part in enumerate(parts):
            if part is None:
                continue

            part = str(part)
            cleaned.append(part.rstrip("/") if i == 0 else part.strip("/"))

        return "/".join(cleaned)

    def get_fs_and_path(self, uri):
        return fsspec.core.url_to_fs(uri, **self.storage_options)

    def exists(self, uri):
        try:
            fs, path = self.get_fs_and_path(uri)
            return fs.exists(path)
        except Exception:
            return False

    def read_json(self, uri):
        fs, path = self.get_fs_and_path(uri)

        with fs.open(path, "r") as f:
            return json.load(f)

    def read_parquet(self, uri):
        fs, path = self.get_fs_and_path(uri)

        with fs.open(path, "rb") as f:
            return pd.read_parquet(f, engine="pyarrow")

    def view_id(self, timepoint, setup):
        return f"timepoint: {int(timepoint)}, setup: {int(setup)}"

    def parse_view_id(self, view_id):
        parts = str(view_id).split(",")

        timepoint = int(parts[0].split(":")[-1].strip())
        setup = int(parts[1].split(":")[-1].strip())

        return timepoint, setup

    def view_label_key(self, timepoint, setup, label):
        return f"{int(timepoint)}/{int(setup)}/{label}"

    def point_relative_path(self, timepoint, setup, label):
        return (
            f"points/"
            f"timepoint={int(timepoint)}/"
            f"setup={int(setup)}/"
            f"label={label}/"
            f"points.parquet"
        )

    def correspondences_relative_path(self, timepoint, setup, label):
        return (
            f"matches/"
            f"timepoint={int(timepoint)}/"
            f"setup={int(setup)}/"
            f"label={label}/"
            f"correspondences.parquet"
        )

    def id_map_relative_path(self, timepoint, setup, label):
        return (
            f"matches/"
            f"timepoint={int(timepoint)}/"
            f"setup={int(setup)}/"
            f"label={label}/"
            f"id_map.json"
        )

    def label_from_path(self, path):
        path = str(path).replace("\\", "/").strip("/")

        for part in path.split("/"):
            if part.startswith("label="):
                return part.split("=", 1)[1]

        parts = path.split("/")
        return parts[1] if len(parts) >= 2 else "beads"

    def load_manifests(self):
        point_manifest_uri = self.join_uri(self.alignment_input_path, "manifest.json")
        match_manifest_uri = self.join_uri(
            self.alignment_input_path,
            "matches",
            "manifest.json",
        )

        self.point_manifest = self.read_json(point_manifest_uri) if self.exists(point_manifest_uri) else {}
        self.match_manifest = self.read_json(match_manifest_uri) if self.exists(match_manifest_uri) else {}

    def normalize_point_index_from_existing_df(self):
        columns = ["timepoint", "setup", "label", "path", "num_points"]

        if self.interest_points_df is None:
            return pd.DataFrame(columns=columns)

        rows = []

        for _, row in self.interest_points_df.iterrows():
            timepoint = int(row["timepoint"])

            if "setup" in row:
                setup = int(row["setup"])
            elif "view_setup" in row:
                setup = int(row["view_setup"])
            else:
                raise KeyError("interest_points_df needs either 'setup' or 'view_setup'")

            if "label" in row and pd.notna(row["label"]):
                label = str(row["label"])
            elif "path" in row and pd.notna(row["path"]):
                label = self.label_from_path(row["path"])
            else:
                label = "beads"

            rows.append(
                {
                    "timepoint": timepoint,
                    "setup": setup,
                    "label": label,
                    "path": self.point_relative_path(timepoint, setup, label),
                    "num_points": -1,
                }
            )

        return pd.DataFrame(rows, columns=columns)

    def load_point_index(self):
        point_index_uri = self.join_uri(
            self.alignment_input_path,
            "point_index.parquet",
        )

        if self.exists(point_index_uri):
            df = self.read_parquet(point_index_uri)
        else:
            df = self.normalize_point_index_from_existing_df()

        if len(df) > 0:
            df["timepoint"] = df["timepoint"].astype("int32")
            df["setup"] = df["setup"].astype("int32")
            df["label"] = df["label"].astype(str)

            if "path" not in df.columns:
                df["path"] = [
                    self.point_relative_path(row.timepoint, row.setup, row.label)
                    for row in df.itertuples(index=False)
                ]

        self.point_index_df = df

    def load_match_index(self):
        match_index_uri = self.join_uri(
            self.alignment_input_path,
            "matches",
            "match_index.parquet",
        )

        if self.exists(match_index_uri):
            df = self.read_parquet(match_index_uri)
        else:
            print(f"⚠️ No match_index.parquet found at: {match_index_uri}")
            df = pd.DataFrame(
                columns=[
                    "timepoint",
                    "setup",
                    "label",
                    "correspondences_path",
                    "id_map_json_path",
                    "id_map_parquet_path",
                    "num_correspondences",
                    "num_target_views",
                ]
            )

        if len(df) > 0:
            df["timepoint"] = df["timepoint"].astype("int32")
            df["setup"] = df["setup"].astype("int32")
            df["label"] = df["label"].astype(str)

        self.match_index_df = df

    def resolve_point_relative_path(self, timepoint, setup, label):
        key = self.view_label_key(timepoint, setup, label)
        manifest_points = self.point_manifest.get("points", {}) or {}

        if key in manifest_points:
            return manifest_points[key]

        if self.point_index_df is not None and len(self.point_index_df) > 0:
            rows = self.point_index_df[
                (self.point_index_df["timepoint"].astype(int) == int(timepoint))
                & (self.point_index_df["setup"].astype(int) == int(setup))
                & (self.point_index_df["label"].astype(str) == str(label))
            ]

            if len(rows) > 0 and "path" in rows.columns:
                return str(rows.iloc[0]["path"])

        return self.point_relative_path(timepoint, setup, label)

    def resolve_point_uri(self, timepoint, setup, label):
        rel_path = self.resolve_point_relative_path(timepoint, setup, label)
        return self.join_uri(self.alignment_input_path, rel_path)

    def resolve_correspondences_uri(self, row):
        if "correspondences_path" in row and pd.notna(row["correspondences_path"]):
            rel_path = str(row["correspondences_path"])
        else:
            rel_path = self.correspondences_relative_path(
                row["timepoint"],
                row["setup"],
                row["label"],
            )

        return self.join_uri(self.alignment_input_path, rel_path)

    def resolve_id_map_json_uri(self, row):
        if "id_map_json_path" in row and pd.notna(row["id_map_json_path"]):
            rel_path = str(row["id_map_json_path"])
        else:
            rel_path = self.id_map_relative_path(
                row["timepoint"],
                row["setup"],
                row["label"],
            )

        return self.join_uri(self.alignment_input_path, rel_path)

    def build_label_map(self):
        self.label_map_global = {}

        if self.point_index_df is None:
            self.load_point_index()

        for row in self.point_index_df.itertuples(index=False):
            view_key = self.view_id(row.timepoint, row.setup)
            self.label_map_global.setdefault(view_key, {})[str(row.label)] = 1.0

    def load_points_for_view_label(self, timepoint, setup, label):
        point_uri = self.resolve_point_uri(timepoint, setup, label)

        try:
            df = self.read_parquet(point_uri)
        except Exception as e:
            print(
                f"⚠️ Failed to load points: "
                f"timepoint={timepoint} setup={setup} label={label}"
            )
            print(f"⚠️ Point URI: {point_uri}")
            print(f"⚠️ Error: {e}")
            return np.empty((0, 3), dtype=np.float64)

        if len(df) == 0:
            return np.empty((0, 3), dtype=np.float64)

        missing = {"x", "y", "z"}.difference(df.columns)

        if missing:
            raise ValueError(
                f"Missing required point columns {sorted(missing)} in {point_uri}"
            )

        return df[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=False)

    def get_all_interest_points_from_parquet(self):
        self.interest_points = {}

        if self.point_index_df is None:
            self.load_point_index()

        for row in self.point_index_df.itertuples(index=False):
            timepoint = int(row.timepoint)
            setup = int(row.setup)
            label = str(row.label)
            view_key = self.view_id(timepoint, setup)

            self.interest_points.setdefault(view_key, {})[label] = (
                self.load_points_for_view_label(timepoint, setup, label)
            )

    def read_id_map_for_match_row(self, row):
        id_map_uri = self.resolve_id_map_json_uri(row)

        if not self.exists(id_map_uri):
            return {}

        try:
            id_map = self.read_json(id_map_uri)
            return id_map if isinstance(id_map, dict) else {}
        except Exception as e:
            print(f"⚠️ Failed reading id map: {id_map_uri}")
            print(f"⚠️ Error: {e}")
            return {}

    def get_connected_views_from_parquet(self):
        self.connected_views = {}

        if self.match_index_df is None:
            self.load_match_index()

        for row in self.match_index_df.itertuples(index=False):
            timepoint = int(row.timepoint)
            setup = int(row.setup)
            label = str(row.label)

            view_key = self.view_id(timepoint, setup)
            id_map = self.read_id_map_for_match_row(row._asdict())

            if not id_map:
                continue

            self.connected_views.setdefault(view_key, {}).update(id_map)

    def get_corresponding_data_from_parquet(self):
        self.corresponding_interest_points = {}

        if self.match_index_df is None:
            self.load_match_index()

        if not self.interest_points:
            self.get_all_interest_points_from_parquet()

        for row in self.match_index_df.itertuples(index=False):
            self.load_correspondences_for_row(row)

    def load_correspondences_for_row(self, row):
        row_dict = row._asdict()

        source_timepoint = int(row.timepoint)
        source_setup = int(row.setup)
        source_label = str(row.label)
        source_view_id = self.view_id(source_timepoint, source_setup)

        corr_uri = self.resolve_correspondences_uri(row_dict)

        if not self.exists(corr_uri):
            print(
                f"⚠️ Skipping {source_view_id}, label={source_label}: "
                f"missing correspondences file."
            )
            return

        try:
            corr_df = self.read_parquet(corr_uri)
        except Exception as e:
            print(
                f"⚠️ Skipping {source_view_id}, label={source_label}: "
                f"could not read correspondences."
            )
            print(f"⚠️ Correspondences URI: {corr_uri}")
            print(f"⚠️ Error: {e}")
            return

        if len(corr_df) == 0:
            return

        required = {
            "source_point_id",
            "target_point_id",
            "target_timepoint",
            "target_setup",
            "target_label",
        }

        missing = required.difference(corr_df.columns)

        if missing:
            print(
                f"⚠️ Skipping {source_view_id}, label={source_label}: "
                f"missing correspondence columns {sorted(missing)}"
            )
            return

        for corr in corr_df.itertuples(index=False):
            self.add_correspondence(
                corr=corr,
                source_view_id=source_view_id,
                source_label=source_label,
            )

    def add_correspondence(self, corr, source_view_id, source_label):
        source_point_id = int(corr.source_point_id)
        target_point_id = int(corr.target_point_id)

        target_view_id = self.view_id(corr.target_timepoint, corr.target_setup)
        target_label = str(corr.target_label)

        source_points = self.interest_points.get(source_view_id, {}).get(source_label)
        target_points = self.interest_points.get(target_view_id, {}).get(target_label)

        if source_points is None or target_points is None:
            return

        if source_point_id < 0 or source_point_id >= len(source_points):
            return

        if target_point_id < 0 or target_point_id >= len(target_points):
            return

        self.corresponding_interest_points.setdefault(source_view_id, []).append(
            {
                "detection_id": source_point_id,
                "detection_p1": source_points[source_point_id],
                "corresponding_detection_id": target_point_id,
                "corresponding_detection_p2": target_points[target_point_id],
                "corresponding_view_id": target_view_id,
                "label": target_label,
            }
        )

    def build_view_id_set(self):
        view_id_set = set()

        for view_key in self.corresponding_interest_points:
            try:
                timepoint, setup = self.parse_view_id(view_key)
                view_id_set.add((str(timepoint), str(setup)))
            except Exception:
                continue

        self.view_id_set = sorted(
            view_id_set,
            key=lambda x: (int(x[0]), int(x[1])),
        )

    def run(self):
        self.load_manifests()
        self.load_point_index()
        self.load_match_index()
        self.build_label_map()
        self.get_all_interest_points_from_parquet()
        self.get_corresponding_data_from_parquet()
        self.get_connected_views_from_parquet()
        self.build_view_id_set()
        print("Points and Metadata Loaded")

        return (
            self.connected_views,
            self.corresponding_interest_points,
            self.interest_points,
            self.label_map_global,
            self.view_id_set,
        )