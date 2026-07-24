import zarr
import fsspec
import s3fs
import os
import json
import shutil
import pandas as pd

class DataPrep:
    def __init__(self, min_alignment_level):
        self.min_alignment_level = min_alignment_level

    def get_zarr_loader(self, root):
        loader = root.find("./SequenceDescription/ImageLoader")

        nested_loader = loader.find("ImageLoader")
        if nested_loader is not None and nested_loader.find("zgroups") is not None:
            return nested_loader

        return loader

    def get_tile_shape(self, zarr_path: str):
        root = zarr.open(fsspec.get_mapper(zarr_path.rstrip("/")), mode="r")
        return root[str(self.min_alignment_level)].shape
    
    def get_voxels(self, zarr_path: str):
        root = zarr.open(fsspec.get_mapper(zarr_path.rstrip("/")), mode="r")
        level_meta = root.attrs["multiscales"][0]["datasets"][self.min_alignment_level]
        scale_transform = next(
            t for t in level_meta["coordinateTransformations"]
            if t["type"] == "scale"
        )
        return tuple(scale_transform["scale"][-3:]) 

    def get_level0_spacing_zyx(self, image_root):
        root = zarr.open(fsspec.get_mapper(image_root.rstrip("/")), mode="r")
        level_meta = next(
            d for d in root.attrs["multiscales"][0]["datasets"]
            if str(d["path"]).strip("/") == "0"
        )
        scale = next(
            t for t in level_meta["coordinateTransformations"]
            if t["type"] == "scale"
        )
        return tuple(float(v) for v in scale["scale"][-3:])

    def combine_interest_point_stores(self, moving_point_store, fixed_point_store, combined_point_store):
        if combined_point_store.startswith("s3://"):
            if not moving_point_store.startswith("s3://") or not fixed_point_store.startswith("s3://"):
                raise ValueError("All interest-point store paths must use S3")

            s3 = s3fs.S3FileSystem(anon=False)

            moving_root = moving_point_store.removeprefix("s3://").rstrip("/")
            fixed_root = fixed_point_store.removeprefix("s3://").rstrip("/")
            combined_root = combined_point_store.removeprefix("s3://").rstrip("/")

            if s3.exists(combined_root):
                s3.rm(combined_root, recursive=True)

            for source_path in s3.find(moving_root):
                relative_path = source_path[len(moving_root):].lstrip("/")
                s3.copy(source_path, f"{combined_root}/{relative_path}")

            fixed_points_root = f"{fixed_root}/points"

            for source_path in s3.find(fixed_points_root):
                relative_path = source_path[len(fixed_points_root):].lstrip("/")
                s3.copy(source_path, f"{combined_root}/points/{relative_path}")

            with s3.open(f"{moving_root}/manifest.json", "r") as file:
                manifest = json.load(file)

            with s3.open(f"{fixed_root}/manifest.json", "r") as file:
                fixed_manifest = json.load(file)

            manifest.setdefault("points", {})
            manifest["points"].update(fixed_manifest.get("points", {}))

            with s3.open(f"{combined_root}/manifest.json", "w") as file:
                json.dump(manifest, file, indent=2)

            with s3.open(f"{moving_root}/point_index.parquet", "rb") as file:
                moving_index = pd.read_parquet(file)

            with s3.open(f"{fixed_root}/point_index.parquet", "rb") as file:
                fixed_index = pd.read_parquet(file)

            combined_index = (
                pd.concat([moving_index, fixed_index], ignore_index=True)
                .drop_duplicates(["timepoint", "setup", "label"], keep="last")
                .sort_values(["timepoint", "setup", "label"])
                .reset_index(drop=True)
            )

            with s3.open(f"{combined_root}/point_index.parquet", "wb") as file:
                combined_index.to_parquet(file, index=False)

        else:
            if os.path.exists(combined_point_store):
                shutil.rmtree(combined_point_store)

            shutil.copytree(moving_point_store, combined_point_store)

            shutil.copytree(
                os.path.join(fixed_point_store, "points"),
                os.path.join(combined_point_store, "points"),
                dirs_exist_ok=True,
            )

            moving_manifest_path = os.path.join(moving_point_store, "manifest.json")
            fixed_manifest_path = os.path.join(fixed_point_store, "manifest.json")
            combined_manifest_path = os.path.join(combined_point_store, "manifest.json")

            with open(moving_manifest_path) as file:
                manifest = json.load(file)

            with open(fixed_manifest_path) as file:
                fixed_manifest = json.load(file)

            manifest.setdefault("points", {})
            manifest["points"].update(fixed_manifest.get("points", {}))

            with open(combined_manifest_path, "w") as file:
                json.dump(manifest, file, indent=2)

            moving_index = pd.read_parquet(
                os.path.join(moving_point_store, "point_index.parquet")
            )
            fixed_index = pd.read_parquet(
                os.path.join(fixed_point_store, "point_index.parquet")
            )

            combined_index = (
                pd.concat([moving_index, fixed_index], ignore_index=True)
                .drop_duplicates(["timepoint", "setup", "label"], keep="last")
                .sort_values(["timepoint", "setup", "label"])
                .reset_index(drop=True)
            )

            combined_index.to_parquet(
                os.path.join(combined_point_store, "point_index.parquet"),
                index=False,
            )

        print(f"Combined interest-point store: {combined_point_store}")
        return combined_point_store
