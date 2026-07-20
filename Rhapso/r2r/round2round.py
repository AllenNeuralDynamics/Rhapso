import zarr
import fsspec
import s3fs
import os
import numpy as np
import copy
import json
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from Rhapso.pipelines.ray.split_dataset import SplitDataset
from Rhapso.pipelines.ray.interest_point_detection import InterestPointDetection
from Rhapso.pipelines.ray.interest_point_matching import InterestPointMatching
from Rhapso.pipelines.ray.solver import Solver

class R2R_Pipeline():
    def __init__(self, fixed_image_multiscale_root, moving_image_multiscale_root, moving_segmentation_zarr_path, min_alignment_level, 
                 max_alignment_level, scratch_directory, results_directory, minimum_alignment_blocksize_physical_units, output_multiscale_level, 
                 multiscale_output, tiling_schedule, tile_edge_filter_enabled, ip_registration_config_file, min_peak_intensity_override, 
                 debug_skip_parameter_sweep, output_transform_path, output_warped_segmentation_path, output_warped_image_path, output_qc_dir,
                 fixed_image_min, fixed_image_max, moving_image_min, moving_image_max, dask_client
    ):
        self.fixed_image_multiscale_root=fixed_image_multiscale_root
        self.moving_image_multiscale_root=moving_image_multiscale_root
        self.moving_segmentation_zarr_path=moving_segmentation_zarr_path
        self.min_alignment_level=min_alignment_level
        self.max_alignment_level=max_alignment_level
        self.scratch_directory=scratch_directory
        self.results_directory=results_directory
        self.minimum_alignment_blocksize_physical_units=minimum_alignment_blocksize_physical_units
        self.output_multiscale_level=output_multiscale_level
        self.multiscale_output=multiscale_output
        self.tiling_schedule=tiling_schedule
        self.tile_edge_filter_enabled=tile_edge_filter_enabled
        self.ip_registration_config_file=ip_registration_config_file
        self.min_peak_intensity_override=min_peak_intensity_override
        self.debug_skip_parameter_sweep=debug_skip_parameter_sweep
        self.output_transform_path=output_transform_path
        self.output_warped_segmentation_path=output_warped_segmentation_path
        self.output_warped_image_path=output_warped_image_path
        self.output_qc_dir=output_qc_dir
        self.fixed_image_min=fixed_image_min
        self.fixed_image_max=fixed_image_max
        self.moving_image_min=moving_image_min
        self.moving_image_max=moving_image_max
        self.dask_client=dask_client
    
    def get_solver_affines_zyx(self, registered_xml_path, fixed_setup_id):
        root = ET.parse(registered_xml_path).getroot()
        tile_affines = {}

        fixed_spacing = self.get_level0_spacing_zyx(
            self.fixed_image_multiscale_root
        )
        moving_spacing = self.get_level0_spacing_zyx(
            self.moving_image_multiscale_root
        )
        world_to_fixed_iso = moving_spacing[1] / fixed_spacing[1]

        for registration in root.findall(
            "./ViewRegistrations/ViewRegistration"
        ):
            setup_id = int(registration.get("setup"))

            if setup_id == fixed_setup_id:
                continue

            solver_transform = next(
                (
                    transform
                    for transform in registration.findall("ViewTransform")
                    if any(
                        name in transform.findtext("Name", "").lower()
                        for name in ("rigidmodel3d", "affinemodel3d")
                    )
                ),
                None,
            )

            if solver_transform is None:
                raise ValueError(
                    f"No solver transform found for moving setup {setup_id}"
                )

            values = np.fromstring(
                solver_transform.findtext("affine", ""),
                dtype=np.float64,
                sep=" ",
            )

            if values.size != 12:
                raise ValueError(
                    f"Expected 12 affine values for setup {setup_id}, "
                    f"found {values.size}"
                )

            affine_xyz = np.eye(4, dtype=np.float64)
            affine_xyz[:3, :] = values.reshape(3, 4)

            # BigStitcher solver: moving -> fixed, XYZ.
            # BigStream displacement: fixed -> moving, ZYX.
            affine_xyz = np.linalg.inv(affine_xyz)

            axis_order = [2, 1, 0, 3]
            affine_zyx = affine_xyz[np.ix_(axis_order, axis_order)]

            # Convert BigStitcher world translation to fixed isotropic units.
            affine_zyx[:3, 3] *= world_to_fixed_iso

            tile_affines[setup_id] = affine_zyx

        if not tile_affines:
            raise ValueError("No moving solver transforms found")

        return tile_affines


    def get_tile_index_map(
        self,
        tile_bboxes_zyx,
        tile_size_zyx,
        overlap_zyx,
    ):
        origin_zyx = tuple(
            min(bbox[0][axis] for bbox in tile_bboxes_zyx.values())
            for axis in range(3)
        )
        stride_zyx = tuple(
            tile_size_zyx[axis] - overlap_zyx[axis]
            for axis in range(3)
        )

        if any(stride <= 0 for stride in stride_zyx):
            raise ValueError(
                f"Invalid tile stride: tile_size={tile_size_zyx}, "
                f"overlap={overlap_zyx}"
            )

        tile_index_map = {}

        for tile_id, (tile_min_zyx, _) in tile_bboxes_zyx.items():
            tile_index = tuple(
                int(
                    round(
                        (tile_min_zyx[axis] - origin_zyx[axis])
                        / stride_zyx[axis]
                    )
                )
                for axis in range(3)
            )
            tile_index_map[tile_id] = tile_index

        return tile_index_map


    def fill_missing_tile_affines(
        self,
        tile_affines,
        all_tile_bboxes_zyx,
    ):
        identity = np.eye(4, dtype=np.float64)

        survivors = {
            tile_id: affine
            for tile_id, affine in tile_affines.items()
            if not np.allclose(affine, identity, atol=1e-6)
        }

        if not survivors:
            raise RuntimeError(
                "All moving tiles have identity transforms"
            )

        missing_ids = [
            tile_id
            for tile_id in all_tile_bboxes_zyx
            if tile_id not in survivors
        ]

        if not missing_ids:
            return tile_affines

        survivor_fraction = len(survivors) / len(all_tile_bboxes_zyx)

        centroids = {
            tile_id: 0.5 * (
                np.asarray(all_tile_bboxes_zyx[tile_id][0], dtype=float)
                + np.asarray(all_tile_bboxes_zyx[tile_id][1], dtype=float)
            )
            for tile_id in survivors
        }

        result = dict(tile_affines)

        for tile_id in missing_ids:
            if survivor_fraction < 0.5:
                result[tile_id] = identity.copy()
                continue

            centroid = 0.5 * (
                np.asarray(all_tile_bboxes_zyx[tile_id][0], dtype=float)
                + np.asarray(all_tile_bboxes_zyx[tile_id][1], dtype=float)
            )

            nearest_id = min(
                survivors,
                key=lambda survivor_id: np.linalg.norm(
                    centroids[survivor_id] - centroid
                ),
            )

            affine = survivors[nearest_id].copy()

            # Keep local rotation/scale, but do not copy a tile-local
            # translation into a different tile.
            affine[:3, 3] = 0.0
            result[tile_id] = affine

        return result


    def get_displacement_weight_block(
        self,
        tile_min_zyx,
        tile_max_zyx,
        block_min_zyx,
        block_max_zyx,
        tile_index_zyx,
        block_grid_zyx,
        overlap_zyx,
    ):
        axis_weights = []

        for axis in range(3):
            tile_length = tile_max_zyx[axis] - tile_min_zyx[axis]

            positions = np.arange(
                block_min_zyx[axis] - tile_min_zyx[axis],
                block_max_zyx[axis] - tile_min_zyx[axis],
                dtype=np.float32,
            )

            weights = np.ones(positions.shape, dtype=np.float32)

            low_overlap = (
                overlap_zyx[axis]
                if tile_index_zyx[axis] > 0
                else 0
            )
            high_overlap = (
                overlap_zyx[axis]
                if tile_index_zyx[axis] < block_grid_zyx[axis] - 1
                else 0
            )

            if low_overlap > 0:
                mask = positions < low_overlap
                weights[mask] = (
                    positions[mask] + 1.0
                ) / (low_overlap + 1.0)

            if high_overlap > 0:
                high_start = tile_length - high_overlap
                mask = positions >= high_start
                weights[mask] = np.minimum(
                    weights[mask],
                    (tile_length - positions[mask])
                    / (high_overlap + 1.0),
                )

            axis_weights.append(weights)

        return (
            axis_weights[0][:, None, None]
            * axis_weights[1][None, :, None]
            * axis_weights[2][None, None, :]
        )

    def matrix_to_displacement_field(self, affine, shape_zyx, spacing_zyx):
        spacing_zyx = np.asarray(spacing_zyx, dtype=np.float64)

        z = (
            np.arange(shape_zyx[0], dtype=np.float64)
            * spacing_zyx[0]
        )[:, None, None]
        y = (
            np.arange(shape_zyx[1], dtype=np.float64)
            * spacing_zyx[1]
        )[None, :, None]
        x = (
            np.arange(shape_zyx[2], dtype=np.float64)
            * spacing_zyx[2]
        )[None, None, :]

        matrix = affine[:3, :3]
        translation = affine[:3, 3]

        displacement = np.empty(
            shape_zyx + (3,),
            dtype=np.float32,
        )

        displacement[..., 0] = (
            matrix[0, 0] * z
            + matrix[0, 1] * y
            + matrix[0, 2] * x
            + translation[0]
            - z
        )
        displacement[..., 1] = (
            matrix[1, 0] * z
            + matrix[1, 1] * y
            + matrix[1, 2] * x
            + translation[1]
            - y
        )
        displacement[..., 2] = (
            matrix[2, 0] * z
            + matrix[2, 1] * y
            + matrix[2, 2] * x
            + translation[2]
            - x
        )

        return displacement

    def move_affine_origin(
        self,
        affine,
        block_min_zyx,
        physical_spacing_zyx,
    ):
        origin = (
            np.asarray(block_min_zyx, dtype=np.float64)
            * physical_spacing_zyx
        )

        local_affine = affine.copy()
        local_affine[:3, 3] = (
            affine[:3, :3] @ origin
            + affine[:3, 3]
            - origin
        )

        return local_affine


    def create_displacement_field(
        self,
        registered_xml_path,
        fixed_setup_id,
        block_grid_zyx,
        tile_size_zyx,
        overlap_zyx,
        all_tile_bboxes_zyx,
        output_path,
        block_shape_zyx=(32, 128, 128),
    ):
        fixed_root = zarr.open(
            fsspec.get_mapper(
                self.fixed_image_multiscale_root.rstrip("/")
            ),
            mode="r",
        )
        fixed_shape_zyx = tuple(
            int(value) for value in fixed_root["0"].shape[-3:]
        )

        fixed_spacing_zyx = self.get_level0_spacing_zyx(
            self.fixed_image_multiscale_root
        )
        physical_spacing_zyx = np.asarray(
            (
                fixed_spacing_zyx[0] / fixed_spacing_zyx[1],
                1.0,
                1.0,
            ),
            dtype=np.float64,
        )

        tile_affines = self.get_solver_affines_zyx(
            registered_xml_path,
            fixed_setup_id,
        )
        tile_affines = self.fill_missing_tile_affines(
            tile_affines,
            all_tile_bboxes_zyx,
        )

        tile_index_map = self.get_tile_index_map(
            all_tile_bboxes_zyx,
            tile_size_zyx,
            overlap_zyx,
        )

        chunks_zyx = tuple(
            min(block_shape_zyx[axis], fixed_shape_zyx[axis])
            for axis in range(3)
        )

        temp_path = os.path.join(
            self.scratch_directory,
            "_displacement_accumulators.zarr",
        )

        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        temp_root = zarr.open_group(temp_path, mode="w")

        displacement_sum = temp_root.create_array(
            "displacement_sum",
            shape=fixed_shape_zyx + (3,),
            chunks=chunks_zyx + (3,),
            dtype=np.float32,
            fill_value=0,
            overwrite=True,
        )
        weight_sum = temp_root.create_array(
            "weight_sum",
            shape=fixed_shape_zyx,
            chunks=chunks_zyx,
            dtype=np.float32,
            fill_value=0,
            overwrite=True,
        )

        try:
            for tile_id, affine in tile_affines.items():
                tile_min_zyx, tile_max_zyx = (
                    all_tile_bboxes_zyx[tile_id]
                )
                tile_index_zyx = tile_index_map[tile_id]

                clipped_min_zyx = tuple(
                    max(0, tile_min_zyx[axis])
                    for axis in range(3)
                )
                clipped_max_zyx = tuple(
                    min(fixed_shape_zyx[axis], tile_max_zyx[axis])
                    for axis in range(3)
                )

                if any(
                    clipped_max_zyx[axis] <= clipped_min_zyx[axis]
                    for axis in range(3)
                ):
                    continue

                for z0 in range(
                    clipped_min_zyx[0],
                    clipped_max_zyx[0],
                    chunks_zyx[0],
                ):
                    z1 = min(
                        z0 + chunks_zyx[0],
                        clipped_max_zyx[0],
                    )

                    for y0 in range(
                        clipped_min_zyx[1],
                        clipped_max_zyx[1],
                        chunks_zyx[1],
                    ):
                        y1 = min(
                            y0 + chunks_zyx[1],
                            clipped_max_zyx[1],
                        )

                        for x0 in range(
                            clipped_min_zyx[2],
                            clipped_max_zyx[2],
                            chunks_zyx[2],
                        ):
                            x1 = min(
                                x0 + chunks_zyx[2],
                                clipped_max_zyx[2],
                            )

                            block_min_zyx = (z0, y0, x0)
                            block_max_zyx = (z1, y1, x1)
                            block_shape = (
                                z1 - z0,
                                y1 - y0,
                                x1 - x0,
                            )

                            weights = self.get_displacement_weight_block(
                                tile_min_zyx=tile_min_zyx,
                                tile_max_zyx=tile_max_zyx,
                                block_min_zyx=block_min_zyx,
                                block_max_zyx=block_max_zyx,
                                tile_index_zyx=tile_index_zyx,
                                block_grid_zyx=block_grid_zyx,
                                overlap_zyx=overlap_zyx,
                            )

                            local_affine = self.move_affine_origin(
                                affine,
                                block_min_zyx,
                                physical_spacing_zyx,
                            )

                            block_displacement = self.matrix_to_displacement_field(
                                affine=local_affine,
                                shape_zyx=block_shape,
                                spacing_zyx=physical_spacing_zyx,
                            )

                            spatial_slice = (
                                slice(z0, z1),
                                slice(y0, y1),
                                slice(x0, x1),
                            )
                            vector_slice = spatial_slice + (
                                slice(None),
                            )

                            accumulated_displacement = np.asarray(
                                displacement_sum[vector_slice]
                            )
                            accumulated_weights = np.asarray(
                                weight_sum[spatial_slice]
                            )

                            accumulated_displacement += (
                                block_displacement
                                * weights[..., None]
                            )
                            accumulated_weights += weights

                            displacement_sum[vector_slice] = (
                                accumulated_displacement
                            )
                            weight_sum[spatial_slice] = (
                                accumulated_weights
                            )

            output_path = output_path.rstrip("/")

            if output_path.startswith("s3://"):
                output_store = fsspec.get_mapper(output_path, create=True)
                output_root = zarr.open_group(output_store, mode="w")
            else:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                if os.path.exists(output_path):
                    shutil.rmtree(output_path)

                os.makedirs(output_path, exist_ok=True)
                output_root = zarr.open_group(output_path, mode="w")
                
            output = output_root.create_array(
                "0",
                shape=fixed_shape_zyx + (3,),
                chunks=chunks_zyx + (3,),
                dtype=np.float32,
                fill_value=0,
                overwrite=True,
            )

            uncovered_voxels = 0

            for z0 in range(0, fixed_shape_zyx[0], chunks_zyx[0]):
                z1 = min(z0 + chunks_zyx[0], fixed_shape_zyx[0])

                for y0 in range(0, fixed_shape_zyx[1], chunks_zyx[1]):
                    y1 = min(y0 + chunks_zyx[1], fixed_shape_zyx[1])

                    for x0 in range(
                        0,
                        fixed_shape_zyx[2],
                        chunks_zyx[2],
                    ):
                        x1 = min(
                            x0 + chunks_zyx[2],
                            fixed_shape_zyx[2],
                        )

                        spatial_slice = (
                            slice(z0, z1),
                            slice(y0, y1),
                            slice(x0, x1),
                        )
                        vector_slice = spatial_slice + (
                            slice(None),
                        )

                        displacement = np.asarray(
                            displacement_sum[vector_slice]
                        )
                        weights = np.asarray(
                            weight_sum[spatial_slice]
                        )

                        uncovered_voxels += int(
                            np.count_nonzero(weights == 0)
                        )

                        normalized = np.zeros_like(displacement)

                        np.divide(
                            displacement,
                            weights[..., None],
                            out=normalized,
                            where=weights[..., None] > 0,
                        )

                        output[vector_slice] = normalized

            metadata = {
                "physical_spacing_zyx": (
                    physical_spacing_zyx.tolist()
                ),
                "creation_level": 0,
                "unit_system": "isotropic-scale-0",
                "vector_axis_order": "zyx",
            }
            output_root.attrs.update(metadata)
            output.attrs.update(metadata)

            if uncovered_voxels:
                print(
                    f"Warning: {uncovered_voxels} displacement "
                    f"voxels had zero weight"
                )

        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

        print(f"Displacement field saved: {output_path}")
        print(f"Displacement field shape: {output.shape}")

        return output_path

    def combine_interest_point_stores(self, moving_point_store, fixed_point_store, combined_point_store):
        if os.path.exists(combined_point_store):
            shutil.rmtree(combined_point_store)

        shutil.copytree(moving_point_store, combined_point_store)

        shutil.copytree(
            os.path.join(fixed_point_store, "points"),
            os.path.join(combined_point_store, "points"),
            dirs_exist_ok=True,
        )

        moving_manifest_path = os.path.join(
            moving_point_store, "manifest.json"
        )
        fixed_manifest_path = os.path.join(
            fixed_point_store, "manifest.json"
        )
        combined_manifest_path = os.path.join(
            combined_point_store, "manifest.json"
        )

        with open(moving_manifest_path) as f:
            manifest = json.load(f)

        with open(fixed_manifest_path) as f:
            fixed_manifest = json.load(f)

        manifest.setdefault("points", {})
        manifest["points"].update(fixed_manifest.get("points", {}))

        with open(combined_manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        moving_index = pd.read_parquet(
            os.path.join(moving_point_store, "point_index.parquet")
        )
        fixed_index = pd.read_parquet(
            os.path.join(fixed_point_store, "point_index.parquet")
        )

        combined_index = (
            pd.concat([moving_index, fixed_index], ignore_index=True)
            .drop_duplicates(
                ["timepoint", "setup", "label"],
                keep="last",
            )
            .sort_values(["timepoint", "setup", "label"])
            .reset_index(drop=True)
        )

        combined_index.to_parquet(
            os.path.join(combined_point_store, "point_index.parquet"),
            index=False,
        )

        print(
            f"Combined interest-point store: "
            f"{combined_point_store}"
        )

        return combined_point_store
    
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

    def set_xml_calibration(self, xml_path, calibration_xyz):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        x, y, z = calibration_xyz

        for registration in root.findall("./ViewRegistrations/ViewRegistration"):
            transform = next(
                (
                    t for t in registration.findall("ViewTransform")
                    if t.findtext("Name") == "calibration"
                ),
                None,
            )

            if transform is None:
                transform = ET.Element("ViewTransform", type="affine")
                ET.SubElement(transform, "Name").text = "calibration"
                registration.insert(0, transform)

            affine = transform.find("affine")
            if affine is None:
                affine = ET.SubElement(transform, "affine")

            affine.text = f"{x} 0 0 0 0 {y} 0 0 0 0 {z} 0"

        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="UTF-8", xml_declaration=True)

    def normalize_loop0_calibration(self, fixed_detected_xml_path, moving_detected_xml_path):
        fixed_spacing = self.get_level0_spacing_zyx(self.fixed_image_multiscale_root)
        moving_spacing = self.get_level0_spacing_zyx(self.moving_image_multiscale_root)

        scale_zyx = tuple(f / m for f, m in zip(fixed_spacing, moving_spacing))
        moving_z_scale = moving_spacing[0] / moving_spacing[1]

        fixed_cal_xyz = (
            scale_zyx[2],
            scale_zyx[1],
            scale_zyx[0] * moving_z_scale,
        )
        moving_cal_xyz = (1.0, 1.0, moving_z_scale)

        self.set_xml_calibration(fixed_detected_xml_path, fixed_cal_xyz)
        self.set_xml_calibration(moving_detected_xml_path, moving_cal_xyz)

        print(f"Fixed calibration XYZ: {fixed_cal_xyz}")
        print(f"Moving calibration XYZ: {moving_cal_xyz}")
    
    def get_split_ids(self, xml_path):
        root = ET.parse(xml_path).getroot()

        return [
            int(definition.findtext("NewId"))
            for definition in root.findall(
                "./SequenceDescription/ImageLoader/SetupIds/SetupIdDefinition"
            )
            if int(definition.findtext("OldId")) == 0
        ]

    def renumber_fixed_setup(self, fixed_xml_path, new_id):
        tree = ET.parse(fixed_xml_path)
        root = tree.getroot()
        new_id = str(new_id)

        for zgroup in root.findall("./SequenceDescription/ImageLoader/zgroups/zgroup"):
            zgroup.set("setup", new_id)

        root.find("./SequenceDescription/ViewSetups/ViewSetup/id").text = new_id

        for registration in root.findall("./ViewRegistrations/ViewRegistration"):
            registration.set("setup", new_id)

        ET.indent(tree, space="  ")
        tree.write(fixed_xml_path, encoding="UTF-8", xml_declaration=True)

        print(f"Renumbered fixed setup ID to {new_id}")

    def detect_interest_points(
        self, input_xml_path, image_root, output_xml_path,
        point_store_dir, min_intensity, max_intensity
    ):
        os.makedirs(point_store_dir, exist_ok=True)
        image_prefix = image_root.rstrip("/").rsplit("/", 1)[0] + "/"

        detection = InterestPointDetection(
            dsxy=8, dsz=8,
            min_intensity=min_intensity, max_intensity=max_intensity,
            sigma=1.8, threshold=0.01, file_type="zarr",
            xml_file_path=input_xml_path,
            image_file_prefix=image_prefix,
            xml_output_file_path=output_xml_path,
            n5_output_file_prefix=point_store_dir.rstrip("/") + "/",
            combine_distance=8, chunks_per_bound=1,
            run_type="ray", max_spots=30000,
            median_filter=8, overlap_only=False,
        )

        detection.run()
        return output_xml_path
    
    def remove_empty_moving_tiles(self, moving_split_xml_path, exclusion_percentage):
        tree = ET.parse(moving_split_xml_path)
        root = tree.getroot()

        tile_bboxes_zyx = {}

        for definition in root.findall("./SequenceDescription/ImageLoader/SetupIds/SetupIdDefinition"):
            if int(definition.findtext("OldId")) != 0:
                continue

            tile_id = int(definition.findtext("NewId"))
            min_xyz = tuple(map(int, definition.findtext("min").split()))
            max_xyz = tuple(map(int, definition.findtext("max").split()))       

            tile_bboxes_zyx[tile_id] = (
                tuple(reversed(min_xyz)),
                tuple(value + 1 for value in reversed(max_xyz)),
            )

        seg = zarr.open(
            fsspec.get_mapper(self.moving_segmentation_zarr_path.rstrip("/")),
            mode="r",
        )

        if not hasattr(seg, "shape"):
            level = str(self.min_alignment_level)

            if level in seg:
                seg = seg[level]
            elif "0" in seg:
                seg = seg["0"]
            else:
                seg = seg[next(iter(seg.array_keys()))]

        moving = zarr.open(
            fsspec.get_mapper(self.moving_image_multiscale_root.rstrip("/")),
            mode="r",
        )["0"]

        moving_shape_zyx = np.asarray(moving.shape[-3:], dtype=float)
        seg_shape_zyx = np.asarray(seg.shape[-3:], dtype=int)
        downsample_zyx = moving_shape_zyx / seg_shape_zyx

        empty_tile_ids = []

        for tile_id, (tile_min_zyx, tile_max_zyx) in tile_bboxes_zyx.items():
            seg_min = np.floor(np.asarray(tile_min_zyx) / downsample_zyx).astype(int)
            seg_max = np.ceil(np.asarray(tile_max_zyx) / downsample_zyx).astype(int)

            seg_min = np.maximum(seg_min, 0)
            seg_max = np.minimum(seg_max, seg_shape_zyx)

            margin = ((seg_max - seg_min) * exclusion_percentage).astype(int)
            center_min = seg_min + margin
            center_max = seg_max - margin

            if np.any(center_max <= center_min):
                center_min = seg_min
                center_max = seg_max

            slices = tuple(
                slice(int(center_min[i]), int(center_max[i]))
                for i in range(3)
            )

            prefix = (0,) * (seg.ndim - 3)

            if not np.any(np.asarray(seg[prefix + slices])):
                empty_tile_ids.append(tile_id)

        ids_to_remove = {str(tile_id) for tile_id in empty_tile_ids}

        setup_ids = root.find("./SequenceDescription/ImageLoader/SetupIds")
        for definition in list(setup_ids.findall("SetupIdDefinition")):
            if definition.findtext("NewId") in ids_to_remove:
                setup_ids.remove(definition)

        view_setups = root.find("./SequenceDescription/ViewSetups")
        for view_setup in list(view_setups.findall("ViewSetup")):
            if view_setup.findtext("id") in ids_to_remove:
                view_setups.remove(view_setup)

        tile_attributes = view_setups.find('Attributes[@name="tile"]')
        for tile in list(tile_attributes.findall("Tile")):
            if tile.findtext("id") in ids_to_remove:
                tile_attributes.remove(tile)

        view_registrations = root.find("./ViewRegistrations")
        for registration in list(view_registrations.findall("ViewRegistration")):
            if registration.get("setup") in ids_to_remove:
                view_registrations.remove(registration)

        view_interest_points = root.find("./ViewInterestPoints")
        if view_interest_points is not None:
            for entry in list(view_interest_points.findall("ViewInterestPointsFile")):
                if entry.get("setup") in ids_to_remove:
                    view_interest_points.remove(entry)

        ET.indent(tree, space="  ")
        tree.write(moving_split_xml_path, encoding="UTF-8", xml_declaration=True)

        print(f"Removed {len(empty_tile_ids)} empty moving tiles")

        return tile_bboxes_zyx
    
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
    
    def get_block_size_and_overlap_1d(self, size, num_blocks):
        if num_blocks == 1:
            return size, 0

        for block_size in range(size, 0, -1):
            total_overlap = (num_blocks * block_size) - size

            if total_overlap < 0:
                continue

            if total_overlap % (num_blocks - 1) != 0:
                continue

            overlap = total_overlap // (num_blocks - 1)

            if overlap <= block_size // 2:
                return block_size, overlap

        raise ValueError(f"Could not tile size={size} with num_blocks={num_blocks}")

    def get_tile_schedule(self, moving_image_shape, moving_image_voxel_size):
        shape_zyx = tuple(moving_image_shape[-3:])
        spacing_zyx = tuple(moving_image_voxel_size[-3:])

        schedule = []

        loop_index = 0
        while True:
            if loop_index == 0:
                grid_zyx = (1, 1, 1)
            else:
                xy = 2 ** (loop_index + 1)
                z = max(1, round(xy * shape_zyx[0] / shape_zyx[1]))
                grid_zyx = (z, xy, xy)

            block_size_zyx = []
            overlap_zyx = []

            for size, blocks in zip(shape_zyx, grid_zyx):
                block_size, overlap = self.get_block_size_and_overlap_1d(size, blocks)
                block_size_zyx.append(block_size)
                overlap_zyx.append(overlap)

            block_size_zyx = tuple(block_size_zyx)
            overlap_zyx = tuple(overlap_zyx)

            block_size_um_zyx = tuple(
                block_size_zyx[i] * spacing_zyx[i]
                for i in range(3)
            )

            if any(size_um < self.minimum_alignment_blocksize_physical_units for size_um in block_size_um_zyx):
                break

            schedule.append(grid_zyx)
            loop_index += 1

        return schedule

    def build_dataset_xml(self, image_root, xml_filename, xml_output_dir, setup_0_size_xyz, setup_0_voxel_size_xyz):
        root_path = image_root.rstrip("/")

        if root_path.startswith("s3://"):
            s3_path = root_path.removeprefix("s3://")
            s3_bucket, object_key = s3_path.split("/", 1)
            setup_0_path, setup_0_name = object_key.rsplit("/", 1)
        else:
            s3_bucket = None
            setup_0_path, setup_0_name = root_path.rsplit("/", 1)

        size_text = " ".join(map(str, setup_0_size_xyz))
        voxel_text = " ".join(map(str, setup_0_voxel_size_xyz))

        root = ET.Element("SpimData", version="0.2")
        ET.SubElement(root, "BasePath", type="relative").text = "."

        sequence = ET.SubElement(root, "SequenceDescription")

        loader = ET.SubElement(sequence, "ImageLoader", format="bdv.multimg.zarr", version="1.0")

        if s3_bucket is not None:
            ET.SubElement(loader, "s3bucket").text = s3_bucket

        ET.SubElement(loader, "zarr", type="absolute").text = setup_0_path

        zgroups = ET.SubElement(loader, "zgroups")
        zgroup = ET.SubElement(zgroups, "zgroup", setup="0", timepoint="0")
        ET.SubElement(zgroup, "path").text = setup_0_name

        view_setups = ET.SubElement(sequence, "ViewSetups")
        view_setup = ET.SubElement(view_setups, "ViewSetup")

        ET.SubElement(view_setup, "id").text = "0"
        ET.SubElement(view_setup, "size").text = size_text

        voxel_size = ET.SubElement(view_setup, "voxelSize")
        ET.SubElement(voxel_size, "unit").text = "micrometer"
        ET.SubElement(voxel_size, "size").text = voxel_text

        attributes = ET.SubElement(view_setup, "attributes")
        ET.SubElement(attributes, "illumination").text = "0"
        ET.SubElement(attributes, "channel").text = "0"
        ET.SubElement(attributes, "tile").text = "0"
        ET.SubElement(attributes, "angle").text = "0"

        for name, element_name in [
            ("illumination", "Illumination"),
            ("channel", "Channel"),
            ("tile", "Tile"),
            ("angle", "Angle"),
        ]:
            attribute_group = ET.SubElement(view_setups, "Attributes", name=name)
            attribute = ET.SubElement(attribute_group, element_name)
            ET.SubElement(attribute, "id").text = "0"
            ET.SubElement(attribute, "name").text = "0"

        timepoints = ET.SubElement(sequence, "Timepoints", type="range")
        ET.SubElement(timepoints, "first").text = "0"
        ET.SubElement(timepoints, "last").text = "0"
        ET.SubElement(sequence, "MissingViews")

        registrations = ET.SubElement(root, "ViewRegistrations")
        registration = ET.SubElement(registrations, "ViewRegistration", timepoint="0", setup="0")

        transform = ET.SubElement(registration, "ViewTransform", type="affine")
        ET.SubElement(transform, "Name").text = "calibration"

        voxel_x, voxel_y, voxel_z = setup_0_voxel_size_xyz
        ET.SubElement(transform, "affine").text = (
            f"{voxel_x} 0 0 0 "
            f"0 {voxel_y} 0 0 "
            f"0 0 {voxel_z} 0"
        )

        for section in [
            "ViewInterestPoints",
            "BoundingBoxes",
            "PointSpreadFunctions",
            "StitchingResults",
            "IntensityAdjustments",
        ]:
            ET.SubElement(root, section)

        os.makedirs(xml_output_dir, exist_ok=True)
        xml_path = os.path.join(xml_output_dir, xml_filename)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="UTF-8", xml_declaration=True)

        return xml_path

    def create_dataset_xml(self, image_root, xml_filename, xml_output_dir):
        root_path = image_root.rstrip("/")

        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root=root_path, s3=s3, check=False)
        group = zarr.open_group(store, mode="r")

        level_0_image = group["0"]
        setup_0_size_xyz = list(level_0_image.shape[-3:][::-1])

        level_0_metadata = next(
            dataset
            for dataset in group.attrs["multiscales"][0]["datasets"]
            if str(dataset["path"]).strip("/") == "0"
        )

        level_0_scale = next(
            transform
            for transform in level_0_metadata["coordinateTransformations"]
            if transform["type"] == "scale"
        )

        setup_0_voxel_size_xyz = list(level_0_scale["scale"][-3:][::-1])

        return self.build_dataset_xml(
            image_root,
            xml_filename,
            xml_output_dir,
            setup_0_size_xyz,
            setup_0_voxel_size_xyz,
        )
    
    def create_tile_schedule(self, image_root):
        root = zarr.open(fsspec.get_mapper(image_root.rstrip("/")), mode="r")

        level = str(self.min_alignment_level)
        image_shape_zyx = tuple(root[level].shape[-3:])

        level_metadata = next(
            dataset
            for dataset in root.attrs["multiscales"][0]["datasets"]
            if str(dataset["path"]).strip("/") == level
        )

        scale_transform = next(
            transform
            for transform in level_metadata["coordinateTransformations"]
            if transform["type"] == "scale"
        )

        voxel_size_zyx = tuple(scale_transform["scale"][-3:])

        return self.get_tile_schedule(image_shape_zyx, voxel_size_zyx)
    
    def split_moving_dataset(self, input_xml_path, output_xml_path, block_grid_zyx):
        fixed_root = zarr.open(fsspec.get_mapper(self.fixed_image_multiscale_root.rstrip("/")), mode="r")
        moving_root = zarr.open(fsspec.get_mapper(self.moving_image_multiscale_root.rstrip("/")), mode="r")

        fixed_shape_zyx = tuple(fixed_root["0"].shape[-3:])
        moving_shape_zyx = tuple(moving_root["0"].shape[-3:])

        max_shape_zyx = tuple(
            max(fixed_shape_zyx[i], moving_shape_zyx[i])
            for i in range(3)
        )

        tile_size_zyx = []
        overlap_zyx = []

        for size, blocks in zip(max_shape_zyx, block_grid_zyx):
            block_size, overlap = self.get_block_size_and_overlap_1d(size, blocks)
            tile_size_zyx.append(block_size)
            overlap_zyx.append(overlap)

        tile_size_zyx = tuple(tile_size_zyx)
        overlap_zyx = tuple(overlap_zyx)

        print(f"Fixed shape ZYX: {fixed_shape_zyx}")
        print(f"Moving shape ZYX: {moving_shape_zyx}")
        print(f"Split grid ZYX: {block_grid_zyx}")
        print(f"Tile size ZYX: {tile_size_zyx}")
        print(f"Overlap ZYX: {overlap_zyx}")

        split_dataset = SplitDataset(
            xml_file_path=input_xml_path,
            xml_output_file_path=output_xml_path,
            n5_path="",
            point_density=0,
            min_points=0,
            max_points=0,
            error=0.0,
            exclude_radius=0,
            target_image_size=tuple(reversed(tile_size_zyx)),
            target_overlap=tuple(reversed(overlap_zyx)),
        )

        split_dataset.run()

        return output_xml_path, tile_size_zyx, overlap_zyx
    
    def align_interest_points(self, combined_xml_path, fixed_setup_id, loop_output_dir, point_store_path):
        point_store_path = point_store_path.rstrip("/") + "/"
        registered_xml_path = os.path.join(
            loop_output_dir, "ip_registered.xml"
        )

        matching = InterestPointMatching(
            xml_input_path=combined_xml_path,
            n5_output_path=point_store_path,
            input_type="zarr",
            match_type="rigid",
            num_neighbors=3, redundancy=0, significance=3,
            search_radius=300, num_required_neighbors=3,
            ransac_sample_size=3, model_min_inliers=6,
            inlier_threshold=30, min_inlier_ratio=0.1,
            num_iterations=10000, regularization_weight=1.0,
            image_file_prefix=self.moving_image_multiscale_root,
        )
        matching.run()

        metrics_output_path = os.path.join(
            loop_output_dir, "metrics", "metrics.json"
        )
        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

        solver = Solver(
            xml_file_path_output=registered_xml_path,
            n5_input_path=point_store_path,
            xml_file_path=combined_xml_path,
            run_type="rigid",
            relative_threshold=3.5,
            absolute_threshold=7.0,
            max_cleanup_rounds=3,
            min_matches=3,
            damp=1.0,
            regularization_weight=1.0,
            max_iterations=10000,
            max_allowed_error=float("inf"),
            max_plateauwidth=200,
            metrics_output_path=metrics_output_path,
            fixed_tile=f"timepoint: 0, setup: {fixed_setup_id}",
        )
        solver.run()

        return registered_xml_path
    
    def get_zarr_loader(self, root):
        loader = root.find("./SequenceDescription/ImageLoader")

        nested_loader = loader.find("ImageLoader")
        if nested_loader is not None and nested_loader.find("zgroups") is not None:
            return nested_loader

        return loader

    def combine_detected_xmls(self, moving_detected_xml_path, fixed_detected_xml_path, output_xml_path):
        moving_tree = ET.parse(moving_detected_xml_path)
        moving_root = moving_tree.getroot()
        fixed_root = ET.parse(fixed_detected_xml_path).getroot()

        moving_view_setups = moving_root.find("./SequenceDescription/ViewSetups")
        fixed_view_setup = fixed_root.find("./SequenceDescription/ViewSetups/ViewSetup")

        existing_setup_ids = {
            setup.findtext("id")
            for setup in moving_view_setups.findall("ViewSetup")
        }

        if fixed_view_setup.findtext("id") not in existing_setup_ids:
            insert_index = next(
                (i for i, child in enumerate(moving_view_setups) if child.tag == "Attributes"),
                len(moving_view_setups),
            )
            moving_view_setups.insert(insert_index, copy.deepcopy(fixed_view_setup))

        moving_registrations = moving_root.find("./ViewRegistrations")
        existing_registration_ids = {
            (registration.get("timepoint"), registration.get("setup"))
            for registration in moving_registrations.findall("ViewRegistration")
        }

        for registration in fixed_root.findall("./ViewRegistrations/ViewRegistration"):
            key = (registration.get("timepoint"), registration.get("setup"))
            if key not in existing_registration_ids:
                moving_registrations.append(copy.deepcopy(registration))

        moving_interest_points = moving_root.find("./ViewInterestPoints")
        if moving_interest_points is None:
            moving_interest_points = ET.SubElement(moving_root, "ViewInterestPoints")

        existing_interest_point_ids = {
            (entry.get("timepoint"), entry.get("setup"), entry.get("label"))
            for entry in moving_interest_points.findall("ViewInterestPointsFile")
        }

        for entry in fixed_root.findall("./ViewInterestPoints/ViewInterestPointsFile"):
            key = (entry.get("timepoint"), entry.get("setup"), entry.get("label"))
            if key not in existing_interest_point_ids:
                moving_interest_points.append(copy.deepcopy(entry))

        ET.indent(moving_tree, space="  ")
        moving_tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)

        print(f"Combined detected XML: {output_xml_path}")
        return output_xml_path

    def run(self):
        loop_output_dir = os.path.join(self.scratch_directory, "loop_0")
        os.makedirs(loop_output_dir, exist_ok=True)

        fixed_xml_path = self.create_dataset_xml(self.fixed_image_multiscale_root, "fixed.dataset.xml", loop_output_dir)
        moving_xml_path = self.create_dataset_xml(self.moving_image_multiscale_root, "moving.dataset.xml", loop_output_dir)

        moving_tile_schedule = self.create_tile_schedule(self.moving_image_multiscale_root)
        block_grid_zyx = moving_tile_schedule[0]
        moving_split_xml_path = os.path.join(loop_output_dir, "moving.split.dataset.xml")

        
        moving_split_xml_path, tile_size_zyx, overlap_zyx, = self.split_moving_dataset(moving_xml_path, moving_split_xml_path, block_grid_zyx)
        all_tile_bboxes_zyx = self.remove_empty_moving_tiles(moving_split_xml_path, 0.25)

        moving_tile_ids = self.get_split_ids(moving_split_xml_path)
        fixed_setup_id = max(moving_tile_ids) + 1
        self.renumber_fixed_setup(fixed_xml_path, fixed_setup_id)

        fixed_point_store = os.path.join(loop_output_dir, "fixed_interestpoints")
        moving_point_store = os.path.join(loop_output_dir, "moving_interestpoints")
        combined_point_store = os.path.join(loop_output_dir, "interestpoints.n5")

        fixed_detected_xml_path = self.detect_interest_points(fixed_xml_path, self.fixed_image_multiscale_root, os.path.join(loop_output_dir, "fixed.detected.xml"), 
                                                              fixed_point_store, self.fixed_image_min, self.fixed_image_max)

        moving_detected_xml_path = self.detect_interest_points(moving_split_xml_path, self.moving_image_multiscale_root, os.path.join(loop_output_dir, "moving.detected.xml"), 
                                                               moving_point_store, self.moving_image_min, self.moving_image_max)

        self.normalize_loop0_calibration(fixed_detected_xml_path, moving_detected_xml_path)

        combined_xml_path = self.combine_detected_xmls(moving_detected_xml_path, fixed_detected_xml_path, os.path.join(loop_output_dir, "combined.detected.xml"))
        self.combine_interest_point_stores(moving_point_store, fixed_point_store, combined_point_store)
        
        registered_xml_path = self.align_interest_points(combined_xml_path, fixed_setup_id, loop_output_dir, combined_point_store)

        displacement_field_path = self.create_displacement_field(registered_xml_path, fixed_setup_id, block_grid_zyx, tile_size_zyx, 
                                                                 overlap_zyx, all_tile_bboxes_zyx, self.output_transform_path)
        
        print(f"Registered XML: {registered_xml_path}")
        print(f"Displacement field: {displacement_field_path}")
