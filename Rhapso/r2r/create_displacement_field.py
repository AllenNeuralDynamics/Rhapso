import zarr
import fsspec
import os
import numpy as np
import shutil
import xml.etree.ElementTree as ET
import base64
import ray
import shlex
import subprocess
import textwrap
from Rhapso.r2r.registration_and_alignment import REMOTE_PYTHON, CLUSTER_DIR, CLUSTER_YML

class CreateDisplacementField():
    def __init__(self, fixed_image_multiscale_root, moving_image_multiscale_root, registered_xml_path, fixed_setup_id, block_grid_zyx, 
                 tile_size_zyx, overlap_zyx, all_tile_bboxes_zyx, output_path, scratch_directory, scale_level):
        self.fixed_image_multiscale_root = fixed_image_multiscale_root
        self.moving_image_multiscale_root = moving_image_multiscale_root
        self.registered_xml_path = registered_xml_path 
        self.fixed_setup_id = fixed_setup_id
        self.block_grid_zyx = block_grid_zyx 
        self.tile_size_zyx = tile_size_zyx 
        self.overlap_zyx = overlap_zyx
        self.all_tile_bboxes_zyx = all_tile_bboxes_zyx 
        self.output_path = output_path 
        self.scratch_directory = scratch_directory
        self.scale_level = scale_level
        self.block_shape_zyx = (32, 128, 128)

    def scale_bbox_to_field_grid(self, bbox_zyx, scale_zyx):
        tile_min_zyx, tile_max_zyx = bbox_zyx

        tile_min_zyx = tuple(
            int(np.floor(tile_min_zyx[axis] / scale_zyx[axis]))
            for axis in range(3)
        )
        tile_max_zyx = tuple(
            int(np.ceil(tile_max_zyx[axis] / scale_zyx[axis]))
            for axis in range(3)
        )

        return tile_min_zyx, tile_max_zyx

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

    def get_spacing_zyx(self, image_root):
        root = zarr.open(
            fsspec.get_mapper(image_root.rstrip("/")),
            mode="r",
        )

        level_metadata = next(
            dataset
            for dataset in root.attrs["multiscales"][0]["datasets"]
            if str(dataset["path"]).strip("/") == str(self.scale_level)
        )

        scale_transform = next(
            transform
            for transform in level_metadata["coordinateTransformations"]
            if transform["type"] == "scale"
        )

        return tuple(
            float(value)
            for value in scale_transform["scale"][-3:]
        )

    def get_solver_affines_zyx(self, registered_xml_path, fixed_setup_id):
        if registered_xml_path.startswith("s3://"):
            with fsspec.open(registered_xml_path, "rb") as file:
                root = ET.parse(file).getroot()
        else:
            root = ET.parse(registered_xml_path).getroot()

        tile_affines = {}

        fixed_spacing = self.get_level0_spacing_zyx(self.fixed_image_multiscale_root)
        moving_spacing = self.get_level0_spacing_zyx(self.moving_image_multiscale_root)
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

            # moving -> fixed, XYZ.
            # fixed -> moving, ZYX.
            affine_xyz = np.linalg.inv(affine_xyz)

            axis_order = [2, 1, 0, 3]
            affine_zyx = affine_xyz[np.ix_(axis_order, axis_order)]

            # world translation to fixed isotropic units.
            affine_zyx[:3, 3] *= world_to_fixed_iso

            tile_affines[setup_id] = affine_zyx

        if not tile_affines:
            raise ValueError("No moving solver transforms found")

        return tile_affines

    def get_tile_index_map(self, tile_bboxes_zyx, tile_size_zyx, overlap_zyx):
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


    def fill_missing_tile_affines(self, tile_affines, all_tile_bboxes_zyx):
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


    def get_displacement_weight_block(self, tile_min_zyx, tile_max_zyx, block_min_zyx, block_max_zyx, 
                                      tile_index_zyx, block_grid_zyx, overlap_zyx):
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

    def move_affine_origin(self, affine, block_min_zyx, physical_spacing_zyx):
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
    
    def create_displacement_field(self):
        fixed_root = zarr.open(
            fsspec.get_mapper(self.fixed_image_multiscale_root.rstrip("/")),
            mode="r",
        )

        fixed_level0_spacing_zyx = np.asarray(
            self.get_level0_spacing_zyx(self.fixed_image_multiscale_root),
            dtype=np.float64,
        )
        fixed_level_spacing_zyx = np.asarray(
            self.get_spacing_zyx(self.fixed_image_multiscale_root),
            dtype=np.float64,
        )
        moving_level0_spacing_zyx = np.asarray(
            self.get_level0_spacing_zyx(self.moving_image_multiscale_root),
            dtype=np.float64,
        )

        fixed_shape_zyx = tuple(
            int(value)
            for value in fixed_root[str(self.scale_level)].shape[-3:]
        )

        field_grid_scale_zyx = (
            fixed_level_spacing_zyx
            / moving_level0_spacing_zyx
        )

        physical_spacing_zyx = (
            fixed_level_spacing_zyx
            / fixed_level0_spacing_zyx[1]
        )

        field_overlap_zyx = tuple(
            0 if overlap == 0 else int(
                np.ceil(overlap / field_grid_scale_zyx[axis])
            )
            for axis, overlap in enumerate(self.overlap_zyx)
        )

        tile_affines = self.get_solver_affines_zyx(
            self.registered_xml_path,
            self.fixed_setup_id,
        )
        tile_affines = self.fill_missing_tile_affines(
            tile_affines,
            self.all_tile_bboxes_zyx,
        )
        tile_index_map = self.get_tile_index_map(
            self.all_tile_bboxes_zyx,
            self.tile_size_zyx,
            self.overlap_zyx,
        )

        chunks_zyx = tuple(
            min(self.block_shape_zyx[axis], fixed_shape_zyx[axis])
            for axis in range(3)
        )

        output_path = self.output_path.rstrip("/")

        if output_path.startswith("s3://"):
            output_root = zarr.open_group(
                fsspec.get_mapper(output_path, create=True),
                mode="w",
            )
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if os.path.exists(output_path):
                shutil.rmtree(output_path)

            output_root = zarr.open_group(output_path, mode="w")

        output = output_root.create_dataset(
            "0",
            shape=fixed_shape_zyx + (3,),
            chunks=chunks_zyx + (3,),
            dtype=np.float32,
            fill_value=0,
            overwrite=True,
        )

        metadata = {
            "physical_spacing_zyx": physical_spacing_zyx.tolist(),
            "creation_level": int(self.scale_level),
            "unit_system": "isotropic-scale-0",
            "vector_axis_order": "zyx",
        }
        output_root.attrs.update(metadata)
        output.attrs.update(metadata)

        tile_data = {}

        for tile_id, affine in tile_affines.items():
            tile_min_zyx, tile_max_zyx = self.scale_bbox_to_field_grid(
                self.all_tile_bboxes_zyx[tile_id],
                field_grid_scale_zyx,
            )

            tile_data[tile_id] = {
                "affine": affine,
                "tile_min_zyx": tile_min_zyx,
                "tile_max_zyx": tile_max_zyx,
                "tile_index_zyx": tile_index_map[tile_id],
            }

        tile_data_ref = ray.put(tile_data)

        @ray.remote
        def process_field_block(block_index, block_min_zyx, block_max_zyx,
                                intersecting_tile_ids, tile_data):
            try:
                block_shape_zyx = tuple(
                    block_max_zyx[axis] - block_min_zyx[axis]
                    for axis in range(3)
                )

                displacement_sum = np.zeros(
                    block_shape_zyx + (3,),
                    dtype=np.float32,
                )
                weight_sum = np.zeros(
                    block_shape_zyx,
                    dtype=np.float32,
                )

                for tile_id in intersecting_tile_ids:
                    tile = tile_data[tile_id]
                    tile_min_zyx = tile["tile_min_zyx"]
                    tile_max_zyx = tile["tile_max_zyx"]

                    intersection_min_zyx = tuple(
                        max(block_min_zyx[axis], tile_min_zyx[axis])
                        for axis in range(3)
                    )
                    intersection_max_zyx = tuple(
                        min(block_max_zyx[axis], tile_max_zyx[axis])
                        for axis in range(3)
                    )

                    if any(
                        intersection_max_zyx[axis] <= intersection_min_zyx[axis]
                        for axis in range(3)
                    ):
                        continue

                    intersection_shape_zyx = tuple(
                        intersection_max_zyx[axis] - intersection_min_zyx[axis]
                        for axis in range(3)
                    )

                    weights = self.get_displacement_weight_block(
                        tile_min_zyx=tile_min_zyx,
                        tile_max_zyx=tile_max_zyx,
                        block_min_zyx=intersection_min_zyx,
                        block_max_zyx=intersection_max_zyx,
                        tile_index_zyx=tile["tile_index_zyx"],
                        block_grid_zyx=self.block_grid_zyx,
                        overlap_zyx=field_overlap_zyx,
                    )

                    local_affine = self.move_affine_origin(
                        tile["affine"],
                        intersection_min_zyx,
                        physical_spacing_zyx,
                    )

                    block_displacement = self.matrix_to_displacement_field(
                        affine=local_affine,
                        shape_zyx=intersection_shape_zyx,
                        spacing_zyx=physical_spacing_zyx,
                    )

                    local_slice = tuple(
                        slice(
                            intersection_min_zyx[axis] - block_min_zyx[axis],
                            intersection_max_zyx[axis] - block_min_zyx[axis],
                        )
                        for axis in range(3)
                    )
                    local_vector_slice = local_slice + (slice(None),)

                    displacement_sum[local_vector_slice] += (
                        block_displacement
                        * weights[..., None]
                    )
                    weight_sum[local_slice] += weights

                normalized = np.zeros_like(displacement_sum)

                np.divide(
                    displacement_sum,
                    weight_sum[..., None],
                    out=normalized,
                    where=weight_sum[..., None] > 0,
                )

                if output_path.startswith("s3://"):
                    worker_root = zarr.open_group(
                        fsspec.get_mapper(output_path),
                        mode="r+",
                    )
                else:
                    worker_root = zarr.open_group(output_path, mode="r+")

                spatial_slice = tuple(
                    slice(block_min_zyx[axis], block_max_zyx[axis])
                    for axis in range(3)
                )
                worker_root["0"][spatial_slice + (slice(None),)] = normalized

                return {
                    "block_index": block_index,
                    "uncovered_voxels": int(np.count_nonzero(weight_sum == 0)),
                }
            except Exception as error:
                return {
                    "block_index": block_index,
                    "block_min_zyx": block_min_zyx,
                    "block_max_zyx": block_max_zyx,
                    "error": str(error),
                }

        futures = []
        block_index = 0

        for z0 in range(0, fixed_shape_zyx[0], chunks_zyx[0]):
            z1 = min(z0 + chunks_zyx[0], fixed_shape_zyx[0])

            for y0 in range(0, fixed_shape_zyx[1], chunks_zyx[1]):
                y1 = min(y0 + chunks_zyx[1], fixed_shape_zyx[1])

                for x0 in range(0, fixed_shape_zyx[2], chunks_zyx[2]):
                    x1 = min(x0 + chunks_zyx[2], fixed_shape_zyx[2])

                    block_min_zyx = (z0, y0, x0)
                    block_max_zyx = (z1, y1, x1)

                    intersecting_tile_ids = [
                        tile_id
                        for tile_id, tile in tile_data.items()
                        if all(
                            tile["tile_max_zyx"][axis] > block_min_zyx[axis]
                            and tile["tile_min_zyx"][axis] < block_max_zyx[axis]
                            for axis in range(3)
                        )
                    ]

                    futures.append(
                        process_field_block.remote(
                            block_index,
                            block_min_zyx,
                            block_max_zyx,
                            intersecting_tile_ids,
                            tile_data_ref,
                        )
                    )
                    block_index += 1

        results = ray.get(futures)
        failed_blocks = [
            result
            for result in results
            if "error" in result
        ]

        if failed_blocks:
            raise RuntimeError(
                f"{len(failed_blocks)} displacement blocks failed: "
                f"{failed_blocks[:3]}"
            )

        uncovered_voxels = sum(
            result["uncovered_voxels"]
            for result in results
        )

        if uncovered_voxels:
            print(
                f"Warning: {uncovered_voxels} displacement "
                f"voxels had zero weight"
            )

        print(f"Displacement field saved: {output_path}")
        print(f"Displacement field shape: {fixed_shape_zyx + (3,)}")
        print(f"Displacement blocks written: {len(results)}")

        return output_path

    def run(self):
        script = textwrap.dedent(f"""
            import ray
            from Rhapso.r2r.create_displacement_field import CreateDisplacementField

            ray.init(address="auto")

            field = CreateDisplacementField(
                {self.fixed_image_multiscale_root!r},
                {self.moving_image_multiscale_root!r},
                {self.registered_xml_path!r},
                {self.fixed_setup_id!r},
                {self.block_grid_zyx!r},
                {self.tile_size_zyx!r},
                {self.overlap_zyx!r},
                {self.all_tile_bboxes_zyx!r},
                {self.output_path!r},
                {self.scratch_directory!r},
                {self.scale_level!r},
            )

            field.create_displacement_field()
        """).strip()

        encoded = base64.b64encode(script.encode()).decode()
        command = (
            f"printf %s {shlex.quote(encoded)} | base64 -d | "
            f"{shlex.quote(REMOTE_PYTHON)}"
        )

        print(f"\n=== Create displacement field level {self.scale_level} ===")
        subprocess.run(
            ["ray", "exec", CLUSTER_YML, command],
            check=True,
            cwd=CLUSTER_DIR,
        )

        return self.output_path.rstrip("/")

    # def create_displacement_field(self):
    #     fixed_root = zarr.open(
    #         fsspec.get_mapper(
    #             self.fixed_image_multiscale_root.rstrip("/")
    #         ),
    #         mode="r",
    #     )

    #     fixed_level0_spacing_zyx = np.asarray(
    #         self.get_level0_spacing_zyx(self.fixed_image_multiscale_root),
    #         dtype=np.float64,
    #     )
    #     fixed_level_spacing_zyx = np.asarray(
    #         self.get_spacing_zyx(self.fixed_image_multiscale_root),
    #         dtype=np.float64,
    #     )
    #     moving_level0_spacing_zyx = np.asarray(
    #         self.get_level0_spacing_zyx(self.moving_image_multiscale_root),
    #         dtype=np.float64,
    #     )

    #     fixed_shape_zyx = tuple(
    #         int(value)
    #         for value in fixed_root[str(self.scale_level)].shape[-3:]
    #     )

    #     # Converts moving level-0 tile coordinates into fixed field-grid coordinates.
    #     field_grid_scale_zyx = (
    #         fixed_level_spacing_zyx
    #         / moving_level0_spacing_zyx
    #     )

    #     # Coordinates and displacement vectors remain in fixed level-0
    #     # isotropic pixel units.
    #     physical_spacing_zyx = (
    #         fixed_level_spacing_zyx
    #         / fixed_level0_spacing_zyx[1]
    #     )

    #     field_overlap_zyx = tuple(
    #         0 if overlap == 0 else int(np.ceil(
    #             overlap / field_grid_scale_zyx[axis]
    #         ))
    #         for axis, overlap in enumerate(self.overlap_zyx)
    #     )

    #     tile_affines = self.get_solver_affines_zyx(self.registered_xml_path, self.fixed_setup_id)
    #     tile_affines = self.fill_missing_tile_affines(tile_affines, self.all_tile_bboxes_zyx)
    #     tile_index_map = self.get_tile_index_map(self.all_tile_bboxes_zyx, self.tile_size_zyx, self.overlap_zyx)

    #     chunks_zyx = tuple(
    #         min(self.block_shape_zyx[axis], fixed_shape_zyx[axis])
    #         for axis in range(3)
    #     )

    #     temp_path = os.path.join(self.scratch_directory, "_displacement_accumulators.zarr")

    #     if os.path.exists(temp_path):
    #         shutil.rmtree(temp_path)

    #     temp_root = zarr.open_group(temp_path, mode="w")

    #     displacement_sum = temp_root.create_dataset(
    #         "displacement_sum",
    #         shape=fixed_shape_zyx + (3,),
    #         chunks=chunks_zyx + (3,),
    #         dtype=np.float32,
    #         fill_value=0,
    #         overwrite=True,
    #     )

    #     weight_sum = temp_root.create_dataset(
    #         "weight_sum",
    #         shape=fixed_shape_zyx,
    #         chunks=chunks_zyx,
    #         dtype=np.float32,
    #         fill_value=0,
    #         overwrite=True,
    #     )

    #     try:
    #         for tile_id, affine in tile_affines.items():
    #             tile_min_zyx, tile_max_zyx = self.scale_bbox_to_field_grid(
    #                 self.all_tile_bboxes_zyx[tile_id],
    #                 field_grid_scale_zyx,
    #             )
    #             tile_index_zyx = tile_index_map[tile_id]

    #             clipped_min_zyx = tuple(
    #                 max(0, tile_min_zyx[axis])
    #                 for axis in range(3)
    #             )
    #             clipped_max_zyx = tuple(
    #                 min(fixed_shape_zyx[axis], tile_max_zyx[axis])
    #                 for axis in range(3)
    #             )

    #             if any(
    #                 clipped_max_zyx[axis] <= clipped_min_zyx[axis]
    #                 for axis in range(3)
    #             ):
    #                 continue

    #             for z0 in range(
    #                 clipped_min_zyx[0],
    #                 clipped_max_zyx[0],
    #                 chunks_zyx[0],
    #             ):
    #                 z1 = min(
    #                     z0 + chunks_zyx[0],
    #                     clipped_max_zyx[0],
    #                 )

    #                 for y0 in range(
    #                     clipped_min_zyx[1],
    #                     clipped_max_zyx[1],
    #                     chunks_zyx[1],
    #                 ):
    #                     y1 = min(
    #                         y0 + chunks_zyx[1],
    #                         clipped_max_zyx[1],
    #                     )

    #                     for x0 in range(
    #                         clipped_min_zyx[2],
    #                         clipped_max_zyx[2],
    #                         chunks_zyx[2],
    #                     ):
    #                         x1 = min(
    #                             x0 + chunks_zyx[2],
    #                             clipped_max_zyx[2],
    #                         )

    #                         block_min_zyx = (z0, y0, x0)
    #                         block_max_zyx = (z1, y1, x1)
    #                         block_shape = (
    #                             z1 - z0,
    #                             y1 - y0,
    #                             x1 - x0,
    #                         )

    #                         weights = self.get_displacement_weight_block(
    #                             tile_min_zyx=tile_min_zyx,
    #                             tile_max_zyx=tile_max_zyx,
    #                             block_min_zyx=block_min_zyx,
    #                             block_max_zyx=block_max_zyx,
    #                             tile_index_zyx=tile_index_zyx,
    #                             block_grid_zyx=self.block_grid_zyx,
    #                             overlap_zyx=field_overlap_zyx,
    #                         )

    #                         local_affine = self.move_affine_origin(
    #                             affine,
    #                             block_min_zyx,
    #                             physical_spacing_zyx,
    #                         )

    #                         block_displacement = self.matrix_to_displacement_field(
    #                             affine=local_affine,
    #                             shape_zyx=block_shape,
    #                             spacing_zyx=physical_spacing_zyx,
    #                         )

    #                         spatial_slice = (
    #                             slice(z0, z1),
    #                             slice(y0, y1),
    #                             slice(x0, x1),
    #                         )
    #                         vector_slice = spatial_slice + (
    #                             slice(None),
    #                         )

    #                         accumulated_displacement = np.asarray(
    #                             displacement_sum[vector_slice]
    #                         )
    #                         accumulated_weights = np.asarray(
    #                             weight_sum[spatial_slice]
    #                         )

    #                         accumulated_displacement += (
    #                             block_displacement
    #                             * weights[..., None]
    #                         )
    #                         accumulated_weights += weights  

    #                         displacement_sum[vector_slice] = (
    #                             accumulated_displacement
    #                         )
    #                         weight_sum[spatial_slice] = (
    #                             accumulated_weights
    #                         )

    #         output_path = self.output_path.rstrip("/")

    #         if output_path.startswith("s3://"):
    #             output_store = fsspec.get_mapper(output_path, create=True)
    #             output_root = zarr.open_group(output_store, mode="w")
    #         else:
    #             os.makedirs(os.path.dirname(output_path), exist_ok=True)

    #             if os.path.exists(output_path):
    #                 shutil.rmtree(output_path)

    #             os.makedirs(output_path, exist_ok=True)
    #             output_root = zarr.open_group(output_path, mode="w")
                
    #         output = output_root.create_dataset(
    #             "0",
    #             shape=fixed_shape_zyx + (3,),
    #             chunks=chunks_zyx + (3,),
    #             dtype=np.float32,
    #             fill_value=0,
    #             overwrite=True,
    #         )

    #         uncovered_voxels = 0

    #         for z0 in range(0, fixed_shape_zyx[0], chunks_zyx[0]):
    #             z1 = min(z0 + chunks_zyx[0], fixed_shape_zyx[0])

    #             for y0 in range(0, fixed_shape_zyx[1], chunks_zyx[1]):
    #                 y1 = min(y0 + chunks_zyx[1], fixed_shape_zyx[1])

    #                 for x0 in range(
    #                     0,
    #                     fixed_shape_zyx[2],
    #                     chunks_zyx[2],
    #                 ):
    #                     x1 = min(
    #                         x0 + chunks_zyx[2],
    #                         fixed_shape_zyx[2],
    #                     )

    #                     spatial_slice = (
    #                         slice(z0, z1),
    #                         slice(y0, y1),
    #                         slice(x0, x1),
    #                     )
    #                     vector_slice = spatial_slice + (
    #                         slice(None),
    #                     )

    #                     displacement = np.asarray(
    #                         displacement_sum[vector_slice]
    #                     )
    #                     weights = np.asarray(
    #                         weight_sum[spatial_slice]
    #                     )

    #                     uncovered_voxels += int(
    #                         np.count_nonzero(weights == 0)
    #                     )

    #                     normalized = np.zeros_like(displacement)

    #                     np.divide(
    #                         displacement,
    #                         weights[..., None],
    #                         out=normalized,
    #                         where=weights[..., None] > 0,
    #                     )

    #                     output[vector_slice] = normalized

    #         metadata = {
    #             "physical_spacing_zyx": physical_spacing_zyx.tolist(),
    #             "creation_level": int(self.scale_level),
    #             "unit_system": "isotropic-scale-0",
    #             "vector_axis_order": "zyx",
    #         }
    #         output_root.attrs.update(metadata)
    #         output.attrs.update(metadata)

    #         if uncovered_voxels:
    #             print(
    #                 f"Warning: {uncovered_voxels} displacement "
    #                 f"voxels had zero weight"
    #             )

    #     finally:
    #         shutil.rmtree(temp_path, ignore_errors=True)

    #     print(f"Displacement field saved: {output_path}")
    #     print(f"Displacement field shape: {output.shape}")

    #     return output_path

    # def run(self):
    #     self.create_displacement_field()
