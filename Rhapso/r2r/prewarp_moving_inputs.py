import os
import shutil
import fsspec
import zarr
import numpy as np
import base64
import shlex
import subprocess
import textwrap
from scipy.ndimage import map_coordinates

from Rhapso.r2r.registration_and_alignment import REMOTE_PYTHON, CLUSTER_DIR, CLUSTER_YML

class PrewarpMovingInputs:
    def __init__(self, fixed_image_multiscale_root, original_moving_root, original_segmentation_root,
                 displacement_field_paths, output_dir, loop_index, block_shape_zyx, scale_level):
        self.fixed_image_multiscale_root = fixed_image_multiscale_root.rstrip("/")
        self.original_moving_root = original_moving_root.rstrip("/")
        self.original_segmentation_root = original_segmentation_root.rstrip("/")
        self.displacement_field_paths = [path.rstrip("/") for path in displacement_field_paths]
        self.output_dir = output_dir.rstrip("/")
        self.loop_index = loop_index
        self.block_shape_zyx = block_shape_zyx
        self.scale_level = scale_level

    def get_store(self, path, create=False):
        if path.startswith("s3://"):
            return fsspec.get_mapper(path, create=create)

        return path

    def open_root(self, path, mode="r"):
        return zarr.open(
            self.get_store(path, create=mode != "r"),
            mode=mode,
        )

    def get_array(self, root_path):
        root = self.open_root(root_path, mode="r")

        if hasattr(root, "shape"):
            return root

        return root[str(self.scale_level)]

    def get_spatial_shape_zyx(self, array):
        return tuple(int(value) for value in array.shape[-3:])

    def get_field_shape_zyx(self, field):
        if field.shape[-1] != 3:
            raise ValueError(
                f"Expected displacement field ending in vector axis 3, got {field.shape}"
            )

        return tuple(int(value) for value in field.shape[-4:-1])

    def read_spatial_block(self, array, slices_zyx):
        prefix = (0,) * (array.ndim - 3)
        return np.asarray(array[prefix + slices_zyx])

    def read_field_block(self, field, slices_zyx):
        prefix = (0,) * (field.ndim - 4)
        return np.asarray(field[prefix + slices_zyx + (slice(None),)])

    def get_level_spacing_zyx(self, root_path, level="0"):
        root = self.open_root(root_path, mode="r")

        if "multiscales" in root.attrs:
            level_metadata = next(
                dataset
                for dataset in root.attrs["multiscales"][0]["datasets"]
                if str(dataset["path"]).strip("/") == str(level)
            )

            scale = next(
                transform["scale"]
                for transform in level_metadata["coordinateTransformations"]
                if transform["type"] == "scale"
            )

            return tuple(float(value) for value in scale[-3:])

        if "physical_spacing_zyx" in root.attrs:
            return tuple(float(value) for value in root.attrs["physical_spacing_zyx"])

        array = self.get_array(root_path)

        if "physical_spacing_zyx" in array.attrs:
            return tuple(float(value) for value in array.attrs["physical_spacing_zyx"])

        return None

    def detect_zarr_format(self, path):
        fs, filesystem_path = fsspec.core.url_to_fs(path)
        filesystem_path = filesystem_path.rstrip("/")

        if fs.exists(f"{filesystem_path}/zarr.json"):
            return 3

        return 2

    def remove_existing_output(self, path):
        if path.startswith("s3://"):
            fs, filesystem_path = fsspec.core.url_to_fs(path)

            if fs.exists(filesystem_path):
                fs.rm(filesystem_path, recursive=True)

            return

        if os.path.exists(path):
            shutil.rmtree(path)

        os.makedirs(os.path.dirname(path), exist_ok=True)

    def create_output_root(self, output_path, dtype):
        self.remove_existing_output(output_path)

        fixed_root = self.open_root(self.fixed_image_multiscale_root, mode="r")
        output_format = self.detect_zarr_format(self.fixed_image_multiscale_root)
        fixed_multiscale = fixed_root.attrs["multiscales"][0]
        fixed_datasets = {
            str(dataset["path"]).strip("/"): dataset
            for dataset in fixed_multiscale["datasets"]
        }

        output_root = zarr.open_group(
            store=self.get_store(output_path, create=True),
            mode="w",
            zarr_format=output_format,
        )

        datasets = []

        for level in range(self.scale_level + 1):
            level_path = str(level)
            fixed_array = fixed_root[level_path]

            shape = tuple(int(value) for value in fixed_array.shape)
            leading_shape = shape[:-3]
            shape_zyx = shape[-3:]
            chunks_zyx = tuple(min(self.block_shape_zyx[axis], shape_zyx[axis]) for axis in range(3))
            chunks = (1,) * len(leading_shape) + chunks_zyx

            output_root.create_array(
                level_path,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                fill_value=0,
                overwrite=True,
            )

            coordinate_transformations = []

            for transform in fixed_datasets[level_path].get("coordinateTransformations", []):
                copied_transform = dict(transform)

                if "scale" in copied_transform:
                    copied_transform["scale"] = [float(value) for value in copied_transform["scale"]]

                if "translation" in copied_transform:
                    copied_transform["translation"] = [float(value) for value in copied_transform["translation"]]

                coordinate_transformations.append(copied_transform)

            datasets.append(
                {
                    "path": level_path,
                    "coordinateTransformations": coordinate_transformations,
                }
            )

        output_root.attrs["multiscales"] = [
            {
                "version": fixed_multiscale.get("version", "0.4"),
                "name": os.path.basename(output_path.rstrip("/")),
                "axes": [dict(axis) for axis in fixed_multiscale["axes"]],
                "datasets": datasets,
            }
        ]

        return output_root

    def build_output_blocks(self, fixed_shape_zyx):
        blocks = []

        for z0 in range(0, fixed_shape_zyx[0], self.block_shape_zyx[0]):
            z1 = min(z0 + self.block_shape_zyx[0], fixed_shape_zyx[0])

            for y0 in range(0, fixed_shape_zyx[1], self.block_shape_zyx[1]):
                y1 = min(y0 + self.block_shape_zyx[1], fixed_shape_zyx[1])

                for x0 in range(0, fixed_shape_zyx[2], self.block_shape_zyx[2]):
                    x1 = min(x0 + self.block_shape_zyx[2], fixed_shape_zyx[2])

                    blocks.append(
                        (
                            slice(z0, z1),
                            slice(y0, y1),
                            slice(x0, x1),
                        )
                    )

        return blocks

    def create_fixed_coordinates(self, output_slices, fixed_spacing_iso_zyx):
        coordinate_axes = [
            np.arange(output_slice.start, output_slice.stop, dtype=np.float32)
            * fixed_spacing_iso_zyx[axis]
            for axis, output_slice in enumerate(output_slices)
        ]

        z_coordinates, y_coordinates, x_coordinates = np.meshgrid(
            coordinate_axes[0],
            coordinate_axes[1],
            coordinate_axes[2],
            indexing="ij",
        )

        return np.stack(
            (
                z_coordinates,
                y_coordinates,
                x_coordinates,
            ),
            axis=0,
        )

    def open_displacement_fields(self, default_spacing_zyx):
        fields = []

        for field_path in self.displacement_field_paths:
            root = self.open_root(field_path, mode="r")
            field = root["0"] if not hasattr(root, "shape") else root

            spacing = (
                root.attrs.get("physical_spacing_zyx")
                if not hasattr(root, "shape")
                else None
            )

            if spacing is None:
                spacing = field.attrs.get(
                    "physical_spacing_zyx",
                    default_spacing_zyx,
                )

            fields.append(
                {
                    "path": field_path,
                    "array": field,
                    "spacing_zyx": np.asarray(spacing, dtype=np.float32),
                }
            )

        return fields

    def sample_displacement_field(self, field, field_spacing_zyx, coordinates_zyx):
        field_coordinates = (
            coordinates_zyx
            / field_spacing_zyx[:, None, None, None]
        )

        flattened = field_coordinates.reshape(3, -1)
        field_shape_zyx = np.asarray(
            self.get_field_shape_zyx(field),
            dtype=int,
        )

        source_start = np.floor(
            np.min(flattened, axis=1)
        ).astype(int) - 1

        source_stop = np.ceil(
            np.max(flattened, axis=1)
        ).astype(int) + 2

        clipped_start = np.maximum(source_start, 0)
        clipped_stop = np.minimum(source_stop, field_shape_zyx)

        if np.any(clipped_stop <= clipped_start):
            return np.zeros_like(coordinates_zyx, dtype=np.float32)

        source_slices = tuple(
            slice(int(clipped_start[axis]), int(clipped_stop[axis]))
            for axis in range(3)
        )

        field_block = self.read_field_block(
            field,
            source_slices,
        ).astype(np.float32, copy=False)

        local_coordinates = (
            field_coordinates
            - clipped_start[:, None, None, None]
        )

        displacement = np.empty_like(
            coordinates_zyx,
            dtype=np.float32,
        )

        for component in range(3):
            displacement[component] = map_coordinates(
                field_block[..., component],
                local_coordinates,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )

        return displacement

    def apply_displacement_chain(self, coordinates_zyx, displacement_fields):
        transformed_coordinates = coordinates_zyx

        # Each later residual maps fixed space into the previously warped
        # moving image. Apply newest residual first, then walk back through
        # the earlier fields to reach the original moving image.
        for field_data in reversed(displacement_fields):
            displacement = self.sample_displacement_field(
                field=field_data["array"],
                field_spacing_zyx=field_data["spacing_zyx"],
                coordinates_zyx=transformed_coordinates,
            )

            transformed_coordinates = (
                transformed_coordinates + displacement
            )

        return transformed_coordinates

    def get_source_bounds(self, source_coordinates_zyx, source_shape_zyx, interpolation_order):
        flattened = source_coordinates_zyx.reshape(3, -1)
        padding = 2 if interpolation_order == 1 else 1

        source_start = np.floor(
            np.min(flattened, axis=1)
        ).astype(int) - padding

        source_stop = np.ceil(
            np.max(flattened, axis=1)
        ).astype(int) + padding + 1

        clipped_start = np.maximum(source_start, 0)
        clipped_stop = np.minimum(
            source_stop,
            np.asarray(source_shape_zyx, dtype=int),
        )

        return clipped_start, clipped_stop
    
    def coordinate_ranges(self, coordinates):
        return tuple(
            (
                float(np.nanmin(coordinates[axis])),
                float(np.nanmax(coordinates[axis])),
            )
            for axis in range(3)
        )

    def inside_fraction(self, coordinates, shape_zyx):
        inside = np.ones(coordinates.shape[1:], dtype=bool)

        for axis in range(3):
            inside &= coordinates[axis] >= 0
            inside &= coordinates[axis] <= shape_zyx[axis] - 1

        return float(np.mean(inside))
    
    def warp_single_block(self, source, output, output_slices, fixed_spacing_iso_zyx,
                      source_spacing_iso_zyx, displacement_fields, interpolation_order):
        fixed_coordinates = self.create_fixed_coordinates(
            output_slices,
            fixed_spacing_iso_zyx,
        )

        transformed_coordinates = self.apply_displacement_chain(
            fixed_coordinates,
            displacement_fields,
        )

        source_coordinates = (
            transformed_coordinates
            / source_spacing_iso_zyx[:, None, None, None]
        )

        source_shape_zyx = self.get_spatial_shape_zyx(source)

        source_start, source_stop = self.get_source_bounds(
            source_coordinates,
            source_shape_zyx,
            interpolation_order,
        )

        if np.any(source_stop <= source_start):
            output[(0, 0) + output_slices] = np.zeros(
                tuple(output_slice.stop - output_slice.start for output_slice in output_slices),
                dtype=output.dtype,
            )
            return

        source_slices = tuple(
            slice(int(source_start[axis]), int(source_stop[axis]))
            for axis in range(3)
        )

        source_block = self.read_spatial_block(
            source,
            source_slices,
        )

        local_coordinates = (
            source_coordinates
            - source_start[:, None, None, None]
        )

        warped_block = map_coordinates(
            source_block,
            local_coordinates,
            order=interpolation_order,
            mode="constant",
            cval=0,
            prefilter=False,
        )

        if np.issubdtype(output.dtype, np.integer):
            warped_block = np.rint(warped_block)

        output[(0, 0) + output_slices] = warped_block.astype(
            output.dtype,
            copy=False,
        )

    def warp_source(self, output_path, fixed_shape_zyx, fixed_spacing_iso_zyx, source_spacing_iso_zyx, interpolation_order):
        source_path = (
            self.original_moving_root
            if interpolation_order == 1
            else self.original_segmentation_root
        )

        script = textwrap.dedent(f"""
            import ray
            import numpy as np
            from Rhapso.r2r.prewarp_moving_inputs import PrewarpMovingInputs

            ray.init(address="auto")

            prewarp = PrewarpMovingInputs(
                {self.fixed_image_multiscale_root!r}, {self.original_moving_root!r},
                {self.original_segmentation_root!r}, {self.displacement_field_paths!r},
                {self.output_dir!r}, {self.loop_index!r}, {self.block_shape_zyx!r},
                {self.scale_level!r}
            )

            source = prewarp.get_array({source_path!r})
            displacement_fields = prewarp.open_displacement_fields(
                np.asarray({tuple(float(value) for value in fixed_spacing_iso_zyx)!r}, dtype=np.float32)
            )

            output_root = prewarp.create_output_root(
                output_path={output_path!r},
                dtype=source.dtype,
            )

            output = output_root[str(prewarp.scale_level)]
            output_shape_zyx = prewarp.get_spatial_shape_zyx(output)
            fixed_shape_zyx = {tuple(int(value) for value in fixed_shape_zyx)!r}

            if output_shape_zyx != fixed_shape_zyx:
                raise ValueError(
                    f"Prewarp output shape mismatch: expected {{fixed_shape_zyx}}, "
                    f"got {{output_shape_zyx}}"
                )

            output_blocks = prewarp.build_output_blocks(output_shape_zyx)
            fixed_spacing_iso_zyx = np.asarray(
                {tuple(float(value) for value in fixed_spacing_iso_zyx)!r},
                dtype=np.float32,
            )
            source_spacing_iso_zyx = np.asarray(
                {tuple(float(value) for value in source_spacing_iso_zyx)!r},
                dtype=np.float32,
            )
            interpolation_order = {interpolation_order!r}

            @ray.remote
            def process_warp_task(block_index, output_slices):
                try:
                    prewarp.warp_single_block(
                        source=source,
                        output=output,
                        output_slices=output_slices,
                        fixed_spacing_iso_zyx=fixed_spacing_iso_zyx,
                        source_spacing_iso_zyx=source_spacing_iso_zyx,
                        displacement_fields=displacement_fields,
                        interpolation_order=interpolation_order,
                    )

                    return {{
                        "block_index": block_index,
                        "output_slices": output_slices,
                    }}
                except Exception as e:
                    return {{
                        "error": str(e),
                        "block_index": block_index,
                        "output_slices": output_slices,
                    }}

            futures = [
                process_warp_task.remote(block_index, output_slices)
                for block_index, output_slices in enumerate(output_blocks)
            ]

            results = ray.get(futures)
            completed_blocks = [result for result in results if "error" not in result]
            failed_blocks = [result for result in results if "error" in result]

            if failed_blocks:
                raise RuntimeError(
                    f"{{len(failed_blocks)}} prewarp blocks failed: {{failed_blocks[:3]}}"
                )

            print(f"[Prewarp] Wrote {{len(completed_blocks)}}/{{len(output_blocks)}} blocks")
        """).strip()

        encoded = base64.b64encode(script.encode()).decode()
        command = f"printf %s {shlex.quote(encoded)} | base64 -d | {shlex.quote(REMOTE_PYTHON)}"

        print(f"\n=== Loop {self.loop_index} prewarp ===")
        subprocess.run(["ray", "exec", CLUSTER_YML, command], check=True, cwd=CLUSTER_DIR)

        return output_path

    # def warp_source(self, source, output_path, fixed_shape_zyx, fixed_spacing_um_zyx,
    #             fixed_spacing_iso_zyx, source_spacing_iso_zyx,
    #             displacement_fields, interpolation_order):
    #     output_root = self.create_output_root(
    #         output_path=output_path,
    #         dtype=source.dtype,
    #     )

    #     output = output_root[str(self.scale_level)]
    #     output_shape_zyx = self.get_spatial_shape_zyx(output)

    #     if output_shape_zyx != tuple(fixed_shape_zyx):
    #         raise ValueError(
    #             f"Prewarp output shape mismatch: expected {fixed_shape_zyx}, "
    #             f"got {output_shape_zyx}"
    #         )

    #     output_blocks = self.build_output_blocks(output_shape_zyx)

    #     @ray.remote
    #     def process_warp_task(block_index, output_slices):
    #         try:
    #             self.warp_single_block(
    #                 source=source,
    #                 output=output,
    #                 output_slices=output_slices,
    #                 fixed_spacing_iso_zyx=fixed_spacing_iso_zyx,
    #                 source_spacing_iso_zyx=source_spacing_iso_zyx,
    #                 displacement_fields=displacement_fields,
    #                 interpolation_order=interpolation_order,
    #             )

    #             return {
    #                 "block_index": block_index,
    #                 "output_slices": output_slices,
    #             }
    #         except Exception as e:
    #             return {
    #                 "error": str(e),
    #                 "block_index": block_index,
    #                 "output_slices": output_slices,
    #             }


    #     futures = [
    #         process_warp_task.remote(block_index, output_slices)
    #         for block_index, output_slices in enumerate(output_blocks)
    #     ]

    #     results = ray.get(futures)
    #     completed_blocks = [result for result in results if "error" not in result]
    #     failed_blocks = [result for result in results if "error" in result]

    #     if failed_blocks:
    #         raise RuntimeError(
    #             f"{len(failed_blocks)} prewarp blocks failed: {failed_blocks[:3]}"
    #         )

    #     print(f"[Prewarp] Wrote {len(completed_blocks)}/{len(output_blocks)} blocks")

    #     # for block_index, output_slices in enumerate(output_blocks):
    #     #     self.warp_single_block(
    #     #         source=source,
    #     #         output=output,
    #     #         output_slices=output_slices,
    #     #         fixed_spacing_iso_zyx=fixed_spacing_iso_zyx,
    #     #         source_spacing_iso_zyx=source_spacing_iso_zyx,
    #     #         displacement_fields=displacement_fields,
    #     #         interpolation_order=interpolation_order,
    #     #     )

    #     #     print(
    #     #         f"[Prewarp] Block {block_index + 1}/{len(output_blocks)} "
    #     #         f"written: {output_slices}"
    #     #     )

    #     return output_path

    def run(self):
        if not self.displacement_field_paths:
            raise ValueError(
                "Prewarp requires at least one displacement field"
            )

        if not self.output_dir.startswith("s3://"):
            os.makedirs(self.output_dir, exist_ok=True)

        fixed = self.get_array(self.fixed_image_multiscale_root)
        moving = self.get_array(self.original_moving_root)
        segmentation = self.get_array(self.original_segmentation_root)

        fixed_shape_zyx = self.get_spatial_shape_zyx(fixed)

        moving_shape_zyx = np.asarray(
            self.get_spatial_shape_zyx(moving),
            dtype=np.float64,
        )

        segmentation_shape_zyx = np.asarray(
            self.get_spatial_shape_zyx(segmentation),
            dtype=np.float64,
        )

        fixed_level0_spacing_um_zyx = np.asarray(
            self.get_level_spacing_zyx(self.fixed_image_multiscale_root, level="0"),
            dtype=np.float32,
        )

        fixed_spacing_um_zyx = np.asarray(
            self.get_level_spacing_zyx(self.fixed_image_multiscale_root, level=str(self.scale_level)),
            dtype=np.float32,
        )

        moving_spacing_um_zyx = np.asarray(
            self.get_level_spacing_zyx(self.original_moving_root, level=str(self.scale_level)),
            dtype=np.float32,
        )

        segmentation_spacing_um_zyx = self.get_level_spacing_zyx(
            self.original_segmentation_root,
            level=str(self.scale_level),
        )

        fixed_xy_um = float(fixed_level0_spacing_um_zyx[1])

        fixed_spacing_iso_zyx = fixed_spacing_um_zyx / fixed_xy_um
        moving_spacing_iso_zyx = moving_spacing_um_zyx / fixed_xy_um

        if segmentation_spacing_um_zyx is None:
            segmentation_spacing_iso_zyx = (
                moving_spacing_iso_zyx
                * moving_shape_zyx
                / segmentation_shape_zyx
            )
        else:
            segmentation_spacing_iso_zyx = (
                np.asarray(segmentation_spacing_um_zyx, dtype=np.float32)
                / fixed_xy_um
            )

        displacement_fields = self.open_displacement_fields(
            default_spacing_zyx=fixed_spacing_iso_zyx,
        )

        moving_name = os.path.basename(self.original_moving_root.rstrip("/"))

        warped_moving_root = os.path.join(
            self.output_dir,
            moving_name,
        )

        warped_segmentation_root = os.path.join(
            self.output_dir,
            "warped_segmentation.zarr",
        )

        print(
            f"[Prewarp] Loop {self.loop_index}: applying "
            f"{len(displacement_fields)} displacement fields"
        )

        self.warp_source(
            output_path=warped_moving_root,
            fixed_shape_zyx=fixed_shape_zyx,
            fixed_spacing_iso_zyx=fixed_spacing_iso_zyx,
            source_spacing_iso_zyx=moving_spacing_iso_zyx,
            interpolation_order=1,
        )

        self.warp_source(
            output_path=warped_segmentation_root,
            fixed_shape_zyx=fixed_shape_zyx,
            fixed_spacing_iso_zyx=fixed_spacing_iso_zyx,
            source_spacing_iso_zyx=segmentation_spacing_iso_zyx,
            interpolation_order=0,
        )

        print(f"[Prewarp] Moving image: {warped_moving_root}")
        print(f"[Prewarp] Segmentation: {warped_segmentation_root}")

        return warped_moving_root, warped_segmentation_root
    