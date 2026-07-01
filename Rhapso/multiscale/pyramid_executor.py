from typing import Any, Sequence, Tuple
from numcodecs.blosc import Blosc
import math
import dask.array as da
import numpy as np
import numpy.typing as npt
import ray

"""
Handles windowed downsampling and block-wise execution using Ray.
"""

class PyramidExecutor:
    def __init__(self, n_lvls: int, scale_factors, chunk_size: Tuple[int, ...], block_shape_zyx: Tuple[int, int, int], 
                 zarr_path: str, base_level: int, zarr_version) -> None:
        self.n_lvls = n_lvls
        self.scale_factors = scale_factors
        self.chunk_size = chunk_size
        self.block_shape_zyx = block_shape_zyx
        self.zarr_path = zarr_path
        self.base_level = base_level
        self.output_zarr_version = zarr_version

    @staticmethod
    def reshape_windowed(array: npt.NDArray[Any], window_size: Sequence[int]) -> npt.NDArray[Any]:
        if len(window_size) != array.ndim:
            raise ValueError(
                f"Length of window_size must match array dimensionality. "
                f"Got {len(window_size)}, expected {array.ndim}"
            )
        new_shape: tuple[int, ...] = ()
        for s, f in zip(array.shape, window_size):
            new_shape += (s // f, f)
        return array.reshape(new_shape)

    @staticmethod
    def windowed_mean(array: npt.NDArray[Any], window_size: Sequence[int], **kwargs: Any) -> npt.NDArray[Any]:
        reshaped = PyramidExecutor.reshape_windowed(array, window_size)
        result: npt.NDArray[Any] = reshaped.mean(
            axis=tuple(range(1, reshaped.ndim, 2)), **kwargs
        )
        return result
    
    def store(self, channel_group, src_level: int, dst_level: int, src_shape: Tuple[int, ...], dst_shape: Tuple[int, ...],
           block_shape_zyx: Tuple[int, int, int], scale_factors_zyx: Tuple[int, int, int]) -> None:
        """
        Use Ray to process all blocks in parallel, with bounded in-flight writes.
        """
        t_dim, c_dim, z_dim, y_dim, x_dim = 0, 1, 2, 3, 4
        bz, by, bx = block_shape_zyx
        sz, sy, sx = scale_factors_zyx

        # Gate against concurrent partial writes into the same Zarr chunk.
        # This is important because each Ray task writes directly to dst_arr[dst_slices].
        dst_arr_for_gate = channel_group[str(dst_level)]
        dst_chunks = dst_arr_for_gate.chunks

        ct, cc, cz, cy, cx = dst_chunks

        if bz % cz != 0 or by % cy != 0 or bx % cx != 0:
            raise ValueError(
                "[PyramidExecutor] Unsafe multiscale write grid: block_shape_zyx must be "
                "a clean multiple of destination Zarr chunks to avoid parallel partial-chunk write races. "
                f"block_shape_zyx={block_shape_zyx}, dst_chunks_zyx={(cz, cy, cx)}"
            )
        # -------------------------------------------------------------------------------

        t_size = dst_shape[t_dim]
        c_size = dst_shape[c_dim]
        z_size = dst_shape[z_dim]
        y_size = dst_shape[y_dim]
        x_size = dst_shape[x_dim]

        t_slice = slice(0, t_size)
        c_slice = slice(0, c_size)

        n_z_blocks = math.ceil(z_size / bz)
        n_y_blocks = math.ceil(y_size / by)
        n_x_blocks = math.ceil(x_size / bx)
        total_blocks = n_z_blocks * n_y_blocks * n_x_blocks

        if total_blocks == 0:
            print("[PyramidExecutor] store: no blocks to process.")
            return

        if not ray.is_initialized():
            print("[PyramidExecutor] store: initializing Ray for parallel processing...")
            ray.init()

        # Per-level concurrency cap.
        # Higher levels have fewer tasks, but each task can still write many chunks,
        # so we intentionally scale down to avoid writing the whole level at once.
        if dst_level == 1:
            max_in_flight = 450
        elif dst_level == 2:
            max_in_flight = 150
        elif dst_level == 3:
            max_in_flight = 100
        elif dst_level == 4:
            max_in_flight = 50
        else:
            max_in_flight = 5

        print(
            f"[PyramidExecutor] store: dispatching {total_blocks} Ray tasks "
            f"(src={src_level}, dst={dst_level}, max_in_flight={max_in_flight})..."
        )

        block_instructions = []

        z0 = 0
        while z0 < z_size:
            z1 = min(z0 + bz, z_size)
            y0 = 0
            while y0 < y_size:
                y1 = min(y0 + by, y_size)
                x0 = 0
                while x0 < x_size:
                    x1 = min(x0 + bx, x_size)

                    dst_slices = (
                        t_slice,
                        c_slice,
                        slice(z0, z1),
                        slice(y0, y1),
                        slice(x0, x1),
                    )

                    src_z0, src_z1 = z0 * sz, z1 * sz
                    src_y0, src_y1 = y0 * sy, y1 * sy
                    src_x0, src_x1 = x0 * sx, x1 * sx

                    src_slices = (
                        slice(0, src_shape[t_dim]),
                        slice(0, src_shape[c_dim]),
                        slice(src_z0, src_z1),
                        slice(src_y0, src_y1),
                        slice(src_x0, src_x1),
                    )

                    block_instructions.append((src_slices, dst_slices))

                    x0 = x1
                y0 = y1
            z0 = z1

        completed = 0
        submitted = 0
        last_printed_pct = -1
        futures = []

        # Submit only the first bounded batch.
        while submitted < total_blocks and len(futures) < max_in_flight:
            src_slices, dst_slices = block_instructions[submitted]
            futures.append(
                process_block_instruction_remote.remote(
                    src_level,
                    dst_level,
                    src_slices,
                    dst_slices,
                    sz,
                    sy,
                    sx,
                    channel_group,
                )
            )
            submitted += 1

        while futures:
            done, futures = ray.wait(futures, num_returns=1, timeout=1.0)

            if done:
                try:
                    ray.get(done)
                except Exception:
                    print("[PyramidExecutor] ERROR: Ray multiscale block failed.")
                    print(f"[PyramidExecutor] src={src_level} dst={dst_level}")
                    print(f"[PyramidExecutor] submitted={submitted}/{total_blocks}")
                    print(f"[PyramidExecutor] completed={completed}/{total_blocks}")
                    print(f"[PyramidExecutor] remaining_in_flight={len(futures)}")
                    raise

                completed += len(done)

                # Refill the queue, but never exceed max_in_flight.
                while submitted < total_blocks and len(futures) < max_in_flight:
                    src_slices, dst_slices = block_instructions[submitted]
                    futures.append(
                        process_block_instruction_remote.remote(
                            src_level,
                            dst_level,
                            src_slices,
                            dst_slices,
                            sz,
                            sy,
                            sx,
                            channel_group,
                        )
                    )
                    submitted += 1

                progress_pct_int = int((completed / total_blocks) * 100)

                if progress_pct_int > last_printed_pct:
                    last_printed_pct = progress_pct_int
                    print(
                        f"[PyramidExecutor]   Progress: {progress_pct_int}% "
                        f"({completed}/{total_blocks} blocks)"
                    )
    
    def create_level_array(self, channel_group, name: str, shape, chunks, dtype):
        """
        Create one pyramid level using the known output zarr version.
        """
        if self.output_zarr_version == 2:
            compressor = Blosc(cname="zstd", clevel=3, shuffle=2, blocksize=0)

            return channel_group.create_array(
                name=name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=compressor,
                overwrite=True,
                chunk_key_encoding={
                    "name": "v2",
                    "separator": "/",
                },
            )

        return channel_group.create_array(
            name=name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            overwrite=True,
            chunk_key_encoding={
                "name": "default",
                "separator": "/",
            },
        )

    def build_pyramid(self, channel_group) -> None:
        """
        Multiscale loop
        """
        # compressor = Blosc(cname="zstd", clevel=3, shuffle=2, blocksize=0)
        start_level = self.base_level + 1
        levels_to_write = self.n_lvls - start_level

        try:
            _ = channel_group[str(self.base_level)]
            print(f"[PyramidExecutor] Using existing base level {self.base_level} from {self.zarr_path}")
        except KeyError:
            print(f"[PyramidExecutor] ERROR: Level {self.base_level} not found at {self.zarr_path}/{self.base_level}.")

        for level in range(start_level, self.n_lvls):
            prev_ds = channel_group[str(level - 1)]
            previous_scale = da.from_zarr(prev_ds, chunks=prev_ds.chunks)

            # scale_factors index relative to first level above base_level
            scale_index = level - start_level
            per_level_zyx = self.scale_factors[scale_index]
            sz, sy, sx = map(int, per_level_zyx)
            new_scale_factor = ([1] * (previous_scale.ndim - 3)) + [sz, sy, sx]

            print(f"[PyramidExecutor] Level {level}/{self.n_lvls-1}")
            print(f"Computing from level {level-1} with scale factor {new_scale_factor}...")

            t_size, c_size, z_src, y_src, x_src = previous_scale.shape
            z_dst = z_src // sz
            y_dst = y_src // sy
            x_dst = x_src // sx
            dst_shape = (t_size, c_size, z_dst, y_dst, x_dst)
            dst_name = str(level)

            # Ensure destination dataset exists and has the expected shape
            if dst_name in channel_group:
                dst_arr = channel_group[dst_name]
                if dst_arr.shape != dst_shape:
                    raise ValueError(f"Existing level {dst_name} has shape {dst_arr.shape} expected {dst_shape}")
            else:
                self.create_level_array(
                    channel_group=channel_group,
                    name=dst_name,
                    shape=dst_shape,
                    chunks=self.chunk_size,
                    dtype=prev_ds.dtype,
                )

            src_shape = tuple(previous_scale.shape)
            scale_factors_zyx = (sz, sy, sx)

            print(f"[PyramidExecutor] Level {level}/{self.n_lvls-1}: Writing to storage...")

            self.store(channel_group=channel_group, src_level=level - 1, dst_level=level, src_shape=src_shape, dst_shape=dst_shape,
                       block_shape_zyx=self.block_shape_zyx, scale_factors_zyx=scale_factors_zyx)

            print(f"[PyramidExecutor] Level {level}/{self.n_lvls-1}: ✓ Complete ({level-start_level+1}/{levels_to_write} levels done)")

    def run(self, channel_group) -> None:
        """
        Entry point for pyramid execution.
        """
        self.build_pyramid(channel_group)

@ray.remote
def process_block_instruction_remote(src_level: int, dst_level: int, src_slices: Tuple[slice, ...], dst_slices: Tuple[slice, ...],
                                     sz: int, sy: int, sx: int, channel_group):
    src_arr = channel_group[str(src_level)]
    dst_arr = channel_group[str(dst_level)]
    src_block = np.asarray(src_arr[src_slices])
    window_size = (1, 1, sz, sy, sx)
    dst_block = PyramidExecutor.windowed_mean(src_block, window_size=window_size)
    dst_arr[dst_slices] = dst_block