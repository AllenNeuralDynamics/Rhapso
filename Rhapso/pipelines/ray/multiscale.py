from Rhapso.multiscale.array_and_chunk_prep import ArrayAndChunkPrep
from Rhapso.multiscale.ome_metadata import OMEMetadata
from Rhapso.multiscale.block_planner import BlockPlanner
from Rhapso.multiscale.pyramid_executor import PyramidExecutor
from typing import List
import dask.array as da

"""
Multiscle Pipeline
"""

class MultiScale:
    def __init__(self, zarr_path: str, chunk_size: List[int], voxel_size: List[float], n_lvls: int, scale_factor, 
                 target_block_size_mb: int, base_level: int):
        self.zarr_path = zarr_path
        self.chunk_size = chunk_size
        self.voxel_size = voxel_size
        self.n_lvls = n_lvls
        self.scale_factor = scale_factor
        self.target_block_size_mb = target_block_size_mb
        self.base_level = base_level

    def multiscale(self) -> None:
        array = da.from_zarr(f"{self.zarr_path}/{self.base_level}")
        print(f"[MultiScale] Loading base level from {self.zarr_path}/{self.base_level}")

        # Normalize to TCZYX + clamp chunks
        prep = ArrayAndChunkPrep(self.chunk_size, dim=5)
        array, dataset_shape, chunk_size = prep.run(array)
        print(f"[MultiScale] Prepared array shape={dataset_shape}, chunks={chunk_size}")

        # Open root + channel group and write OME metadata
        ome = OMEMetadata(self.zarr_path, list(dataset_shape), self.scale_factor, self.voxel_size, self.n_lvls, list(chunk_size))
        channel_group, stack_name, scale_factors, zarr_version = ome.run()
        print(f"[MultiScale] Using channel group '{stack_name}'")

        # Compute block shape once, using the normalized chunks
        planner = BlockPlanner(self.target_block_size_mb, mode="iso")
        chunked = da.rechunk(array, chunk_size)
        block_shape_zyx = planner.run(chunked) 
        print(f"[MultiScale] Block shape ZYX={block_shape_zyx}")

        # Execute multiscale pyramid
        executor = PyramidExecutor(self.n_lvls, scale_factors, tuple(chunk_size), block_shape_zyx, self.zarr_path, self.base_level, zarr_version)
        executor.run(channel_group)
        print("[MultiScale] Pyramid build complete")

    def run(self) -> None:
        """
        Entry point
        """
        self.multiscale()