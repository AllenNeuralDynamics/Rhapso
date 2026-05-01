import numpy as np
import zarr
import fsspec

"""
Compute fusion grid
"""

class ComputeGrid():
    def __init__(self, dims, block_size, block_scale, zarr_input_prefix, output_path):
        self.dims = dims
        self.block_size = block_size
        self.block_scale = block_scale
        self.zarr_input_prefix = zarr_input_prefix 
        self.output_path = output_path

    def grid_create(self):
        dims = np.asarray(self.dims, dtype=np.int64)
        out_bs = np.asarray(self.block_size, dtype=np.int64)
        bpj = np.asarray(self.block_scale, dtype=np.int64)
        gbs = out_bs * bpj  # computeBlockSize

        n = dims.shape[0]
        offset = np.zeros(n, dtype=np.int64)
        grid_blocks = []

        while True:
            # cropBlockDimensions(dimensions, offset, outBlockSize, gridBlockSize, cropped, gridPos)
            cropped = np.minimum(gbs, dims - offset)
            grid_pos = offset // out_bs

            grid_blocks.append([
                offset.copy(),
                cropped.copy(),
                grid_pos.copy(),
            ])

            advanced = False
            for d in range(n):
                offset[d] += gbs[d]
                if offset[d] < dims[d]:
                    advanced = True
                    break
                offset[d] = 0

            if not advanced:
                break

        return grid_blocks
    
    def init_output_zarr(self, output_shape_zyx):
        # Create output zarr driver, group, and attrs and open group
        root_store = fsspec.get_mapper(self.output_path.rstrip("/"))
        zarr.storage.init_group(store=root_store, overwrite=False)
        root = zarr.open_group(store=root_store, mode="a")
        src_store = fsspec.get_mapper(self.zarr_input_prefix.rstrip("/"))
        src_root = zarr.open_group(store=src_store, mode="r")
        root.attrs.update(dict(src_root.attrs))

        # Create multiscale array 0 at root
        if "0" not in root:
            Z, Y, X = output_shape_zyx
            root.create_dataset(
                "0",
                shape=(1, 1, Z, Y, X),
                chunks=(1, 1, 128, 256, 256),
                dtype=np.uint16,
                overwrite=False,
                fill_value=0,
                dimension_separator="/",
            )
    
    def run(self):
        grid = self.grid_create()
        output_shape_zyx = (int(self.dims[2]), int(self.dims[1]), int(self.dims[0]))
        self.init_output_zarr(output_shape_zyx)

        return grid, output_shape_zyx