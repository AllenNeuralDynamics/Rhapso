import numpy as np
import zarr
import fsspec

"""
Initialze output zarr store/group in s3
"""

class InitializeOutputZarr():
    def __init__(self, output_path, zarr_input_prefix, dims):
        self.output_path = output_path
        self.zarr_input_prefix = zarr_input_prefix
        self.output_shape_zyx = (int(dims[2]), int(dims[1]), int(dims[0]))

    def init_output_zarr(self):
        # Create output zarr driver, group, and attrs and open group
        root_store = fsspec.get_mapper(self.output_path.rstrip("/"))

        # Always overwrite anything already there
        zarr.storage.init_group(store=root_store, overwrite=True)
        root = zarr.open_group(store=root_store, mode="a")

        src_store = fsspec.get_mapper(self.zarr_input_prefix.rstrip("/"))
        src_root = zarr.open_group(store=src_store, mode="r")
        root.attrs.update(dict(src_root.attrs))

        # Always recreate multiscale array 0 
        Z, Y, X = self.output_shape_zyx
        root.create_dataset(
            "0",
            shape=(1, 1, Z, Y, X),
            chunks=(1, 1, 128, 256, 256),
            dtype=np.uint16,
            overwrite=True,
            fill_value=0,
            dimension_separator="/",
        )
    
    def run(self):
        self.init_output_zarr()