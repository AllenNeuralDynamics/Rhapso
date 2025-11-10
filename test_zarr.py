import dask.array as da

# PATH = "s3://sean-fusion/output7/multiscale_channel_488.zarr/5"
PATH = "s3://sean-fusion/exaSPIM_output36/channel_488.zarr/0"


a = da.from_zarr(PATH, storage_options={"anon": False})  # set to {"anon": True} if public
print(f"path:   {PATH}")
print(f"shape:  {a.shape}")
print(f"dtype:  {a.dtype}")
print(f"chunks: {a.chunks}")
print(f"nbytes: {a.nbytes} ({a.nbytes/1024**3:.2f} GB)")