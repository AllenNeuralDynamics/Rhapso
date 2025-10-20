from .multiscale.blocked_array_writer import BlockedArrayWriter
from .multiscale.chunk_utils import ensure_shape_5d, ensure_array_5d
from .multiscale.ome_zarr_ import store_array, downsample_and_store, _get_bytes, write_ome_ngff_metadata

from pathlib import Path 
import logging
import time
import uuid
import yaml
import dask
import s3fs
import zarr

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def run_multiscale(full_res_arr: dask.array, 
                   out_group: zarr.group,
                   voxel_sizes_zyx: tuple,
                   input_path: str
                   ): 
    
    arr = full_res_arr.rechunk((1, 1, 128, 128, 128))
    arr = ensure_array_5d(arr)
    LOGGER.info(f"input array: {arr}")
    LOGGER.info(f"input array size: {arr.nbytes / 2 ** 20} MiB")
    block_shape = ensure_shape_5d(BlockedArrayWriter.get_block_shape(arr))
    LOGGER.info(f"block shape: {block_shape}")
    
    scale_factors = (2, 2, 2) 
    scale_factors = ensure_shape_5d(scale_factors)
    # n_levels = 5
    n_levels = 8   # Need 8 levels for exaspim 
    compressor = None

    write_ome_ngff_metadata(
            out_group,
            arr,
            out_group.path,
            n_levels,
            scale_factors[2:],
            voxel_sizes_zyx,
            origin=None,
        )

    t0 = time.time()
    
    store_array(arr, out_group, "0", block_shape, input_path, compressor)
    # pyramid = downsample_and_store(
    #     arr, out_group, n_levels, scale_factors, block_shape, compressor
    # )
    # write_time = time.time() - t0

    # LOGGER.info(
    #     f"Finished writing tile.\n"
    #     f"Took {write_time}s. {_get_bytes(pyramid) / write_time / (1024 ** 2)} MiB/s"
    # )

def read_config_yaml(yaml_path: str) -> dict:
    if yaml_path.startswith('s3://'):
        print(f"{yaml_path} yaml path")
        fs = s3fs.S3FileSystem(anon=False)  
        with fs.open(yaml_path, "r") as f:
            yaml_dict = yaml.safe_load(f)
    else:
        with open(yaml_path, "r") as f:
            yaml_dict = yaml.safe_load(f)
    
    return yaml_dict

def write_config_yaml(yaml_path: str, yaml_data: dict) -> None:
    with open(yaml_path, "w") as file:
        yaml.dump(yaml_data, file)

def main(): 
    yml_path = 's3://multiscale-sean/multiscale_input.yml'
    params = read_config_yaml(yml_path)
    in_path = params['in_path']
    output_path = params['output_path']
    voxel_sizes_zyx = tuple(params['resolution_zyx'])

    # Run multiscaling
    arr = zarr.open(in_path + "/0", mode='r')
    arr = dask.array.from_zarr(arr)

    s3 = s3fs.S3FileSystem(
        config_kwargs={
            'max_pool_connections': 64,
            'retries': {
                'mode': 'standard',
            }
        }
    )
    store = s3fs.S3Map(root=output_path, s3=s3)
    out_group = zarr.group(store=store, overwrite=True)

    # out_group = zarr.open_group(output_path, mode='w')
    run_multiscale(arr, out_group, voxel_sizes_zyx, in_path)    

    # Dump 'Done' file w/ Unique ID
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)
    unique_file_name = str(Path('/results') / f"file_{timestamp}_{unique_id}.yml")

    output_data = {}
    output_data['output_path'] = output_path
    output_data['resolution_zyx'] = params['resolution_zyx']
    write_config_yaml(yaml_path=unique_file_name, 
                      yaml_data=output_data)


if __name__ == '__main__':
    main()