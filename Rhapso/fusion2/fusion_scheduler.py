"""
Example fusion scheduler.
Reads dataset in S3 following Neural Dynamics dataset convention
and generates yaml files containing configurations
for individual worker runtimes.
An example yaml schema names 'data contract' provided below.
"""

from collections import OrderedDict
from datetime import datetime
import glob
import os
import xmltodict

import numpy as np
from pathlib import Path
import fsspec

import aind_cloud_fusion.io as io
import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.script_utils as script_utils

"""
DATA CONTRACT:
Exaspim scheduler outputs a configuration file
with the following minimal fields.

dataset_type: {BigStitcherDataset, BigStitcherDatasetChannel}
channel: channel number
input_path: 's3://<YOUR INPUT PATH>'
output_path: "s3://<YOUR OUTPUT PATH>"
worker_cells: [list of cell 3-ples]
"""

def create_starter_ymls(xml_path: str,
                        output_path: str,
                        num_workers: int = 64):
    """
    xml_path: Local xml path
    output_path: Local results path
    """

    # Construct I/O cloud paths from local paths in xml
    # This code converts:
    # This: /root/capsule/data/exaSPIM_674191_2023-09-12_12-37-37/exaSPIM.zarr
    # To This (input path): s3://aind-open-data/exaSPIM_674191_2023-09-12_12-37-37/exaSPIM.zarr/
    
    if xml_path.startswith("s3://"):
        with fsspec.open(xml_path, mode="rt") as f:
            data: OrderedDict = xmltodict.parse(f.read())
    else:
        with open(xml_path, "r") as file:
            data: OrderedDict = xmltodict.parse(file.read())
    
    parsed_path = data["SpimData"]["SequenceDescription"]["ImageLoader"][
        "zarr"
    ]["#text"]
    parts = parsed_path.split('/')
    if 'root' in parts:
        parts.remove('root')
    if 'capsule' in parts:
        parts.remove('capsule')
    
    # input_s3_path = 's3://aind-open-data/' + parts[2] + '/' + parts[3] + '/'
    input_s3_path = parsed_path
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #output_s3_path_base = 's3://aind-open-data/' + parts[2] + f'_full_res_{datetime_str}/'
    # output_s3_path_base = 's3://aind-open-data/' + parts[2] + f'/fused_full_res/' 
    output_s3_path_base = "s3://sean-fusion/output/"   

    # Init application objects to init fusion scheduling.
    # Application Object: DATASET
    # xml_path = str(Path(xml_path))
    s3_path = input_s3_path
    DATASET = io.BigStitcherDataset(xml_path, s3_path, datastore=0)

    # Application Object: OUTPUT_PARAMS
    OUTPUT_PARAMS = io.OutputParameters(
        path=output_path,
        chunksize=(1, 1, 128, 128, 128),
        resolution_zyx=DATASET.tile_resolution_zyx,
        datastore=0
    )

    # Application Object: RUNTIME PARAMS
    # Not required here

    # Application Parameter: CELL_SIZE
    CELL_SIZE = [128, 128, 128]

    # Application Parameter: POST_REG_TFMS
    # Exaspim and Dispim do not require deskewing
    POST_REG_TFMS: list[geometry.Affine] = []

    # Application Object: BLENDING_MODULE
    # Not required here

    # Initalize fusion
    _, _, _, _, output_volume_size, _ = fusion.initialize_fusion(
        DATASET, POST_REG_TFMS, OUTPUT_PARAMS
    )

    # Divide work and export into yamls
    z_cnt, y_cnt, x_cnt = fusion.get_cell_count_zyx(
        output_volume_size, CELL_SIZE
    )
    zs = np.arange(0, z_cnt, 1)
    ys = np.arange(0, y_cnt, 1)
    xs = np.arange(0, x_cnt, 1)
    x, y, z = np.meshgrid(xs, ys, zs)
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    cell_coords = np.column_stack((z_flat, y_flat, x_flat))

    # Ex math:
    # 50 cells
    # 6 workers
    # ceil(50 / 6) -> 9
    channels = script_utils.get_unique_channels_for_dataset(input_s3_path)
    total_cells = len(cell_coords) * len(channels)
    n = int(np.ceil(total_cells / num_workers))
    print(f'Each worker assigned {n} cells')

    #divide the workers up by channel
    for worker_id in range(int(num_workers/len(channels))):
        for i, ch in enumerate(channels):
            print(f'Generating Worker {worker_id} Yaml')
            start = worker_id * n
            end = (worker_id + 1) * n
            worker_cells = cell_coords[start:end, :].tolist()

            if i == 0:
                dataset_type = 'BigStitcherDataset'
            else:
                dataset_type = 'BigStitcherDatasetChannel'

            ch_X_configs = {}
            ch_X_configs['dataset_type'] = dataset_type
            ch_X_configs['channel'] = ch
            ch_X_configs['input_path'] = input_s3_path
            ch_X_configs['output_path'] = output_s3_path_base + f'channel_{ch}.zarr'
            ch_X_configs['worker_cells'] = worker_cells
            script_utils.write_config_yaml(str(output_path +'/'+ f'worker_{worker_id}_ch{ch}.yml'), ch_X_configs)


if __name__ == '__main__':
    xml_path = "s3://aind-open-data/HCR_802704_2025-08-30_02-00-00_processed_2025-10-01_21-09-24/image_tile_alignment/bigstitcher.xml"
    output_path = "s3://sean-fusion/HCR_802704/channel_488.zarr"

    print(f'{xml_path=}')
    print(f'{output_path=}')

    create_starter_ymls(xml_path,
                        output_path,
                        num_workers = 3)