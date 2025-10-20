"""
Runs fusion from config file generated
from dispim or exaspim scheduler.
"""

import glob
import os
import uuid
import time
from pathlib import Path
import yaml

import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.io as io
import aind_cloud_fusion.script_utils as script_utils


def execute_job(yml_path: str,
                xml_path: str,
                output_path: str):
    """
    yml_path: Local yml path
    xml_path: Local xml path
    output_path: Local results path
    """

    # Parse information from worker yaml
    # (See scheduler.py data contract)
    configs = script_utils.read_config_yaml(yml_path)
    input_path = configs['input_path']
    output_s3_path = configs['output_path']
    dataset_type = configs['dataset_type']
    channel = int(configs['channel'])
    worker_cells = [tuple(cell) for cell in configs['worker_cells']]

    # Initialize Application Objects
    # Application Object: DATASET
    s3_path = input_path
    if dataset_type == 'BigStitcherDataset':
        dataset = io.BigStitcherDataset(xml_path, s3_path, datastore=0)
    elif dataset_type == 'BigStitcherDatasetChannel':
        dataset = io.BigStitcherDatasetChannel(xml_path, s3_path, channel, datastore=0)
    DATASET = dataset

    # Application Object: OUTPUT_PARAMS
    OUTPUT_PARAMS = io.OutputParameters(
        path=output_s3_path,
        chunksize=(1, 1, 128, 128, 128),    # NOTE: Please select your output chunk size
        resolution_zyx=DATASET.tile_resolution_zyx,   # NOTE: Please select your output resolution
        datastore=0
    )

    # Application Object: RUNTIME_PARAMS
    RUNTIME_PARAMS = io.RuntimeParameters(
        option=2,
        pool_size=int(os.environ.get("CO_CPUS", 1)),
        worker_cells=worker_cells
    )

    # Application Parameter: CELL_SIZE
    CELL_SIZE = [128, 128, 128]     # NOTE: Please set this to = output chunk size

    # Application Parameter: POST_REG_TFMS
    POST_REG_TFMS: list[geometry.Affine] = []   # NOTE: Please add optional post-reg transforms

    _, _, _, tile_aabbs, output_volume_size, _ = fusion.initialize_fusion(
        DATASET, POST_REG_TFMS, OUTPUT_PARAMS
    )

    # Application Object: BLENDING_MODULE
    BLENDING_MODULE = blend.WeightedLinearBlending(tile_aabbs)     # NOTE: Please choose your desired blending
    # BLENDING_MODULE = blend.MaxProjection() 

    # Run fusion
    fusion.run_fusion(
            DATASET,
            OUTPUT_PARAMS,
            RUNTIME_PARAMS,
            CELL_SIZE,
            POST_REG_TFMS,
            BLENDING_MODULE,
    )

    # Log 'done' file for next capsule in pipeline.
    # Unique log filename
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)
    unique_file_name = str(Path(output_path) / f"file_{timestamp}_{unique_id}.yml")

    log_content = {}
    log_content['output_path'] = OUTPUT_PARAMS.path
    log_content['resolution_zyx'] = list(OUTPUT_PARAMS.resolution_zyx)

    # with open(unique_file_name, "w") as file:
    #     yaml.dump(log_content, file)


if __name__ == '__main__':
    yml_path = "/Users/seanfite/Desktop/Fusion/worker_0_ch488.yml"
    xml_path = "s3://aind-open-data/HCR_802704_2025-08-30_02-00-00_processed_2025-10-01_21-09-24/image_tile_alignment/bigstitcher.xml"
    output_path = "s3://sean-fusion/HCR_802704/fusion.zarr"

    print(f'{yml_path=}')
    print(f'{xml_path=}')
    print(f'{output_path=}')

    execute_job(yml_path,
                xml_path,
                output_path)