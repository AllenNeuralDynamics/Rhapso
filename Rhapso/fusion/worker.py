"""
Runs fusion from config file generated
from dispim or exaspim scheduler.
"""

import os

from .affine_fusion import blend as blend
from .affine_fusion import fusion as fusion
from .affine_fusion import geometry as geometry
from .affine_fusion import io as io
from .affine_fusion import script_utils as script_utils

def execute_job(yml_path: str, xml_path: str):
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

    _, _, _, _, tile_aabbs, _, _ = fusion.initialize_fusion(DATASET, POST_REG_TFMS, OUTPUT_PARAMS)

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

if __name__ == '__main__':
    yml_path = "s3://martin-test-bucket/HCR_802704/yaml_configs/worker_0_ch488.yml"
    xml_path = "s3://aind-open-data/HCR_802704_2025-08-30_02-00-00_processed_2025-10-01_21-09-24/image_tile_alignment/bigstitcher.xml"

    print(f'{yml_path=}')
    print(f'{xml_path=}')

    execute_job(yml_path, xml_path)