"""
Runs fusion from config file generated
from dispim or exaspim scheduler.
"""

from .affine_fusion import blend as blend
from .affine_fusion import fusion as fusion
from .affine_fusion import geometry as geometry
from .affine_fusion import io as io
from .affine_fusion import script_utils as script_utils

def execute_job(xml_path: str, image_data_input_path: str, output_s3_path: str, dataset_type: str, channel: int):
    """
    yml_path: Local yml path
    xml_path: Local xml path
    output_path: Local results path
    """

    # Initialize Application Objects
    # Application Object: DATASET
    if dataset_type == 'BigStitcherDataset':
        dataset = io.BigStitcherDataset(xml_path, image_data_input_path, datastore=0)
    elif dataset_type == 'BigStitcherDatasetChannel':
        dataset = io.BigStitcherDatasetChannel(xml_path, image_data_input_path, channel, datastore=0)
    DATASET = dataset

    # Application Object: OUTPUT_PARAMS
    OUTPUT_PARAMS = io.OutputParameters(
        path=output_s3_path,
        chunksize=(1, 1, 640, 256, 256),    # NOTE: Please select your output chunk size
        resolution_zyx=DATASET.tile_resolution_zyx,   # NOTE: Please select your output resolution
        datastore=0
    )

    # Application Parameter: CELL_SIZE
    CELL_SIZE = [640, 256, 256]     # NOTE: Please set this to = output chunk size

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
            CELL_SIZE,
            POST_REG_TFMS,
            BLENDING_MODULE,
    )

if __name__ == '__main__':

    xml_path = "s3://aind-open-data/HCR_802704_2025-08-30_02-00-00_processed_2025-10-01_21-09-24/image_tile_alignment/bigstitcher.xml"
    image_data_input_path = "s3://aind-open-data/HCR_802704_2025-08-30_02-00-00_processed_2025-10-01_21-09-24/image_radial_correction/"
    output_s3_path = "s3://sean-fusion/output5/channel_488.zarr"
    dataset_type = "BigStitcherDataset"
    channel = None                      # list channel num (int) if fusing a specific channel from an xml of multiple channels

    execute_job(xml_path, image_data_input_path, output_s3_path, dataset_type, channel)