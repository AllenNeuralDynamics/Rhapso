"""
How to run this file:
1. install rhapso locally:
    - In the root of this github repo, run 'pip3 install -e .'
2. Install dependencies:
    - In the root of this github repo, run 'pip3 install -r requirements.txt'
3. Run this file:
    - Run this file: 'python3 Rhapso/pipelines/fusion/fusion_test_pipeline.py' 
"""
from Rhapso.fusion.fusion import runFusion

if __name__ == '__main__':

    # Path to the input directory containing unaligned TIFF and XML files for fusion
    INPUT_PATH = "/Users/seanfite/Desktop/TIFF"

    # Path to the output Zarr file where the fused dataset will be stored
    OUTPUT_PATH = "/Users/seanfite/Desktop/TIFF/Output/fused_output.zarr"

    # Path to the output Zarr file where the multiscale representation of the fused dataset will be stored
    MULTISCALE_OUTPUT_PATH = "/Users/seanfite/Desktop/TIFF/Output/fusedMultiscaleOutput.zarr"

    # Path to the XML file describing the dataset structure and metadata
    XML_PATH = "/Users/seanfite/Desktop/TIFF/dataset.xml"

    # Index of the channel to process in the dataset
    CHANNEL = 0

    # Size of each cell in the 3D grid used for fusion
    CELL_SIZE = (128, 128, 128)

    # Number of levels in the multiscale pyramid
    N_LEVELS = 8

    # Scale factors for downsampling in the multiscale pyramid (z, y, x)
    SCALE_FACTORS = (2, 2, 2)

    # Run fusion
    runFusion(
        INPUT_PATH,
        OUTPUT_PATH,
        MULTISCALE_OUTPUT_PATH,
        XML_PATH,
        CHANNEL,
        CELL_SIZE,
        N_LEVELS,
        SCALE_FACTORS
    )