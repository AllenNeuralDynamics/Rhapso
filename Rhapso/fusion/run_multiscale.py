"""
Runs radial correction in a set of tiles provided
to the data directory
"""

from multiscale.radial_correction import main

def run():
    tilename= "tile_000000_ch_488.zarr/"
    output_path = "s3://rhapso-zar-sample"
    tensorstore_driver = "zarr"
    input_s3_dataset_path = 's3://sean-fusion/exaSPIM_720164_2025-07-07_17-55-45_rhapso/fusion/fused.zarr'
    acquisition_prefix = "s3://aind-open-data/exaSPIM_720164_2025-07-07_17-55-45_processed_2025-07-15_16-22-02"
    acquisition_path = f"{acquisition_prefix}/acquisition.json"

    main(input_s3_dataset_path, output_path, acquisition_path, tilename, tensorstore_driver)

if __name__ == "__main__":
    run()
