'''
Worker script with hard-coded parameters to generate Neuroglancer link
'''

from Rhapso.util.neuroglancer_link_gen.main import generate_neuroglancer_link

# Hard-coded parameters
ZARR_PATH = "s3://aind-scratch-data/sean.fite/HCR_823476_s5_12-17/channel_488.zarr/multiscale_fused/"

MULTI_ZARR_PATHS = [
    "s3://aind-open-data/HCR_000000-s107-ls1_2026-01-23_00-00-00_processed_2026-01-24_06-00-53/image_tile_fusing/fused/channel_488.zarr/",
    "s3://aind-open-data/HCR_000000-s107-ls1_2026-01-23_00-00-00_processed_2026-01-24_06-00-53/image_tile_fusing/fused/channel_561.zarr/",
    "s3://aind-open-data/HCR_000000-s107-ls1_2026-01-23_00-00-00_processed_2026-01-24_06-00-53/image_tile_fusing/fused/channel_638.zarr/",
]

USE_MULTI = False
VMIN = 90
VMAX = 400
JSON_UPLOAD_BUCKET = "aind-scratch-data"
JSON_UPLOAD_PATH = "sean.fite/HCR_823476_s5_12-17/channel_488.zarr/multiscale_fused/fused_ng.json"
JSON_LOCAL_OUTPUT = "results"
DATASET_TYPE = "hcr"
OPACITY = 0.5
BLEND = "default"

if __name__ == "__main__":

    if USE_MULTI:
        print("✨ Generating multi-channel HCR Neuroglancer link...")
        generate_neuroglancer_link(
            zarr_path=ZARR_PATH,       
            vmin=VMIN,
            vmax=VMAX,
            json_upload_bucket=JSON_UPLOAD_BUCKET,
            json_upload_path=JSON_UPLOAD_PATH,
            json_local_output=JSON_LOCAL_OUTPUT,
            dataset_type=DATASET_TYPE,
            opacity=OPACITY,
            blend=BLEND,
            zarr_paths=MULTI_ZARR_PATHS,
        )
    else:
        print("✨ Generating single-channel Neuroglancer link...")
        generate_neuroglancer_link(
            zarr_path=ZARR_PATH,
            vmin=VMIN,
            vmax=VMAX,
            json_upload_bucket=JSON_UPLOAD_BUCKET,
            json_upload_path=JSON_UPLOAD_PATH,
            json_local_output=JSON_LOCAL_OUTPUT,
            dataset_type=DATASET_TYPE,
            opacity=OPACITY,
            blend=BLEND,
        )
