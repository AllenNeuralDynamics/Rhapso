'''
Worker script with hard-coded parameters to generate Neuroglancer link
'''

from Rhapso.fusion.neuroglancer_link_gen.main import generate_neuroglancer_link

# Hard-coded parameters
ZARR_PATH = "s3://martin-test-bucket/output7/multiscale_channel_488.zarr"
VMIN = 90
VMAX = 400
JSON_UPLOAD_BUCKET = "martin-test-bucket"
JSON_UPLOAD_PATH = "NG_out.json"
JSON_LOCAL_OUTPUT = "results"
DATASET_TYPE = "hcr"
OPACITY = 0.5
BLEND = "default"

if __name__ == "__main__":
    # Call the function with hard-coded parameters
    generate_neuroglancer_link(
        zarr_path=ZARR_PATH,
        vmin=VMIN,
        vmax=VMAX,
        json_upload_bucket=JSON_UPLOAD_BUCKET,
        json_upload_path=JSON_UPLOAD_PATH,
        json_local_output=JSON_LOCAL_OUTPUT,
        dataset_type=DATASET_TYPE,
        opacity=OPACITY,
        blend=BLEND
    )
