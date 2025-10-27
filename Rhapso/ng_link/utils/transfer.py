"""Transfer file to S3 bucket"""
import subprocess
from pathlib import Path


def copy_to_s3(file_loc: str, bucket: str = None):
    """Copy fileout to s3 bucket, generally from a
    /scratch location to an S3 URI

    Requires AWS CLI to be installed and configured with
    credentials.

    Parameters:
    ------------
    file_loc: str
    The location of the process_output.json to be copied.

    bucket: str
    The S3 Bucket URI that the masks will be copied to.
    """

    print("Copying to s3 bucket")
    file_loc = Path(file_loc)
    assert file_loc.exists(), f"Fileout {file_loc} does not exist."
    file_loc = str(file_loc)
    if bucket is None:
        print(
            f"No bucket specified, segmentation masks at \
                {file_loc} not transfered"
        )
        return
    else:
        bucket = str(bucket)

        cmd = f"aws s3 cp {file_loc} {bucket}"
        try:
            subprocess.run(cmd, shell=True)
            print("*" * 70)
            print("Finished Copy segmentation masks to S3!")
            print("*" * 70)
        except Exception as e:
            print("Error copying to s3 bucket: ", e)
            raise e
