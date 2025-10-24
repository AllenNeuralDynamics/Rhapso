"""
Runs radial correction in a set of tiles provided
to the data directory
"""

import os
import sys
from pathlib import Path

from aind_z1_radial_correction import radial_correction
from aind_z1_radial_correction.utils import utils


def run():
    """
    Main run file in Code Ocean
    """

    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")

    # Worker scheduler path has radial parameters and acquisition
    worker_scheduler_path = list(Path(data_folder).glob("worker*"))[0]

    acquisition_path = f"{worker_scheduler_path}/acquisition.json"
    data_description_path = f"{worker_scheduler_path}/data_description.json"

    radial_correction_parameters_path = (
        f"{worker_scheduler_path}/radial_correction_parameters.json"
    )

    required_input_elements = [
        acquisition_path,
        radial_correction_parameters_path,
        data_description_path,
    ]

    missing_files = utils.validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    radial_correction_parameters = utils.read_json_as_dict(
        radial_correction_parameters_path
    )

    tilenames = radial_correction_parameters.get("tilenames", [])
    worker_id = radial_correction_parameters.get("worker_id", None)
    bucket_name = radial_correction_parameters.get("bucket_name", None)
    input_s3_dataset_path = radial_correction_parameters.get(
        "input_s3_dataset_path", None
    )
    tensorstore_driver = radial_correction_parameters.get(
        "tensorstore_driver", "zarr3"
    )
    write_to_s3 = radial_correction_parameters.get("write_to_s3", True)

    print(f"Worker ID: {worker_id} processing {len(tilenames)} tiles!")

    write_folder = results_folder
    if bucket_name is not None and write_to_s3:
        data_description = utils.read_json_as_dict(data_description_path)
        dataset_name = data_description.get("name", None)
        if not dataset_name:
            raise ValueError(
                f"Dataset name not found in data_description.json: {data_description_path}"
            )

        write_folder = (
            f"s3://{bucket_name}/{dataset_name}/image_radial_correction"
        )

    if input_s3_dataset_path is not None:
        data_folder = f"s3://{bucket_name}/{input_s3_dataset_path}"

    if len(tilenames):
        radial_correction.main(
            data_folder=data_folder,
            results_folder=write_folder,
            acquisition_path=acquisition_path,
            tilenames=tilenames,
            driver=tensorstore_driver,
        )

        # Write the output path to a file
        with open(
            f"{results_folder}/output_path_worker_{worker_id}.txt", "w"
        ) as f:
            f.write(write_folder)

    else:
        print(f"Nothing to do! Tilenames: {tilenames}")


if __name__ == "__main__":
    run()
