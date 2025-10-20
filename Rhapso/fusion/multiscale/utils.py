"""
Utility functions for the radial correction step
"""

import json
import boto3
from pathlib import Path
from typing import List
from urllib.parse import urlparse
from aind_data_schema.core.processing import (DataProcess, Processing)
from aind_data_schema.components.identifiers import Code
from packaging import version

def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: str,
    prefix: str,
    processor_full_name: str,   # kept for compatibility; not used directly in v2
    pipeline_version: str,
):
    """
    Generates processing.json for the output folder (AIND schema v2).
    """

    # Option A: one-liner helper builds a simple dependency graph
    p = Processing.create_with_sequential_process_graph(
        pipelines=[
            Code(
                name="Radial correction pipeline",
                url="",                 # fill if you have a repo URL
                version=pipeline_version,
            )
        ],
        data_processes=data_processes,
        notes=(
            "Metadata for radial correction; to be merged with other steps later."
        ),
    )

    # Write processing.json alongside your outputs
    p.write_standard_file(output_directory=dest_processing, prefix=prefix)

def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    if filepath.startswith("s3://"):
        u = urlparse(filepath)
        bucket = u.netloc
        key = u.path.lstrip("/")
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
    
    return json.loads(obj["Body"].read())

    # dictionary = {}

    # if os.path.exists(filepath):
    #     with open(filepath) as json_file:
    #         dictionary = json.load(json_file)

    # return dictionary


def get_voxel_resolution(acquisition_path: Path) -> List[float]:
    """
    Get the voxel resolution from an acquisition.json file.

    Parameters
    ----------
    acquisition_path: Path
        Path to the acquisition.json file.
    Returns
    -------
    List[float]
        Voxel resolution in the format [z, y, x].
    """

    acquisition_config = read_json_as_dict(acquisition_path)

    schema_version = acquisition_config.get("schema_version")
    print(f"Schema version: {schema_version}")

    if version.parse(schema_version) >= version.parse("2.0.0"):
        return _get_voxel_resolution_v2(acquisition_config)
    else:
        return _get_voxel_resolution_v1(acquisition_config)


def _get_voxel_resolution_v1(acquisition_config: dict) -> List[float]:
    """
    Get the voxel resolution from an acquisition.json file.

    Parameters
    ----------
    acquisition_config: Dict
        Dictionary with the acquisition.json data.
    Returns
    -------
    List[float]
        Voxel resolution in the format [z, y, x].
    """

    if not acquisition_config:
        raise ValueError("acquisition.json file is empty or invalid.")

    # Grabbing a tile with metadata from acquisition - we assume all
    # dataset was acquired with the same resolution
    tile_coord_transforms = acquisition_config["tiles"][0][
        "coordinate_transformations"
    ]

    scale_transform = [
        x["scale"] for x in tile_coord_transforms if x["type"] == "scale"
    ][0]

    x = float(scale_transform[0])
    y = float(scale_transform[1])
    z = float(scale_transform[2])

    return [z, y, x]


def _get_voxel_resolution_v2(acquisition_config: dict) -> List[float]:
    """
    Get the voxel resolution from an acquisition.json in
    aind-data-schema v2 format.

    Parameters
    ----------
    acquisition_config: Dict
        Dictionary with the acquisition.json data.

    Returns
    -------
    List[float]
        Voxel resolution in the format [z, y, x].
    """
    try:
        data_stream = acquisition_config.get("data_streams", [])[0]
        configuration = data_stream.get("configurations", [])[0]
        image = configuration.get("images", [])[0]
        image_to_acquisition_transform = image[
            "image_to_acquisition_transform"
        ]
    except (IndexError, AttributeError, KeyError) as e:
        raise ValueError(
            "acquisition_config structure is invalid or missing "
            "required fields"
        ) from e

    scale_transform = [
        x["scale"]
        for x in image_to_acquisition_transform
        if x["object_type"] == "Scale"
    ][0]

    x = float(scale_transform[0])
    y = float(scale_transform[1])
    z = float(scale_transform[2])

    return [z, y, x]


def validate_capsule_inputs(input_elements: List[str]) -> List[str]:
    """
    Validates input elemts for a capsule in
    Code Ocean.

    Parameters
    -----------
    input_elements: List[str]
        Input elements for the capsule. This
        could be sets of files or folders.

    Returns
    -----------
    List[str]
        List of missing files
    """

    missing_inputs = []
    for required_input_element in input_elements:
        required_input_element = Path(required_input_element)

        if not required_input_element.exists():
            missing_inputs.append(str(required_input_element))

    return missing_inputs


def is_s3_path(path: str) -> bool:
    """
    Checks if a path is an s3 path

    Parameters
    ----------
    path: str
        Provided path

    Returns
    -------
    bool
        True if it is a S3 path,
        False if not.
    """
    parsed = urlparse(str(path))
    return parsed.scheme == "s3"


def get_parent_path(path: str) -> str:
    """
    Gets parent path

    Parameters
    ----------
    path: str
        Provided path

    Returns
    -------
    str
        Parent path
    """
    parsed = urlparse(path)
    if parsed.scheme == "s3":
        # Remove the last part of the S3 key
        parts = parsed.path.strip("/").split("/")
        parent_key = "/".join(parts[:-1])
        return f"s3://{parsed.netloc}/{parent_key}"
    else:
        # Local path fallback
        return str(Path(path).parent)
