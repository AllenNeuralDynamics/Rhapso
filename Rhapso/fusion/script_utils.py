"""
Utilities for scripts.
"""

import re
from pathlib import Path

import boto3
import yaml


def read_config_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


def write_config_yaml(yaml_path: str, yaml_data: dict) -> None:
    with open(yaml_path, "w") as file:
        yaml.dump(yaml_data, file)


def list_all_tiles_in_bucket_path(
    bucket_SPIM_folder: str, bucket_name="aind-open-data"
) -> list:
    """
    Lists all tiles in a given bucket path

    Parameters
    ------------------------
    bucket_SPIM_folder: str
        Path to SPIM folder in bucket.
    bucket_name: str
        Name of bucket.

    Returns
    ------------------------
    list:
        List of all tiles in SPIM folder.
    """
    # s3 = boto3.resource('s3')
    bucket_name, prefix = bucket_SPIM_folder.replace("s3://", "").split("/", 1)

    client = boto3.client("s3")
    result = client.list_objects(
        Bucket=bucket_name, Prefix=prefix, Delimiter="/"
    )

    tiles = []
    for o in result.get("CommonPrefixes"):
        tiles.append(o.get("Prefix"))
    return tiles


def extract_channel_from_tile_path(t_path: str) -> int:
    """
    Extracts channel from tile path naming convention:
    tile_X_####_Y_####_Z_####_ch_####.filetype

    Parameters
    ------------------------
    t_path: str
        Tile path to run regex on.

    Returns
    ------------------------
    int:
        Channel value.

    """

    pattern = r"(ch|CH)_(\d+)"
    match = re.search(pattern, t_path)
    channel = int(match.group(2))
    return channel


def get_unique_channels_for_dataset(dataset_path: str) -> list:
    """
    Extracts a list of channels in a given dataset

    Parameters:
    -----------
    dataset_path: str
        Path to a dataset's zarr folder

    Returns:
    --------
    unique_list_of_channels: list(int)
        A list of int, containing the unique list of channel wavelengths

    """
    # Reference Path: s3://aind-open-data/HCR_677594_2023-10-13_13-55-48/SPIM.ome.zarr/
    # path_parts = dataset_path.split('/')
    # tiles_in_path = list_bucket_directory(path_parts[2], path_parts[3] + '/' + path_parts[4])

    tiles_in_path = list_all_tiles_in_bucket_path(
        dataset_path, "aind-open-data"
    )

    unique_list_of_channels = []
    for tile in tiles_in_path:
        channel = extract_channel_from_tile_path(tile)

        if channel not in unique_list_of_channels:
            unique_list_of_channels.append(channel)

    return unique_list_of_channels


def get_folder_regex(dataset_path, regex):
    dataset_path = Path(dataset_path)
    print(f"Reading data from: {dataset_path}")
    channels = [str(i) for i in list(dataset_path.glob(regex))]
    return channels
