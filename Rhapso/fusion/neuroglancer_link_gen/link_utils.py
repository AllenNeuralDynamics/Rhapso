"""
Utilities for nueroglancer links.
"""
import pathlib
import re
from collections import defaultdict

import boto3
import numpy as np


def calculate_net_transforms(
    view_transforms: dict[int, list[dict]]
) -> dict[int, np.ndarray]:
    """
    Accumulate net transform and net translation for each matrix stack.
    Net translation =
        Sum of translation vectors converted into original nominal basis
    Net transform =
        Product of 3x3 matrices
    NOTE: Translational component (last column) is defined
          wrt to the DOMAIN, not codomain.
          Implementation is informed by this given.

    Parameters
    ------------------------
    view_transforms: dict[int, list[dict]]
        Dictionary of tile ids to transforms associated with each tile.

    Returns
    ------------------------
    dict[int, np.ndarray]:
        Dictionary of tile ids to net transform.

    """

    identity_transform = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    net_transforms: dict[int, np.ndarray] = defaultdict(
        lambda: np.copy(identity_transform)
    )

    for view, tfs in view_transforms.items():
        net_translation = np.zeros(3)
        net_matrix_3x3 = np.eye(3)

        # Tfs is a list of dicts containing transform under 'affine' key
        for tf in tfs:
            nums = [float(val) for val in tf["affine"].split(" ")]
            matrix_3x3 = np.array([nums[0::4], nums[1::4], nums[2::4]])
            translation = np.array(nums[3::4])
            net_translation = net_translation + (translation)
            net_matrix_3x3 = net_matrix_3x3 @ matrix_3x3
        net_transforms[view] = np.hstack(
            (net_matrix_3x3, net_translation.reshape(3, 1))
        )

    return net_transforms


def convert_matrix_3x4_to_5x6(matrix_3x4: np.ndarray) -> np.ndarray:
    """
    Converts classic 3x4 homogeneous coordinates: (x y z T)
    to nueroglancer 5x6 coordinates (t c z y x T)

    Parameters
    ------------------------
    matrix_3x4: np.ndarray
        See description above.

    Returns
    ------------------------
    np.ndarray:
        See description above.
    """

    # Initalize
    matrix_5x6 = np.zeros((5, 6), np.float32)
    np.fill_diagonal(matrix_5x6, 1)

    # Swap Rows 0 and 2; Swap Colums 0 and 2
    patch = np.copy(matrix_3x4)
    patch[[0, 2], :] = patch[[2, 0], :]
    patch[:, [0, 2]] = patch[:, [2, 0]]

    # Place patch in bottom-right corner
    matrix_5x6[2:6, 2:7] = patch

    return matrix_5x6


def list_all_tiles_in_path(SPIM_folder: str) -> list:
    """
    Lists all tiles in a given SPIM folder.

    Parameters
    ------------------------
    SPIM_folder: str
        Path to SPIM folder.

    Returns
    ------------------------
    list:
        List of all tiles in SPIM folder.
    """
    SPIM_folder = pathlib.Path(SPIM_folder)
    # assert SPIM_folder.exists()

    return list(SPIM_folder.glob("*.zarr"))


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
    # my_bucket = s3.Bucket(bucket_name)

    client = boto3.client("s3")
    result = client.list_objects(
        Bucket=bucket_name, Prefix=prefix + "/", Delimiter="/"
    )
    # print(result)
    tiles = []
    for o in result.get("CommonPrefixes"):
        # print 'sub folder : ', o.get('Prefix')
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
    if pathlib.Path(dataset_path).exists():
        tiles_in_path = list_all_tiles_in_path(dataset_path)
    else:
        tiles_in_path = list_all_tiles_in_bucket_path(
            dataset_path, "aind-open-data"
        )
    unique_list_of_channels = []
    for tile in tiles_in_path:
        channel = extract_channel_from_tile_path(tile)

        if channel not in unique_list_of_channels:
            unique_list_of_channels.append(channel)

    return unique_list_of_channels


def wavelength_to_hex(wavelength: int) -> int:
    """
    Converts wavelength to corresponding color hex value.
    Parameters
    ------------------------
    wavelength: int
        Integer value representing wavelength.
    Returns
    ------------------------
    int:
        Hex value color.
    """

    # Each wavelength key is the upper bound to a wavelgnth band.
    # Wavelengths range from 380-750nm.
    # Color map wavelength/hex pairs are generated
    # by sampling along a CIE diagram arc.
    color_map = {
        460: 0x690AFE,  # Purple
        470: 0x3F2EFE,  # Blue-Purple
        480: 0x4B90FE,  # Blue
        490: 0x59D5F8,  # Blue-Green
        500: 0x5DF8D6,  # Green
        520: 0x5AFEB8,  # Green
        540: 0x58FEA1,  # Green
        560: 0x51FF1E,  # Green
        565: 0xBBFB01,  # Green-Yellow
        575: 0xE9EC02,  # Yellow
        580: 0xF5C503,  # Yellow-Orange
        590: 0xF39107,  # Orange
        600: 0xF15211,  # Orange-Red
        620: 0xF0121E,  # Red
        750: 0xF00050,  # Pink
    }

    for ub, hex_val in color_map.items():
        if wavelength < ub:  # Exclusive
            return hex_val
    return hex_val  # hex_val is set to the last color in for loop
