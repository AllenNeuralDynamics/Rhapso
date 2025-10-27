"""
Library for generating dispim link.
"""
import pathlib

import link_utils
import numpy as np
from ng_state import NgState
from parsers import XmlParser
from utils import transfer
from typing import List, Dict, Tuple


def apply_deskewing(matrix_3x4: np.ndarray, theta: float = 45) -> np.ndarray:
    """
    Compounds deskewing transform to input 3x4 matrix:
        Deskewing @ Input_3x4_Matrix.

    Parameters
    ------------------------
    matrix_3x4: np.ndarray
        3x4 numpy array representing transformation matrix applied to tile.

    theta: float
        Angle of lens during acquisition.

    Returns
    ------------------------
    np.ndarray:
        3x4 numpy array composite transform.

    """

    # Deskewing
    # X vector => XZ direction
    deskew_factor = np.tan(np.deg2rad(theta))
    deskew = np.array([[1, 0, 0], [0, 1, 0], [deskew_factor, 0, 1]])
    matrix_3x4 = deskew @ matrix_3x4

    return matrix_3x4


def generate_dispim_link(
    base_channel_xml_path: str,
    # cross_channel_xml_path: str,
    s3_path: str,
    max_dr: int = 800,
    opacity: float = 0.5,
    blend: str = "additive",
    deskew_angle: int = 45,
    output_json_path: str = ".",
    spim_foldername="SPIM.ome.zarr",
) -> None:
    """
    Creates an neuroglancer link to visualize
    registration transforms on dispim dataset pre-fusion.

    Parameters
    ------------------------
    base_channel_xml_path: str
        Path to xml file acquired from tile-to-tile
        registration within the base channel.
        These registrations are reused for
        registering tiles in all other channels.

    cross_channel_xml_path: str
        Path to xml file acquired from channel-to-channel registration.
        These registrations are prepended to each tile registration.

    s3_path: str
        Path of s3 bucket where dipim dataset is located.

    output_json_path: str
        Local path to write process_output.json file that nueroglancer reads.

    Returns
    ------------------------
    None
    """

    # Gather base channel xml info
    vox_sizes: Tuple[float, float, float] = XmlParser.extract_tile_vox_size(
        base_channel_xml_path
    )
    tile_paths: Dict[int, str] = XmlParser.extract_tile_paths(
        base_channel_xml_path
    )
    tile_transforms: Dict[int, List[Dict]] = XmlParser.extract_tile_transforms(
        base_channel_xml_path
    )
    intertile_transforms: Dict[
        int, np.ndarray
    ] = link_utils.calculate_net_transforms(tile_transforms)
    base_channel: int = link_utils.extract_channel_from_tile_path(
        tile_paths[0]
    )

    channels: List[int] = link_utils.get_unique_channels_for_dataset(
        s3_path + spim_foldername
    )

    # Generate input config
    layers = []  # Represent Neuroglancer Tabs
    input_config = {
        "dimensions": {
            "x": {"voxel_size": vox_sizes[0], "unit": "microns"},
            "y": {"voxel_size": vox_sizes[1], "unit": "microns"},
            # reverse the order from bigstitcher again
            "z": {"voxel_size": vox_sizes[2], "unit": "microns"},
            "c'": {"voxel_size": 1, "unit": ""},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": layers,
        "showScaleBar": False,
        "showAxisLines": False,
    }

    for channel in channels:
        # Determine color of this layer
        hex_val: int = link_utils.wavelength_to_hex(channel)
        hex_str = f"#{str(hex(hex_val))[2:]}"

        # Init new list of sources for each channel
        sources = []  # Represent Tiles w/in Tabs
        layers.append(
            {
                "type": "image",  # Optional
                "source": sources,
                "channel": 0,  # Optional
                "shaderControls": {
                    "normalized": {"range": [90, max_dr]}
                },  # Optional
                "shader": {"color": hex_str, "emitter": "RGB", "vec": "vec3",},
                "visible": True,  # Optional
                "opacity": opacity,
                "name": f"CH_{channel}",
                "blend": blend,
            }
        )

        for tile_id in range(len(intertile_transforms)):
            # Get base tile path, modify path across channels
            base_t_path = tile_paths[tile_id]
            t_path = base_t_path.replace(f"{base_channel}", f"{channel}")

            # Get net transform
            intertile_tf = intertile_transforms[tile_id]
            i_matrix_3x3 = intertile_tf[:, 0:3]
            i_translation = intertile_tf[:, 3]

            net_matrix_3x3 = i_matrix_3x3  # NOTE: Right-multiply
            net_translation = i_translation
            net_tf = np.hstack((net_matrix_3x3, net_translation.reshape(3, 1)))

            # net_tf = apply_deskewing(net_tf, deskew_angle)

            # Add (path, transform) source entry
            if s3_path.endswith("/"):
                url = f"{s3_path}{spim_foldername}/{t_path}"
            else:
                url = f"{s3_path}/{spim_foldername}/{t_path}"

            final_transform = link_utils.convert_matrix_3x4_to_5x6(net_tf)
            sources.append(
                {"url": url, "transform_matrix": final_transform.tolist()}
            )

    bucket_name, prefix = s3_path.replace("s3://", "").split("/", 1)
    prefix = prefix[:-1]  # remove trailing '/'
    # Generate the link
    neuroglancer_link = NgState(
        input_config=input_config,
        mount_service="s3",
        bucket_path=f"{bucket_name}",
        output_dir=output_json_path,
        base_url="https://aind-neuroglancer-sauujisjxq-uw.a.run.app/",
    )
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())
    return neuroglancer_link.get_url_link()


def ingest_xml_and_write_ng_link(
    xml_path: str, s3_bucket: str = "aind-open-data"
):
    """A wrapper function that autogenerates the s3_path
     for dispim_link.generate_dispim_link

    Automatically saves process_output.json, which can be
     manually uploaded to S3 bucket/dataset.

    Parameters:
    ----------
    xml_path: str
        Relative path to xml file (bigstitcher format) that
        contains tile position information

    s3_bucket:str
        name of s3 bucket where the dataset lives

    Return:
    -------
    link: str
    Neuroglancer link for xml dataset.


    """
    # read_xml and get dataset prefix for S3
    dataset_path = XmlParser.extract_dataset_path(xml_path)
    dataset_name = dataset_path.split("/")[2]

    # print(f"dataset_path {dataset_path}")
    # print(f"dataset_name {dataset_name}")

    s3_path = f"s3://{s3_bucket}/{dataset_name}/"

    output_folder = f"/results/{dataset_name}/"

    if not pathlib.Path(output_folder).exists():
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    link = generate_dispim_link(
        xml_path,
        s3_path,
        max_dr=400,
        opacity=1.0,
        blend="additive",
        output_json_path=output_folder,
    )

    # copy output json to s3 bucket dataset

    transfer.copy_to_s3(output_folder + "process_output.json", s3_path)

    return link
