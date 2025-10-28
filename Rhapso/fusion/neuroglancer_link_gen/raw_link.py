"""
Library for generating raw link for visualizing tiles in nominal position.
"""
import numpy as np
from parsers import XmlParser
from . import link_utils
from ng_state import NgState


def generate_raw_link(
    xml_path: str,
    s3_path: str,
    max_dr: int = 200,
    opacity: float = 1.0,
    blend: str = "default",
    output_json_path: str = ".",
    bucket_path: str = "aind-open-data",
) -> None:
    """Creates an neuroglancer link to visualize
    raw tile placements of one color channel defined in the input xml.

    Parameters
    ------------------------
    xml_path: str
        Path of xml outputted by BigStitcher.
    s3_path: str
        Path of s3 bucket where exaspim dataset is located.
    output_json_path: str
        Local path to write process_output.json file that nueroglancer reads.
    bucket_path: str
        S3 bucket name where the process_output.json will be uploaded.
        Default is "aind-open-data".

    Returns
    ------------------------
    None
    """
    # Gather xml info
    vox_sizes: tuple[float, float, float] = XmlParser.extract_tile_vox_size(
        xml_path
    )
    tile_paths: dict[int, str] = XmlParser.extract_tile_paths(xml_path)

    # Reference Pathstring:
    # "s3://aind-open-data/diSPIM_647459_2022-12-21_00-39-00/diSPIM.zarr"
    tile_transforms: dict[int, list[dict]] = {}

    s3_list = s3_path.split("/")
    dataset_name = s3_list[4]
    dataset_list = dataset_name.split(".")
    dataset_type = dataset_list[0]

    if dataset_type == "diSPIM":
        tile_transforms: dict[
            int, list[dict]
        ] = XmlParser.extract_tile_transforms(xml_path)

    net_transforms: dict[
        int, np.ndarray
    ] = link_utils.calculate_net_transforms(tile_transforms)

    # Determine color
    channel: int = link_utils.extract_channel_from_tile_path(tile_paths[0])
    hex_val: int = link_utils.wavelength_to_hex(channel)
    hex_str = f"#{str(hex(hex_val))[2:]}"

    # Generate input config
    layers = []  # Nueroglancer Tabs
    input_config = {
        "dimensions": {
            "x": {"voxel_size": vox_sizes[0], "unit": "microns"},
            "y": {"voxel_size": vox_sizes[1], "unit": "microns"},
            "z": {"voxel_size": vox_sizes[2], "unit": "microns"},
            "c'": {"voxel_size": 1, "unit": ""},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": layers,
        "showScaleBar": False,
        "showAxisLines": False,
    }

    sources = []  # Tiles within tabs
    layers.append(
        {
            "type": "image",  # Optional
            "source": sources,
            "channel": 0,  # Optional
            "shaderControls": {
                "normalized": {"range": [0, max_dr]}
            },  # Optional  # Exaspim has low HDR
            "shader": {"color": hex_str, "emitter": "RGB", "vec": "vec3",},
            "visible": True,  # Optional
            "opacity": opacity,
            "name": f"CH_{channel}",
            "blend": blend,
        }
    )

    for tile_id, t_path in tile_paths.items():
        source_dict = {"url": f"{s3_path}/{t_path}"}
        if dataset_type == "diSPIM":
            net_tf = net_transforms[tile_id]
            final_transform = link_utils.convert_matrix_3x4_to_5x6(net_tf)
            source_dict["transform_matrix"] = final_transform.tolist()

        sources.append(source_dict)

    # Generate the link
    neuroglancer_link = NgState(
        input_config=input_config,
        mount_service="s3",
        bucket_path=bucket_path,
        output_dir=output_json_path,
    )
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())
