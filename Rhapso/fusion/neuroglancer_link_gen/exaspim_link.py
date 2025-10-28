"""
Library for generating exaspim link.
"""
from collections import defaultdict
from typing import Optional

from . import link_utils
from .ng_layer import NgLayer
from .ng_state import NgState 
from .parsers import OmeZarrParser, XmlParser


def generate_exaspim_link(
    xml_path: Optional[str] = None,
    s3_path: Optional[str] = None,
    vmin: Optional[float] = 0,
    vmax: Optional[float] = 200,
    opacity: Optional[float] = 1.0,
    blend: Optional[str] = "default",
    output_json_path: Optional[str] = ".",
    dataset_name: Optional[str] = None,
    bucket_path: Optional[str] = "aind-open-data",
) -> None:
    """Creates a neuroglancer link to visualize
    registration transforms on exaspim dataset pre-fusion.

    Parameters
    ------------------------
    xml_path: str
        Path of xml outputted by BigStitcher.
    s3_path: str
        Path of s3 bucket where exaspim dataset is located.
    vmin: float
        Minimum value for shader.
    vmax: float
        Maximum value for shader.
    opacity: float
        Opacity of shader.
    blend: str
        Blend mode of shader.
    output_json_path: str
        Local directory to write process_output.json file that
        neuroglancer reads.
    dataset_name: Optional[str]
        Name of dataset. If None, will be directory name of
        output_json_path.
    bucket_path: Optional[str]
        S3 bucket name where the process_output.json will be uploaded.
        Default is "aind-open-data". Change this to your own bucket name
        if you plan to upload the JSON to a different bucket.

    Returns
    ------------------------
    None
    """

    if xml_path is None and s3_path.endswith(".zarr"):
        vox_sizes, tile_paths, net_transforms = OmeZarrParser.extract_info(
            s3_path
        )
    else:
        vox_sizes, tile_paths, net_transforms = XmlParser.extract_info(
            xml_path
        )

    channel_sources = defaultdict(list)
    for tile_id, _ in enumerate(net_transforms):
        t_path = tile_paths[tile_id]

        channel: int = link_utils.extract_channel_from_tile_path(t_path)

        final_transform = link_utils.convert_matrix_3x4_to_5x6(
            net_transforms[tile_id]
        )

        channel_sources[channel].append(
            {
                "url": f"{s3_path}/{t_path}",
                "transform_matrix": final_transform.tolist(),
            }
        )

    layers = []  # Neuroglancer Tabs
    for i, (channel, sources) in enumerate(channel_sources.items()):
        hex_val: int = link_utils.wavelength_to_hex(channel)
        hex_str = f"#{str(hex(hex_val))[2:]}"

        layers.append(
            {
                "type": "image",  # Optional
                "source": sources,
                "channel": 0,  # Optional
                "shaderControls": {
                    "normalized": {"range": [vmin, vmax]}
                },  # Optional  # Exaspim has low HDR
                "shader": {"color": hex_str, "emitter": "RGB", "vec": "vec3",},
                "visible": True,  # Optional
                "opacity": opacity,
                "name": f"CH_{channel}",
                "blend": blend,
            }
        )

    # Generate input config
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

    # Generate the link
    neuroglancer_link = NgState(
        input_config=input_config,
        mount_service="s3",
        bucket_path=bucket_path,
        output_dir=output_json_path,
        dataset_name=dataset_name,
    )
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())
