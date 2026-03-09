"""
Script to generate CCF + cell counts
"""

import json
import os
from pathlib import Path
from typing import Optional, Union

import boto3
# import neuroglancer
import pandas as pd

from ng_link import NgState
# from ng_link.ng_layer import generate_precomputed_cells
from ng_link.ng_state import get_points_from_xml

# IO types
PathLike = Union[str, Path]


def get_ccf(
    out_path: str,
    bucket_name: Optional[str] = "tissuecyte-visualizations",
    s3_folder: Optional[str] = "data/221205/ccf_annotations/",
):
    """
    Parameters
    ----------
    out_path : str
        path to where the precomputed segmentation map will be stored
    bucket_name: Optional[str]
        Bucket name where the precomputed annotation is stored for the
        CCF
    s3_folder: Optional[str]
        Path inside of the bucket where the annotations are stored

    """

    # location of the data from tissueCyte,
    # but can get our own and change to aind-open-data

    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = os.path.join(out_path, os.path.relpath(obj.key, s3_folder))

        # dont currently need 10um data so we should skip
        if "10000_10000_10000" in obj.key:
            continue

        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))

        # dont try and download folders
        if obj.key[-1] == "/":
            continue

        bucket.download_file(obj.key, target)


def generate_cff_cell_counting(
    input_path: str, output_path: str, ccf_reference_path: Optional[str] = None
):
    """
    Function for creating segmentation layer with cell counts

    Parameters
    -----------------

    input_path: str
        path to file cell_count_by_region.csv
        generated from "aind-smartspim-quantification"
    output_path: str
        path to where you want to save the precomputed files
    """
    # check that save path exists and if not create
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    df_count = pd.read_csv(input_path, index_col=0)
    include = list(df_count["Structure"].values)

    # get CCF id-struct pairings
    if ccf_reference_path is None:
        ccf_reference_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "ccf_ref.csv"
        )

    df_ccf = pd.read_csv(ccf_reference_path)

    keep_ids = []
    keep_struct = []
    for r, irow in df_ccf.iterrows():
        if irow["struct"] in include:
            keep_ids.append(str(irow["id"]))
            total = df_count.loc[
                df_count["Structure"] == irow["struct"], ["Total"]
            ].values.squeeze()
            keep_struct.append(irow["struct"] + " cells: " + str(total))

    # download ccf procomputed format
    get_ccf(output_path)

    # currently using 25um resolution so need to drop 10um data or NG finicky
    with open(os.path.join(output_path, "info"), "r") as f:
        info_file = json.load(f)

    info_file["scales"].pop(0)

    with open(os.path.join(output_path, "info"), "w") as f:
        json.dump(info_file, f, indent=2)

    # build json for segmentation properties
    data = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": keep_ids,
            "properties": [
                {"id": "label", "type": "label", "values": keep_struct}
            ],
        },
    }

    with open(
        os.path.join(output_path, "segment_properties/info"), "w"
    ) as outfile:
        json.dump(data, outfile, indent=2)


def generate_25_um_ccf_cells(params: dict, micron_res: int = 25):
    """
    Generates the visualization link for the
    CCF + Cell counting in the 25 um resolution
    """
    # Generating CCF and cell counting precomputed format

    # Get cells from XML
    cells = get_points_from_xml(params["cells_precomputed"]["xml_path"])
    # cells = random.shuffle(cells)

    generate_cff_cell_counting(
        params["ccf_cells_precomputed"]["input_path"],
        params["ccf_cells_precomputed"]["output_path"],
        params["ccf_cells_precomputed"]["ccf_reference_path"],
    )

    # Creating neuroglancer link
    ccf_cell_count = {
        "dimensions": {
            # check the order
            "z": {"voxel_size": micron_res, "unit": "microns"},
            "y": {"voxel_size": micron_res, "unit": "microns"},
            "x": {"voxel_size": micron_res, "unit": "microns"},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": [
            {
                "source": params["zarr_path"],
                "type": "image",
                "channel": 0,
                "shaderControls": {
                    "normalized": {"range": [0, 500]}
                },  # Optional
                "name": "image_25_um",
            },
            {
                "type": "annotation",
                "source": f"precomputed://{params['cells_precomputed']['output_precomputed']}",
                "tool": "annotatePoint",
                "name": "cell_points",
                "annotations": cells,
            },
            {
                "type": "segmentation",
                "source": f"precomputed:///{params['ccf_cells_precomputed']['output_path']}",
                "tab": "source",
                "name": "cell_counting_in_CCF",
            },
        ],
    }

    neuroglancer_link = NgState(
        input_config=ccf_cell_count,
        mount_service="s3",
        bucket_path="aind-msma-data",
        output_dir="/Users/camilo.laiton/repositories/new_ng_link/aind-ng-link",
    )

    return neuroglancer_link


if __name__ == "__main__":
    params = {
        "ccf_cells_precomputed": {  # Parameters to generate CCF + Cells precomputed format
            "input_path": "/Users/camilo.laiton/Downloads/cell_count_by_region.csv",  # Path where the cell_count.csv is located
            "output_path": "/Users/camilo.laiton/repositories/new_ng_link/aind-ng-link/src/ng_link/scripts/CCF_Cells_Test",  # Path where we want to save the CCF + cell location precomputed
            "ccf_reference_path": None,  # Path where the CCF reference csv is located
        },
        "cells_precomputed": {  # Parameters to generate cell points precomputed format
            "xml_path": "/Users/camilo.laiton/Downloads/transformed_cells.xml",  # Path where the cell points are located
            "output_precomputed": "/Users/camilo.laiton/repositories/new_ng_link/aind-ng-link/src/ng_link/scripts/Cells_Test",  # Path where the precomputed format will be stored
        },
        "zarr_path": "s3://aind-open-data/SmartSPIM_656374_2023-01-27_12-41-55_stitched_2023-01-31_17-28-34/processed/CCF_Atlas_Registration/Ex_445_Em_469/OMEZarr/image.zarr",  # Path where the 25 um zarr image is stored, output from CCF capsule
    }

    generate_25_um_ccf_cells(params)
