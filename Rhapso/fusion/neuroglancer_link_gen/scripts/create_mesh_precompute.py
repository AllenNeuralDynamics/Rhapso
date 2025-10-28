#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:59:40 2023

@author: nicholas.lusk
"""

import json
import os
import re
import shutil
import subprocess
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import trimesh

# IO types
PathLike = Union[str, Path]


class ng_mesh_precompute:
    """
    Class for constructing a mesh layer for neuroglancer from CCF registered
    SWC files. Currently only does single resolution.
    """

    def __init__(
        self,
        save_path: PathLike,
        resolution: List[int],
        dimensions: Optional[List[int]] = [13200, 8000, 11400],
        voxel_offest: Optional[List[int]] = [0, 0, 0],
        tmp_path: Optional[PathLike] = "/scratch/",
    ) -> None:
        """
        Class constructor

        Parameters
        ------------------------
        save_path: Pathlike
            Location for saving the precomputed format
        resolution: list
            resolution of SWC in each dimention (nm): [AP, DV, ML]
        dimentions: Optional[List[int]]
            dimentions of the tissue volume in (um): [AP, DV, ML].
            Default are for a 1um CCF [13200, 8000, 11400]
        voxel_offset: Optional[List[int]]
            Offest of SWC relative to origin. Default is no offset [0, 0, 0]
        """

        self.save_path = save_path
        self.resolution = resolution
        self.dimensions = dimensions
        self.chunk_size = resolution
        self.tmp_path = tmp_path

        # where you install Ultraliser. This is the location if on codeocean
        self.ultra_func = "/home/Ultraliser/build/bin/ultraNeuroMorpho2Mesh"
        self.mesh_path = os.path.join(save_path, "mesh")

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            os.mkdir(self.mesh_path)

    def to_precomputed(m, flip_x=True) -> bytes:
        """
        Create byte string in a neuroglancer precomputed format for meshes

        Parameters
        ----------
        m : trimesh obj
            A triangle mesh object imported using trimesh module
        flip_x : bool, optional
            If the x-axis (i.e. AP needs to be flipped). The default is True.

        Returns
        -------
        m_bytes: byte stream
            Byte stream of verticies and faces in neuroglancer precomputed
            format
        """

        verts = np.asarray(m.vertices)
        faces = np.asarray(m.triangles)

        if flip_x:
            verts[:, 0] = 13200 - verts[:, 0]

        verts *= 10 ** 3

        vertex_index_format = [
            np.uint32(verts.shape[0]),  # Number of vertices (3 coordinates)
            np.float32(verts),
            np.uint32(faces),
        ]

        m_bytes = b"".join(
            [array.tobytes("C") for array in vertex_index_format]
        )

        return m_bytes

    def write_mesh_info(self) -> None:
        """
        Write mesh info files
        """

        # info file that goes within the precomputed folder
        info_file_1 = {
            "num_channels": 1,
            "type": "segmentation",
            "data_type": "uint32",
            "scales": [
                {
                    "encoding": "raw",
                    "key": "1_1_1",
                    "chunk_sizes": [self.chunk_size],
                    "resolution": self.resolution,
                    "voxel_offset": self.offset,
                    "size": self.dimensions,
                }
            ],
            "mesh": "mesh",
        }

        with open(os.path.join(self.save_path, "info"), "w") as f:
            f.write(json.dumps(info_file_1))

        # write second info file
        info_file_2 = {
            "@type": "neuroglancer_legacy_mesh",
        }

        with open(os.path.join(self.mesh_path, "info"), "w") as f:
            f.write(json.dumps(info_file_2))

    def write_fragment_files(self, count: int, m_bytes: bytes) -> None:
        """
        Write fragment files for an individual mesh after being converted to a
        precomputed format

        Parameters
        ----------
        count : int
            Number id of the cell associated with fragment file being created
        m_bytes : byte stream
            Byte stream in neuroglancer precomputed format for mesh being saved

        Returns
        -------
        None

        """

        # write fragment files
        frag_name = str(count + 1) + ":0"
        cell_name = str(count + 1) + ":0:0"

        metadata = {
            "fragments": [cell_name],
        }

        with open(os.path.join(self.mesh_path, frag_name), "w") as f:
            f.write(json.dumps(metadata))

        with open(os.path.join(self.mesh_path, cell_name), "wb") as outfile:
            outfile.write(m_bytes)

    # function to create a single resolution mesh
    def build_mesh(self, fname: PathLike) -> str:
        """
        Runs Ultraliser to produce mesh and saves to scratch

        Parameters
        ----------

        fname : Pathlike
            Path to an individual SWC file for processing

        Returns
        -------
        processed_path
            The location of the newly created ultraliser mesh

        """

        # run SWC to Mesh code
        command_list = [
            self.ultra_func,
            "--morphology",
            fname,
            "--output-directory",
            self.tmp_path,
            "--export-obj-mesh",
            "--ignore-marching-cubes-mesh",
            "--ignore-laplacian-mesh",
            "--solid",
        ]

        subprocess.call(command_list)
        processed_path = glob(os.path.join(self.tmp_path, "meshes", "*.obj"))[
            0
        ]

        return processed_path


def main(params: dict) -> None:
    files = glob(os.path.join(params["swc_path"], "*.swc"))
    ng_mesh = ng_mesh_precompute(params["save_path"], params["resolution"])

    for c, fname in enumerate(files):
        base = os.path.basename(fname)

        # might be specific to Lydia's datasets but don't need soma files
        if len(re.findall("soma", base)) > 0:
            continue

        # run ultraliser and return output path
        processed_path = ng_mesh.build_mesh(fname)

        # create info files for mesh precomputed
        ng_mesh.write_mesh_info()

        # load mesh and convert to precomputed format
        m = trimesh.load_mesh(processed_path)
        m_bytes = ng_mesh.to_precomputed(m)

        ng_mesh.write_fragment_files(c, m_bytes)

        # clear previously created mesh
        shutil.rmtree(params["save_path"])

    return


if __name__ == "__main__":
    """
    This is parameterized for codeocean. The swc path is a data asset
    containing a folder that has all swc files for a given brain
    """

    params = {
        "save_path": "/results/mesh_precompute",
        "swc_path": "/data/swc_mesh_files/swc_files",
        "resolution": [1000, 1000, 1000],
    }

    main(params)
