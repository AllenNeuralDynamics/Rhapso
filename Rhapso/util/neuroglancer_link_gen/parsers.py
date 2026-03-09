"""Module for parsing dataset info."""

from typing import Dict, List, OrderedDict, Tuple

import numpy as np
import xmltodict
import zarr

from . import link_utils


class OmeZarrParser:
    """Class for parsing OME-Zarr datasets."""

    @staticmethod
    def parse_transform(z, res) -> Dict[str, list]:
        """
        Parses scale and translation transformations for a resolution level
        in an OME-Zarr dataset.

        Parameters
        ----------
        z : zarr.core.Array
            Zarr array representing a dataset.
        res : str
            Dataset resolution to parse.

        Returns
        -------
        Dict[str, list] A dictionary containing scale and
        translation data for the first dataset.
        """

        # Read the metadata from .zattrs
        try:
            metadata = z.attrs.asdict()
        except KeyError:
            raise ValueError("OME-Zarr metadata not found.")

        # Extract transformations for the first dataset
        transformations = {}
        multiscales = metadata.get("multiscales", [])

        if multiscales:
            datasets = multiscales[0].get("datasets", [])
            for ds in datasets:
                if ds["path"] == res:
                    coord_transforms = ds.get("coordinateTransformations", [])
                    scale = next(
                        (
                            t["scale"]
                            for t in coord_transforms
                            if t["type"] == "scale"
                        ),
                        None,
                    )
                    translation = next(
                        (
                            t["translation"]
                            for t in coord_transforms
                            if t["type"] == "translation"
                        ),
                        None,
                    )
                    transformations = {
                        "scale": scale,
                        "translation": translation,
                    }
                    break

        return transformations

    @staticmethod
    def extract_info(
        s3_path: str,
    ) -> Tuple[tuple, Dict[int, str], Dict[int, np.ndarray]]:
        """
        Extracts voxel sizes, tile paths, and tile offsets from a given
        OME-Zarr path.

        Parameters
        ----------
        s3_path : str
            Path to the OME-Zarr dataset.

        Returns
        -------
        Tuple[tuple, Dict[int, str], Dict[int, np.ndarray]]
            A tuple containing voxel sizes, tile paths, and tile offsets.
        """
        vox_sizes: tuple[float, float, float] = (
            OmeZarrParser.extract_tile_vox_size(s3_path)
        )
        tile_paths: dict[int, str] = OmeZarrParser.extract_tile_paths(s3_path)
        net_transforms: dict[int, np.ndarray] = (
            OmeZarrParser._get_identity_mats(s3_path)
        )
        return vox_sizes, tile_paths, net_transforms

    @staticmethod
    def extract_tile_paths(zarr_path: str) -> Dict[int, str]:
        """
        Extracts tile paths from a given Zarr dataset.

        Parameters
        ----------
        zarr_path : str
            Path to the Zarr dataset.

        Returns
        -------
        Dict[int, str]
            A dictionary mapping tile indices to their paths.
        """
        z = zarr.open(zarr_path, mode="r")
        return {i: k for i, k in enumerate(sorted(z.keys()))}

    @staticmethod
    def extract_tile_vox_size(zarr_path: str) -> Tuple[float, float, float]:
        """
        Extracts the voxel size of tiles from a given Zarr dataset.

        Parameters
        ----------
        zarr_path : str
            Path to the Zarr dataset.

        Returns
        -------
        Tuple[float, float, float]
            A tuple representing the voxel size in the x, y, and z dimensions.
        """
        z = zarr.open(zarr_path, mode="r")
        first_tile = z[next(iter(z.keys()))]

        return tuple(
            reversed(
                OmeZarrParser.parse_transform(first_tile, "0")["scale"][2:]
            )
        )

    @staticmethod
    def _get_identity_mats(zarr_path: str) -> Dict[int, np.ndarray]:
        """
        Create a homogeneous identity matrix for each tile in the dataset.
        We need to do this because neuroglancer expects the offset to be
        encoded in the .zattrs. The transformation matrix in the viewer
        state should do nothing.

        Parameters
        ----------
        zarr_path : str
            Path to the Zarr dataset.

        Returns
        -------
        Dict[int, np.ndarray]
            A dictionary mapping tile indices to their offset matrices.
        """
        z = zarr.open(zarr_path, mode="r")
        tile_offsets = {}
        for i in range(len(list(z.keys()))):
            # Use the identity matrix since the offset is already encoded in
            # the .zattrs
            tile_offsets[i] = np.hstack((np.eye(3), np.zeros(3).reshape(3, 1)))
        return tile_offsets


class XmlParser:
    """Class for parsing BigStitcher XML datasets."""

    @staticmethod
    def extract_dataset_path(xml_path: str) -> str:
        """
        Parses BDV XML and extracts the dataset path.

        Parameters
        ----------
        xml_path : str
            Path to the XML file.

        Returns
        -------
        str
            Path of the dataset extracted from the XML.
        """

        # view_paths: dict[int, str] = {}
        with open(xml_path, "r") as file:
            data: OrderedDict = xmltodict.parse(file.read())

        dataset_path = data["SpimData"]["SequenceDescription"]["ImageLoader"][
            "zarr"
        ]

        return dataset_path["#text"]

    @staticmethod
    def extract_tile_paths(xml_path: str) -> Dict[int, str]:
        """
        Parses BDV XML and extracts a map of setup IDs to tile paths.

        Parameters
        ----------
        xml_path : str
            Path to the XML file.

        Returns
        -------
        Dict[int, str]
            Dictionary mapping tile IDs to their paths.
        """

        view_paths: dict[int, str] = {}
        with open(xml_path, "r") as file:
            data: OrderedDict = xmltodict.parse(file.read())

        for id, zgroup in enumerate(
            data["SpimData"]["SequenceDescription"]["ImageLoader"]["zgroups"][
                "zgroup"
            ]
        ):
            view_paths[int(id)] = zgroup["path"]

        return view_paths

    @staticmethod
    def extract_tile_vox_size(xml_path: str) -> Tuple[float, float, float]:
        """
        Parses BDV XML and extracts voxel sizes.

        Parameters
        ----------
        xml_path : str
            Path to the XML file.

        Returns
        -------
        Tuple[float, float, float]
            Tuple containing voxel sizes (x, y, z).
        """

        with open(xml_path, "r") as file:
            data: OrderedDict = xmltodict.parse(file.read())

        first_tile_metadata = data["SpimData"]["SequenceDescription"][
            "ViewSetups"
        ]["ViewSetup"][0]
        vox_sizes: str = first_tile_metadata["voxelSize"]["size"]
        return tuple(float(val) for val in vox_sizes.split(" "))

    @staticmethod
    def extract_tile_transforms(xml_path: str) -> Dict[int, List[dict]]:
        """
        Parses BDV XML and extracts a map of setup IDs to lists of
        transformations.

        Parameters
        ----------
        xml_path : str
            Path to the XML file.

        Returns
        -------
        Dict[int, List[dict]]
            Dictionary mapping tile IDs to lists of transformations.
        """

        view_transforms: dict[int, list[dict]] = {}
        with open(xml_path, "r") as file:
            data: OrderedDict = xmltodict.parse(file.read())

        for view_reg in data["SpimData"]["ViewRegistrations"][
            "ViewRegistration"
        ]:
            tfm_stack = view_reg["ViewTransform"]
            if not isinstance(tfm_stack, list):
                tfm_stack = [tfm_stack]
            view_transforms[int(view_reg["@setup"])] = tfm_stack

        view_transforms = {
            view: tfs[::-1] for view, tfs in view_transforms.items()
        }

        return view_transforms

    @staticmethod
    def extract_info(
        xml_path: str,
    ) -> Tuple[tuple, Dict[int, str], Dict[int, np.ndarray]]:
        """
        Extracts voxel sizes, tile paths, and tile transforms from a given
        XML path.

        Parameters
        ----------
        xml_path : str
            Path to the BDV XML dataset.

        Returns
        -------
        Tuple[tuple, Dict[int, str], Dict[int, np.ndarray]]
            A tuple containing voxel sizes, tile paths, and tile offsets.
        """
        vox_sizes: tuple[float, float, float] = (
            XmlParser.extract_tile_vox_size(xml_path)
        )
        tile_paths: dict[int, str] = XmlParser.extract_tile_paths(xml_path)
        tile_transforms: dict[int, list[dict]] = (
            XmlParser.extract_tile_transforms(xml_path)
        )
        XmlParser.omit_initial_offsets(tile_transforms)
        net_transforms: dict[int, np.ndarray] = (
            link_utils.calculate_net_transforms(tile_transforms)
        )
        return vox_sizes, tile_paths, net_transforms

    @staticmethod
    def omit_initial_offsets(view_transforms: dict[int, list[dict]]) -> None:
        """
        For OME-Zarr datasets, inital offsets are
        already encoded in the metadata and extracted my neuroglancer.
        This function removes the duplicate transform.

        Parameters
        ------------------------
        view_transforms: dict[int, list[dict]]
            Dictionary of tile ids to list of transforms.

        Returns
        ------------------------
        None
        """

        for view, tfs in view_transforms.items():
            tfs.pop(0)
