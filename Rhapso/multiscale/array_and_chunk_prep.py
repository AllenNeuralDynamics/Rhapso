from typing import List, Tuple
from numpy.typing import ArrayLike
import numpy as np
import xml.etree.ElementTree as ET
import s3fs

"""
Pads arrays to TCZYX and clamps chunks to data shape.
"""

class ArrayAndChunkPrep:
    def __init__(self, chunk_size: List[int], xml_path=None, dim: int = 5,
                 voxel_size: List[float] = None) -> None:
        """
        Parameters
        ----------
        voxel_size : list[float], optional
            Explicit ZYX voxel size (in micrometers). When provided, skips
            the BigStitcher XML read. Useful when the caller already has
            the base-level physical spacing in hand (e.g. from an OME-NGFF
            multiscales entry).
        xml_path : str, optional
            BigStitcher XML path. Required if ``voxel_size`` is not given;
            ignored otherwise.
        """
        self.chunk_size = chunk_size
        self.xml_path = xml_path
        self.dim = dim
        self.voxel_size = voxel_size

    def voxel_size_zyx_from_xml(self) -> list[float]:
        if self.xml_path.startswith("s3://"):
            fs = s3fs.S3FileSystem(anon=True)
            with fs.open(self.xml_path, "rb") as f:
                xml_text = f.read()
            root = ET.fromstring(xml_text)
        else:
            root = ET.parse(self.xml_path).getroot()

        size_text = root.find(".//ViewSetup/voxelSize/size").text  # "X Y Z"
        x, y, z = map(float, size_text.split())
        return [z, y, x]

    def _pad_array_n_d(self, arr: ArrayLike) -> ArrayLike:
        if self.dim > 5:
            raise ValueError("Padding more than 5 dimensions is not supported.")
        while arr.ndim < self.dim:
            arr = arr[np.newaxis, ...]
        return arr

    def _compute_dataset_shape(self, arr: ArrayLike) -> Tuple[int, ...]:
        dataset_shape = tuple(i for i in arr.shape if i != 1)
        extra_axes = (1,) * (self.dim - len(dataset_shape))
        return extra_axes + dataset_shape

    def _clamp_chunks(self, dataset_shape: Tuple[int, ...]) -> List[int]:
        full_chunks = ([1] * (self.dim - len(self.chunk_size))) + list(self.chunk_size)
        for i, val in enumerate(dataset_shape):
            if full_chunks[i] > val:
                full_chunks[i] = val
        return full_chunks

    def run(self, data: ArrayLike):
        """
        Entry point
        """
        if self.voxel_size is not None:
            voxel_size = [float(v) for v in self.voxel_size]
        elif self.xml_path is not None:
            voxel_size = self.voxel_size_zyx_from_xml()
        else:
            raise ValueError(
                "ArrayAndChunkPrep requires either voxel_size or xml_path"
            )
        arr = self._pad_array_n_d(data)
        dataset_shape = self._compute_dataset_shape(arr)
        full_chunks = self._clamp_chunks(dataset_shape)

        return arr, dataset_shape, full_chunks, voxel_size