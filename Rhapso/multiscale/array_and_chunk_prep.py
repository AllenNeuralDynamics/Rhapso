from typing import List, Tuple
from numpy.typing import ArrayLike
import numpy as np

"""
Pads arrays to TCZYX and clamps chunks to data shape.
"""

class ArrayAndChunkPrep:
    def __init__(self, chunk_size: List[int], dim: int = 5) -> None:
        self.chunk_size = chunk_size
        self.dim = dim

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
        arr = self._pad_array_n_d(data)
        dataset_shape = self._compute_dataset_shape(arr)
        full_chunks = self._clamp_chunks(dataset_shape)
        return arr, dataset_shape, full_chunks