from typing import Tuple, Optional
import numpy as np
import dask.array as da

"""
Computes block_shape_zyx for multiscale blocks.
"""

class BlockPlanner:
    def __init__(self, target_block_size_mb: int = 409600, mode: str = "iso") -> None:
        self.target_block_size_bytes = int(target_block_size_mb * 1024**2)
        self.mode = mode

    @staticmethod
    def _get_size(shape: Tuple[int, ...], itemsize: int) -> int:
        if any(s <= 0 for s in shape):
            raise ValueError("shape must be > 0 in all dimensions")
        return int(np.prod(shape)) * itemsize

    def _closer_to_target(self, shape1: Tuple[int, ...], shape2: Tuple[int, ...], target_bytes: int, itemsize: int) -> Tuple[int, ...]:
        size1 = float(self._get_size(shape1, itemsize))
        size2 = float(self._get_size(shape2, itemsize))
        if abs(size1 - target_bytes) < abs(size2 - target_bytes):
            return shape1
        return shape2

    def expand_chunks(self, chunks: Tuple[int, int, int], data_shape: Tuple[int, int, int], target_size: int,
                      itemsize: int, mode: str) -> Tuple[int, int, int]:
        if any(c < 1 for c in chunks):
            raise ValueError("chunks must be >= 1 for all dimensions")
        if any(s < 1 for s in data_shape):
            raise ValueError("data_shape must be >= 1 for all dimensions")
        if any(c > s for c, s in zip(chunks, data_shape)):
            raise ValueError("chunks cannot be larger than data_shape in any dimension")
        if target_size <= 0:
            raise ValueError("target_size must be > 0")
        if itemsize <= 0:
            raise ValueError("itemsize must be > 0")

        if mode == "cycle":
            current = np.array(chunks, dtype=np.uint64)
            prev = current.copy()
            idx = 0
            ndims = len(current)

            while self._get_size(tuple(current), itemsize) < target_size:
                prev = current.copy()
                current[idx % ndims] = min(
                    data_shape[idx % ndims], current[idx % ndims] * 2
                )
                idx += 1
                if all(c >= s for c, s in zip(current, data_shape)):
                    break

            expanded = self._closer_to_target(
                tuple(int(d) for d in current),
                tuple(int(d) for d in prev),
                target_size,
                itemsize,
            )

        elif mode == "iso":
            initial = np.array(chunks, dtype=np.uint64)
            current = initial
            prev = current
            i = 2

            while self._get_size(tuple(current), itemsize) < target_size:
                prev = current
                current = initial * i
                current = (
                    min(data_shape[0], current[0]),
                    min(data_shape[1], current[1]),
                    min(data_shape[2], current[2]),
                )
                i += 1
                if all(c >= s for c, s in zip(current, data_shape)):
                    break

            expanded = self._closer_to_target(
                tuple(int(d) for d in current),
                tuple(int(d) for d in prev),
                target_size,
                itemsize,
            )
        else:
            raise ValueError(f"Invalid mode {mode}")

        return tuple(int(d) for d in expanded)

    def get_block_shape(self, arr: da.Array, chunks: Optional[Tuple[int, int, int]] = None) -> Tuple[int, int, int]:
        if chunks is None:
            # use the array's chunks, take the last 3 dims (ZYX)
            chunks = arr.chunksize if hasattr(arr, "chunksize") else arr.chunks
        chunks_zyx = tuple(int(c) for c in chunks[-3:])
        data_shape_zyx = tuple(int(s) for s in arr.shape[-3:])

        return self.expand_chunks(
            chunks=chunks_zyx,
            data_shape=data_shape_zyx,
            target_size=self.target_block_size_bytes,
            itemsize=arr.itemsize,
            mode=self.mode,
        )
    
    def run(self, arr: da.Array, chunks: Optional[Tuple[int, int, int]] = None) -> Tuple[int, int, int]:
        """
        Entry point
        """
        return self.get_block_shape(arr=arr, chunks=chunks)