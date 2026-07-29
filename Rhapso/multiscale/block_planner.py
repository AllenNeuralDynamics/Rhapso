from typing import Tuple
import dask.array as da

"""
Computes a chunk-aligned block_shape_zyx for multiscale blocks.
"""

class BlockPlanner:
    @staticmethod
    def expand_chunks(chunks: Tuple[int, int, int], data_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        block_shape = []

        for chunk, size in zip(chunks, data_shape):
            aligned_limit = (size // chunk) * chunk
            block_shape.append(min(chunk * 3, aligned_limit))

        return tuple(block_shape)

    def get_block_shape(self, arr: da.Array) -> Tuple[int, int, int]:
        chunks_zyx = tuple(int(c) for c in arr.chunksize[-3:])
        data_shape_zyx = tuple(int(s) for s in arr.shape[-3:])
        return self.expand_chunks(chunks_zyx, data_shape_zyx)

    def run(self, arr: da.Array) -> Tuple[int, int, int]:
        return self.get_block_shape(arr)
    