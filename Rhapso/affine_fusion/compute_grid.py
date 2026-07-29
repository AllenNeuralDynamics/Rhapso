import numpy as np

"""
Compute fusion grid
"""

class ComputeGrid():
    def __init__(self, dims, block_size):
        self.dims = dims
        self.block_size = block_size

    def grid_create(self):
        dims = np.asarray(self.dims, dtype=np.int64)
        grid_block_size = np.asarray(self.block_size, dtype=np.int64)

        n = dims.shape[0]
        offset = np.zeros(n, dtype=np.int64)
        grid_blocks = []

        while True:
            cropped = np.minimum(grid_block_size, dims - offset)
            grid_pos = offset // grid_block_size

            grid_blocks.append([
                offset.copy(),
                cropped.copy(),
                grid_pos.copy(),
            ])

            advanced = False
            for d in range(n):
                offset[d] += grid_block_size[d]
                if offset[d] < dims[d]:
                    advanced = True
                    break
                offset[d] = 0

            if not advanced:
                break

        return grid_blocks
    
    def run(self):
        grid = self.grid_create()

        return grid