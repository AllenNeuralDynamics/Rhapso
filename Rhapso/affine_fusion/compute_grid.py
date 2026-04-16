import numpy as np

"""
Compute fusion grid
"""

class ComputeGrid():
    def __init__(self, dims, block_size, block_scale):
        self.dims = dims
        self.block_size = block_size
        self.block_scale = block_scale

    def grid_create(self):
        dims = np.asarray(self.dims, dtype=np.int64)
        out_bs = np.asarray(self.block_size, dtype=np.int64)
        bpj = np.asarray(self.block_scale, dtype=np.int64)
        gbs = out_bs * bpj  # computeBlockSize

        n = dims.shape[0]
        offset = np.zeros(n, dtype=np.int64)
        grid_blocks = []

        while True:
            # cropBlockDimensions(dimensions, offset, outBlockSize, gridBlockSize, cropped, gridPos)
            cropped = np.minimum(gbs, dims - offset)
            grid_pos = offset // out_bs

            grid_blocks.append([
                offset.copy(),
                cropped.copy(),
                grid_pos.copy(),
            ])

            advanced = False
            for d in range(n):
                offset[d] += gbs[d]
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