import numpy as np
import zarr
import fsspec
import time
from scipy.ndimage import map_coordinates

"""
Core fusion implementation, sample chunks and blend into final cell
"""

class FuseCell:
    def __init__(self, image_instructions, blocks, per_view_transforms, output_path,
                 grid_block, fusion_min_global, fusion_max_global, overlap_strategy):
        self.image_instructions = image_instructions
        self.blocks = blocks
        self.per_view_transforms = per_view_transforms
        self.output_path = output_path
        self.grid_block = grid_block
        self.fusion_min_global = fusion_min_global
        self.fusion_max_global = fusion_max_global
        self.overlap_strategy = overlap_strategy

    def open_zarr_array(self, path: str, mode: str = "r"):
        path = path.rstrip("/") + "/0"
        store = fsspec.get_mapper(path)
        return zarr.open(store, mode=mode)

    def open_view_dataset(self, view_id, mode="r"):
        path = self.per_view_transforms[view_id]["path"].rstrip("/") + "/0"
        store = fsspec.get_mapper(path)
        return zarr.open(store, mode=mode)
    
    def write_block(self, fused_block_zyx, out_offset_xyz):
        out = self.open_zarr_array(self.output_path, mode="r+")
        x0, y0, z0 = map(int, out_offset_xyz)
        z_len, y_len, x_len = fused_block_zyx.shape

        out[0, 0,
            z0:z0 + z_len,
            y0:y0 + y_len,
            x0:x0 + x_len] = fused_block_zyx.astype(np.uint16, copy=False)
        
    def ramp(self, v, b0, b1, b2, b3):
        inv_left  = 1.0 / max(b1 - b0, 1e-9)
        inv_right = 1.0 / max(b3 - b2, 1e-9)
        left  = np.clip((v - b0) * inv_left,  0.0, 1.0)
        right = np.clip((b3 - v) * inv_right, 0.0, 1.0)
        return np.minimum(left, right).astype(np.float32, copy=False)

    def build_fused_points(self, block_min, block_max):
        """
        Build fused-local XYZ coordinates for the requested output block.
        """
        xs = np.arange(block_min[0], block_max[0] + 1, dtype=np.float32)
        ys = np.arange(block_min[1], block_max[1] + 1, dtype=np.float32)
        zs = np.arange(block_min[2], block_max[2] + 1, dtype=np.float32)

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")  # XYZ
        pts = np.stack(
            [X.ravel(), Y.ravel(), Z.ravel(), np.ones(X.size, dtype=np.float32)],
            axis=1
        )

        return pts, len(xs), len(ys), len(zs)
    
    def evaluate_avg_blend_weights_xyz(self, image_instructions, SX, SY, SZ):
        b0, b1, b2, b3 = (image_instructions[k] for k in ("b0", "b1", "b2", "b3"))
        wx = self.ramp(SX, b0[0], b1[0], b2[0], b3[0])
        wy = self.ramp(SY, b0[1], b1[1], b2[1], b3[1])
        wz = self.ramp(SZ, b0[2], b1[2], b2[2], b3[2])
        return (wx * wy * wz).astype(np.float32, copy=False)  
    
    def load_source_chunk_for_view(self, view_key, src_min_xyz, src_max_xyz):
        """
        Load a source chunk for a view given SOURCE-space inclusive XYZ bounds.
        """
        x0, y0, z0 = map(int, src_min_xyz)
        x1, y1, z1 = map(int, src_max_xyz)

        arr = self.open_view_dataset(view_key, mode="r")  # [t,c,z,y,x]
        az, ay, ax = map(int, arr.shape[-3:])

        # clip to dataset bounds
        cx0 = max(0, min(x0, ax - 1))
        cy0 = max(0, min(y0, ay - 1))
        cz0 = max(0, min(z0, az - 1))

        cx1 = max(0, min(x1, ax - 1))
        cy1 = max(0, min(y1, ay - 1))
        cz1 = max(0, min(z1, az - 1))

        if (cx1 < cx0) or (cy1 < cy0) or (cz1 < cz0):
            return np.zeros((0, 0, 0), dtype=np.float32), np.array([cx0, cy0, cz0], dtype=np.int64)

        chunk = arr[0, 0, cz0:cz1 + 1, cy0:cy1 + 1, cx0:cx1 + 1]
        chunk_np = np.asarray(chunk, dtype=np.float32)

        if chunk_np.size == 0:
            return chunk_np, np.array([cx0, cy0, cz0], dtype=np.int64)

        return chunk_np, np.array([cx0, cy0, cz0], dtype=np.int64)
    
    def evaluate_mask_xyz(self, instr, SX, SY, SZ):
        # Hard 0/1 mask: inside border => 1, otherwise 0
        b0 = instr["b0"]
        b3 = instr["b3"]
        mx = (SX >= b0[0]) & (SX <= b3[0])
        my = (SY >= b0[1]) & (SY <= b3[1])
        mz = (SZ >= b0[2]) & (SZ <= b3[2])
        return mx & my & mz
    
    def lowest_view_wins(self, out_shape_zyx, final_blocks, images_dict, X, Y, Z, max_slab_n, slab, nz, ny, nx):
        """
        Lowest view wins / first writer wins.

        For split views:
        - SX/SY/SZ are split-local coords used for mask evaluation.
        - SX_read/SY_read/SZ_read are full old-tile coords used to read/sample the source zarr.
        """
        out = np.zeros(out_shape_zyx, dtype=np.float32)
        filled = np.zeros(out_shape_zyx, dtype=bool)

        # Iterate views in lowest-first order
        for view_key in sorted(final_blocks.keys()):
            split_def = self.per_view_transforms[view_key].get("split_def")

            if split_def is not None:
                split_min = np.asarray(split_def["split_min_xyz"], dtype=np.float32)
            else:
                split_min = np.array([0, 0, 0], dtype=np.float32)

            instr = images_dict[view_key]
            inv_t = np.asarray(instr["inv_t"], dtype=np.float32)
            A = inv_t[:3, :3]
            t = inv_t[:3, 3]

            # These are split-local coords if this is a split view
            SX = A[0, 0] * X + A[0, 1] * Y + A[0, 2] * Z + t[0]
            SY = A[1, 0] * X + A[1, 1] * Y + A[1, 2] * Z + t[1]
            SZ = A[2, 0] * X + A[2, 1] * Y + A[2, 2] * Z + t[2]

            # These are the coords actually used to read from the OLD tile zarr
            SX_read = SX + split_min[0]
            SY_read = SY + split_min[1]
            SZ_read = SZ + split_min[2]

            src_min = np.floor([SX_read.min(), SY_read.min(), SZ_read.min()]).astype(np.int64)
            src_max = np.ceil([SX_read.max(), SY_read.max(), SZ_read.max()]).astype(np.int64)

            src_chunk_zyx, src_min_xyz = self.load_source_chunk_for_view(view_key, src_min, src_max)
            if src_chunk_zyx.size == 0:
                del SX, SY, SZ, SX_read, SY_read, SZ_read
                continue

            # Mask should stay in split-local coords
            mask_zyx = self.evaluate_mask_xyz(instr, SX, SY, SZ)

            # Convert READ coords to chunk-relative in-place
            SX_read -= src_min_xyz[0]
            SY_read -= src_min_xyz[1]
            SZ_read -= src_min_xyz[2]

            out_buf = np.empty(max_slab_n, dtype=np.float32)

            for z0 in range(0, nz, slab):
                z1 = min(nz, z0 + slab)
                n_slab = (z1 - z0) * ny * nx

                map_coordinates(
                    src_chunk_zyx,
                    [SZ_read[z0:z1].ravel(), SY_read[z0:z1].ravel(), SX_read[z0:z1].ravel()],
                    order=1,
                    mode="nearest",
                    prefilter=False,
                    output=out_buf[:n_slab],
                )

                sampled_zyx = out_buf[:n_slab].reshape((z1 - z0, ny, nx))

                # First-writer-wins within split-local mask
                sel = mask_zyx[z0:z1] & (~filled[z0:z1])
                out[z0:z1][sel] = sampled_zyx[sel]
                filled[z0:z1][sel] = True

            del SX, SY, SZ, SX_read, SY_read, SZ_read, mask_zyx, out_buf

        return out
    
    def avg_blend(self, final_blocks, images_dict, X, Y, Z, max_slab_n, slab, nz, ny, nx, numerator, denominator):
        """
        Avg blend
        """
        for view_key, _ in final_blocks.items():
            split_def = self.per_view_transforms[view_key].get("split_def")

            if split_def is not None:
                split_min = np.asarray(split_def["split_min_xyz"], dtype=np.float32)
            else:
                split_min = np.array([0, 0, 0], dtype=np.float32)

            instr = images_dict[view_key]
            inv_t = np.asarray(instr["inv_t"], dtype=np.float32)
            A = inv_t[:3, :3]
            t = inv_t[:3, 3]

            # These are split-local coords if this is a split view
            SX = A[0, 0] * X + A[0, 1] * Y + A[0, 2] * Z + t[0]
            SY = A[1, 0] * X + A[1, 1] * Y + A[1, 2] * Z + t[1]
            SZ = A[2, 0] * X + A[2, 1] * Y + A[2, 2] * Z + t[2]

            # These are the coords actually used to read from the OLD tile zarr
            SX_read = SX + split_min[0]
            SY_read = SY + split_min[1]
            SZ_read = SZ + split_min[2]

            src_min = np.floor([SX_read.min(), SY_read.min(), SZ_read.min()]).astype(np.int64)
            src_max = np.ceil([SX_read.max(), SY_read.max(), SZ_read.max()]).astype(np.int64)

            src_chunk_zyx, src_min_xyz = self.load_source_chunk_for_view(view_key, src_min, src_max)
            if src_chunk_zyx.size == 0:
                del SX, SY, SZ, SX_read, SY_read, SZ_read
                continue

            # Weights should stay in split-local coords
            weights_zyx = self.evaluate_avg_blend_weights_xyz(instr, SX, SY, SZ)

            # Convert READ coords to chunk-relative in-place
            SX_read -= src_min_xyz[0]
            SY_read -= src_min_xyz[1]
            SZ_read -= src_min_xyz[2]

            out_buf = np.empty(max_slab_n, dtype=np.float32)

            for z0 in range(0, nz, slab):
                z1 = min(nz, z0 + slab)
                n_slab = (z1 - z0) * ny * nx

                map_coordinates(
                    src_chunk_zyx,
                    [SZ_read[z0:z1].ravel(), SY_read[z0:z1].ravel(), SX_read[z0:z1].ravel()],
                    order=1,
                    mode="nearest",
                    prefilter=False,
                    output=out_buf[:n_slab],
                )

                sampled_zyx = out_buf[:n_slab].reshape((z1 - z0, ny, nx))
                w_slab = weights_zyx[z0:z1]

                numerator[z0:z1] += sampled_zyx * w_slab
                denominator[z0:z1] += w_slab

            del SX, SY, SZ, SX_read, SY_read, SZ_read, weights_zyx, out_buf

        np.divide(numerator, denominator, out=numerator, where=denominator > 0)
        numerator[denominator == 0] = 0
        return numerator
    
    def render_fused_block(self, images_dict, final_blocks, block_min, block_max):
        block_min = np.asarray(block_min, dtype=np.int64)
        block_max = np.asarray(block_max, dtype=np.int64)

        xs = np.arange(block_min[0], block_max[0] + 1, dtype=np.float32)
        ys = np.arange(block_min[1], block_max[1] + 1, dtype=np.float32)
        zs = np.arange(block_min[2], block_max[2] + 1, dtype=np.float32)
        nx, ny, nz = len(xs), len(ys), len(zs)

        # Build coords in ZYX so z-slabs are contiguous
        Z = zs[:, None, None]   # (nz,1,1)
        Y = ys[None, :, None]   # (1,ny,1)
        X = xs[None, None, :]   # (1,1,nx)
        # Broadcast result shapes: (nz, ny, nx)

        out_shape_zyx = (nz, ny, nx)
        numerator = np.zeros(out_shape_zyx, dtype=np.float32)
        denominator = np.zeros(out_shape_zyx, dtype=np.float32)

        slab = 16
        max_slab_n = slab * ny * nx  # maximum elements per slab

        if self.overlap_strategy == "avg_blend":
            return self.avg_blend(final_blocks, images_dict, X, Y, Z, max_slab_n, slab, nz, ny, nx, numerator, denominator)
        
        elif self.overlap_strategy == "lowest_view_wins":
            return self.lowest_view_wins(out_shape_zyx, final_blocks, images_dict, X, Y, Z, max_slab_n, slab, nz, ny, nx)        

    def run(self):
        block_min = self.grid_block[0]
        block_max = [0] * len(block_min) 

        interval = (self.fusion_max_global - self.fusion_min_global)
        for d in range(len(block_min)):
            block_max[d] = min(int(interval[d]), int(block_min[d] + self.grid_block[1][d] - 1))

        fused_block = self.render_fused_block(
            images_dict=self.image_instructions,
            final_blocks=self.blocks,
            block_min=block_min,
            block_max=block_max,
        )

        fused_u16 = np.clip(np.rint(fused_block), 0, 65535).astype(np.uint16)
        self.write_block(fused_u16, self.grid_block[0])