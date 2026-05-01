import numpy as np
import zarr
import fsspec
from scipy.ndimage import map_coordinates
import os

# TSV_HEADER = (
#     "x0\ty0\tz0\tx1\ty1\tz1\t"
#     "gx0\tgy0\tgz0\tgx1\tgy1\tgz1\t"
#     "cand\tmin\tmax\tmean\tnonzero\tzero\tfrac_nonzero\tsumU16\ttotal\n"
# )

class FuseCell:
    def __init__(self, image_instructions, blocks, per_view_transforms, output_path,
                 grid_block, fusion_min_global, fusion_max_global):
        self.image_instructions = image_instructions
        self.blocks = blocks
        self.per_view_transforms = per_view_transforms
        self.output_path = output_path
        self.grid_block = grid_block
        self.fusion_min_global = fusion_min_global
        self.fusion_max_global = fusion_max_global

    # def append_block_tsv(self, tsv_path, row_values):
    #     # Write header once if file is new/empty
    #     need_header = (not os.path.exists(tsv_path)) or (os.path.getsize(tsv_path) == 0)
    #     with open(tsv_path, "a", buffering=1024 * 1024) as f:
    #         if need_header:
    #             f.write(TSV_HEADER)
    #         f.write("\t".join(str(v) for v in row_values) + "\n")

    def _open_zarr_array(self, path: str, mode: str = "r"):
        path = path.rstrip("/") + "/0"
        store = fsspec.get_mapper(path)
        return zarr.open(store, mode=mode)

    def _open_view_dataset(self, view_id, mode="r"):
        path = self.per_view_transforms[view_id]["path"].rstrip("/") + "/0"
        store = fsspec.get_mapper(path)
        return zarr.open(store, mode=mode)
    
    def write_block(self, fused_block_zyx, out_offset_xyz):
        out = self._open_zarr_array(self.output_path, mode="r+")
        x0, y0, z0 = map(int, out_offset_xyz)
        z_len, y_len, x_len = fused_block_zyx.shape

        out[0, 0,
            z0:z0 + z_len,
            y0:y0 + y_len,
            x0:x0 + x_len] = fused_block_zyx.astype(np.uint16, copy=False)
        
    def _ramp(self, v, b0, b1, b2, b3):
        out = np.zeros_like(v, dtype=np.float32)

        m = (v >= b0) & (v < b1)
        if b1 > b0:
            out[m] = (v[m] - b0) / (b1 - b0)

        m = (v >= b1) & (v <= b2)
        out[m] = 1.0

        m = (v > b2) & (v <= b3)
        if b3 > b2:
            out[m] = (b3 - v[m]) / (b3 - b2)

        return np.clip(out, 0.0, 1.0)

    def _build_fused_points(self, block_min, block_max):
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

    def _evaluate_avg_blend_weights(self, image_instructions, src_pts, nx, ny, nz):
        """
        Evaluate AVG_BLEND weights at the sampled source positions.
        image_instructions is your dict with b0,b1,b2,b3.
        """
        wx = self._ramp(
            src_pts[:, 0],
            image_instructions["b0"][0],
            image_instructions["b1"][0],
            image_instructions["b2"][0],
            image_instructions["b3"][0],
        )
        wy = self._ramp(
            src_pts[:, 1],
            image_instructions["b0"][1],
            image_instructions["b1"][1],
            image_instructions["b2"][1],
            image_instructions["b3"][1],
        )
        wz = self._ramp(
            src_pts[:, 2],
            image_instructions["b0"][2],
            image_instructions["b1"][2],
            image_instructions["b2"][2],
            image_instructions["b3"][2],
        )

        w = (wx * wy * wz).astype(np.float32)
        w_xyz = w.reshape((nx, ny, nz))
        w_zyx = w_xyz.transpose(2, 1, 0)

        return w_zyx
    
    def _load_source_chunk_for_view(self, view_key, src_min_xyz, src_max_xyz):
        """
        Load a source chunk for a view given SOURCE-space inclusive XYZ bounds.
        """
        x0, y0, z0 = map(int, src_min_xyz)
        x1, y1, z1 = map(int, src_max_xyz)

        arr = self._open_view_dataset(view_key, mode="r")  # [t,c,z,y,x]
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

    def render_fused_block(self, images_dict, final_blocks, block_min, block_max):
        block_min = np.asarray(block_min, dtype=np.int64)
        block_max = np.asarray(block_max, dtype=np.int64)

        pts, nx, ny, nz = self._build_fused_points(block_min, block_max)

        out_shape_zyx = (nz, ny, nx)
        numerator = np.zeros(out_shape_zyx, dtype=np.float32)
        denominator = np.zeros(out_shape_zyx, dtype=np.float32)

        for view_key, block_info in final_blocks.items():
            instr = images_dict[view_key]
            inv_t = np.asarray(instr["inv_t"], dtype=np.float64)

            # Compute SOURCE points for this fused block
            src_pts = (inv_t @ pts.T).T[:, :3]  # absolute source XYZ floats

            # Source bbox needed to load (use floor/ceil so we contain all sample points)
            src_min = np.floor(src_pts.min(axis=0)).astype(np.int64)
            src_max = np.ceil(src_pts.max(axis=0)).astype(np.int64)

            src_chunk_zyx, src_min_xyz = self._load_source_chunk_for_view(view_key, src_min, src_max)
            if src_chunk_zyx.size == 0:
                continue

            # Now sample using the SAME src_pts we already computed
            rel_x = src_pts[:, 0] - src_min_xyz[0]
            rel_y = src_pts[:, 1] - src_min_xyz[1]
            rel_z = src_pts[:, 2] - src_min_xyz[2]

            sampled = map_coordinates(
                src_chunk_zyx,
                [rel_z, rel_y, rel_x],
                order=1,
                mode="nearest",
            ).astype(np.float32)

            sampled_xyz = sampled.reshape((nx, ny, nz))
            sampled_zyx = sampled_xyz.transpose(2, 1, 0)

            weights_zyx = self._evaluate_avg_blend_weights(
                image_instructions=instr,
                src_pts=src_pts,
                nx=nx,
                ny=ny,
                nz=nz,
            )

            numerator += sampled_zyx * weights_zyx
            denominator += weights_zyx

        fused_block = np.zeros_like(numerator, dtype=np.float32)
        valid = denominator > 0
        fused_block[valid] = numerator[valid] / denominator[valid]

        denom_max = float(denominator.max()) if denominator.size else 0.0

        return fused_block, denom_max

    def run(self):
        block_min = self.grid_block[0]
        block_max = [0] * len(block_min) 

        interval = (self.fusion_max_global - self.fusion_min_global)
        for d in range(len(block_min)):
            block_max[d] = min(int(interval[d]), int(block_min[d] + self.grid_block[1][d] - 1))

        fused_block, denom_max = self.render_fused_block(
            images_dict=self.image_instructions,
            final_blocks=self.blocks,
            block_min=block_min,
            block_max=block_max,
        )

        fused_u16 = np.clip(np.rint(fused_block), 0, 65535).astype(np.uint16)

        # ----- stats (match Java TSV semantics) -----
        # vmin = float(fused_u16.min())
        # vmax = float(fused_u16.max())
        # mean = float(fused_u16.mean())

        # total = int(fused_u16.size)
        # nonzero = int(np.count_nonzero(fused_u16))
        # zero = total - nonzero
        # frac_nonzero = (nonzero / total) if total else 0.0
        # sumU16 = int(fused_u16.astype(np.uint64).sum())

        # cand = int(len(self.blocks))  # candidate views for this block in your pipeline

        # # ----- bounds -----
        # # local bounds (x0..z1)
        # x0, y0, z0 = (int(block_min[0]), int(block_min[1]), int(block_min[2]))
        # x1, y1, z1 = (int(block_max[0]), int(block_max[1]), int(block_max[2]))

        # # global bounds: add fusion_min_global (your bbMin equivalent)
        # # NOTE: block_min/max in your code are "fused-local"; global = local + fusion_min_global
        # bbMin = np.asarray(self.fusion_min_global, dtype=np.int64)

        # gmin = np.asarray(block_min, dtype=np.int64) + bbMin
        # gmax = np.asarray(block_max, dtype=np.int64) + bbMin

        # gx0, gy0, gz0 = (int(gmin[0]), int(gmin[1]), int(gmin[2]))
        # gx1, gy1, gz1 = (int(gmax[0]), int(gmax[1]), int(gmax[2]))

        self.write_block(fused_u16, self.grid_block[0])

        # return [
        #     x0, y0, z0, x1, y1, z1,
        #     gx0, gy0, gz0, gx1, gy1, gz1,
        #     cand, vmin, vmax, mean,
        #     nonzero, zero, frac_nonzero,
        #     sumU16, total
        # ]