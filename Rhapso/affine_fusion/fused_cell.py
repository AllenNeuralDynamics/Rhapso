import numpy as np
import zarr
import fsspec

"""
AVG_BLEND fusion
"""

class FusedCell:
    def __init__(
        self,
        per_view_transforms,
        overlapping_views,
        fused_min, fused_max,                
        output_path,
        out_offset_xyz,                  
        fusion_min_global,                 
        output_shape_zyx,                    
        dtype=np.uint16,
        chunks_5d=(1, 1, 128, 256, 256),
        border_zyx=(0.0, 0.0, 0.0),
        blending_zyx=(40.0, 40.0, 40.0),
    ):
        self.per_view_transforms = per_view_transforms
        self.overlapping_views = overlapping_views
        self.fused_min = np.asarray(fused_min, dtype=np.int64)
        self.fused_max = np.asarray(fused_max, dtype=np.int64)
        self.output_path = output_path
        self.out_offset_xyz = np.asarray(out_offset_xyz, dtype=np.int64)
        self.fusion_min_global = np.asarray(fusion_min_global, dtype=np.int64)
        self.output_shape_zyx = tuple(int(x) for x in output_shape_zyx)
        self.dtype = np.dtype(dtype)
        self.chunks_5d = tuple(int(x) for x in chunks_5d)
        self.border_zyx = np.asarray(border_zyx, dtype=np.float32)
        self.blending_zyx = np.asarray(blending_zyx, dtype=np.float32)

        self._T_world_to_local = np.eye(4, dtype=np.float64)
        self._T_world_to_local[0, 3] = -float(self.fusion_min_global[0])
        self._T_world_to_local[1, 3] = -float(self.fusion_min_global[1])
        self._T_world_to_local[2, 3] = -float(self.fusion_min_global[2])

    def _open_zarr_array(self, path: str, mode: str = "r"):
        path = path.rstrip("/") + "/0"
        store = fsspec.get_mapper(path)
        return zarr.open(store, mode=mode)

    def _open_view_dataset(self, view_id, mode="r"):
        path = self.per_view_transforms[view_id]["path"].rstrip("/") + "/0"
        store = fsspec.get_mapper(path)
        return zarr.open(store, mode=mode)

    def _fetch_view_crop_zyx(self, view_id, pts_zyx: np.ndarray):
        arr = self._open_view_dataset(view_id, mode="r")

        if arr.ndim != 5:
            raise ValueError(f"Unexpected source zarr ndim={arr.ndim} for {self.per_view_transforms[view_id]['path']}")

        Z, Y, X = int(arr.shape[2]), int(arr.shape[3]), int(arr.shape[4])
        src_shape_zyx = (Z, Y, X)

        z = pts_zyx[:, 0]
        y = pts_zyx[:, 1]
        x = pts_zyx[:, 2]

        m = (z >= -1) & (z <= Z) & (y >= -1) & (y <= Y) & (x >= -1) & (x <= X)
        if not np.any(m):
            return np.zeros((0, 0, 0), dtype=np.float32), np.array([0, 0, 0], dtype=np.int64), src_shape_zyx

        z = z[m]; y = y[m]; x = x[m]

        z0 = int(np.floor(z.min())) - 1
        y0 = int(np.floor(y.min())) - 1
        x0 = int(np.floor(x.min())) - 1
        z1 = int(np.ceil(z.max())) + 2
        y1 = int(np.ceil(y.max())) + 2
        x1 = int(np.ceil(x.max())) + 2

        z0 = max(z0, 0); y0 = max(y0, 0); x0 = max(x0, 0)
        z1 = min(z1, Z); y1 = min(y1, Y); x1 = min(x1, X)

        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return np.zeros((0, 0, 0), dtype=np.float32), np.array([z0, y0, x0], dtype=np.int64), src_shape_zyx

        vol = np.asarray(arr[0, 0, z0:z1, y0:y1, x0:x1])
        if vol.size == 0:
            raise ValueError(f"Empty source zarr data for {self.per_view_transforms[view_id]['path']} (shape={vol.shape})")

        return vol, np.array([z0, y0, x0], dtype=np.int64), src_shape_zyx

    def _trilinear_sample(self, vol_zyx: np.ndarray, pts_zyx: np.ndarray):
        z = pts_zyx[:, 0]
        y = pts_zyx[:, 1]
        x = pts_zyx[:, 2]

        z0 = np.floor(z).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x0 = np.floor(x).astype(np.int64)
        z1 = z0 + 1
        y1 = y0 + 1
        x1 = x0 + 1

        Z, Y, X = vol_zyx.shape
        valid = (z0 >= 0) & (y0 >= 0) & (x0 >= 0) & (z1 < Z) & (y1 < Y) & (x1 < X)

        vals = np.zeros((pts_zyx.shape[0],), dtype=np.float32)
        if not np.any(valid):
            return vals, valid

        zv = z[valid]; yv = y[valid]; xv = x[valid]
        z0v = z0[valid]; y0v = y0[valid]; x0v = x0[valid]
        z1v = z1[valid]; y1v = y1[valid]; x1v = x1[valid]

        dz = (zv - z0v).astype(np.float32)
        dy = (yv - y0v).astype(np.float32)
        dx = (xv - x0v).astype(np.float32)

        c000 = vol_zyx[z0v, y0v, x0v].astype(np.float32, copy=False)
        c001 = vol_zyx[z0v, y0v, x1v].astype(np.float32, copy=False)
        c010 = vol_zyx[z0v, y1v, x0v].astype(np.float32, copy=False)
        c011 = vol_zyx[z0v, y1v, x1v].astype(np.float32, copy=False)
        c100 = vol_zyx[z1v, y0v, x0v].astype(np.float32, copy=False)
        c101 = vol_zyx[z1v, y0v, x1v].astype(np.float32, copy=False)
        c110 = vol_zyx[z1v, y1v, x0v].astype(np.float32, copy=False)
        c111 = vol_zyx[z1v, y1v, x1v].astype(np.float32, copy=False)

        c00 = c000 * (1 - dx) + c001 * dx
        c01 = c010 * (1 - dx) + c011 * dx
        c10 = c100 * (1 - dx) + c101 * dx
        c11 = c110 * (1 - dx) + c111 * dx

        c0 = c00 * (1 - dy) + c01 * dy
        c1 = c10 * (1 - dy) + c11 * dy

        out = c0 * (1 - dz) + c1 * dz
        vals[valid] = out
        return vals, valid

    def _blend_weights_source(self, pts_zyx: np.ndarray, src_shape_zyx):
        Z, Y, X = src_shape_zyx
        z = pts_zyx[:, 0].astype(np.float32)
        y = pts_zyx[:, 1].astype(np.float32)
        x = pts_zyx[:, 2].astype(np.float32)

        b0z, b0y, b0x = self.border_zyx
        blz, bly, blx = self.blending_zyx

        def ramp(pos, dim_len, b0, bl):
            b3 = float(dim_len - 1) - float(b0)
            b1 = float(b0) + float(bl)
            b2 = float(dim_len - 1) - float(b0) - float(bl)

            w = np.ones_like(pos, dtype=np.float32)
            w = np.where((pos < 0.0) | (pos > float(dim_len - 1)), 0.0, w)
            w = np.where((pos < float(b0)) | (pos > b3), 0.0, w)

            if bl > 0:
                win = (pos - float(b0)) / float(bl)
                w = np.where((pos >= float(b0)) & (pos < b1), np.clip(win, 0.0, 1.0), w)
                wout = (b3 - pos) / float(bl)
                w = np.where((pos > b2) & (pos <= b3), np.clip(wout, 0.0, 1.0), w)

            return w

        wz = ramp(z, Z, b0z, blz)
        wy = ramp(y, Y, b0y, bly)
        wx = ramp(x, X, b0x, blx)

        return (wz * wy * wx).astype(np.float32, copy=False)

    def fuse_avg_blend(self):
        out_shape_xyz = self.fused_max - self.fused_min + 1
        sx, sy, sz = map(int, out_shape_xyz)

        local_min = self.fused_min - self.fusion_min_global
        local_max = self.fused_max - self.fusion_min_global

        x = (np.arange(local_min[0], local_max[0] + 1, dtype=np.float64) + 0.5)
        y = (np.arange(local_min[1], local_max[1] + 1, dtype=np.float64) + 0.5)
        z = (np.arange(local_min[2], local_max[2] + 1, dtype=np.float64) + 0.5)

        yy, xx, zz = np.meshgrid(y, x, z, indexing="ij")  # (Y,X,Z)
        local_xyz = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        local_h = np.concatenate([local_xyz, np.ones((local_xyz.shape[0], 1), dtype=np.float64)], axis=1)

        sum_yxz = np.zeros((sy, sx, sz), dtype=np.float32)
        wsum_yxz = np.zeros((sy, sx, sz), dtype=np.float32)

        for view_id in self.overlapping_views:
            T_world = np.asarray(self.per_view_transforms[view_id]["transform"], dtype=np.float64)
            T_local = self._T_world_to_local @ T_world
            invT_local = np.linalg.inv(T_local)

            src_h = local_h @ invT_local.T
            src_xyz = src_h[:, :3] - 0.5

            pts_zyx = np.stack([src_xyz[:, 2], src_xyz[:, 1], src_xyz[:, 0]], axis=1)

            vol_crop_zyx, crop0, src_shape_zyx = self._fetch_view_crop_zyx(view_id, pts_zyx)
            if vol_crop_zyx.size == 0:
                continue

            w = self._blend_weights_source(pts_zyx, src_shape_zyx)

            pts_local_zyx = pts_zyx - crop0[None, :]
            vals, valid = self._trilinear_sample(vol_crop_zyx, pts_local_zyx)

            wv = (w * valid.astype(np.float32)).astype(np.float32, copy=False)

            vals_img = vals.reshape((sy, sx, sz))
            w_img = wv.reshape((sy, sx, sz))

            sum_yxz += vals_img * w_img
            wsum_yxz += w_img

        denom = np.where(wsum_yxz > 0, wsum_yxz, 1.0).astype(np.float32, copy=False)
        fused_yxz = sum_yxz / denom
        fused_zyx = np.transpose(fused_yxz, (2, 0, 1)).astype(np.float32, copy=False)
        return fused_zyx

    def write_block(self, fused_block_zyx):
        out = self._open_zarr_array(self.output_path, mode="r+")

        x0, y0, z0 = map(int, self.out_offset_xyz)
        z_len, y_len, x_len = fused_block_zyx.shape

        out[0, 0, z0:z0 + z_len, y0:y0 + y_len, x0:x0 + x_len] = fused_block_zyx.astype(self.dtype, copy=False)

    def run(self):
        fused_block_zyx = self.fuse_avg_blend()
        self.write_block(fused_block_zyx)
        return fused_block_zyx