import Rhapso.translation_fusion.geometry as geometry
import numpy as np

"""
Interface for generic blending.
"""

class BlendingModule:
    def blend(self, chunks: list[np.ndarray], kwargs=None) -> np.ndarray:
        raise NotImplementedError


class MaxProjection(BlendingModule):
    def blend(self, chunks: list[np.ndarray], kwargs=None) -> np.ndarray:
        # normalize shapes to (z,y,x)
        norm = []
        for c in chunks:
            c = np.asarray(c)
            if c.ndim == 5:
                c = c[0, 0]
            elif c.ndim != 3:
                raise ValueError(f"Expected (z,y,x) or (1,1,z,y,x), got {c.shape}")
            norm.append(c)

        fused = np.maximum.reduce(norm).astype(np.float32, copy=False)
        return fused[None, None, :, :, :]  # keep (1,1,z,y,x) like your pipeline

class WeightedLinearBlending(BlendingModule):
    """
    Linear Blending with distance-based weights.
    NOTE: Only supports translation-only registration on square tiles.
    To modify for affine registration:
    - Forward transform overlap weights into output volume.
    - Inverse transform for local weights.
    """

    def __init__(
        self,
        tile_aabbs: dict[int, geometry.AABB],
    ) -> None:
        super().__init__()
        """
        tile_aabbs: dict of tile_id -> AABB, defined in fusion initalization.
        """
        self.tile_aabbs = tile_aabbs

    def blend(self, chunks: list[np.ndarray], kwargs={}) -> np.ndarray:
        # Trivial no blending case
        if len(chunks) == 1:
            return chunks[0]

        chunk_tile_ids = kwargs["chunk_tile_ids"]
        cell_box = kwargs["cell_box"]  # (z0,z1,y0,y1,x0,x1)

        # Normalize chunk shapes to (z,y,x) for weighting
        chunks_zyx = []
        for c in chunks:
            c = np.asarray(c)
            if c.ndim == 5:
                c = c[0, 0]
            elif c.ndim != 3:
                raise ValueError(f"Expected chunk (z,y,x) or (1,1,z,y,x), got {c.shape}")
            chunks_zyx.append(c)

        z0, z1, y0, y1, x0, x1 = cell_box
        z_len = int(z1 - z0)
        y_len = int(y1 - y0)
        x_len = int(x1 - x0)

        # Build grids (only x/y are needed for weights; z is broadcast)
        y_idx = (np.arange(y0, y1, dtype=np.float32) + 0.5)
        x_idx = (np.arange(x0, x1, dtype=np.float32) + 0.5)
        y_grid, x_grid = np.meshgrid(y_idx, x_idx, indexing="ij")  # (y,x)

        local_weights = []
        total_weight = np.zeros((z_len, y_len, x_len), dtype=np.float32)

        for tile_id, chunk in zip(chunk_tile_ids, chunks_zyx):
            tile_aabb = self.tile_aabbs[tile_id]
            x_min = float(tile_aabb[4])
            cy = (float(tile_aabb[3]) + float(tile_aabb[2])) / 2.0
            cx = (float(tile_aabb[5]) + float(tile_aabb[4])) / 2.0

            # weights in (y,x)
            weights_yx = (cx - x_min) - np.maximum(np.abs(x_grid - cx), np.abs(y_grid - cy))
            # broadcast to (z,y,x)
            weights = np.broadcast_to(weights_yx[None, :, :], (z_len, y_len, x_len)).astype(np.float32, copy=False)

            # signal mask like torch.clamp(chunk,0,1) for uint16/int/float
            signal_mask = np.clip(chunk, 0, 1).astype(np.float32, copy=False)

            inbound_weights = weights * signal_mask
            local_weights.append(inbound_weights)
            total_weight += inbound_weights

        fused = np.zeros((z_len, y_len, x_len), dtype=np.float32)

        # Avoid divide-by-zero where total_weight==0 (should be rare, but safe)
        denom = np.where(total_weight > 0, total_weight, 1.0).astype(np.float32, copy=False)

        for w, c in zip(local_weights, chunks_zyx):
            fused += (w / denom) * c.astype(np.float32, copy=False)

        # Return in same shape style as your pipeline expects.
        return fused[None, None, :, :, :]