import Rhapso.fusion.geometry as geometry
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


# class BlendingModule:
#     """
#     Minimal interface for modular blending function.
#     Subclass can define arbitrary constructors/attributes/members as necessary.
#     """

#     def blend(
#         self, chunks: list[torch.Tensor], device: torch.device, kwargs={}
#     ) -> torch.Tensor:
#         """
#         chunks:
#             Chunks to blend into snowball_chunk
#         kwargs:
#             Extra keyword arguments
#         """

#         raise NotImplementedError(
#             "Please implement in BlendingModule subclass."
#         )


# class MaxProjection(BlendingModule):
#     """
#     Simplest blending implementation possible. No constructor needed.
#     """

#     def blend(
#         self, chunks: list[torch.Tensor], device: torch.device, kwargs={}
#     ) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         chunks: list of 3D tensors to combine. Contains 2 or more elements.

#         Returns
#         -------
#         fused_chunk: combined chunk
#         """

#         fused_chunk = chunks[0].to(device)
#         for c in chunks[1:]:
#             c = c.to(device)
#             fused_chunk = torch.maximum(fused_chunk, c)

#         return fused_chunk


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

    # def blend(
    #     self, chunks: list[torch.Tensor], device: torch.device, kwargs={}
    # ) -> torch.Tensor:
    #     """
    #     Parameters
    #     ----------
    #     snowball chunk: 5d tensor in 11zyx order
    #     chunks: 5d tensor(s) in 11zyx order
    #     kwargs:
    #         chunk_tile_ids:
    #             list of tile ids corresponding to each chunk
    #         cell_box:
    #             cell AABB in output volume/absolute coordinates

    #     Returns
    #     -------
    #     fused_chunk: combined chunk
    #     """

    #     # Trivial no blending case -- non-overlaping region.
    #     if len(chunks) == 1:
    #         return chunks[0]

    #     # For 2+ chunks, within an overlapping region:
    #     chunk_tile_ids = kwargs["chunk_tile_ids"]
    #     cell_box = kwargs["cell_box"]

    #     # Calculate local weight masks
    #     local_weights: list[torch.Tensor] = []
    #     total_weight: torch.Tensor = torch.zeros(chunks[0].shape)
    #     for tile_id, chunk in zip(chunk_tile_ids, chunks):
    #         tile_aabb = self.tile_aabbs[tile_id]
    #         x_min = tile_aabb[4]
    #         cy = (tile_aabb[3] + tile_aabb[2]) / 2
    #         cx = (tile_aabb[5] + tile_aabb[4]) / 2

    #         z_indices = torch.arange(cell_box[0], cell_box[1], step=1) + 0.5
    #         y_indices = torch.arange(cell_box[2], cell_box[3], step=1) + 0.5
    #         x_indices = torch.arange(cell_box[4], cell_box[5], step=1) + 0.5

    #         z_grid, y_grid, x_grid = torch.meshgrid(
    #             z_indices,
    #             y_indices,
    #             x_indices,
    #             indexing="ij",  # {z_grid, y_grid, x_grid} are 3D Tensors
    #         )

    #         # Weight formula:
    #         # 1) Apply pyramid function wrt to center of square tile.
    #         # For each incoming chunk, a chunk may only have partial signal,
    #         # representing cells that lie between two tiles.
    #         # 2) After calculating pyramid weights, confine weights to actual boundary
    #         # of image, represented by position of non-zero values in chunk.
    #         weights = (cx - x_min) - torch.max(
    #             torch.abs(x_grid - cx), torch.abs(y_grid - cy)
    #         )
    #         signal_mask = torch.clamp(chunk, 0, 1)
    #         inbound_weights = weights * signal_mask

    #         local_weights.append(inbound_weights)
    #         total_weight += inbound_weights

    #     # Calculate fused chunk
    #     fused_chunk = torch.zeros(chunks[0].shape)

    #     for w, c in zip(local_weights, chunks):
    #         w /= total_weight
    #         w = w.to(device)
    #         c = c.to(device)
    #         fused_chunk += w * c

    #     return fused_chunk

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

    # Cleaner Code: 
    # def blend(self, chunks: list[torch.Tensor], device: str, kwargs={}) -> torch.Tensor:
    #     if len(chunks) == 1:
    #         return chunks[0]
        
    #     chunk_tile_ids = kwargs["chunk_tile_ids"]
    #     cell_box = kwargs["cell_box"]
        
    #     # Create coordinate grids once
    #     z_indices = torch.arange(cell_box[0], cell_box[1], step=1, device=device) + 0.5
    #     y_indices = torch.arange(cell_box[2], cell_box[3], step=1, device=device) + 0.5
    #     x_indices = torch.arange(cell_box[4], cell_box[5], step=1, device=device) + 0.5
    #     z_grid, y_grid, x_grid = torch.meshgrid(z_indices, y_indices, x_indices, indexing="ij")
        
    #     # Pre-allocate tensors
    #     chunks_tensor = torch.stack([chunk.to(device) for chunk in chunks])
    #     weights_tensor = torch.zeros((len(chunks),) + chunks[0].shape, device=device)
        
    #     # Vectorized weight calculation
    #     for i, tile_id in enumerate(chunk_tile_ids):
    #         tile_aabb = self.tile_aabbs[tile_id]
    #         x_min = tile_aabb[4]
    #         cy = (tile_aabb[3] + tile_aabb[2]) / 2
    #         cx = (tile_aabb[5] + tile_aabb[4]) / 2
            
    #         weights = (cx - x_min) - torch.max(
    #             torch.abs(x_grid - cx), torch.abs(y_grid - cy)
    #         )
    #         signal_mask = torch.clamp(chunks_tensor[i], 0, 1)
    #         weights_tensor[i] = weights * signal_mask
        
    #     # Normalize weights
    #     total_weight = weights_tensor.sum(dim=0, keepdim=True)
    #     normalized_weights = weights_tensor / total_weight
        
    #     # Final blending
    #     return (normalized_weights * chunks_tensor).sum(dim=0)