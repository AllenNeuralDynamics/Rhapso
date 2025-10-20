"""
Interface for generic blending.
"""

import dask.array as da
import numpy as np
import torch
import xmltodict

from collections import defaultdict

import aind_cloud_fusion.geometry as geometry


class BlendingModule:
    """
    Minimal interface for modular blending function.
    Subclass can define arbitrary constructors/attributes/members as necessary.
    """

    def blend(self,
              chunks: list[torch.Tensor],
              device: torch.device,
              kwargs = {}
    ) -> torch.Tensor:
        """
        chunks:
            Chunks to blend into snowball_chunk
        kwargs:
            Extra keyword arguments
        """

        raise NotImplementedError(
            "Please implement in BlendingModule subclass."
        )


class MaxProjection(BlendingModule):
    """
    Simplest blending implementation possible. No constructor needed.
    """

    def blend(self,
              chunks: list[torch.Tensor],
              device: torch.device,
              kwargs = {}
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        chunks: list of 3D tensors to combine. Contains 2 or more elements.

        Returns
        -------
        fused_chunk: combined chunk
        """

        fused_chunk = chunks[0].to(device)
        for c in chunks[1:]:
            c = c.to(device)
            fused_chunk = torch.maximum(fused_chunk, c)

        return fused_chunk


def get_overlap_regions(tile_layout: list[list[int]],
                        tile_aabbs: dict[int, geometry.AABB]
                        ) -> tuple[dict[int, list[int]],
                                   dict[int, geometry.AABB]]:
    """
    Input:
    tile_layout: array of tile ids arranged corresponding to stage coordinates
    tile_aabbs: dict of tile_id -> AABB, defined in fusion initalization.

    Output:
    tile_to_overlap_ids: Maps tile_id to associated overlap region id
    overlaps: Maps overlap_id to actual overlap region AABB

    Access pattern:
    tile_id -> overlap_id -> overlaps
    """

    def _get_overlap_aabb(aabb_1: geometry.AABB,
                          aabb_2: geometry.AABB):
        """
        Utility for finding overlapping regions between tiles and chunks.
        """

        # Check AABB's are colliding, meaning they colllide in all 3 axes
        assert (aabb_1[1] > aabb_2[0] and aabb_1[0] < aabb_2[1]) and \
               (aabb_1[3] > aabb_2[2] and aabb_1[2] < aabb_2[3]) and \
               (aabb_1[5] > aabb_2[4] and aabb_1[4] < aabb_2[5]), \
               f'Input AABBs are not colliding: {aabb_1=}, {aabb_2=}'

        # Between two colliding intervals A and B,
        # the overlap interval is the maximum of (A_min, B_min)
        # and the minimum of (A_max, B_max).
        overlap_aabb = (np.max([aabb_1[0], aabb_2[0]]),
                        np.min([aabb_1[1], aabb_2[1]]),
                        np.max([aabb_1[2], aabb_2[2]]),
                        np.min([aabb_1[3], aabb_2[3]]),
                        np.max([aabb_1[4], aabb_2[4]]),
                        np.min([aabb_1[5], aabb_2[5]]))

        return overlap_aabb

    # Output Data Structures
    tile_to_overlap_ids: dict[int, list[int]] = defaultdict(list)
    overlaps: dict[int, geometry.AABB] = {}

    # 1) Find all unique edges
    edges: list[tuple[int, int]] = []
    x_length = len(tile_layout)
    y_length = len(tile_layout[0])
    directions = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),         (0, 1),
                    (1, -1), (1, 0), (1, 1)]
    for x in range(x_length):
        for y in range(y_length):
            for (dx, dy) in directions:
                nx = x + dx
                ny = y + dy
                if (0 <= nx and nx < x_length and
                    0 <= ny and ny < y_length and   # Boundary conditions
                    tile_layout[x][y] != -1 and
                    tile_layout[nx][ny] != -1):  # Spacer conditions

                    id_1 = tile_layout[x][y]
                    id_2 = tile_layout[nx][ny]
                    e = tuple(sorted([id_1, id_2]))
                    edges.append(e)
    edges = sorted(list(set(edges)), key=lambda x: (x[0], x[1]))

    # 2) Find overlap regions
    overlap_id = 0
    for (id_1, id_2) in edges:
        aabb_1 = tile_aabbs[id_1]
        aabb_2 = tile_aabbs[id_2]

        try:
            o_aabb = _get_overlap_aabb(aabb_1, aabb_2)
        except:
            continue

        overlaps[overlap_id] = o_aabb
        tile_to_overlap_ids[id_1].append(overlap_id)
        tile_to_overlap_ids[id_2].append(overlap_id)
        overlap_id += 1

    return tile_to_overlap_ids, overlaps


class WeightedLinearBlending(BlendingModule):
    """
    Linear Blending with distance-based weights.
    NOTE: Only supports translation-only registration on square tiles. 
    To modify for affine registration:
    - Forward transform overlap weights into output volume.
    - Inverse transform for local weights.
    """

    def __init__(self,
                 tile_aabbs: dict[int, geometry.AABB],
                 ) -> None:
        super().__init__()
        """
        tile_aabbs: dict of tile_id -> AABB, defined in fusion initalization.
        """
        self.tile_aabbs = tile_aabbs

    def blend(self,
              chunks: list[torch.Tensor],
              device: torch.device,
              kwargs = {}
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        snowball chunk: 5d tensor in 11zyx order
        chunks: 5d tensor(s) in 11zyx order
        kwargs:
            chunk_tile_ids:
                list of tile ids corresponding to each chunk
            cell_box:
                cell AABB in output volume/absolute coordinates

        Returns
        -------
        fused_chunk: combined chunk
        """

        # Trivial no blending case -- non-overlaping region.
        if len(chunks) == 1:
            return chunks[0]

        # For 2+ chunks, within an overlapping region:
        chunk_tile_ids = kwargs['chunk_tile_ids']
        cell_box = kwargs['cell_box']

        # Calculate local weight masks
        local_weights: list[torch.Tensor] = []
        total_weight: torch.Tensor = torch.zeros(chunks[0].shape)
        for tile_id, chunk in zip(chunk_tile_ids, chunks):
            tile_aabb = self.tile_aabbs[tile_id]
            x_min = tile_aabb[4]
            cy = (tile_aabb[3] + tile_aabb[2]) / 2
            cx = (tile_aabb[5] + tile_aabb[4]) / 2

            z_indices = torch.arange(cell_box[0], cell_box[1], step=1) + 0.5
            y_indices = torch.arange(cell_box[2], cell_box[3], step=1) + 0.5
            x_indices = torch.arange(cell_box[4], cell_box[5], step=1) + 0.5

            z_grid, y_grid, x_grid = torch.meshgrid(
                z_indices, y_indices, x_indices, indexing="ij"  # {z_grid, y_grid, x_grid} are 3D Tensors
            )

            # Weight formula:
            # 1) Apply pyramid function wrt to center of square tile.
            # For each incoming chunk, a chunk may only have partial signal,
            # representing cells that lie between two tiles.
            # 2) After calculating pyramid weights, confine weights to actual boundary
            # of image, represented by position of non-zero values in chunk.
            weights = (cx - x_min) - torch.max(torch.abs(x_grid - cx), torch.abs(y_grid - cy))
            signal_mask = torch.clamp(chunk, 0, 1)
            inbound_weights = weights * signal_mask

            local_weights.append(inbound_weights)
            total_weight += inbound_weights

        # Calculate fused chunk
        fused_chunk = torch.zeros(chunks[0].shape)

        for w, c in zip(local_weights, chunks):
            w /= total_weight
            w = w.to(device)
            c = c.to(device)
            fused_chunk += (w * c)

        return fused_chunk

def parse_yx_tile_layout(xml_path: str) -> list[list[int]]:
    """
    Utility for parsing tile layout from a bigstitcher xml
    requested by some blending modules.

    tile_layout follows axis convention:
    +--- +x
    |
    |
    +y

    Tile ids in output tile_layout uses the same tile ids
    defined in the xml file. Spaces denoted with tile id '-1'.
    """

    # Parse stage positions
    with open(xml_path, "r") as file:
        data = xmltodict.parse(file.read())
    stage_positions_xyz: dict[int, tuple[float, float, float]] = {}
    for d in data['SpimData']['ViewRegistrations']['ViewRegistration']:
        tile_id = d['@setup']

        view_transform = d['ViewTransform']
        if isinstance(view_transform, list):
            view_transform = view_transform[-1]

        nums = [float(val) for val in view_transform["affine"].split(" ")]
        stage_positions_xyz[tile_id] = tuple(nums[3::4])

    # Calculate delta_x and delta_y
    positions_arr_xyz = np.array([pos for pos in stage_positions_xyz.values()])
    x_pos = list(set(positions_arr_xyz[:, 0]))
    x_pos = sorted(x_pos)
    delta_x = x_pos[1] - x_pos[0]
    y_pos = list(set(positions_arr_xyz[:, 1]))
    y_pos = sorted(y_pos)
    delta_y = y_pos[1] - y_pos[0]

    # Fill tile_layout
    tile_layout = np.ones((len(y_pos), len(x_pos))) * -1
    for tile_id, s_pos in stage_positions_xyz.items():
        ix = int(s_pos[0] / delta_x)
        iy = int(s_pos[1] / delta_y)

        tile_layout[iy, ix] = tile_id

    tile_layout = tile_layout.astype(int)

    return tile_layout
