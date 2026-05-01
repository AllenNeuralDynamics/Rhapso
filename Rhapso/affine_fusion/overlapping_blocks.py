import numpy as np

class OverlappingBlocks:
    def __init__(self, per_view_transforms, overlapping_views, super_block_offset, fused_min, fused_max, grid_block):
        self.per_view_transforms = per_view_transforms
        self.overlapping_views = overlapping_views
        self.super_block_offset = super_block_offset
        self.fused_min = np.asarray(fused_min, dtype=np.int64)
        self.fused_max = np.asarray(fused_max, dtype=np.int64)
        self.grid_block = grid_block
        self.expand = 1
        self.cell_dimensions = np.array([256, 256, 128], dtype=np.int64)

    def expand_interval(self):
        b = int(self.expand)
        expanded_min = self.fused_min.copy()
        expanded_max = self.fused_max.copy()
        expanded_min -= b
        expanded_max += b
        return expanded_min, expanded_max

    def intersect(self, a_min, a_max, b_min, b_max):
        inter_min = np.maximum(a_min, b_min)
        inter_max = np.minimum(a_max, b_max)
        return inter_min, inter_max

    def is_empty(self, inter_min, inter_max):
        return np.any(inter_min > inter_max)

    def overlaps(self, a_min, a_max, b_min, b_max):
        inter_min, inter_max = self.intersect(a_min, a_max, b_min, b_max)
        return not self.is_empty(inter_min, inter_max)

    def for_best_resolution(self):
        best_level = 0
        mipmap_transform = np.eye(4, dtype=np.float64)  # identity 
        return best_level, mipmap_transform

    def transformed_bounding_box_from_minmax(self, T4, interval_min_xyz, interval_max_xyz):
        """
        Transform an arbitrary local interval [min..max] by transforming 8 corners.
        Returns integer world bounds (floor min, ceil max), inclusive.
        """
        x0, y0, z0 = map(float, interval_min_xyz)
        x1, y1, z1 = map(float, interval_max_xyz)

        corners = np.array([
            [x0, y0, z0],
            [x0, y0, z1],
            [x0, y1, z0],
            [x0, y1, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
        ], dtype=np.float64)

        A = np.asarray(T4[:3, :3], dtype=np.float64)
        t = np.asarray(T4[:3, 3], dtype=np.float64)

        world = corners @ A.T + t
        bounds_min = np.floor(world.min(axis=0)).astype(np.int64)
        bounds_max = np.ceil(world.max(axis=0)).astype(np.int64)
        return bounds_min, bounds_max

    def _cellgrid_params_xyz(self, size_xyz):
        size_xyz = np.asarray(size_xyz, dtype=np.int64)
        cd = self.cell_dimensions

        num_cells = (size_xyz + cd - 1) // cd
        border_size = size_xyz - (num_cells - 1) * cd
        return num_cells, border_size

    def _get_cell_interval_xyz(self, grid_pos_xyz, num_cells_xyz, border_size_xyz):
        gp = np.asarray(grid_pos_xyz, dtype=np.int64)
        cd = self.cell_dimensions
        nc = np.asarray(num_cells_xyz, dtype=np.int64)
        bs = np.asarray(border_size_xyz, dtype=np.int64)

        cell_min = gp * cd
        is_last = (gp + 1 == nc)
        cell_dim = np.where(is_last, bs, cd)
        cell_max = cell_min + cell_dim - 1
        return cell_min, cell_max

    # def find_overlapping_blocks(self, model, size):
    #     prefetch = []

    #     dims_xyz = np.asarray(size, dtype=np.int64)
    #     cd = np.asarray(self.cell_dimensions, dtype=np.int64)
    #     grid_dims = (dims_xyz + cd - 1) // cd
    #     b = int(self.expand)

    #     _, mipmap_transform = self.for_best_resolution()
    #     img_to_world = np.asarray(model, dtype=np.float64) @ np.asarray(mipmap_transform, dtype=np.float64)

    #     # --- NEW: compute candidate cell index range by inverse-mapping fused bbox to local ---
    #     inv = np.linalg.inv(img_to_world)

    #     # Use expanded fused interval (same logic as expand_interval but local here)
    #     fused_min = self.fused_min.astype(np.float64) - b
    #     fused_max = self.fused_max.astype(np.float64) + b

    #     # 8 corners of the fused/world interval
    #     x0, y0, z0 = fused_min
    #     x1, y1, z1 = fused_max
    #     world_corners = np.array([
    #         [x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
    #         [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1],
    #     ], dtype=np.float64)

    #     Ainv = inv[:3, :3]
    #     tinv = inv[:3, 3]
    #     local = world_corners @ Ainv.T + tinv

    #     local_min = np.floor(local.min(axis=0)).astype(np.int64)
    #     local_max = np.ceil(local.max(axis=0)).astype(np.int64)

    #     # Clamp to image bounds (inclusive)
    #     local_min = np.maximum(local_min, 0)
    #     local_max = np.minimum(local_max, dims_xyz - 1)

    #     # Convert local voxel range -> cell index range
    #     gmin = np.maximum(local_min // cd, 0)
    #     gmax = np.minimum(local_max // cd, grid_dims - 1)

    #     # --- loop only candidate grid positions ---
    #     for gx in range(int(gmin[0]), int(gmax[0]) + 1):
    #         for gy in range(int(gmin[1]), int(gmax[1]) + 1):
    #             for gz in range(int(gmin[2]), int(gmax[2]) + 1):
    #                 grid_pos = np.array([gx, gy, gz], dtype=np.int64)

    #                 cell_min = grid_pos * cd
    #                 cell_max = np.minimum(cell_min + cd, dims_xyz) - 1  # inclusive

    #                 expanded_min = cell_min - b
    #                 expanded_max = cell_max + b

    #                 bounds_min, bounds_max = self.transformed_bounding_box_from_minmax(
    #                     img_to_world, expanded_min, expanded_max
    #                 )

    #                 if self.overlaps(bounds_min, bounds_max, self.fused_min, self.fused_max):
    #                     prefetch.append({"cell_min": cell_min.copy()})

    #     return prefetch

    def find_overlapping_blocks(self, model, size):
        prefetch = []

        # Treat `size` as the source tile/image dims (XYZ)
        dims_xyz = np.asarray(size, dtype=np.int64)
        cd = np.asarray(self.cell_dimensions, dtype=np.int64)

        _, mipmap_transform = self.for_best_resolution()

        # Java: imgToWorld = model.copy(); imgToWorld.concatenate(best.mipmapTransform)
        img_to_world = np.asarray(model, dtype=np.float64) @ np.asarray(mipmap_transform, dtype=np.float64)

        # Java: grid.getGridDimensions()
        grid_dims = (dims_xyz + cd - 1) // cd  # ceil(dims/cellDims)

        b = int(self.expand)

        for gx in range(int(grid_dims[0])):
            for gy in range(int(grid_dims[1])):
                for gz in range(int(grid_dims[2])):
                    grid_pos = np.array([gx, gy, gz], dtype=np.int64)

                    # Java: grid.getCellInterval(gridPos, cellMin, cellMax)
                    cell_min = grid_pos * cd
                    cell_max = np.minimum(cell_min + cd, dims_xyz) - 1  # inclusive, border-clamped

                    # Java: expand(cellBBox, expand, projectedCellBBox)
                    expanded_min = cell_min - b
                    expanded_max = cell_max + b

                    # Java: bounds = smallestContainingInterval(imgToWorld.estimateBounds(...))
                    bounds_min, bounds_max = self.transformed_bounding_box_from_minmax(
                        img_to_world, expanded_min, expanded_max
                    )

                    if self.overlaps(bounds_min, bounds_max, self.fused_min, self.fused_max):
                        prefetch.append({"cell_min": cell_min.copy()})

        return prefetch

    def find(self):
        expanded_min, expanded_max = self.expand_interval()

        blocks_by_view = {}
        for view_id in self.overlapping_views:
            info = self.per_view_transforms[view_id]
            model = info["transform"]
            size = info["size"]

            size = np.asarray(size, dtype=np.int64)
            if size.shape[0] > 3:
                size_xyz = size[:3]
            else:
                size_xyz = size

            # Whole-view screening
            bounds_min, bounds_max = self.transformed_bounding_box_from_minmax(
                np.asarray(model, dtype=np.float64),
                np.array([0, 0, 0], dtype=np.int64),
                size_xyz - 1
            )

            if self.overlaps(expanded_min, expanded_max, bounds_min, bounds_max):
                blocks = self.find_overlapping_blocks(model, size_xyz)
                if blocks:
                    blocks_by_view[view_id] = blocks

        return blocks_by_view

    def run(self):
        return self.find()