import numpy as np

"""
Find overlapping views for one fusion worker block
"""

class OverlappingViews:
    def __init__(self, super_block_offset, super_block_size, per_view_transforms):
        self.super_block_offset = super_block_offset
        self.super_block_size = super_block_size
        self.per_view_transforms = per_view_transforms

    def expand_interval(self, min_xyz, max_xyz):
        pad = 2
        return (min_xyz - pad, max_xyz + pad)

    def overlaps(self, a_min, a_max, b_min, b_max):
        # inclusive overlap check per dimension
        return np.all(a_min <= b_max) and np.all(b_min <= a_max)

    def estimate_bounds_minmax_3d(self, T4, size_xyz):
        """
        Uses interval [0..sx-1], [0..sy-1], [0..sz-1] (inclusive)
        then floor(min)/ceil(max).
        """
        A = T4[:3, :3]
        t = T4[:3, 3]

        sx, sy, sz = int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2])

        s0 = float(sx - 1)
        s1 = float(sy - 1)
        s2 = float(sz - 1)

        tt0, tt1, tt2 = t[0], t[1], t[2]

        # row 0
        rMin0 = rMax0 = tt0
        rMin0 += s0*A[0,0] if A[0,0] < 0 else 0.0; rMax0 += 0.0 if A[0,0] < 0 else s0*A[0,0]
        rMin0 += s1*A[0,1] if A[0,1] < 0 else 0.0; rMax0 += 0.0 if A[0,1] < 0 else s1*A[0,1]
        rMin0 += s2*A[0,2] if A[0,2] < 0 else 0.0; rMax0 += 0.0 if A[0,2] < 0 else s2*A[0,2]

        # row 1
        rMin1 = rMax1 = tt1
        rMin1 += s0*A[1,0] if A[1,0] < 0 else 0.0; rMax1 += 0.0 if A[1,0] < 0 else s0*A[1,0]
        rMin1 += s1*A[1,1] if A[1,1] < 0 else 0.0; rMax1 += 0.0 if A[1,1] < 0 else s1*A[1,1]
        rMin1 += s2*A[1,2] if A[1,2] < 0 else 0.0; rMax1 += 0.0 if A[1,2] < 0 else s2*A[1,2]

        # row 2
        rMin2 = rMax2 = tt2
        rMin2 += s0*A[2,0] if A[2,0] < 0 else 0.0; rMax2 += 0.0 if A[2,0] < 0 else s0*A[2,0]
        rMin2 += s1*A[2,1] if A[2,1] < 0 else 0.0; rMax2 += 0.0 if A[2,1] < 0 else s1*A[2,1]
        rMin2 += s2*A[2,2] if A[2,2] < 0 else 0.0; rMax2 += 0.0 if A[2,2] < 0 else s2*A[2,2]

        min_xyz = np.array([np.floor(rMin0), np.floor(rMin1), np.floor(rMin2)], dtype=np.int64)
        max_xyz = np.array([np.ceil (rMax0), np.ceil (rMax1), np.ceil (rMax2)], dtype=np.int64)
        return min_xyz, max_xyz

    def find_overlapping_views(self):
        fused_min = self.super_block_offset
        fused_max = self.super_block_offset + self.super_block_size - 1

        # ---- DEBUG: is this block covered by ANY tile? (Case1 vs Case2) ----
        # Put behind a guard so you don't spam:
        # e.g. only check blocks at the global Z-min slab or only first N calls.
        # if fused_min[2] == 0:  # example guard: only blocks at Z==0 slab (your black stripe)
        #     world_x = 0.5 * (float(fused_min[0]) + float(fused_max[0])) + 0.5
        #     world_y = 0.5 * (float(fused_min[1]) + float(fused_max[1])) + 0.5
        #     world_z = 0.5 * (float(fused_min[2]) + float(fused_max[2])) + 0.5
        #     world_pt = np.array([world_x, world_y, world_z, 1.0], dtype=np.float64)

        #     hits = 0
        #     for (tp, setup), info in self.per_view_transforms.items():
        #         T = np.asarray(info["transform"], dtype=np.float64)
        #         sx, sy, sz = (int(info["size"][0]), int(info["size"][1]), int(info["size"][2]))

        #         tile_pt = (np.linalg.inv(T) @ world_pt)[:3] - 0.5  # voxel-index convention
        #         inside = (
        #             (0.0 <= tile_pt[0] < float(sx)) and
        #             (0.0 <= tile_pt[1] < float(sy)) and
        #             (0.0 <= tile_pt[2] < float(sz))
        #         )
        #         if inside:
        #             print(f"[BLOCK_COVERED] block_center={world_pt[:3]} by=({tp},{setup}) tile_pt={tile_pt}")
        #             hits += 1
        #             if hits >= 3:
        #                 break

        #     if hits == 0:
        #         print(f"[BLOCK_UNCOVERED] block_center={world_pt[:3]} fused_min={tuple(fused_min)} fused_max={tuple(fused_max)}")
        
        exp_min, exp_max = self.expand_interval(fused_min, fused_max)

        overlapping = []
        for (tp, setup), view_info in self.per_view_transforms.items():
            T = view_info["transform"]
            size = view_info["size"]

            view_min, view_max = self.estimate_bounds_minmax_3d(T, size)

            if self.overlaps(exp_min, exp_max, view_min, view_max):
                overlapping.append((tp, setup))

        return overlapping, fused_min, fused_max

    def run(self):
        return self.find_overlapping_views()