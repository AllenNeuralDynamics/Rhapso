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