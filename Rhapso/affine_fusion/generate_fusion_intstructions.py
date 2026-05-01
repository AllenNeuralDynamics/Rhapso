import numpy as np

"""
Create fuse instructions
"""

class GenerateFusionInstructions():
    def __init__(self, per_view_transforms, grid_block, fusion_min_global, fusion_max_global):
        self.per_view_transforms = per_view_transforms
        self.grid_block = grid_block
        self.fusion_min_global = fusion_min_global
        self.fusion_max_global = fusion_max_global

    def estimate_bounds(self, t, dim_xyz):
        A = np.asarray(t[:3, :3], dtype=np.float64)
        t = np.asarray(t[:3, 3], dtype=np.float64)

        sx, sy, sz = map(float, np.asarray(dim_xyz, dtype=np.float64))

        # FinalInterval(dim): min=(0,0,0), max=(sx-1, sy-1, sz-1)
        t0 = 0.0
        t1 = 0.0
        t2 = 0.0

        s0 = (sx - 1.0) - t0  # == sx-1
        s1 = (sy - 1.0) - t1  # == sy-1
        s2 = (sz - 1.0) - t2  # == sz-1

        # tt = A * t + tvec (but t is zeros here; kept for parity)
        tt0 = A[0, 0] * t0 + A[0, 1] * t1 + A[0, 2] * t2 + t[0]
        tt1 = A[1, 0] * t0 + A[1, 1] * t1 + A[1, 2] * t2 + t[1]
        tt2 = A[2, 0] * t0 + A[2, 1] * t1 + A[2, 2] * t2 + t[2]

        # Row 0
        rMin0 = rMax0 = tt0
        rMin0 += s0 * A[0, 0] if A[0, 0] < 0 else 0.0
        rMax0 += 0.0 if A[0, 0] < 0 else s0 * A[0, 0]
        rMin0 += s1 * A[0, 1] if A[0, 1] < 0 else 0.0
        rMax0 += 0.0 if A[0, 1] < 0 else s1 * A[0, 1]
        rMin0 += s2 * A[0, 2] if A[0, 2] < 0 else 0.0
        rMax0 += 0.0 if A[0, 2] < 0 else s2 * A[0, 2]

        # Row 1
        rMin1 = rMax1 = tt1
        rMin1 += s0 * A[1, 0] if A[1, 0] < 0 else 0.0
        rMax1 += 0.0 if A[1, 0] < 0 else s0 * A[1, 0]
        rMin1 += s1 * A[1, 1] if A[1, 1] < 0 else 0.0
        rMax1 += 0.0 if A[1, 1] < 0 else s1 * A[1, 1]
        rMin1 += s2 * A[1, 2] if A[1, 2] < 0 else 0.0
        rMax1 += 0.0 if A[1, 2] < 0 else s2 * A[1, 2]

        # Row 2
        rMin2 = rMax2 = tt2
        rMin2 += s0 * A[2, 0] if A[2, 0] < 0 else 0.0
        rMax2 += 0.0 if A[2, 0] < 0 else s0 * A[2, 0]
        rMin2 += s1 * A[2, 1] if A[2, 1] < 0 else 0.0
        rMax2 += 0.0 if A[2, 1] < 0 else s1 * A[2, 1]
        rMin2 += s2 * A[2, 2] if A[2, 2] < 0 else 0.0
        rMax2 += 0.0 if A[2, 2] < 0 else s2 * A[2, 2]

        rMin = np.array([rMin0, rMin1, rMin2], dtype=np.float64)
        rMax = np.array([rMax0, rMax1, rMax2], dtype=np.float64)
        return rMin, rMax
    
    def largest_contained_interval(self, ri):
        rmin, rmax = ri
        rmin = np.asarray(rmin, dtype=np.float64)
        rmax = np.asarray(rmax, dtype=np.float64)

        imin = np.ceil(rmin).astype(np.int64)
        imax = np.floor(rmax).astype(np.int64)
        return imin, imax
    
    def expand_interval(self, interval_min, interval_max, border: int):
        border = int(border)
        return interval_min - border, interval_max + border
    
    def set_bounds(self, bb, i, interval_min, interval_max, num_dimensions=3):
        o_min = num_dimensions * (2 * i)
        o_max = num_dimensions * (2 * i + 1)
        for d in range(num_dimensions):
            bb[o_min + d] = int(interval_min[d])
            bb[o_max + d] = int(interval_max[d])
    
    def is_overlapping(self, i, f_min, f_max, bb):
        num_dims = 3
        o_min = num_dims * (2 * i)
        o_max = num_dims * (2 * i + 1)

        for d in range(num_dims):
            bounds_min = bb[o_min + d]   
            bounds_max = bb[o_max + d]   

            if f_min[d] > bounds_max or f_max[d] < bounds_min:  
                return False

        return True

    def get_overlapping_view_indices(self, f_min, f_max, view_ids, bb):
        indices = []
        num_views = len(view_ids)
        for i in range(num_views):
            if self.is_overlapping(i, f_min, f_max, bb):
                indices.append(i)
        return indices

    def filter(self, bb):
        num_dimensions = 3
        f_min = self.fusion_min_global 
        f_max = self.fusion_max_global
        view_ids = sorted(self.per_view_transforms.keys())

        indices = self.get_overlapping_view_indices(f_min, f_max, view_ids, bb)
        filtered_num_views = len(indices)

        filtered_views = []  
        filtered_bb = np.empty((filtered_num_views * num_dimensions * 2,), dtype=np.int64)

        for i in indices:
            o = num_dimensions * (2 * i)
            fo = num_dimensions * (2 * len(filtered_views))   

            for k in range(2 * num_dimensions):
                filtered_bb[fo + k] = bb[o + k]

            filtered_views.append(view_ids[i])

        return filtered_views, filtered_bb
    
    def overlap(self):
        num_dims = 3
        expand_overlap = 2

        bounds = []
        for (tp, setup), view_info in self.per_view_transforms.items():
            t = view_info["transform"]
            dim = view_info["size"]
            ri = self.estimate_bounds(t, dim)
            bb_min, bb_max = self.largest_contained_interval(ri)
            bb_min, bb_max = self.expand_interval(bb_min, bb_max, expand_overlap)
            bounds.append((bb_min, bb_max))

        num_views = len(bounds)
        bb = np.empty((num_views * num_dims * 2,), dtype=np.int64)
        for i, (bb_min, bb_max) in enumerate(bounds):
            self.set_bounds(bb, i, bb_min, bb_max, num_dimensions=num_dims)
        
        return bb
    
    def offset(self, offset_xyz, bb, views):
        num_views = len(views)
        num_dimensions = 3
        offsetBB = np.empty(bb.shape, dtype=np.int64)

        for i in range(2 * num_views):
            o = num_dimensions * i
            for d in range(num_dimensions):
                offsetBB[o + d] = bb[o + d] - offset_xyz[d]

        return offsetBB
    
    def concatenate_bounding_box_offset(self, model, fusion_min):
        model = np.asarray(model, dtype=np.float64)

        t = np.eye(4, dtype=np.float64)
        t[0, 3] = -float(fusion_min[0])
        t[1, 3] = -float(fusion_min[1])
        t[2, 3] = -float(fusion_min[2])

        return t @ model
    
    def get_scale(self, transform):
        T = np.asarray(transform, dtype=np.float64)
        return np.array([abs(T[0, 0]), abs(T[1, 1]), abs(T[2, 2])], dtype=np.float64)

    def adjust_blending(self, blending, border, transform):
        blending = np.asarray(blending, dtype=np.float32)
        border = np.asarray(border, dtype=np.float32)

        scale = self.get_scale(transform).astype(np.float32)
        blending /= scale
        border /= scale

        return blending, border
    
    def create_avg_blend(self, dim_xyz, border, blending, transform_from_source, interval_min_xyz=(0, 0, 0)):
        n = 3

        m = np.eye(4, dtype=np.float64)
        m[0, 3] = float(interval_min_xyz[0])
        m[1, 3] = float(interval_min_xyz[1])
        m[2, 3] = float(interval_min_xyz[2])

        T = np.asarray(transform_from_source, dtype=np.float64)
        t = T @ m

        inv_t = np.linalg.inv(t)

        # d(0) is the direction vector for x: first column of linear part of inv(t)
        d0 = inv_t[:3, 0].copy()

        # ---- compute b0..b3 from interval dimensions ----
        dim_xyz = np.asarray(dim_xyz, dtype=np.float64)
        border = np.asarray(border, dtype=np.float64)
        blending = np.asarray(blending, dtype=np.float64)

        b0 = np.empty((n,), dtype=np.float64)
        b1 = np.empty((n,), dtype=np.float64)
        b2 = np.empty((n,), dtype=np.float64)
        b3 = np.empty((n,), dtype=np.float64)

        for d in range(n):
            dim = float(dim_xyz[d])  # interval.dimension(d)

            b0[d] = border[d]
            b1[d] = border[d] + blending[d]
            b2[d] = dim - 1.0 - border[d] - blending[d]
            b3[d] = dim - 1.0 - border[d]

            # if (b1 > b2) collapse inside region
            if b1[d] > b2[d]:
                mid = 0.5 * (b1[d] + b2[d])
                b1[d] = mid
                b2[d] = mid

        return {
            "t": t,
            "inv_t": inv_t,
            "d0": d0,
            "b0": b0.astype(np.float32),
            "b1": b1.astype(np.float32),
            "b2": b2.astype(np.float32),
            "b3": b3.astype(np.float32),
            "blending": blending.astype(np.float32).copy(),
            "border": border.astype(np.float32).copy(),
        }
    
    def fuse(self):
        fusion_type = "avg_blend"
        num_dimensions = 3

        bb = self.overlap()
        filtered_views, filtered_bb = self.filter(bb)
        offset_filtered_bb = self.offset(self.fusion_min_global, filtered_bb, filtered_views)

        image_instructions = {}
        blocks = {}

        # Get interval for fused output cell
        block_min = self.grid_block[0]
        block_max = [0] * len(block_min) 
        interval = (self.fusion_max_global - self.fusion_min_global)
        for d in range(len(block_min)):
            block_max[d] = min(int(interval[d]), int(block_min[d] + self.grid_block[1][d] - 1))

        # Iterate through overlap views to get bboxs
        for i, view in enumerate(filtered_views):
            model = self.per_view_transforms[view]['transform']
            dim = self.per_view_transforms[view]['size']

            base = i * 2 * num_dimensions
            bb_min = np.array(offset_filtered_bb[base: base + num_dimensions], dtype=np.int64)
            bb_max = np.array(offset_filtered_bb[base + num_dimensions: base + 2 * num_dimensions], dtype=np.int64)

            inter_min = np.maximum(block_min, bb_min)
            inter_max = np.minimum(block_max, bb_max)

            if np.all(inter_min <= inter_max):
                blocks[view] = (inter_min, inter_max)
            
            # Get fused interval in regular space
            transform = self.concatenate_bounding_box_offset(model, self.fusion_min_global)

            blending = [40, 40, 40]
            border = [0, 0, 0]

            blending, border = self.adjust_blending(blending, border, transform)
            
            if fusion_type == "avg_blend":
                weights = self.create_avg_blend(dim, border, blending, transform, interval_min_xyz=(0, 0, 0))

            image_instructions[view] = weights
        
        return image_instructions, blocks

    def run(self):
        return self.fuse()