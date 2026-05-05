import numpy as np
from bioio import BioImage
import bioio_tifffile
import zarr
import s3fs
import dask.array as da
import math
import os

"""
Overlap Detection figures out where image tile overlap. 
"""

# TIFF reader wants to be used as an abstract class
class CustomBioImage(BioImage):
    def standard_metadata(self):
        pass
    
    def scale(self):
        pass
    
    def time_interval(self):
        pass

class OverlapDetection():
    def __init__(self, transform_models, dataframes, dsxy, dsz, prefix, file_type, overlapping_only=True):
        self.transform_models = transform_models
        self.image_loader_df = dataframes['image_loader']
        self.dsxy, self.dsz = dsxy, dsz
        self.prefix = prefix
        self.file_type = file_type
        self.overlapping_only = overlapping_only
        self.to_process = {}
        self.image_shape_cache = {}
        self.max_interval_size = 0
    
    def create_mipmap_transform(self):
        """
        Build a 4×4 homogeneous scaling matrix for the mipmap level
        """
        scale_matrix = np.array([
            [self.dsxy, 0, 0, 0],  
            [0, self.dsxy, 0, 0],  
            [0, 0, self.dsz, 0],  
            [0, 0, 0, 1]          
        ])
        
        return scale_matrix
    
    def load_image_metadata(self, file_path):
        if file_path in self.image_shape_cache:
            return self.image_shape_cache[file_path]
        
        if self.file_type == 'zarr':
            s3 = s3fs.S3FileSystem(anon=True)
            print(f"[OverlapDetection] Opening root zarr: {file_path}")
            try:
                store = s3fs.S3Map(root=file_path, s3=s3)
                zarr_obj = zarr.open(store, mode='r')
                if isinstance(zarr_obj, zarr.hierarchy.Group):
                    print(f"[OverlapDetection] Opened zarr Group. Available levels: {list(zarr_obj.keys())}")
                    zarr_arr = zarr_obj['0']
                else:
                    print(f"[OverlapDetection] Opened zarr Array directly.")
                    zarr_arr = zarr_obj
                dask_array = da.from_zarr(zarr_arr)
                dask_array = da.expand_dims(dask_array, axis=2)
                shape = dask_array.shape
                self.image_shape_cache[file_path] = shape
                print(f"[OverlapDetection] Shape (after expand): {shape}")
            except Exception as e:
                print(f"[OverlapDetection] ERROR opening root zarr: {e}")
                raise

        elif self.file_type == 'tiff':
            img = CustomBioImage(file_path, reader=bioio_tifffile.Reader)
            data = img.get_dask_stack()
            shape = data.shape
            self.image_shape_cache[file_path] = shape
        
        return shape

    def _split_tile_shape(self, row):
        """Derive 6D shape tuple from split tile crop bounds.

        Parameters
        ----------
        row : pd.Series
            Row from image_loader_df with 'crop_min' and 'crop_max' columns.
            Values are space-separated "X Y Z" strings.

        Returns
        -------
        tuple
            6D shape tuple (1, 1, 1, Z, Y, X) matching load_image_metadata format.
        """
        cmin = list(map(int, row['crop_min'].split()))
        cmax = list(map(int, row['crop_max'].split()))
        x_size = cmax[0] - cmin[0] + 1
        y_size = cmax[1] - cmin[1] + 1
        z_size = cmax[2] - cmin[2] + 1
        return (1, 1, 1, z_size, y_size, x_size)

    # def open_and_downsample(self, shape):
    #     X = int(shape[5])
    #     Y = int(shape[4])
    #     Z = int(shape[3])

    #     dsx = int(self.dsxy)
    #     dsy = int(self.dsxy)
    #     dsz = int(self.dsz)

    #     def ceil_half_chain(n, f):
    #         out = int(n)
    #         while f >= 2:
    #             out = (out + 1) // 2  # ceil(n/2)
    #             f //= 2
    #         return out

    #     x_new = ceil_half_chain(X, dsx)
    #     y_new = ceil_half_chain(Y, dsy)
    #     z_new = ceil_half_chain(Z, dsz)

    #     mipmap_transform = self.create_mipmap_transform()
    #     return ((0, 0, 0), (x_new, y_new, z_new)), mipmap_transform
    
    def open_and_downsample(self, shape, dsxy, dsz):
        """
        Downsample a 3D volume by powers of two by repeatedly halving along each axis
        """
        dsx = dsxy
        dsy = dsxy

        # downsample x dimension
        x_new = shape[5]
        while dsx > 1:
            x_new = x_new // 2 if x_new % 2 == 0 else (x_new // 2) + 1
            dsx //= 2

        # downsample y dimension
        y_new = shape[4]
        while dsy > 1:
            y_new = y_new // 2 if y_new % 2 == 0 else (y_new // 2) + 1
            dsy //= 2

        # downsample z dimension
        z_new = shape[3]
        while dsz > 1:
            z_new = z_new // 2 if z_new % 2 == 0 else (z_new // 2) + 1
            dsz //= 2

        return ((0, 0, 0), (x_new, y_new, z_new))
    
    def get_inverse_mipmap_transform(self, mipmap_transform):
        """
        Compute the inverse of the given mipmap transform
        """
        try:
            inverse_scale_matrix = np.linalg.inv(mipmap_transform)
        except np.linalg.LinAlgError:
            print("Matrix cannot be inverted.")
            return None
        
        return inverse_scale_matrix    
    
    def estimate_bounds(self, a, interval):
        """
        Transform an axis-aligned box through a 4x4 affine
        """
        # set lower bounds
        t0, t1, t2 = 0, 0, 0
        
        # set upper bounds
        if self.file_type == 'zarr':
            s0 = interval[5] - t0
            s1 = interval[4] - t1
            s2 = interval[3] - t2 
        elif self.file_type == 'tiff':
            s0 = interval[5] - t0
            s1 = interval[4] - t1
            s2 = interval[3] - t2

        # get dot product of uppper bounds and inverted downsampling matrix
        matrix = np.array(a) 
        tt = np.dot(matrix[:, :3], [t0, t1, t2]) + matrix[:, 3]
        r_min = np.copy(tt)
        r_max = np.copy(tt)

        # set upper and lower bounds using inverted downsampling matrix
        for i in range(3):
            if matrix[i, 0] < 0:
                r_min[i] += s0 * matrix[i, 0]
            else:
                r_max[i] += s0 * matrix[i, 0]
            
            if matrix[i, 1] < 0:
                r_min[i] += s1 * matrix[i, 1]
            else:
                r_max[i] += s1 * matrix[i, 1]

            if matrix[i, 2] < 0:
                r_min[i] += s2 * matrix[i, 2]
            else:
                r_max[i] += s2 * matrix[i, 2]
        
        return r_min[:3], r_max[:3]

    def calculate_intersection(self, bbox1, bbox2):
        """
        Compute the axis-aligned intersection of two 3D boxes given as (min, max) coordinates
        """
        intersect_min = np.maximum(bbox1[0], bbox2[0])
        intersect_max = np.minimum(bbox1[1], bbox2[1])
        
        return (intersect_min, intersect_max)

    def calculate_new_dims(self, lower_bound, upper_bound):
        """
        Compute per-axis lengths from bounds
        """
        new_dims = []
        for lb, ub in zip(lower_bound, upper_bound):
            if lb == 0:
                new_dims.append(ub + 1)
            else:
                new_dims.append(ub - lb)
        
        return new_dims
    
    def floor_log2(self, n):
        """
        Return ⌊log2(n)⌋ - clamps n ≤ 1 to 1 so the result is 0 for n ≤ 1
        """
        return max(0, int(math.floor(math.log2(max(1, n)))))

    def choose_zarr_level(self):
        """Pick the actual pyramid level closest to the requested dsxy/dsz.

        Reads per-axis downsample factors from the parent zarr group's
        OME-zarr ``coordinateTransformations.scale`` metadata (the
        pyramid writer's declared per-axis sampling-density ratio) —
        NOT from an isotropic ``2 ** level`` assumption, NOT from
        array-shape ratios. Metadata is the right primitive: it
        explicitly encodes whatever anisotropy the pyramid has, and is
        immune to integer-flooring slack at odd L0 extents (e.g.
        dataset A L0_z=220, L4_z=13 → shape-ratio 16.92 vs metadata
        16.0; metadata is the writer's intent).

        Picks the level whose per-axis ds is the largest possible while
        still ``≤`` the request on every axis (so the remaining
        downsampling can be done in software without ever upsampling),
        preferring the smallest leftover product to minimize redundant
        downsampling work.

        Anisotropic-pyramid example (HCR_823476_s5, request dsxy=16, dsz=4):
          L0: ds=(1,1,1)
          L1: ds=(2,2,1)
          L2: ds=(4,4,1)
          L3: ds=(8,8,2)
          L4: ds=(16,16,4)  ← exact match, leftover=(1,1,1) — picked
          L5: ds=(32,32,8)  ← rejected (over-downsamples on every axis)
        Legacy ``min(log2_xy, log2_z)`` would have picked L2 with
        leftover ``(4, 4, 1)`` — pulling 64× more bytes from S3 and
        re-doing the antialiasing in software.

        Falls back to legacy isotropic ``min(log2(dsxy), log2(dsz))`` when
        the parent zarr's metadata can't be parsed (preserves prior
        behavior on non-OME-zarr inputs / tests). Tuple convention for
        ``leftovers`` is preserved as ``(_, dsxy_leftover, dsz_leftover)``
        so the call site at ``__call__`` is untouched — only the first
        slot is unused-but-kept-for-shape.
        """
        max_level = 7
        try:
            root = zarr.open(self.prefix, mode='r')
            scale_l0 = self._ome_scale_zyx(root, "0")
            if scale_l0 is None:
                raise ValueError("no scale metadata at L0")

            # Iterate every level declared in the multiscales metadata
            # (NOT array_keys() — the metadata is what defines the
            # pyramid's intent).
            attrs = root.attrs.asdict()
            datasets = attrs.get("multiscales", [{}])[0].get("datasets", [])
            level_records = []
            for d in datasets:
                level_name = str(d.get("path"))
                if not level_name.isdigit():
                    continue
                lvl_int = int(level_name)
                if lvl_int > max_level:
                    continue
                scale_ln = self._ome_scale_zyx(root, level_name)
                if scale_ln is None:
                    continue
                # ds_axis = scale(L)[axis] / scale(L0)[axis]. Rounded to
                # int because the request and downstream downsamplers
                # are integer-valued.
                ds_z = max(1, int(round(scale_ln[0] / max(scale_l0[0], 1e-12))))
                ds_y = max(1, int(round(scale_ln[1] / max(scale_l0[1], 1e-12))))
                ds_x = max(1, int(round(scale_ln[2] / max(scale_l0[2], 1e-12))))
                level_records.append((lvl_int, ds_x, ds_y, ds_z))

            if not level_records:
                raise ValueError("no usable pyramid levels in metadata")

            req_xy = max(1, int(self.dsxy))
            req_z = max(1, int(self.dsz))

            # Eligibility: every axis's ds must be ≤ the request, so the
            # remaining downsampling can be done in software without
            # ever needing to upsample. With metadata-declared ds (no
            # rounding slack), this is the strict comparison we want.
            eligible = [
                rec for rec in level_records
                if rec[1] <= req_xy
                and rec[2] <= req_xy
                and rec[3] <= req_z
            ]
            if not eligible:
                # Request is finer than even L0. Pick L0 and pass through.
                eligible = [(0, 1, 1, 1)]

            # Score each candidate: (leftover_x * leftover_y * leftover_z).
            # Exact match → score=1 (perfect). Lower is better; tiebreak
            # by deeper level (cheaper S3 reads).
            def _score(rec):
                lvl, dsx, dsy, dsz = rec
                lo_x = max(1, req_xy // max(dsx, 1))
                lo_y = max(1, req_xy // max(dsy, 1))
                lo_z = max(1, req_z // max(dsz, 1))
                return (lo_x * lo_y * lo_z, -lvl)

            best_lvl, dsx_lvl, dsy_lvl, dsz_lvl = min(eligible, key=_score)

            # ``leftovers`` tuple convention: (unused, leftover_xy, leftover_z).
            # leftover_xy = max(leftover_x, leftover_y) so subsequent code
            # that applies a single dsxy-factor never under-downsamples
            # either axis. In the typical case dsx==dsy so this is just
            # ``req_xy // dsx``.
            leftover_x = max(1, req_xy // max(dsx_lvl, 1))
            leftover_y = max(1, req_xy // max(dsy_lvl, 1))
            leftover_z = max(1, req_z // max(dsz_lvl, 1))
            leftover_xy = max(leftover_x, leftover_y)
            leftovers = (leftover_xy, leftover_xy, leftover_z)
            return best_lvl, leftovers
        except Exception as e:
            # Legacy fallback: assume isotropic 2**level pyramid. Safe
            # for unit tests + any pyramid with that structure that
            # lacks parseable OME-zarr metadata.
            print(
                f"[OverlapDetection] choose_zarr_level falling back to "
                f"legacy isotropic picker (metadata not readable: {e!r})"
            )
            lvl_xy = self.floor_log2(self.dsxy)
            lvl_z = self.floor_log2(self.dsz)
            best = min(lvl_xy, lvl_z, max_level)
            factor = 1 << best
            leftovers = (
                max(1, self.dsxy // factor),
                max(1, self.dsxy // factor),
                max(1, self.dsz // factor),
            )
            return best, leftovers
    
    def _per_axis_pyramid_scale(self, level):
        """Return (sx, sy, sz) — the level→L0 voxel-grid scale per axis.

        Reads the OME-zarr ``coordinateTransformations.scale`` metadata
        (the writer's declared per-axis ds) — same source of truth as
        ``choose_zarr_level``. Falls back to isotropic ``2 ** level`` if
        the parent zarr's metadata can't be parsed.
        """
        if level <= 0:
            return 1.0, 1.0, 1.0
        try:
            root = zarr.open(self.prefix, mode='r')
            scale_l0 = self._ome_scale_zyx(root, "0")
            scale_ln = self._ome_scale_zyx(root, str(level))
            if scale_l0 is None or scale_ln is None:
                raise ValueError("scale metadata missing")
            sz = max(1.0, scale_ln[0] / max(scale_l0[0], 1e-12))
            sy = max(1.0, scale_ln[1] / max(scale_l0[1], 1e-12))
            sx = max(1.0, scale_ln[2] / max(scale_l0[2], 1e-12))
            return float(sx), float(sy), float(sz)
        except Exception as e:
            print(
                f"[OverlapDetection] _per_axis_pyramid_scale fallback to "
                f"isotropic 2**{level}: metadata not readable ({e!r})"
            )
            s = float(2 ** level)
            return s, s, s

    @staticmethod
    def _ome_scale_zyx(root_group, level_name: str):
        """Return (scale_z, scale_y, scale_x) from OME-zarr multiscales metadata.

        Reads ``coordinateTransformations[type==scale]`` for the named
        level. Slices the trailing ZYX entries from a 3- or 5-axis
        scale declaration. Returns ``None`` when the metadata is missing
        or malformed — caller falls back to legacy heuristic.
        """
        try:
            attrs = root_group.attrs.asdict()
            multiscales = attrs.get("multiscales", [])
            if not multiscales:
                return None
            for d in multiscales[0].get("datasets", []):
                if str(d.get("path")) != str(level_name):
                    continue
                for ct in d.get("coordinateTransformations", []):
                    if ct.get("type") == "scale":
                        s = ct.get("scale", [])
                        if len(s) == 5:
                            return float(s[2]), float(s[3]), float(s[4])
                        if len(s) == 3:
                            return float(s[0]), float(s[1]), float(s[2])
                        return None
        except Exception:
            return None
        return None

    def affine_with_half_pixel_shift(self, sx, sy, sz):
        """
        Build a 4x4 scaling affine that also shifts by 0.5·(scale-1) per axis so voxel centers stay aligned after 
        resampling (half-pixel compensation)
        """
        # translation = 0.5 * (scale - 1) per axis
        tx = 0.5 * (sx - 1.0)
        ty = 0.5 * (sy - 1.0)
        tz = 0.5 * (sz - 1.0)
        
        return np.array([
            [sx, 0.0, 0.0, tx],
            [0.0, sy, 0.0, ty],
            [0.0, 0.0, sz, tz],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
    
    def size_interval(self, lb, ub):
        """
        Find the number of voxels in a 3D box with inclusive bounds
        """
        return int((int(ub[0]) - int(lb[0]) + 1) *
                (int(ub[1]) - int(lb[1]) + 1) *
                (int(ub[2]) - int(lb[2]) + 1))

    def find_overlapping_area(self):
        """
        Compute XY Z overlap intervals against every other view, accounting for mipmap/downsampling and per-view affine transforms
        """
        is_split = 'crop_min' in self.image_loader_df.columns

        for i, row_i in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row_i['timepoint']}, setup: {row_i['view_setup']}"
            
            # get inverted matrice of downsampling
            all_intervals = []
            if self.file_type == 'zarr':
                level, leftovers = self.choose_zarr_level()
                print(f"[OverlapDetection] view={view_id}: chosen level={level}, leftovers={leftovers}, dsxy={self.dsxy}, dsz={self.dsz}")

                if is_split:
                    dim_base = self._split_tile_shape(row_i)
                else:
                    dim_base = self.load_image_metadata(self.prefix)

                # Per-axis pyramid scale. The legacy ``s = 2 ** level``
                # is wrong for anisotropic pyramids that preserve one
                # axis at coarse levels (e.g. HCR_823476_s5 keeps Z at
                # full-res through L2 while halving XY). Read the actual
                # shape ratio from the parent zarr; fall back to
                # isotropic on lookup failure.
                sx, sy, sz = self._per_axis_pyramid_scale(level)

                mipmap_of_downsample = self.affine_with_half_pixel_shift(sx, sy, sz)

                # leftovers are returned by ``choose_zarr_level`` as
                # (ds_x, ds_y, ds_z) — what remains to be applied as
                # software downsampling on top of the chosen level.
                _, dsxy, dsz = leftovers
                
            elif self.file_type == 'tiff':
                dim_base = self.load_image_metadata(os.path.join(self.prefix, row_i['file_path']))
                mipmap_of_downsample = self.create_mipmap_transform()
                dsxy, dsz = self.dsxy, self.dsz
                level = None

            downsampled_dim_base = self.open_and_downsample(dim_base, dsxy, dsz)
            t1 = self.get_inverse_mipmap_transform(mipmap_of_downsample)

            if self.overlapping_only:
                # compare with all view_ids
                for j, row_j in self.image_loader_df.iterrows():
                    if i == j: continue

                    view_id_other = f"timepoint: {row_j['timepoint']}, setup: {row_j['view_setup']}"

                    if self.file_type == 'zarr':
                        if is_split:
                            dim_other = self._split_tile_shape(row_j)
                        else:
                            dim_other = self.load_image_metadata(self.prefix)
                    elif self.file_type == 'tiff':
                        dim_other = self.load_image_metadata(os.path.join(self.prefix, row_j['file_path']))

                    # get transforms matrix from both view_ids and downsampling matrices
                    matrix = self.transform_models.get(view_id)
                    matrix_other = self.transform_models.get(view_id_other)

                    if self.file_type == 'zarr':
                        s = float(2 ** level)
                        mipmap_of_downsample_other = self.affine_with_half_pixel_shift(s, s, s)
                    elif self.file_type == 'tiff':
                        mipmap_of_downsample_other = self.create_mipmap_transform()

                    inverse_mipmap_of_downsample_other = self.get_inverse_mipmap_transform(mipmap_of_downsample_other)
                    inverse_matrix = self.get_inverse_mipmap_transform(matrix)

                    concatenated_matrix = np.dot(inverse_matrix, matrix_other)
                    t2 = np.dot(inverse_mipmap_of_downsample_other, concatenated_matrix)

                    intervals = self.estimate_bounds(t1, dim_base)
                    intervals_other = self.estimate_bounds(t2, dim_other)

                    bounding_boxes = tuple(map(lambda x: np.round(x).astype(int), intervals))
                    bounding_boxes_other = tuple(map(lambda x: np.round(x).astype(int), intervals_other))

                    # find upper and lower bounds of intersection
                    if np.all((bounding_boxes[1] >= bounding_boxes_other[0]) & (bounding_boxes_other[1] >= bounding_boxes[0])):
                        intersected_boxes = self.calculate_intersection(bounding_boxes, bounding_boxes_other)
                        intersect = self.calculate_intersection(downsampled_dim_base, intersected_boxes)
                        intersect_dict = {
                            'lower_bound': intersect[0],
                            'upper_bound': intersect[1],
                            'span': self.calculate_new_dims(intersect[0], intersect[1])
                        }

                        lb, ub = intersect[0], intersect[1]
                        sz = self.size_interval(lb, ub)
                        if sz > self.max_interval_size:
                            self.max_interval_size = sz

                        # add max size
                        all_intervals.append(intersect_dict)

                # Single-view dataset: no pairwise overlaps exist, so use the
                # full downsampled volume as the processing region.
                if not all_intervals and len(self.image_loader_df) == 1:
                    lb = np.array(downsampled_dim_base[0])
                    ub = np.array(downsampled_dim_base[1])
                    all_intervals.append({
                        'lower_bound': lb,
                        'upper_bound': ub,
                        'span': self.calculate_new_dims(lb, ub),
                    })
                    sz = self.size_interval(lb, ub)
                    if sz > self.max_interval_size:
                        self.max_interval_size = sz

            else:
                # Full-volume mode: use the entire downsampled tile as the
                # processing region (for registration, not stitching).
                lb = np.array(downsampled_dim_base[0])
                ub = np.array(downsampled_dim_base[1])
                all_intervals.append({
                    'lower_bound': lb,
                    'upper_bound': ub,
                    'span': self.calculate_new_dims(lb, ub),
                })
                sz = self.size_interval(lb, ub)
                if sz > self.max_interval_size:
                    self.max_interval_size = sz

            self.to_process[view_id] = all_intervals
        
        return dsxy, dsz, level, mipmap_of_downsample
                
    def run(self):
        """
        Executes the entry point of the script.
        """
        dsxy, dsz, level, mipmap_of_dowsample = self.find_overlapping_area()
        return self.to_process, dsxy, dsz, level, self.max_interval_size, mipmap_of_dowsample
