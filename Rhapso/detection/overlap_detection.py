import numpy as np
from bioio import BioImage
import bioio_tifffile
import zarr
import s3fs
import dask.array as da
import math

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
    def __init__(self, transform_models, dataframes, dsxy, dsz, prefix, file_type):
        self.transform_models = transform_models
        self.image_loader_df = dataframes['image_loader']
        self.dsxy, self.dsz = dsxy, dsz
        self.prefix = prefix
        self.file_type = file_type
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
            s3 = s3fs.S3FileSystem(anon=False)
            store = s3fs.S3Map(root=file_path, s3=s3)
            zarr_array = zarr.open(store, mode='r')
            dask_array = da.from_zarr(zarr_array)
            dask_array = da.expand_dims(dask_array, axis=2)
            shape = dask_array.shape
            self.image_shape_cache[file_path] = shape

        elif self.file_type == 'tiff':
            img = CustomBioImage(file_path, reader=bioio_tifffile.Reader)
            data = img.get_dask_stack()
            shape = data.shape
            self.image_shape_cache[file_path] = shape
        
        return shape
    
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
        """
        pick the highest power-of-two pyramid level ( ≤ 7) compatible with dsxy/dsz
        """
        max_level = 7
        lvl_xy = self.floor_log2(self.dsxy)
        lvl_z  = self.floor_log2(self.dsz)
        best = min(lvl_xy, lvl_z, max_level)
        factor = 1 << best  
        leftovers = (max(1, self.dsxy // factor), max(1, self.dsxy // factor), max(1, self.dsz // factor))
        return best, leftovers
    
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
        for i, row_i in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row_i['timepoint']}, setup: {row_i['view_setup']}"
            
            # get inverted matrice of downsampling
            all_intervals = []        
            if self.file_type == 'zarr':
                level, leftovers = self.choose_zarr_level()
                dim_base = self.load_image_metadata(self.prefix + row_i['file_path'] + f'/{0}')

                # isotropic pyramid
                s = float(2 ** level)  
                mipmap_of_downsample = self.affine_with_half_pixel_shift(s, s, s)

                # TODO - update mipmap with leftovers if other than 1
                _, dsxy, dsz = leftovers
                
            elif self.file_type == 'tiff':
                dim_base = self.load_image_metadata(self.prefix + row_i['file_path'])
                mipmap_of_downsample = self.create_mipmap_transform()
                dsxy, dsz = self.dsxy, self.dsz
                level = None

            downsampled_dim_base = self.open_and_downsample(dim_base, dsxy, dsz)
            t1 = self.get_inverse_mipmap_transform(mipmap_of_downsample) 

            # compare with all view_ids
            for j, row_j in self.image_loader_df.iterrows():
                if i == j: continue
                
                view_id_other = f"timepoint: {row_j['timepoint']}, setup: {row_j['view_setup']}"

                if self.file_type == 'zarr':
                    dim_other = self.load_image_metadata(self.prefix + row_j['file_path'] + f'/{0}')
                elif self.file_type == 'tiff':
                    dim_other = self.load_image_metadata(self.prefix + row_j['file_path'])
                
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
    
            self.to_process[view_id] = all_intervals
        
        return dsxy, dsz, level, mipmap_of_downsample
                
    def run(self):
        """
        Executes the entry point of the script.
        """
        dsxy, dsz, level, mipmap_of_dowsample = self.find_overlapping_area()
        return self.to_process, dsxy, dsz, level, self.max_interval_size, mipmap_of_dowsample
