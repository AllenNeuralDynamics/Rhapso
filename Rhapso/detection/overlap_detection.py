import numpy as np
from bioio import BioImage
import bioio_tifffile

# This components uses transforms models to find areas of overlap between image tiles

class OverlapDetection():

    def __init__(self, transform_models, dataframes, dsxy, dsz, prefix):
        self.transform_models = transform_models
        self.image_loader_df = dataframes['image_loader']
        self.dsxy, self.dsz = dsxy, dsz
        self.prefix = prefix
        self.to_process = {}
    
    def load_image_metadata(self, file_path):
        img = BioImage(file_path, reader=bioio_tifffile.Reader)
        data = img.get_dask_stack()
        self.image_data = np.zeros(data.shape[3:], dtype=data.dtype)
        return data

    def create_mipmap_transform(self):
        scale_matrix = np.array([
            [self.dsxy, 0, 0, 0],  
            [0, self.dsxy, 0, 0],  
            [0, 0, self.dsz, 0],  
            [0, 0, 0, 1]          
        ])
        return scale_matrix
    
    def open_and_downsample(self, shape):
        dsx = self.dsxy
        dsy = self.dsxy
        dsz = self.dsz

        mipmap_transform = self.create_mipmap_transform()

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

        return ((0, 0, 0), (x_new, y_new, z_new)), mipmap_transform
    
    def get_inverse_mipmap_transform(self, mipmap_transform):
        try:
            inverse_scale_matrix = np.linalg.inv(mipmap_transform)
        except np.linalg.LinAlgError:
            print("Matrix cannot be inverted.")
            return None
        return inverse_scale_matrix
    
    def estimate_bounds(self, a, interval):
        assert len(interval) >= 6, "Interval dimensions do not match." 

        # set lower bounds
        t0, t1, t2 = 0, 0, 0
        
        # set upper bounds
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
        intersect_min = np.maximum(bbox1[0], bbox2[0])
        intersect_max = np.minimum(bbox1[1], bbox2[1])
        return (intersect_min, intersect_max)

    def calculate_new_dims(self, lower_bound, upper_bound):
        new_dims = []
        for lb, ub in zip(lower_bound, upper_bound):
            if lb == 0:
                new_dims.append(ub + 1)
            else:
                new_dims.append(ub - lb)
        return new_dims

    def find_overlapping_area(self):
        # iterate through each view_id
        for i, row_i in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row_i['timepoint']}, setup: {row_i['view_setup']}"
            all_intervals = []
            
            # get inverted matrice of downsampling
            img_base = self.load_image_metadata(self.prefix + row_i['file_path'])
            dim_base = img_base.shape
            downsampled_dim_base, mipmap_of_downsample = self.open_and_downsample(dim_base)
            t1 = self.get_inverse_mipmap_transform(mipmap_of_downsample) 

            # compare with all view_ids
            for j, row_j in self.image_loader_df.iterrows():
                if i == j: continue
                
                view_id_other = f"timepoint: {row_j['timepoint']}, setup: {row_j['view_setup']}"
                img_other = self.load_image_metadata(self.prefix + row_j['file_path'])
                dim_other = img_other.shape
                
                # get transforms matrix from both view_ids and downsampling matrices
                matrix = self.transform_models.get(view_id)
                matrix_other = self.transform_models.get(view_id_other)
                inverse_matrix = self.get_inverse_mipmap_transform(matrix)
                concatenated_matrix = np.dot(inverse_matrix, matrix_other) 
                t2 = np.dot(t1, concatenated_matrix)

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
                    all_intervals.append(intersect_dict)        
                    
            self.to_process[view_id] = all_intervals
                
    def run(self):
        self.find_overlapping_area()
        return self.to_process