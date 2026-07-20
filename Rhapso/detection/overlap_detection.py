import numpy as np
from bioio import BioImage
import bioio_tifffile
import zarr
import s3fs
import math
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

"""
Overlap Detection figures out where image tile overlap. 
"""

# TIFF reader wants to be used as an abstract class
# class CustomBioImage(BioImage):
#     def standard_metadata(self):
#         pass
    
#     def scale(self):
#         pass
    
#     def time_interval(self):
#         pass

class CustomBioImage(BioImage):
    @property
    def standard_metadata(self):
        return self.reader.standard_metadata

    @property
    def scale(self):
        return self.reader.scale

    @property
    def time_interval(self):
        return self.reader.time_interval

    @property
    def dimension_properties(self):
        return self.reader.dimension_properties

class OverlapDetection():
    def __init__(self, transform_models, dataframes, dsxy, dsz, prefix, file_type, overlap_only):
        self.transform_models = transform_models
        self.image_loader_df = dataframes['image_loader']
        self.dsxy, self.dsz = dsxy, dsz
        self.prefix = prefix
        self.file_type = file_type
        self.to_process = {}
        self.image_shape_cache = {}
        self.max_interval_size = 0
        self.overlap_only = overlap_only
    
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
    
    def read_s3_json(self, s3_path):
        no_scheme = s3_path.replace("s3://", "", 1)
        bucket, key = no_scheme.split("/", 1)

        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

        keys_to_try = [key]

        # If caller asks for v3 metadata, fallback to v2 metadata in same folder.
        if key.endswith("/zarr.json"):
            base = key[: -len("/zarr.json")]
            keys_to_try.append(base + "/.zarray")  # v2 array metadata
            keys_to_try.append(base + "/.zattrs")  # v2 group/root attrs

        last_error = None

        for candidate_key in keys_to_try:
            try:
                response = s3.get_object(Bucket=bucket, Key=candidate_key)
                body = response["Body"]

                try:
                    return json.loads(body.read().decode("utf-8"))
                finally:
                    body.close()

            except ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code in ("NoSuchKey", "404", "NotFound"):
                    last_error = e
                    continue
                raise

        raise last_error

    def normalize_zarr_shape(self, raw_shape):
        if len(raw_shape) == 5:
            t, c, z, y, x = raw_shape
            return (t, c, 1, z, y, x)

        if len(raw_shape) == 3:
            z, y, x = raw_shape
            return (1, 1, 1, z, y, x)

        return tuple(raw_shape)
    
    def close_s3(self, s3):
        if s3 is None: return
        try:
            s3creator = getattr(s3, "_s3creator", None)
            if s3creator is not None:
                s3.close_session(s3.loop, s3creator)
        except Exception:
            pass
    
    def open_zarr(self, full_path):
        full_path = str(full_path).rstrip("/")
        root_path = full_path.replace("s3://", "", 1)

        def has_aws_credentials():
            import os
            return bool(
                os.environ.get("AWS_ACCESS_KEY_ID")
                and os.environ.get("AWS_SECRET_ACCESS_KEY")
            )

        def open_with_anon(anon):
            s3 = s3fs.S3FileSystem(
                anon=anon,
                skip_instance_cache=True,
            )

            try:
                mapper = s3fs.S3Map(
                    root=root_path,
                    s3=s3,
                    check=False,
                )

                if hasattr(zarr.storage, "FsspecStore"):
                    store = zarr.storage.FsspecStore.from_mapper(mapper)
                    return zarr.open(store, mode="r"), s3

                return zarr.open(mapper, mode="r"), s3

            except Exception:
                self.close_s3(s3)
                raise

        try:
            return open_with_anon(anon=True)
        except Exception as anon_error:
            if not has_aws_credentials():
                raise RuntimeError(
                    f"Anonymous S3 access failed for public Zarr store: {full_path}"
                ) from anon_error

            return open_with_anon(anon=False)

    def load_image_metadata(self, file_path):
        file_path = str(file_path).rstrip("/")

        if file_path in self.image_shape_cache:
            return self.image_shape_cache[file_path]

        if self.file_type == "zarr":
            meta = self.read_s3_json(file_path + "/zarr.json")
            shape = self.normalize_zarr_shape(meta["shape"])
            self.image_shape_cache[file_path] = shape
            return shape

        if self.file_type == "tiff":
            img = CustomBioImage(file_path, reader=bioio_tifffile.Reader)
            dask_array = img.get_dask_stack()
            shape = dask_array.shape
            self.image_shape_cache[file_path] = shape
            return shape

        raise ValueError(f"Unsupported file_type: {self.file_type}")

    def open_and_downsample(self, shape, dsxy, dsz):
        """
        Calculate the final XYZ shape after additional in-memory downsampling.
        """
        x_new, y_new, z_new = shape[5], shape[4], shape[3]

        f = dsxy
        while f > 1:
            x_new //= 2
            f //= 2

        f = dsxy
        while f > 1:
            y_new //= 2
            f //= 2

        f = dsz
        while f > 1:
            z_new //= 2
            f //= 2

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
        return [int(ub - lb + 1) for lb, ub in zip(lower_bound, upper_bound)]
    
    def floor_log2(self, n):
        """
        Return ⌊log2(n)⌋ - clamps n ≤ 1 to 1 so the result is 0 for n ≤ 1
        """
        return max(0, int(math.floor(math.log2(max(1, n)))))
    
    def get_zarr_multiscales(self, zarr_root_path):
        meta = self.read_s3_json(str(zarr_root_path).rstrip("/") + "/zarr.json")
        attrs = meta.get("attributes", meta)

        multiscales = attrs.get("multiscales") or attrs.get("ome", {}).get("multiscales")
        if not multiscales:
            raise ValueError(f"No multiscales metadata found at {zarr_root_path}")

        return multiscales[0]
    
    def get_zarr_num_levels(self, zarr_root_path):
        """
        Read number of saved OME-Zarr pyramid levels from root zarr.json.
        """
        meta = self.read_s3_json(str(zarr_root_path).rstrip("/") + "/zarr.json")
        attrs = meta.get("attributes", meta)

        multiscales = attrs.get("multiscales") or attrs.get("ome", {}).get("multiscales")
        if not multiscales:
            raise ValueError(f"No multiscales metadata found at {zarr_root_path}")

        return len(multiscales[0]["datasets"])

    def choose_zarr_level(self, zarr_root_path):
        """
        pick the highest power-of-two pyramid level compatible with dsxy/dsz
        """
        max_level = self.get_zarr_num_levels(zarr_root_path) - 1
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
        Compute XYZ overlap intervals using the actual stored Zarr pyramid scaling
        plus any remaining in-memory downsampling.
        """
        for i, row_i in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row_i['timepoint']}, setup: {row_i['view_setup']}"
            all_intervals = []

            if self.file_type == "zarr":
                zarr_root_path = self.prefix + row_i["file_path"]

                level, leftovers = self.choose_zarr_level(zarr_root_path)
                multiscale = self.get_zarr_multiscales(zarr_root_path)
                base_path = str(multiscale["datasets"][0]["path"])
                level_path = str(multiscale["datasets"][level]["path"])

                dim_base = self.load_image_metadata(zarr_root_path + f"/{base_path}")
                dim_selected_level = self.load_image_metadata(zarr_root_path + f"/{level_path}")

                dsxy = leftovers[0]
                dsz = leftovers[2]

                s = float(2 ** level)
                mipmap_of_downsample = self.affine_with_half_pixel_shift(s, s, s)
                downsampled_dim_base = self.open_and_downsample(dim_selected_level, dsxy, dsz)

            elif self.file_type == "tiff":
                dim_base = self.load_image_metadata(self.prefix + row_i["file_path"])
                mipmap_of_downsample = self.create_mipmap_transform()
                dsxy, dsz = self.dsxy, self.dsz
                level = None
                downsampled_dim_base = self.open_and_downsample(dim_base, dsxy, dsz)

            else:
                raise ValueError(f"Unsupported file_type: {self.file_type}")

            if not self.overlap_only:
                lower_bound = np.array(downsampled_dim_base[0], dtype=int)
                upper_bound = np.array(downsampled_dim_base[1], dtype=int) - 1

                interval = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "span": (upper_bound - lower_bound + 1).tolist(),
                }

                self.to_process[view_id] = [interval]
                self.max_interval_size = max(
                    self.max_interval_size,
                    self.size_interval(lower_bound, upper_bound),
                )
                continue

            t1 = self.get_inverse_mipmap_transform(mipmap_of_downsample)

            for j, row_j in self.image_loader_df.iterrows():
                if i == j:
                    continue

                view_id_other = f"timepoint: {row_j['timepoint']}, setup: {row_j['view_setup']}"

                if self.file_type == "zarr":
                    zarr_root_path_other = self.prefix + row_j["file_path"]
                    level_other, _ = self.choose_zarr_level(zarr_root_path_other)
                    multiscale_other = self.get_zarr_multiscales(zarr_root_path_other)
                    base_path_other = str(multiscale_other["datasets"][0]["path"])
                    dim_other = self.load_image_metadata(zarr_root_path_other + f"/{base_path_other}")
                    s_other = float(2 ** level_other)
                    mipmap_of_downsample_other = self.affine_with_half_pixel_shift(s_other, s_other, s_other)

                else:
                    dim_other = self.load_image_metadata(self.prefix + row_j["file_path"])
                    mipmap_of_downsample_other = self.create_mipmap_transform()

                matrix = self.transform_models.get(view_id)
                matrix_other = self.transform_models.get(view_id_other)

                inverse_mipmap_other = self.get_inverse_mipmap_transform(mipmap_of_downsample_other)
                inverse_matrix = self.get_inverse_mipmap_transform(matrix)

                concatenated_matrix = np.dot(inverse_matrix, matrix_other)
                t2 = np.dot(inverse_mipmap_other, concatenated_matrix)

                intervals = self.estimate_bounds(t1, dim_base)
                intervals_other = self.estimate_bounds(t2, dim_other)

                bounding_boxes = tuple(np.round(v).astype(int) for v in intervals)
                bounding_boxes_other = tuple(np.round(v).astype(int) for v in intervals_other)

                overlaps = np.all(
                    (bounding_boxes[1] >= bounding_boxes_other[0])
                    & (bounding_boxes_other[1] >= bounding_boxes[0])
                )

                if not overlaps:
                    continue

                intersected_boxes = self.calculate_intersection(bounding_boxes, bounding_boxes_other)
                volume_bounds = (
                    np.array(downsampled_dim_base[0], dtype=int),
                    np.array(downsampled_dim_base[1], dtype=int) - 1,
                )
                intersect = self.calculate_intersection(volume_bounds, intersected_boxes)

                lb, ub = intersect
                intersect_dict = {
                    "lower_bound": lb,
                    "upper_bound": ub,
                    "span": self.calculate_new_dims(lb, ub),
                }

                self.max_interval_size = max(
                    self.max_interval_size,
                    self.size_interval(lb, ub),
                )

                all_intervals.append(intersect_dict)

            self.to_process[view_id] = all_intervals

        return dsxy, dsz, level, mipmap_of_downsample
                
    def run(self):
        """
        Executes the entry point of the script.
        """
        dsxy, dsz, level, mipmap_of_dowsample = self.find_overlapping_area()
        print("Overlapping Area Detected")
        
        return self.to_process, dsxy, dsz, level, self.max_interval_size, mipmap_of_dowsample
