from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType
from awsglue.utils import getResolvedOptions
from pyspark.sql import functions as F
from awsglue.job import Job
from io import BytesIO
from io import StringIO
import boto3
import json
import numpy as np
import zipfile
import pandas as pd
import os
import subprocess
from bioio import BioImage
import bioio_tifffile
import dask.array as da
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from dask import delayed
from dask import compute
import sys

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

class SparkETLInterestPointDetection:
    def __init__(self, bucket_name, s3):
        self.sc = SparkContext.getOrCreate()  
        self.spark = SparkSession.builder.appName("Interest Point Detection").getOrCreate()
        # self.spark = SparkSession.builder.master("local[*]").appName("MyApp").getOrCreate()

        self.glue_context = GlueContext(self.sc)
        self.params = {}
        self.dsxy = None
        self.dsz = None
        self.prefix = None
        self.bounds = None
        self.image_tile_paths = None
        self.bucket_name = bucket_name
        self.s3 = s3
        self.merged_df = None

        self.localization = 1
        self.downsample_z = 2
        self.downsample_xy = 4
        self.image_sigma_x = 0.5
        self.image_sigma_y = 0.5
        self.image_sigma_z = 0.5
        self.min_intensity = 0.0
        self.max_intensity = 2048.0
        self.block_size = 1024,1024,1024
        self.sigma = 1.8
        self.threshold = 0.008
        self.find_min = False
        self.find_max = True
        self.image_data = None
        self.k_min_1_inv = None
        self.mask_float = None
        self.interest_points = []
    
    def load_dataframes_from_s3(self):
        paginator = self.s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix='dataframes/')
        
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    file_key = obj['Key']
                    if 'bounds.csv' in file_key:
                        # Load bounds.csv into DataFrame
                        response = self.s3.get_object(Bucket=self.bucket_name, Key=file_key)
                        bounds_content = response['Body'].read().decode('utf-8')
                        self.bounds = pd.read_csv(StringIO(bounds_content))
                    elif 'image_loader.csv' in file_key:
                        # Load image_loader.csv into DataFrame
                        response = self.s3.get_object(Bucket=self.bucket_name, Key=file_key)
                        image_loader_content = response['Body'].read().decode('utf-8')
                        self.image_tile_paths = pd.read_csv(StringIO(image_loader_content))
    
    def load_params_from_s3(self):
        try:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key='params')
            params_body = obj['Body'].read().decode('utf-8')
            params = json.loads(params_body)
            self.dsxy = params.get('dsxy')
            self.dsz = params.get('dsz')
            self.prefix = params.get('prefix')

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {str(e)}")
            raise
        except Exception as e:
            print(f"Failed to load 'params' file: {str(e)}")
            raise
    
    def merge_dataframes(self):
        self.merged_df = pd.merge(self.bounds, self.image_tile_paths, on=['timepoint', 'view_setup'], how='left')  

    def dynamic_frame(self):
        # Group bounds into viewIDs
        grouped_bounds_dict = self.bounds.groupby(['timepoint', 'view_setup']).apply(lambda x: x.drop(['timepoint', 'view_setup'], axis=1).to_dict('records')).to_dict()
        data_dict = self.image_tile_paths.to_dict(orient='records')
        for record in data_dict:
            key = (record['timepoint'], record['view_setup'])
            bounds_data = grouped_bounds_dict.get(key, [])
            record['bounds'] = bounds_data
            # add broadcast params here so we dont have to deal with passing in globals
        
        bounds_element_schema = StructType([
            StructField("span_z", IntegerType(), True),
            StructField("upper_bound_z", IntegerType(), True),
            StructField("span_y", IntegerType(), True),
            StructField("span_x", IntegerType(), True),
            StructField("upper_bound_x", IntegerType(), True),
            StructField("upper_bound_y", IntegerType(), True),
            StructField("lower_bound_x", IntegerType(), True),
            StructField("lower_bound_y", IntegerType(), True),
            StructField("lower_bound_z", IntegerType(), True)
        ])

        schema = StructType([
            StructField("view_setup", IntegerType(), True),
            StructField("timepoint", IntegerType(), True),
            StructField("series", IntegerType(), True),
            StructField("channel", IntegerType(), True),
            StructField("file_path", StringType(), True),
            StructField("bounds", ArrayType(bounds_element_schema), True)
        ])

        # Convert to Dynamic Frame
        df = self.spark.createDataFrame(data_dict, schema=schema)
        dyf = DynamicFrame.fromDF(df, self.glue_context, "image_data")
        return dyf
    
    def interest_point_detection(self):
        
        def process_group(group): 

            prefix = 's3://rhapso-tif-sample/IP_TIFF_XML/'
            dsxy = 4
            dsz = 2  
            sigma = 1.8
            threshold = 0.008
            min_intensity = 0
            max_intensity = 2048
            
            def compute_sigma_difference(sigma, image_sigma):
                print("we are in comput sigma difference")
                steps = len(sigma) - 1
                sigma_diff = np.zeros(steps + 1)
                sigma_diff[0] = np.sqrt(sigma[0]**2 - image_sigma**2)

                for i in range(1, steps + 1):
                    sigma_diff[i] = np.sqrt(sigma[i]**2 - image_sigma**2)

                return sigma_diff
            
            def compute_sigma(steps, k, initial_sigma):
                print("we are in compute sigma")
                sigma = np.zeros(steps + 1)
                sigma[0] = initial_sigma

                for i in range(1, steps + 1):
                    sigma[i] = sigma[i - 1] * k

                return sigma
    
            def compute_sigmas(initial_sigma, shape):
                print("we are in computer sigmas")
                k = 2 ** (1 / 4)
                k_min_1_inv = 1.0 / (k - 1.0)
                steps = 3
                sigma = np.zeros((2, shape))

                for i in range(shape):
                    sigma_steps_x = compute_sigma(steps, k, initial_sigma)
                    sigma_steps_diff_x = compute_sigma_difference(sigma_steps_x, 0.5)
                    sigma[0][i] = sigma_steps_diff_x[0]  
                    sigma[1][i] = sigma_steps_diff_x[1]
                
                print("we are returning sigma")
            
                return (sigma, k_min_1_inv)

            def normalize_image(image, min_intensity, max_intensity):
                normalized_image = (image - min_intensity) / (max_intensity - min_intensity)
                return normalized_image
            
            def apply_gaussian_blur(input_float, sigma, shape):
                blurred_image = input_float
                print("we are in apply blur")
                print("this is the shape:")
                print(shape)
                print(sigma)
                
                for i in range(shape):
                    print(f"Starting gaussian_filter for index {i} with sigma={sigma[i]}")
                    blurred_image = gaussian_filter(blurred_image, sigma=sigma[i], mode='reflect')
                    print("we are in for loop in blur")
                
                print("we are returning blur")
                return blurred_image
            
            def gaussian_3d(xyz, amplitude, xo, yo, zo, sigma_x, sigma_y, sigma_z, offset):
                x, y, z = xyz
                g = offset + amplitude * np.exp(
                    -(((x - xo) ** 2) / (2 * sigma_x ** 2) +
                    ((y - yo) ** 2) / (2 * sigma_y ** 2) +
                    ((z - zo) ** 2) / (2 * sigma_z ** 2)))
                return g.ravel()

            def refine_peaks(peaks, image):
                refined_peaks = []
                window_size = 5

                for peak in peaks:
                    x, y, z = peak
                    # Check if the peak is too close to any border
                    if (x < window_size or x >= image.shape[0] - window_size or
                        y < window_size or y >= image.shape[1] - window_size or
                        z < window_size or z >= image.shape[2] - window_size):
                        continue

                    # Extract a volume around the peak
                    patch = image[x-window_size:x+window_size+1,
                                y-window_size:y+window_size+1,
                                z-window_size:z+window_size+1]

                    # Prepare the data for fitting
                    x_grid, y_grid, z_grid = np.mgrid[
                        -window_size:window_size+1,
                        -window_size:window_size+1,
                        -window_size:window_size+1]
                    initial_guess = (patch.max(), 0, 0, 0, 1, 1, 1, 0)  # Amplitude, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset

                    try:
                        popt, _ = curve_fit(
                            gaussian_3d, (x_grid, y_grid, z_grid), patch.ravel(),
                            p0=initial_guess)
                        refined_x = x + popt[1]
                        refined_y = y + popt[2]
                        refined_z = z + popt[3]
                        refined_peaks.append((refined_x, refined_y, refined_z))
                    except Exception as e:
                        refined_peaks.append((x, y, z))

                return refined_peaks

            def compute_difference_of_gaussian(image, shape, sigma, threshold, min_intensity, max_intensity):

                print("starting for: ", group['file_path'])
                initial_sigma = sigma
                min_peak_value = threshold
                min_initial_peak_value = min_peak_value / 3.0
                k = 2 ** (1 / 4)

                # normalize image using min/max intensities
                input_float = normalize_image(image, min_intensity, max_intensity)  
                print("we just got input float")                                            

                # calculate gaussian blur levels 
                sigmas, k_min_1_inv = compute_sigmas(initial_sigma, shape)  
                print("we just got sigmas")

                sigma_1 = sigmas[0]
                sigma_2 = sigmas[1]                     

                # # apply gaussian blur
                blurred_image_1 = apply_gaussian_blur(input_float, sigma_1, shape)                
                blurred_image_2 = apply_gaussian_blur(input_float, sigma_2, shape)
                print("we just blurred images")

                # subtract blurred images
                dog = blurred_image_1 - blurred_image_2
                print("we just subtracted the blur")

                # detect peaks
                peaks = peak_local_max(dog, threshold_rel=min_initial_peak_value)
                print("we just got peaks")

                # refine localization 
                # final_peaks = refine_peaks(peaks, image)   
                # print("final peaks:")
                # print(final_peaks)

                print("peaks")
                print(peaks)
                
                return peaks                       

            def apply_dog_to_all_slices(downsampled_slices, shape, sigma, threshold, file_path):
                results = []
                
                for slice_dask in downsampled_slices:
                    print("our file path: ", file_path)
                    dog_result = compute_difference_of_gaussian(slice_dask, shape, sigma, threshold, min_intensity, max_intensity)
                    results.append(dog_result)

                return results
            
            def downsample(data, factor_dx, factor_dy, factor_dz, axes):
                for axis in axes:
                    if axis == 0: 
                        while factor_dz > 1:
                            data = da.coarsen(np.mean, data, {0:2}, trim_excess=True)
                            factor_dz //= 2  
                    if axis == 1: 
                        while factor_dx > 1:
                            data = da.coarsen(np.mean, data, {1:2}, trim_excess=True)
                            factor_dx //= 2
                    if axis == 2:
                        while factor_dy > 1:
                            data = da.coarsen(np.mean, data, {2:2}, trim_excess=True)
                            factor_dy //= 2
                return data   

            def load_and_process_slices(file_name, bounds, dsxy, dsz, prefix):
                file_path = prefix + file_name
                img = BioImage(file_path, reader=bioio_tifffile.Reader)
                full_dask_stack = img.get_dask_stack()[0, 0, 0, :, :, :]
                shape = full_dask_stack.shape
                downsampled_slices = []

                for interval in bounds:
                    z_start = interval['lower_bound_z']
                    z_stop = interval['upper_bound_z'] + 1
                    
                    y_start = interval['lower_bound_y']
                    y_stop = interval['upper_bound_y'] + 1
                    
                    x_start = interval['lower_bound_x']
                    x_stop = interval['upper_bound_x'] + 1

                    slice = full_dask_stack[z_start:z_stop, y_start:y_stop, x_start:x_stop]
                    downsampled_slice = downsample(slice, dsxy, dsxy, dsz, axes=[0, 1, 2])
                    downsampled_slices.append(downsampled_slice)
                
                del full_dask_stack

                return (downsampled_slices, shape)

            images, shape = load_and_process_slices(group['file_path'], group['bounds'], dsxy, dsz, prefix)
            final_peaks = apply_dog_to_all_slices(images, len(shape), sigma, threshold, group['file_path'])        
            
            return final_peaks

        dyf = self.dynamic_frame()
        output = dyf.map(process_group, transformation_ctx="process_group_ctx")
        return output
    
    def send_interest_points_to_s3(self, output):
        data_frame = output.toDF()
        folder_key = "interest_points/data.json"  
        path = f"s3://{self.bucket_name}/{folder_key}"
        data_frame.write.format("json").mode("overwrite").save(path)

    def run(self):
        self.load_dataframes_from_s3()
        self.load_params_from_s3()
        self.merge_dataframes()
        output = self.interest_point_detection()
        output.show()

s3 = boto3.client('s3')
bucket_name = 'interest-point-detection'

spark_etl = SparkETLInterestPointDetection(bucket_name, s3)
spark_etl.run()

job.commit()


# Local Testing in Spark Env:

# AWS GLUE DOCKER CONTAINER
# docker run -d --name glue --user root -v /Users/seanfite/Desktop/Rhapso:/app/logs -it amazon/aws-glue-libs:glue_libs_4.0.0_image_01

# ACCESS DOCKER CONTAINER TO ADD DIRECTORY:
# docker exec -it glue /bin/bash
# mkdir -p /app/spark_etl

# STAY IN DOCKER CONTAINER AND ADD PACKAGES:
# curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# python3 get-pip.py
# pip install boto3
# pip install bioio
# pip install bioio-tifffile
# pip install matplotlib
# aws configure

# UPDATE RUN SCRIPT IN DOCKER CONTAINER
# docker cp /Users/seanfite/Desktop/Rhapso/interest_point_detection/spark_etl/interest_point_detection_spark.py glue:/app/spark_etl/interest_point_detection.py

# RUN DOCKER CONTAINER 
# docker exec -it glue python3 /app/spark_etl/interest_point_detection.py --JOB_NAME your_job_name