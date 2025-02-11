from pyspark.sql import SparkSession
from bioio import BioImage
import bioio_tifffile
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import boto3
import json

# TODO -
# Integrate image loading functionality from the base python version
# Once running, refactor it to use a map function
# Then update code in git and push new wheel file actions update
# Run spark-etl pipeline script and ensure we can get it running there too

# AWS GLUE DOCKER CONTAINER
# Create aws glue docker container

# Access docker container to update:
# docker exec -it glue /bin/bash 

# Update docker container script
# docker cp /Users/seanfite/Desktop/Rhapso/interest_point_detection/spark_etl/interest_point_detection.py glue:/app/spark_etl/interest_point_detection.py

# Run docker container 
# docker exec -it glue python3 /app/spark_etl/interest_point_detection.py

class TiffImageReader:
    def __init__(self, dsxy, dsz, overlapping_interval, process_intervals, file_path):
        self.dsxy, self.dsz = dsxy, dsz
        self.overlapping_interval = overlapping_interval
        self.process_intervals = process_intervals
        self.file_path = file_path
        
        self.downsampled_dask_images = None
        self.image_data = {}

    # downsampling by factor of 2 for X,Y,Z
    def downsample(self, data, factor_dx, factor_dy, factor_dz, axes):
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
        
    # get image data and downsample
    def load_image_metadata(self, file_path):
        img = BioImage(file_path, reader=bioio_tifffile.Reader)
        initial_dask_images = img.get_dask_stack()[0, 0, 0, :, :, :]
        self.downsampled_dask_images = self.downsample(initial_dask_images, self.dsxy, self.dsxy, self.dsz, axes=[0, 1, 2])    

    # get image as a slice within bounds 
    def fetch_image_slice(self, z, interval):
        try: 
            lb, ub = interval['lower_bound'], interval['upper_bound']
            data_slice = self.downsampled_dask_images[z, lb[1]:ub[1], lb[2]:ub[2]].compute()
            return data_slice

        except Exception as e:
            print(f"Error fetching image slice: {e}")
            return
        
    # iterate through all intervals and then z values
    def fetch_all_slices(self):
        z_values = set()
        for interval in self.process_intervals:
            lower_z = interval['lower_bound'][2]
            upper_z = interval['upper_bound'][2]
            z_values.update(range(lower_z, upper_z + 1))
            for z in sorted(z_values):
                bounds_key = (z, tuple(interval['lower_bound']), tuple(interval['upper_bound']))
                self.image_data[bounds_key] = self.fetch_image_slice(z, interval)
                # self.visualize_slice(self.image_data[bounds_key])

    # display image slice
    def visualize_slice(self, image):
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(image, cmap='gray')  
            plt.title(f"Image Slice")
            plt.colorbar()
            plt.show()
        except Exception as e:
            print(f"Error fetching image slice: {e}")

    def run(self):
        self.load_image_metadata(self.file_path)
        self.fetch_all_slices()      
        return self.image_data

class SparkETLInterestPointDetection:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.tiff_image_reader = TiffImageReader
        self.spark = SparkSession.builder.appName("Interest Point Detection").getOrCreate()
        # args = getResolvedOptions(sys.argv, ['bucket_name'])
        # bucket_name = args['bucket_name']
        self.bucket_name = 'interest-point-detection'
        self.dataframes = {}
        self.params = {}
        self.load_dataframes_from_s3()
        self.load_params_from_s3()
        self.image_loader_df = self.dataframes.get('dataframes/image_loader.csv')
        self.dsxy = self.params['dsxy']
        self.dsz = self.params['dsz']
        self.prefix = self.params['prefix']
    
    def load_dataframes_from_s3(self):
        print("Retrieving dataframes from the 'dataframes/' folder:")
        paginator = self.s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix='dataframes/')
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    self.dataframes[obj['Key']] = obj['Size']
    
    def load_params_from_s3(self):
        print("Loading parameters from the 'params' file:")
        try:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key='params')
            params_body = obj['Body'].read().decode('utf-8')
            self.params = json.loads(params_body)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {str(e)}")
            raise
        except Exception as e:
            print(f"Failed to load 'params' file: {str(e)}")
            raise
    
    def create_empty_folder_in_s3(self):
        folder_key = "future_folder_of_interest_points/"  
        self.s3.put_object(Bucket=self.bucket_name, Key=folder_key)

    def run(self):
        print(self.dataframes)
        print(self.params)
        self.create_empty_folder_in_s3()

if __name__ == "__main__":
    etl_job = SparkETLInterestPointDetection()
    etl_job.run()