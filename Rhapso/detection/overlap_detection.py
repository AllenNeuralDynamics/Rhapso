import tifffile
import dask.array as da
import io

# This component recieves image data, downsamples it, and detects overlap - using a cache based lazy-load system

class OverlapDetection:
    def __init__(self, data_frame, s3):
        self.data_frame_dict = data_frame
        self.s3 = s3
        self.image_cache = {}
    
    def load_image_from_s3_to_buffer(self, bucket_name, object_key):
        response = self.s3.get_object(Bucket=bucket_name, Key=object_key)
        image_data = response['Body'].read()
        return image_data

    def load_image_to_dask(self, bucket_name, object_key, series):
        cache_key = (bucket_name, object_key, series)

        if cache_key not in self.image_cache:
            image_data = self.load_image_from_s3_to_buffer(bucket_name, object_key)
            # Read the TIFF image directly from the bytes buffer
            with tifffile.TiffFile(io.BytesIO(image_data)) as tif:
                img_series = tif.series[series].aszarr()
                # Load as a Dask array
                img_dask = da.from_zarr(img_series)  

                # Cache the dask array for this series
                self.image_cache[cache_key] = img_dask

        return self.image_cache[cache_key]

    def get_image_slice(self, bucket_name, object_key, series, channel, timepoint, z):
        img_dask = self.load_image_to_dask(bucket_name, object_key, series)
        # Fetch the specific slice from the dask array
        img_slice = img_dask[channel, z, :, :, timepoint]
        
        return img_slice.compute()

    def process_images(self, image_loader_df):
        for _, row in image_loader_df.iterrows():
            bucket_name, object_key = row['file_path'].split('/', 1)
            self.get_image_slice(
                bucket_name=bucket_name,
                object_key=object_key,
                series=row['series'],
                channel=row['channel'],
                timepoint=row['timepoint'],
                z=0  
            )

    def run(self):
        image_loader_df = self.data_frame_dict.get('image_loader')
        self.process_images(image_loader_df)

        # print(image_loader_df)
