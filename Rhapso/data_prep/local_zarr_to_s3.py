import zarr
import s3fs
import dask.array as da

class LocalZarrToS3():
    def __init__(self):
        # Local file path where the Zarr dataset is stored
        self.local_path = '/Users/seanfite/Desktop/Rhapso-Final-Output/'
        # S3 path where you want to save the dataset
        self.s3_path = 's3://interest-point-detection/big-stitcher-output'

    def fetch_zarr_data(self):
        # Open the local Zarr file
        zarr_local = zarr.open(self.local_path, mode='r')
        # Convert to a Dask array for potentially large data handling
        dask_array = da.from_zarr(zarr_local)
        return dask_array

    def save_to_s3(self, dask_array):
        # Set up the S3 connection
        s3 = s3fs.S3FileSystem(anon=False)
        # Create an S3 map to the specified path
        store = s3fs.S3Map(root=self.s3_path, s3=s3)
        # Use Dask array to store data back to Zarr, this step may require compute to trigger the write
        da.to_zarr(dask_array, store, compute=True)

    def run(self):
        # Fetch the data from local Zarr
        data = self.fetch_zarr_data()
        # Save it to S3
        self.save_to_s3(data)

# Create an instance of the class and run the process
if __name__ == "__main__":
    process = LocalZarrToS3()
    process.run()
