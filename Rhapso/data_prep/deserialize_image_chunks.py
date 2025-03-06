import numpy as np
import pyarrow.parquet as pq

class DeserializeImageChunks:
    def __init__(self):
        pass

    def process_image_data(self, record):
        shape = tuple(record['shape'])
        image_data = []
        slice_keys = [k for k in record.keys() if k.startswith('slice_')]
        sorted_slice_keys = sorted(slice_keys, key=lambda x: int(x.split('_')[1]))

        for key in sorted_slice_keys:
            slice_data = record[key]
            if slice_data is not None:
                if slice_data is None:
                    break
                image_data.extend(slice_data)
            else:
                break
        
        if image_data:
            image_array = np.array(image_data)
            reshaped_image = image_array.reshape(shape)
            return reshaped_image
        else:
            print("No valid data for image, cannot reshape.")
            return None

    def run(self, image_chunks_dyf):
        return self.process_image_data(image_chunks_dyf)  

# DEBUGGING HELPERS  
# ----------------------

# self.file_path = 's3://interest-point-detection/ipd-staging/'
# self.partition_key =  1

# def fetch_data_by_partition_key(self):
#     try:
#         table = pq.read_table(
#             self.file_path,
#             filters=[('partition_key', '=', self.partition_key)]
#         )
#         df = table.to_pandas()
#         return df
#     except Exception as e:
#         print("Error reading parquet data from:", self.file_path, "Error:", e)