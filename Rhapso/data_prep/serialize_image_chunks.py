import pyarrow as pa
import pyarrow.parquet as pq
import dask.array as da
import numpy as np

# This component flattens 3D images into 2D and parquets into S3

class SerializeImageChunks:
    def __init__(self, image_chunks, parquet_bucket_path):
        self.image_chunks = image_chunks
        self.parquet_bucket_path = parquet_bucket_path
        self.current_chunk_count = 0
        self.partition_count_per_file = 0
        self.file_count = 0
        self.columns = []
        self.shapes = []
        self.partitions = []
        self.partition_key = 0

    def serialize_image_chunk_as_table(self, image_chunk):
        if isinstance(image_chunk, da.Array):
            image_chunk = image_chunk.compute()

        arrays = []
        fields = []

        for i in range(image_chunk.shape[0]):  
            slice_flattened = image_chunk[i].ravel()
            pa_slice = pa.array([slice_flattened], type=pa.large_list(pa.float32()))
            arrays.append(pa_slice)
            fields.append(pa.field(f'slice_{i}', pa.large_list(pa.float32())))
        
        shape_array = pa.array([image_chunk.shape], type=pa.list_(pa.int32()))
        arrays.append(shape_array)
        fields.append(pa.field('shape', pa.list_(pa.int32())))

        schema = pa.schema(fields)
        table = pa.Table.from_arrays(arrays, schema=schema)

        np.set_printoptions(threshold=np.inf)
        return table

    def process_chunks_to_parquet(self):
        for image in self.image_chunks:
            
            chunk_table = self.serialize_image_chunk_as_table(image['image_chunk'])
            self.columns.append(chunk_table)
            
            self.current_chunk_count += 1

            # 6 image chunks = 1 partition 
            if self.current_chunk_count >= 6:
                
                # stack the 6 image chunk tables
                combined_table = pa.concat_tables(self.columns, promote=True)
                
                # set partition key for the group
                partition_keys = [str(self.partition_key)] * combined_table.num_rows
                partition_key_column = pa.array(partition_keys, type=pa.string())
                combined_table = combined_table.append_column('partition_key', partition_key_column)

                self.partitions.append(combined_table)
                self.columns = []
                self.shapes = []  
                self.current_chunk_count = 0 
                self.partition_count_per_file += 1
                self.partition_key += 1

                # 8 partitions = 1 parquet file 
                if self.partition_count_per_file >= 8: 

                    # combine partitions into 1 table
                    final_table = pa.concat_tables(self.partitions, promote=True)
                    filename = f'{self.file_count}.parquet'
                    
                    pq.write_to_dataset(
                        final_table, 
                        self.parquet_bucket_path + filename,
                        partition_cols=['partition_key'], 
                        compression='ZSTD'
                    )
                    
                    self.file_count += 1
                    self.partitions = []
                    self.partition_count_per_file = 0

    def run(self):
        self.process_chunks_to_parquet()