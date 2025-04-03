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

    def serialize_image_chunk_as_table(self, image_chunk, view_id, interval):
        if isinstance(image_chunk, da.Array):
            image_chunk = image_chunk.compute()

        arrays = []
        fields = []

        for i in range(image_chunk.shape[0]):  
            slice_flattened = image_chunk[i].ravel()
            pa_slice = pa.array([slice_flattened], type=pa.large_list(pa.float32()))
            arrays.append(pa_slice)
            fields.append(pa.field(f'slice_{i}', pa.large_list(pa.float32())))
        
        lower_bound, upper_bound, span = interval

        lower_bound_flat = list(lower_bound)
        upper_bound_flat = list(upper_bound)
        span_flat = list(span)
        shape_flat = list(image_chunk.shape)

        # Prepare metadata as a struct
        metadata_struct = [
            pa.array([view_id], type=pa.string()),
            pa.array([lower_bound_flat], type=pa.list_(pa.int32())),
            pa.array([upper_bound_flat], type=pa.list_(pa.int32())),
            pa.array([span_flat], type=pa.list_(pa.int32())),
            pa.array([shape_flat], type=pa.list_(pa.int32())),
        ]

        metadata_array = pa.StructArray.from_arrays(metadata_struct, names=['View ID', 'Lower Bound', 'Upper Bound', 'Span', 'Shape'])
        arrays.append(metadata_array)
        fields.append(pa.field('metadata', metadata_array.type))

        schema = pa.schema(fields)
        table = pa.Table.from_arrays(arrays, schema=schema)

        return table

    def process_chunks_to_parquet(self):
        for image in self.image_chunks:
            image_chunk = image['image_chunk']
            view_id = image['view_id']
            interval = image['interval_key']  
            chunk_table = self.serialize_image_chunk_as_table(image_chunk, view_id, interval)
            self.columns.append(chunk_table)
            
            self.current_chunk_count += 1

            # 6 image chunks = 1 partition (128mb)
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

                # 8 partitions = 1 parquet file (1gb)
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
                
        if self.columns:
            # There are remaining columns that didn't make up a full partition
            combined_table = pa.concat_tables(self.columns, promote=True)
            partition_keys = [str(self.partition_key)] * combined_table.num_rows
            partition_key_column = pa.array(partition_keys, type=pa.string())
            combined_table = combined_table.append_column('partition_key', partition_key_column)

            self.partitions.append(combined_table)
            self.columns = []
            self.partition_key += 1

        # Now check for any remaining partitions that haven't been written to a parquet file
        if self.partitions:
            final_table = pa.concat_tables(self.partitions, promote=True)
            filename = f'{self.file_count}.parquet'
            
            pq.write_to_dataset(
                final_table,
                self.parquet_bucket_path + filename,
                partition_cols=['partition_key'],
                compression='ZSTD'
            )

    def run(self):
        self.process_chunks_to_parquet()