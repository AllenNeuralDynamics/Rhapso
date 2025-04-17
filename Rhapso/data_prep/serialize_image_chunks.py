import pyarrow as pa
import pyarrow.parquet as pq
import dask.array as da

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

    def flatten_image_chunk_as_table(self, image_chunk, view_id, interval):
        """
        Flattens each slice of a multi-dimensional image chunk and compiles the flattened data
        and associated metadata into a structured Apache Arrow table. This table includes
        detailed information about the image view and spatial dimensions of the intervals processed.
        """
        if isinstance(image_chunk, da.Array): image_chunk = image_chunk.compute()

        arrays = []
        fields = []

        # Flatten each slice of the image chunk and create an Apache Arrow array for each 
        for i in range(image_chunk.shape[0]):  
            slice_flattened = image_chunk[i].ravel()
            pa_slice = pa.array([slice_flattened], type=pa.large_list(pa.float32()))
            arrays.append(pa_slice)
            fields.append(pa.field(f'slice_{i}', pa.large_list(pa.float32())))
        
        # Extract interval information to be included in metadata
        lower_bound, upper_bound, span = interval
        lower_bound_flat = list(lower_bound)
        upper_bound_flat = list(upper_bound)
        span_flat = list(span)
        shape_flat = list(image_chunk.shape)

        # Prepare and structure the metadata as an Apache Arrow struct array
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

        # Construct the final table schema and build the table from the arrays
        schema = pa.schema(fields)
        table = pa.Table.from_arrays(arrays, schema=schema)

        return table

    def process_chunks_to_parquet(self):
        """
        Processes and groups image chunks into partitions and writes them to Parquet files with structured 
        partitioning.
        """
        for image in self.image_chunks:
            # Extract image chunk and metadata from the current image
            image_chunk = image['image_chunk']
            view_id = image['view_id']
            interval = image['interval_key']  

            # Flatten the image chunk into a structured table format
            chunk_table = self.flatten_image_chunk_as_table(image_chunk, view_id, interval)
            self.columns.append(chunk_table)
            
            self.current_chunk_count += 1

            # Combine chunks into a partition after every 6 chunks
            if self.current_chunk_count >= 6:
                
                # Concatenate all chunk tables into one table
                combined_table = pa.concat_tables(self.columns, promote=True)
                
                # Assign a partition key for efficient data management
                partition_keys = [str(self.partition_key)] * combined_table.num_rows
                partition_key_column = pa.array(partition_keys, type=pa.string())
                combined_table = combined_table.append_column('partition_key', partition_key_column)

                # Add the combined table to the list of partitions
                self.partitions.append(combined_table)
                
                # Reset the columns for the next group of chunks
                self.columns = []
                self.shapes = []  
                self.current_chunk_count = 0 
                self.partition_count_per_file += 1
                self.partition_key += 1

                # Write to a Parquet file after every 8 partitions 
                if self.partition_count_per_file >= 8: 

                    # Combine all partitions into a single table
                    final_table = pa.concat_tables(self.partitions, promote=True)
                    filename = f'{self.file_count}.parquet'
                    
                    # Write the table to a dataset with partitioning
                    pq.write_to_dataset(
                        final_table, 
                        self.parquet_bucket_path + filename,
                        partition_cols=['partition_key'], 
                        compression='ZSTD'
                    )
                    
                    # Update the file count and reset partitions for the next file
                    self.file_count += 1
                    self.partitions = []
                    self.partition_count_per_file = 0

        # Handle any remaining columns that did not complete a partition  
        if self.columns:
            combined_table = pa.concat_tables(self.columns, promote=True)
            partition_keys = [str(self.partition_key)] * combined_table.num_rows
            partition_key_column = pa.array(partition_keys, type=pa.string())
            combined_table = combined_table.append_column('partition_key', partition_key_column)

            self.partitions.append(combined_table)
            self.columns = []
            self.partition_key += 1

        # Check for any remaining partitions that have not been written to a parquet file
        if self.partitions:
            final_table = pa.concat_tables(self.partitions, promote=True)
            filename = f'{self.file_count}.parquet'
            
            # Final write operation for any leftover partitions
            pq.write_to_dataset(
                final_table,
                self.parquet_bucket_path + filename,
                partition_cols=['partition_key'],
                compression='ZSTD'
            )

    def run(self):
        """
        Executes the entry point of the script.
        """
        self.process_chunks_to_parquet()