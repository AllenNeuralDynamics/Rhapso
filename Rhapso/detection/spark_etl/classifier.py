from bioio import BioImage
import bioio_tifffile
import numpy as np
import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
from collections import defaultdict

def group_slices(slices):
    grouped = defaultdict(list)
    for slice in slices:
        grouped[slice.shape].append(slice)
    return grouped

def convert_slice_to_1d(grouped_images):
    dataframes = {}
    for dims, group in grouped_images.items():
        for img_index, image in enumerate(group):
            z_slices = image.shape[0]
            for z in range(z_slices):
                slice_2d = image[z, :, :]
                slice_flattened = slice_2d.reshape(-1)
                num_columns = slice_flattened.shape[0]
                column_names = [f'pixel_{j}' for j in range(num_columns)]
                ddf = dd.from_dask_array(slice_flattened.reshape(1, -1), columns=column_names)
                dataframes[(dims, img_index, z)] = ddf

    return dataframes

def convert_chunk_to_1d(grouped_images):
    dataframes = {}
    for dims, group in grouped_images.items():
        for img_index, image in enumerate(group):
            flattened_image = image.ravel()
            num_columns = flattened_image.shape[0]
            column_names = [f'pixel_{j}' for j in range(num_columns)]
            ddf = dd.from_dask_array(flattened_image.reshape(1, -1), columns=column_names)
            dataframes[dims, img_index] = ddf
            
    return dataframes

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
    
# Get image data and load within a bound and downsampled
def load_and_process_slices(process_intervals, file_path, dsxy, dsz):
    img = BioImage(file_path, reader=bioio_tifffile.Reader)
    full_dask_stack = img.get_dask_stack()[0, 0, 0, :, :, :]
    downsampled_slices = []

    for interval in process_intervals:
        z_start, z_stop = interval['lower_bound'][2], interval['upper_bound'][2] + 1
        y_start, y_stop = interval['lower_bound'][1], interval['upper_bound'][1] + 1
        x_start, x_stop = interval['lower_bound'][0], interval['upper_bound'][0] + 1

        slice = full_dask_stack[z_start:z_stop, y_start:y_stop, x_start:x_stop]
        downsampled_slice = downsample(slice, dsxy, dsxy, dsz, axes=[0, 1, 2])
        downsampled_slices.append(downsampled_slice)
    
    grouped_slices = group_slices(downsampled_slices)
    dataframes = convert_chunk_to_1d(grouped_slices)
    return dataframes

def save_to_parquet(dataframes, bucket, path, timepoint, setup):
    for (dims, img_index), ddf in dataframes.items():
        dimension_info = f"{dims[0]}x{dims[1]}x{dims[2]}"
        target_path = f'{path}/{timepoint}_{setup}_{dimension_info}_copy{img_index}'
        full_path = f's3://{bucket}/{target_path}'
        ddf.to_parquet(full_path, engine='pyarrow')

def main():
    dsxy, dsz = 4, 2
    file_path = '/Users/seanfite/Desktop/AllenInstitute/Rhapso/Data/IP_TIFF_XML/spim_TL30_Angle225.tif'
    bucket = 'interest-point-detection'
    path = 'flattened-images-2'
    timepoint = 'timepoint_30'
    setup = 'setup_5'
    process_intervals = {
        'timepoint_30_setup_5': [
            {'lower_bound': [0, 14, 0], 'upper_bound': [347, 246, 46], 'span': [348, 232, 47]},
            {'lower_bound': [0, 11, 0], 'upper_bound': [347, 249, 46], 'span': [348, 238, 47]},
            {'lower_bound': [0, 0, 0], 'upper_bound': [347, 260, 45], 'span': [348, 261, 46]},
            {'lower_bound': [0, 18, 0], 'upper_bound': [347, 242, 46], 'span': [348, 224, 47]},
            {'lower_bound': [0, 99, 0], 'upper_bound': [347, 161, 46], 'span': [348, 62, 47]},
            {'lower_bound': [0, 13, 0], 'upper_bound': [347, 247, 46], 'span': [348, 234, 47]},
            {'lower_bound': [0, 99, 0], 'upper_bound': [347, 161, 46], 'span': [348, 62, 47]},
            {'lower_bound': [0, 13, 0], 'upper_bound': [347, 247, 46], 'span': [348, 234, 47]},
            {'lower_bound': [0, 0, 0], 'upper_bound': [347, 260, 45], 'span': [348, 261, 46]},
            {'lower_bound': [0, 18, 0], 'upper_bound': [347, 242, 46], 'span': [348, 224, 47]},
            {'lower_bound': [0, 11, 0], 'upper_bound': [347, 249, 46], 'span': [348, 238, 47]},
            {'lower_bound': [0, 0, 0], 'upper_bound': [347, 260, 46], 'span': [348, 261, 47]},
            {'lower_bound': [0, 14, 0], 'upper_bound': [347, 246, 46], 'span': [348, 232, 47]}
        ]
    }
    intervals = process_intervals['timepoint_30_setup_5']
    image_dataframes = load_and_process_slices(intervals, file_path, dsxy, dsz)
    save_to_parquet(image_dataframes, bucket, path, timepoint, setup)

main()
