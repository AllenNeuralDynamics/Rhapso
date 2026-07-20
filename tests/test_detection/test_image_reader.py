import unittest
from unittest.mock import patch, MagicMock
import dask.array as da
import numpy as np

from Rhapso.detection.image_reader import ImageReader


class TestImageReader(unittest.TestCase):
    def test_fetch_image_data_crop_applied_before_downsampling(self):
        """Test that crop is applied after transpose, before downsampling"""
        reader = ImageReader(file_type='zarr')

        # Create a mock record with crop bounds
        record = {
            'view_id': 'timepoint: 0, setup: 0',
            'file_path': 's3://bucket/test.zarr/0',
            'interval_key': ((0, 0, 0), (50, 50, 25), (51, 51, 26)),
            'offset': 0,
            'lb': (0, 0, 0),
            'crop_min': [2, 2, 2],
            'crop_max': [7, 7, 7]
        }

        # Mock the zarr opening to return a known dask array (10x10x10)
        mock_array = da.ones((1, 1, 10, 10, 10), dtype=np.float32)

        with patch('zarr.open') as mock_zarr, \
             patch('s3fs.S3FileSystem'), \
             patch('s3fs.S3Map'), \
             patch('dask.array.from_zarr', return_value=mock_array):

            # Call fetch_image_data
            view_id, interval_key, chunk, offset, lower_bound = reader.fetch_image_data(
                record, dsxy=1, dsz=1
            )

            # Verify crop was applied: array should be [2:8, 2:8, 2:8] = 6x6x6
            self.assertEqual(chunk.shape, (6, 6, 6))
            self.assertEqual(view_id, 'timepoint: 0, setup: 0')

    def test_fetch_image_data_without_crop(self):
        """Test backward compatibility: records without crop fields work normally"""
        reader = ImageReader(file_type='zarr')

        # Record without crop fields
        record = {
            'view_id': 'timepoint: 0, setup: 0',
            'file_path': 's3://bucket/test.zarr/0',
            'interval_key': ((0, 0, 0), (50, 50, 25), (51, 51, 26)),
            'offset': 0,
            'lb': (0, 0, 0),
            'crop_min': None,
            'crop_max': None
        }

        # Mock the zarr opening to return a known dask array
        mock_array = da.ones((1, 1, 10, 10, 10), dtype=np.float32)

        with patch('zarr.open') as mock_zarr, \
             patch('s3fs.S3FileSystem'), \
             patch('s3fs.S3Map'), \
             patch('dask.array.from_zarr', return_value=mock_array):

            # Call fetch_image_data - should not raise an error
            view_id, interval_key, chunk, offset, lower_bound = reader.fetch_image_data(
                record, dsxy=1, dsz=1
            )

            # Should succeed without crop
            self.assertEqual(view_id, 'timepoint: 0, setup: 0')

    def test_fetch_image_data_tiff_no_crop_error(self):
        """Test that tiff mode without crop works (no changes to tiff path)"""
        reader = ImageReader(file_type='tiff')

        record = {
            'view_id': 'timepoint: 0, setup: 0',
            'file_path': '/path/to/test.tif',
            'interval_key': ((0, 0, 0), (50, 50, 25), (51, 51, 26)),
            'offset': 0,
            'lb': (0, 0, 0),
            'crop_min': None,
            'crop_max': None
        }

        # Mock the BioImage reader
        mock_bioimage = MagicMock()
        mock_dask_array = da.ones((1, 1, 1, 10, 10, 10), dtype=np.float32)
        mock_bioimage.get_dask_stack.return_value = mock_dask_array

        with patch('Rhapso.detection.image_reader.CustomBioImage', return_value=mock_bioimage):
            # Should not raise an error
            view_id, interval_key, chunk, offset, lower_bound = reader.fetch_image_data(
                record, dsxy=1, dsz=1
            )

            self.assertEqual(view_id, 'timepoint: 0, setup: 0')

    def test_fetch_image_data_crop_bounds_validation(self):
        """Test that crop bounds exceeding array dimensions raise clear error"""
        reader = ImageReader(file_type='zarr')

        # Record with crop_max exceeding array dimensions
        record = {
            'view_id': 'timepoint: 0, setup: 0',
            'file_path': 's3://bucket/test.zarr/0',
            'interval_key': ((0, 0, 0), (50, 50, 25), (51, 51, 26)),
            'offset': 0,
            'lb': (0, 0, 0),
            'crop_min': [0, 0, 0],
            'crop_max': [15, 5, 5]  # Exceeds dimension 0 (10x10x10 array)
        }

        # Mock the zarr opening to return a known dask array (10x10x10)
        mock_array = da.ones((1, 1, 10, 10, 10), dtype=np.float32)

        with patch('zarr.open') as mock_zarr, \
             patch('s3fs.S3FileSystem'), \
             patch('s3fs.S3Map'), \
             patch('dask.array.from_zarr', return_value=mock_array):

            # Should raise ValueError with clear message
            with self.assertRaises(ValueError) as context:
                reader.fetch_image_data(record, dsxy=1, dsz=1)
            
            error_msg = str(context.exception)
            self.assertIn('crop_max[0]=15 exceeds array dimension 0', error_msg)
            self.assertIn('(shape=10)', error_msg)

    def test_fetch_image_data_negative_crop_min(self):
        """Test that negative crop_min values raise clear error"""
        reader = ImageReader(file_type='zarr')

        record = {
            'view_id': 'timepoint: 0, setup: 0',
            'file_path': 's3://bucket/test.zarr/0',
            'interval_key': ((0, 0, 0), (50, 50, 25), (51, 51, 26)),
            'offset': 0,
            'lb': (0, 0, 0),
            'crop_min': [-1, 0, 0],
            'crop_max': [5, 5, 5]
        }

        mock_array = da.ones((1, 1, 10, 10, 10), dtype=np.float32)

        with patch('zarr.open') as mock_zarr, \
             patch('s3fs.S3FileSystem'), \
             patch('s3fs.S3Map'), \
             patch('dask.array.from_zarr', return_value=mock_array):

            with self.assertRaises(ValueError) as context:
                reader.fetch_image_data(record, dsxy=1, dsz=1)
            
            self.assertIn('crop_min[0]=-1 is negative', str(context.exception))

    def test_fetch_image_data_crop_min_greater_than_crop_max(self):
        """Test that crop_min > crop_max raises clear error"""
        reader = ImageReader(file_type='zarr')

        record = {
            'view_id': 'timepoint: 0, setup: 0',
            'file_path': 's3://bucket/test.zarr/0',
            'interval_key': ((0, 0, 0), (50, 50, 25), (51, 51, 26)),
            'offset': 0,
            'lb': (0, 0, 0),
            'crop_min': [5, 0, 0],
            'crop_max': [3, 5, 5]  # crop_min[0] > crop_max[0]
        }

        mock_array = da.ones((1, 1, 10, 10, 10), dtype=np.float32)

        with patch('zarr.open') as mock_zarr, \
             patch('s3fs.S3FileSystem'), \
             patch('s3fs.S3Map'), \
             patch('dask.array.from_zarr', return_value=mock_array):

            with self.assertRaises(ValueError) as context:
                reader.fetch_image_data(record, dsxy=1, dsz=1)
            
            self.assertIn('crop_min[0]=5 > crop_max[0]=3', str(context.exception))


if __name__ == "__main__":
    unittest.main()
