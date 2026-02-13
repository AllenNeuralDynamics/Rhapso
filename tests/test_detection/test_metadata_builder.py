import unittest
import pandas as pd
import numpy as np

from Rhapso.detection.metadata_builder import MetadataBuilder


class TestMetadataBuilder(unittest.TestCase):
    def test_build_paths_split_uses_zarr_base_path(self):
        """Test that split mode uses zarr_base_path for file path construction"""
        # Create mock image_loader_df with split columns
        image_loader_df = pd.DataFrame({
            'view_setup': ['0', '1'],
            'timepoint': ['0', '0'],
            'file_path': ['Tile_X_0000_Y_0000_Z_0000_ch_405.zarr', 'Tile_X_0000_Y_0000_Z_0000_ch_405.zarr'],
            'crop_min': ['0 0 0', '300 0 0'],
            'crop_max': ['499 499 99', '799 499 99'],
            'zarr_base_path': ['s3://bucket/SPIM.ome.zarr/', 's3://bucket/SPIM.ome.zarr/']
        })

        # Mock overlapping_area
        overlapping_area = {
            'timepoint: 0, setup: 0': [{'lower_bound': np.array([0, 0, 0]), 'upper_bound': np.array([100, 100, 50])}],
            'timepoint: 0, setup: 1': [{'lower_bound': np.array([50, 0, 0]), 'upper_bound': np.array([150, 100, 50])}]
        }

        dataframes = {'image_loader': image_loader_df}
        builder = MetadataBuilder(
            dataframes=dataframes,
            overlapping_area=overlapping_area,
            image_file_prefix='s3://bucket/SPIM.ome.zarr/',
            file_type='zarr',
            dsxy=1.0,
            dsz=1.0,
            chunks_per_bound=1,
            sigma=1.0,
            run_type='ray',
            level=0
        )
        builder.build_paths()

        # Check that file_path uses zarr_base_path
        self.assertTrue(
            'zarr' in builder.metadata[0]['file_path'],
            f"File path should contain zarr path: {builder.metadata[0]['file_path']}"
        )

    def test_build_paths_split_passes_crop_bounds(self):
        """Test that crop bounds are included in metadata records"""
        image_loader_df = pd.DataFrame({
            'view_setup': ['0'],
            'timepoint': ['0'],
            'file_path': ['Tile_X_0000_Y_0000_Z_0000_ch_405.zarr'],
            'crop_min': ['0 0 0'],
            'crop_max': ['499 499 99'],
            'zarr_base_path': ['s3://bucket/SPIM.ome.zarr/']
        })

        overlapping_area = {
            'timepoint: 0, setup: 0': [{'lower_bound': np.array([0, 0, 0]), 'upper_bound': np.array([100, 100, 50])}]
        }

        dataframes = {'image_loader': image_loader_df}
        builder = MetadataBuilder(
            dataframes=dataframes,
            overlapping_area=overlapping_area,
            image_file_prefix='s3://bucket/SPIM.ome.zarr/',
            file_type='zarr',
            dsxy=1.0,
            dsz=1.0,
            chunks_per_bound=0,  # No chunking
            sigma=1.0,
            run_type='ray',
            level=0
        )
        builder.build_paths()

        # Check that crop_min and crop_max are in metadata
        self.assertIn('crop_min', builder.metadata[0])
        self.assertIn('crop_max', builder.metadata[0])
        self.assertEqual(builder.metadata[0]['crop_min'], [0, 0, 0])
        self.assertEqual(builder.metadata[0]['crop_max'], [499, 499, 99])

    def test_build_paths_split_scales_crop_bounds_by_level(self):
        """Test that crop bounds are scaled by 2^level"""
        image_loader_df = pd.DataFrame({
            'view_setup': ['0'],
            'timepoint': ['0'],
            'file_path': ['Tile_X_0000_Y_0000_Z_0000_ch_405.zarr'],
            'crop_min': ['300 0 0'],
            'crop_max': ['799 499 99'],
            'zarr_base_path': ['s3://bucket/SPIM.ome.zarr/']
        })

        overlapping_area = {
            'timepoint: 0, setup: 0': [{'lower_bound': np.array([0, 0, 0]), 'upper_bound': np.array([100, 100, 50])}]
        }

        dataframes = {'image_loader': image_loader_df}
        # level=2 means scale by 2^2 = 4
        builder = MetadataBuilder(
            dataframes=dataframes,
            overlapping_area=overlapping_area,
            image_file_prefix='s3://bucket/SPIM.ome.zarr/',
            file_type='zarr',
            dsxy=1.0,
            dsz=1.0,
            chunks_per_bound=0,
            sigma=1.0,
            run_type='ray',
            level=2
        )
        builder.build_paths()

        # 300 // 4 = 75, 799 // 4 = 199, etc.
        self.assertEqual(builder.metadata[0]['crop_min'], [75, 0, 0])
        self.assertEqual(builder.metadata[0]['crop_max'], [199, 124, 24])

    def test_build_paths_regular_zarr_no_crop(self):
        """Test backward compatibility: regular zarr has no crop fields"""
        image_loader_df = pd.DataFrame({
            'view_setup': ['0'],
            'timepoint': ['0'],
            'file_path': ['test.zarr']
        })

        overlapping_area = {
            'timepoint: 0, setup: 0': [{'lower_bound': np.array([0, 0, 0]), 'upper_bound': np.array([100, 100, 50])}]
        }

        dataframes = {'image_loader': image_loader_df}
        builder = MetadataBuilder(
            dataframes=dataframes,
            overlapping_area=overlapping_area,
            image_file_prefix='s3://bucket/',
            file_type='zarr',
            dsxy=1.0,
            dsz=1.0,
            chunks_per_bound=0,
            sigma=1.0,
            run_type='ray',
            level=0
        )
        builder.build_paths()

        # Regular zarr should have None for crop fields
        self.assertIsNone(builder.metadata[0]['crop_min'])
        self.assertIsNone(builder.metadata[0]['crop_max'])


if __name__ == "__main__":
    unittest.main()
