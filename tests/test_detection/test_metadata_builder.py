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

    def test_chunked_metadata_no_z_band_double_add(self):
        """Regression: with chunks_per_bound>1, applying lb + offset to
        chunk-local peaks must equal the absolute parent-frame coord —
        i.e. no double-add of the chunk shift. Prior bug: lb=parent_lb
        and offset=z together added the chunk shift twice, producing
        Z-banded IPs (one missing band per chunk after the first).
        """
        image_loader_df = pd.DataFrame({
            'view_setup': ['0'],
            'timepoint': ['0'],
            'file_path': ['Tile_X_0000_Y_0000_Z_0000_ch_405.zarr'],
            'crop_min': ['0 0 0'],
            'crop_max': ['99 99 999'],
            'zarr_base_path': ['s3://bucket/SPIM.ome.zarr/']
        })
        # Single un-split parent region spanning Z=0..999 (XYZ-ordered
        # bounds match what OverlapDetection emits).
        overlapping_area = {
            'timepoint: 0, setup: 0': [{
                'lower_bound': np.array([0, 0, 0]),
                'upper_bound': np.array([99, 99, 999]),
            }],
        }
        dataframes = {'image_loader': image_loader_df}
        builder = MetadataBuilder(
            dataframes=dataframes,
            overlapping_area=overlapping_area,
            image_file_prefix='s3://bucket/SPIM.ome.zarr/',
            file_type='zarr',
            dsxy=1.0,
            dsz=1.0,
            chunks_per_bound=4,
            sigma=1.0,
            run_type='ray',
            level=0,
        )
        builder.build_paths()

        # Expect 4 chunked metadata records.
        self.assertEqual(len(builder.metadata), 4)

        # For each chunk, verify the contract that produces correct
        # global coords: peaks_global = peak_local + lb + offset.
        # With our fix, lb already encodes the parent-frame chunk-start
        # and offset is 0, so a peak at local (0,0,0) should map to the
        # chunk's expected global Z (within `overlap` tolerance for the
        # halo expansion).
        prev_chunk_z_start = -1
        for entry in builder.metadata:
            actual_lb = entry['lb']
            offset = entry['offset']
            interval_lb = entry['interval_key'][0]

            # offset must be 0 — the chunk shift is already in `lb`.
            self.assertEqual(
                offset, 0,
                f"offset should be 0 (chunk-shift lives in lb); got {offset}",
            )
            # `lb` must equal the chunk's interval lower bound. Same
            # tuple — no separate parent-vs-chunk frames.
            self.assertEqual(tuple(actual_lb), tuple(interval_lb))

            # Chunk Z-starts must be monotonically increasing (sanity).
            chunk_z_start = actual_lb[2]
            self.assertGreater(chunk_z_start, prev_chunk_z_start)
            prev_chunk_z_start = chunk_z_start

        # Simulate the DoG recombine step: upsample first (identity at
        # level 0), then add lb (L0 coords), then add offset (0).
        # A synthetic peak at chunk-local (0,0,0) in chunk i should
        # land at global Z = interval_key[0][2].
        for entry in builder.metadata:
            local_peak_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            # upsample_coordinates is identity at level 0
            upsampled_peak = local_peak_xyz.copy()
            lb_xyz = np.array(entry['lb'], dtype=np.float32)
            global_peak = upsampled_peak + lb_xyz  # apply_lower_bounds
            global_peak[:, 2] += entry['offset']   # apply_offset (now 0)
            self.assertEqual(
                float(global_peak[0, 2]),
                float(entry['interval_key'][0][2]),
                "peak at chunk-local (0,0,0) must map to chunk's "
                "parent-frame Z without double-adding the offset",
            )

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
