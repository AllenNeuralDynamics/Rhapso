from Rhapso.pipelines.ray.r2r import R2R

fixed_root = "s3://aind-open-data/HCR_772643-3a-1_2025-03-19_10-00-00/flipped/SPIM.ome.zarr/Tile_X_0000_Y_0000_Z_0000_ch_405.zarr"
moving_root = "s3://aind-open-data/HCR_772643-3a-1_2025-02-26_10-00-00/SPIM.ome.zarr/Tile_X_0000_Y_0000_Z_0000_ch_405.zarr/"
seg_root = "s3://aind-scratch-data/sean.fite/r2r/segmentation_mask_orig_res.zarr"
output_dir = f"s3://aind-scratch-data/sean.fite/round-2-round/HCR_772643"
alignment_config = "Rhapso/pipelines/ray/param/r2r/HCR_772643.yml"
min_level = 0
min_block_size = 50
r2r_res_levels = [4, 3, 2, 1]

pipeline = R2R(fixed_root, moving_root, seg_root, min_level, output_dir, min_block_size, alignment_config, r2r_res_levels)
pipeline.run()
