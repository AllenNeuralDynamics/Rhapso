from Rhapso.r2r.round2round import R2R_Pipeline

run_id = 0
fixed_image_multiscale_root = "s3://aind-open-data/HCR_772643-3a-1_2025-03-19_10-00-00/flipped/SPIM.ome.zarr/Tile_X_0000_Y_0000_Z_0000_ch_405.zarr"
moving_image_multiscale_root = "s3://aind-open-data/HCR_772643-3a-1_2025-02-26_10-00-00/SPIM.ome.zarr/Tile_X_0000_Y_0000_Z_0000_ch_405.zarr/"
moving_segmentation_zarr_path = "/Users/sean.fite/Desktop/segmentation_mask_orig_res.zarr"
scratch_directory = f"/Users/sean.fite/Desktop/r2r_scratch/{run_id}"
results_directory = f"/Users/sean.fite/Desktop/r2r_results/{run_id}"
min_alignment_level= 0
max_alignment_level = 0
output_multiscale_level=0
multiscale_output=True
fixed_range = (50.0, 300.0)   
moving_range = (50.0, 600.0)
minimum_alignment_blocksize_physical_units = 50
output_transform_path = '/Users/sean.fite/Desktop/r2r_output/0/transform.zarr'
output_warped_segmentation_path = '/Users/sean.fite/Desktop/r2r_output/0/warped_segmentation.zarr'
output_warped_image_path = '/Users/sean.fite/Desktop/r2r_output/0/warped_image.zarr'
output_qc_directory = '/Users/sean.fite/Desktop/r2r_output/0/qc'
tile_edge_filter_enabled = False
min_peak_intensity = None
tiling_schedule = None
debug_skip = None
ip_registration_config_file = False
output_directory = '/Users/sean.fite/Desktop/r2r_output/0'

pipeline = R2R_Pipeline(
    fixed_image_multiscale_root=fixed_image_multiscale_root,
    moving_image_multiscale_root=moving_image_multiscale_root,
    moving_segmentation_zarr_path=moving_segmentation_zarr_path,
    min_alignment_level=min_alignment_level,
    max_alignment_level=max_alignment_level,
    scratch_directory=scratch_directory,
    results_directory=results_directory,
    minimum_alignment_blocksize_physical_units=minimum_alignment_blocksize_physical_units,
    output_multiscale_level=output_multiscale_level,
    multiscale_output=multiscale_output,
    tiling_schedule=tiling_schedule,
    tile_edge_filter_enabled=tile_edge_filter_enabled,
    ip_registration_config_file=ip_registration_config_file,
    min_peak_intensity_override=min_peak_intensity,
    debug_skip_parameter_sweep=debug_skip,
    output_transform_path=output_transform_path,
    output_warped_segmentation_path=output_warped_segmentation_path,
    output_warped_image_path=output_warped_image_path,
    output_qc_dir=output_qc_directory,
    fixed_image_min=fixed_range[0] if fixed_range else None,
    fixed_image_max=fixed_range[1] if fixed_range else None,
    moving_image_min=moving_range[0] if moving_range else None,
    moving_image_max=moving_range[1] if moving_range else None,
    dask_client=None,  # let the pipeline create its own LocalCluster
)

pipeline.run()
