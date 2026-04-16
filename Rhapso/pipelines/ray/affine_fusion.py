from Rhapso.affine_fusion.compute_bbox import ComputeBBox
from Rhapso.affine_fusion.compute_grid import ComputeGrid
from Rhapso.affine_fusion.overlapping_views import OverlappingViews
from Rhapso.affine_fusion.overlapping_blocks import OverlappingBlocks
from Rhapso.affine_fusion.fused_cell import FusedCell
import ray
import time
import zarr
import fsspec
import numpy as np

# This class implements the affine fusion pipeline

class AffineFusion:
    def __init__(self):
        # Z1
        # self.aligned_xml_path = "s3://aind-open-data/HCR_823476-s1-ls2_2025-11-18_00-00-00_processed_2026-03-03_22-58-55/image_tile_alignment/bigstitcher.xml"
        # self.zarr_input_prefix = "s3://aind-open-data/HCR_823476-s1-ls2_2025-11-18_00-00-00_processed_2026-03-03_22-58-55/image_radial_correction"
        
        # exaSPIM
        self.aligned_xml_path = "s3://aind-scratch-data/sean.fite/bigstitcher_kept.xml"
        self.zarr_input_prefix = "s3://aind-open-data/exaSPIM_720164_2025-07-07_17-55-45_processed_2025-07-15_16-22-02/flatfield_correction/SPIM.ome.zarr"

        self.output_path = "s3://aind-scratch-data/sean.fite/affine_fusion/test_19/fusion/fused.zarr" 
        self.block_size = [256, 256, 128] 
        self.intensity_range = [0, 65535]    # UINT16 default
        self.block_scale = [2, 2, 1]

    def affine_fusion(self):
        ray.init()

        # Compute bounding box for fused volume based on alignment xml
        compute_bbox = ComputeBBox(self.aligned_xml_path, self.zarr_input_prefix)
        bb_min, bb_max, per_view_transforms = compute_bbox.run() 
        dims = (bb_max - bb_min) + 1

        # Compute grid for fused volume
        compute_grid = ComputeGrid(dims, self.block_size, self.block_scale)
        grid = compute_grid.run()

        dims_xyz = (bb_max - bb_min) + 1
        output_shape_zyx = (int(dims_xyz[2]), int(dims_xyz[1]), int(dims_xyz[0]))

        # --- Create output zarr driver (match reference layout) ---
        root_store = fsspec.get_mapper(self.output_path.rstrip("/"))

        # Ensure fused.zarr is a zarr group (creates fused.zarr/.zgroup)
        zarr.storage.init_group(store=root_store, overwrite=False)

        # Open destination root group
        root = zarr.open_group(store=root_store, mode="a")

        # Copy root attrs from input (to get the big .zattrs like the reference)
        src_store = fsspec.get_mapper(self.zarr_input_prefix.rstrip("/"))
        src_root = zarr.open_group(store=src_store, mode="r")
        root.attrs.update(dict(src_root.attrs))

        # IMPORTANT: create array "0" at ROOT (fused.zarr/0/.zarray)
        if "0" not in root:
            Z, Y, X = output_shape_zyx
            root.create_dataset(
                "0",
                shape=(1, 1, Z, Y, X),
                chunks=(1, 1, 128, 256, 256),
                dtype=np.uint16,
                overwrite=False,
                fill_value=0,
                dimension_separator="/",
            )

        print("[fusion] output zarr initialized", flush=True)

        # Distributed approach
        @ray.remote(num_cpus=2)
        def fuse_grid_block(grid_block, bb_min, per_view_transforms, output_path, output_shape_zyx):
            # The min coordinates and size of the block this job renders
            super_block_offset = grid_block[0] + bb_min
            super_block_size   = grid_block[1]

            # Find overlapping views for this job
            find_overlapping_views = OverlappingViews(super_block_offset, super_block_size, per_view_transforms)
            overlapping_views, fused_min, fused_max = find_overlapping_views.run()

            if not overlapping_views:
                return

            # Use overlapping view to find overlapping blocks for this job
            find_overlapping_blocks = OverlappingBlocks(per_view_transforms, overlapping_views, super_block_offset, 
                                                        fused_min, fused_max)
            blocks = find_overlapping_blocks.run()

            if not blocks:
                return

            # Fuse overlapping blocks
            fused_cell = FusedCell(per_view_transforms, overlapping_views, fused_min, fused_max, output_path, 
                                   grid_block[0], bb_min, output_shape_zyx)
            fused_cell.run()

        # progress
        futures = []
        total_cells = len(grid)
        completed = 0
        failed = 0
        t_run0 = time.perf_counter()
        last_pct_printed = -1

        print(f"[fusion] submitting {total_cells} tasks", flush=True)

        for i, grid_block in enumerate(grid, start=1):
            futures.append(
                fuse_grid_block.remote(
                    grid_block, bb_min, per_view_transforms, self.output_path, output_shape_zyx
                )
            )

            # drain completions while submitting
            done, futures = ray.wait(futures, num_returns=1, timeout=0)
            while done:
                try:
                    ray.get(done[0])   
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"[fusion][ERROR] task failed: {type(e).__name__}: {e}", flush=True)

                pct_int = int((completed / max(total_cells, 1)) * 100.0)
                if pct_int > last_pct_printed:
                    last_pct_printed = pct_int
                    elapsed = time.perf_counter() - t_run0
                    rate = completed / max(elapsed, 1e-9)
                    eta_s = (total_cells - completed) / max(rate, 1e-9)
                    print(
                        f"[fusion] Progress: ok={completed - failed} failed={failed} total={total_cells} ({pct_int}%) "
                        f"elapsed={elapsed/60:.1f}m rate={rate:.2f} cells/s eta={eta_s/60:.1f}m",
                        flush=True,
                    )

                done, futures = ray.wait(futures, num_returns=1, timeout=0)

        # finish remaining tasks 
        while futures:
            done, futures = ray.wait(futures, num_returns=1, timeout=1.0)
            if not done:
                continue

            try:
                ray.get(done[0])
                completed += 1
            except Exception as e:
                failed += 1
                print(f"[fusion][ERROR] task failed: {type(e).__name__}: {e}", flush=True)

            pct_int = int((completed / max(total_cells, 1)) * 100.0)
            if pct_int > last_pct_printed:
                last_pct_printed = pct_int
                elapsed = time.perf_counter() - t_run0
                rate = completed / max(elapsed, 1e-9)
                eta_s = (total_cells - completed) / max(rate, 1e-9)
                print(
                    f"[fusion] Progress: ok={completed - failed} failed={failed} total={total_cells} ({pct_int}%) "
                    f"elapsed={elapsed/60:.1f}m rate={rate:.2f} cells/s eta={eta_s/60:.1f}m",
                    flush=True,
                )

        print("Fusion done", flush=True)

    def run(self):
        self.affine_fusion()

affine_fusion = AffineFusion()
affine_fusion.run()

# Iterative approach (dev)
# for grid_block in grid:
    
#     # The min coordinates and size of the block this job renders
#     super_block_offset = grid_block[0] + bb_min   
#     super_block_size   = grid_block[1] 

#     # Find overlapping views for this job
#     find_overlapping_views = OverlappingViews(super_block_offset, super_block_size, per_view_transforms)
#     overlapping_views, fused_min, fused_max = find_overlapping_views.run()

#     if not overlapping_views:
#         continue
    
#     # Use overlapping view to find overlapping blocks for this job
#     find_overlapping_blocks = OverlappingBlocks(per_view_transforms, overlapping_views, super_block_offset, fused_min, fused_max)
#     blocks = find_overlapping_blocks.run()

#     if not blocks:
#         continue
    
#     # Fuse overlapping blocks
#     dims_xyz = (bb_max - bb_min) + 1
#     output_shape_zyx = (int(dims_xyz[2]), int(dims_xyz[1]), int(dims_xyz[0]))
#     fused_cell = FusedCell(per_view_transforms, overlapping_views, fused_min, fused_max, self.output_path, grid_block[0], bb_min, output_shape_zyx)
#     fused_cell.run()