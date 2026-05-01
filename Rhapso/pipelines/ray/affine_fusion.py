from Rhapso.affine_fusion.compute_bbox import ComputeBBox
from Rhapso.affine_fusion.compute_grid import ComputeGrid
from Rhapso.affine_fusion.overlapping_views import OverlappingViews
from Rhapso.affine_fusion.overlapping_blocks import OverlappingBlocks
from Rhapso.affine_fusion.generate_fusion_intstructions import GenerateFusionInstructions
from Rhapso.affine_fusion.fuse_cell import FuseCell
import ray
import time

# This class implements the affine fusion pipeline

class AffineFusion:
    def __init__(self, aligned_xml_path, zarr_input_prefix, output_path, block_size, intensity_range, block_scale):
        self.aligned_xml_path = aligned_xml_path
        self.zarr_input_prefix = zarr_input_prefix
        self.output_path = output_path 
        self.block_size = block_size
        self.intensity_range = intensity_range
        self.block_scale = block_scale

    def run_grid_with_progress(self, grid, fuse_task_remote, *task_args):
        """
        Submits ray task and prints progress as tasks finish.
        """
        futures = []
        total_cells = len(grid)
        completed = 0
        failed = 0
        t_run0 = time.perf_counter()
        last_pct_printed = -1

        print(f"[fusion] submitting {total_cells} tasks", flush=True)

        for grid_block in grid:
            futures.append(fuse_task_remote.remote(grid_block, *task_args))

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

    def affine_fusion(self):
        ray.init()

        # Compute bounding box for fused volume based on alignment xml
        compute_bbox = ComputeBBox(self.aligned_xml_path, self.zarr_input_prefix)
        bb_min, bb_max, per_view_transforms = compute_bbox.run() 
        dims = (bb_max - bb_min) + 1

        # Compute grid for fused volume
        compute_grid = ComputeGrid(dims, self.block_size, self.block_scale, self.zarr_input_prefix, self.output_path)
        grid, _ = compute_grid.run()

        # Distribute grid and implement core fusion work
        @ray.remote
        def fuse_grid_block(grid_block, bb_min, per_view_transforms, output_path):
            # The min coordinates and size of the block this job renders
            super_block_offset = grid_block[0] + bb_min
            super_block_size   = grid_block[1]

            # Find and gate for overlapping views 
            find_overlapping_views = OverlappingViews(super_block_offset, super_block_size, per_view_transforms)
            overlapping_views, fused_min, fused_max = find_overlapping_views.run()
            if not overlapping_views:
                return

            # Find and gate for overlapping blocks
            find_overlapping_blocks = OverlappingBlocks(
                per_view_transforms, overlapping_views, super_block_offset, fused_min, fused_max, grid_block
            )
            blocks = find_overlapping_blocks.run()
            if not any(blocks.values()):
                return
            
            # Define instructions for fusing contributing image blocks
            map_fusion_instructions = GenerateFusionInstructions(per_view_transforms, grid_block, bb_min, bb_max)
            image_instructions, blocks = map_fusion_instructions.run()

            # Fuse blocks into cell
            fuse_cell = FuseCell(
                image_instructions, blocks, per_view_transforms, output_path, grid_block, bb_min, bb_max
            )
            fuse_cell.run()

        # run jobs with progress prints
        self.run_grid_with_progress(grid, fuse_grid_block, bb_min, per_view_transforms, self.output_path)

    def run(self):
        self.affine_fusion()


# ITERATIVE APPROACH
# for grid_block in grid:
#     # Left off: [16896, 5632, 128]
    
#     # DEBUG: run only this specific fused grid block ----
#     # TARGET_OFFSET = (65024, 512, 0)
#     # TARGET_OFFSET = (1024, 19968, 128)
#     # TARGET_OFFSET = (512, 20480, 128)

#     # gb_off = tuple(int(x) for x in grid_block[0]) # (ox,oy,oz)
#     # if gb_off != TARGET_OFFSET:
#     #     continue

#     # The min coordinates and size of the block this job renders
#     super_block_offset = grid_block[0] + bb_min
#     super_block_size   = grid_block[1]

#     # Find overlapping views 
#     find_overlapping_views = OverlappingViews(super_block_offset, super_block_size, per_view_transforms)
#     overlapping_views, fused_min, fused_max = find_overlapping_views.run()

#     if not overlapping_views:
#         continue

#     # Use overlapping view to find overlapping blocks for this job
#     find_overlapping_blocks = OverlappingBlocks(
#         per_view_transforms, overlapping_views, super_block_offset, fused_min, fused_max, grid_block
#     )
#     blocks = find_overlapping_blocks.run()

#     if not any(blocks.values()):
#         continue
    
#     # Map instructions for fusing images
#     map_fusion_instructions = GenerateFusionInstructions(per_view_transforms, grid_block, bb_min, bb_max)
#     image_instructions, blocks = map_fusion_instructions.run()

#     # Fuse blocks into cell
#     fuse_cell = FuseCell(
#         image_instructions, blocks, per_view_transforms, self.output_path, grid_block, bb_min, bb_max
#     )
#     fuse_cell.run()