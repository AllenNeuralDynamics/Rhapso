from Rhapso.affine_fusion.compute_bbox import ComputeBBox
from Rhapso.affine_fusion.initialize_output_zarr import InitializeOutputZarr
from Rhapso.affine_fusion.compute_grid import ComputeGrid
from Rhapso.affine_fusion.overlapping_views import OverlappingViews
from Rhapso.affine_fusion.overlapping_blocks import OverlappingBlocks
from Rhapso.affine_fusion.generate_fusion_instructions import GenerateFusionInstructions
from Rhapso.affine_fusion.fuse_cell import FuseCell
from Rhapso.pipelines.ray.util.fusion_progress import FusionProgress
import ray

# This class implements the affine fusion pipeline

class AffineFusion:
    def __init__(self, aligned_xml_path, zarr_input_prefix, output_path, block_size, intensity_range, block_scale, overlap_strategy):
        self.aligned_xml_path = aligned_xml_path
        self.zarr_input_prefix = zarr_input_prefix
        self.output_path = output_path 
        self.block_size = block_size
        self.intensity_range = intensity_range
        self.block_scale = block_scale
        self.overlap_strategy = overlap_strategy

    def affine_fusion(self):
        ray.init()

        print("Newest approach")

        # Compute bounding box for fused volume based on alignment xml
        compute_bbox = ComputeBBox(self.aligned_xml_path, self.zarr_input_prefix)
        bb_min, bb_max, per_view_transforms = compute_bbox.run() 
        dims = (bb_max - bb_min) + 1
        print("Bbox of fused volume computed")

        # Initialize zarr datastore in s3 for fused output
        initialize_output_zarr = InitializeOutputZarr(self.output_path, self.zarr_input_prefix, dims)
        initialize_output_zarr.run()
        print("Zarr group/store initialized with bbox dims")

        # Compute grid for fused volume
        compute_grid = ComputeGrid(dims, self.block_size, self.block_scale)
        grid = compute_grid.run()
        print("Output grid computed")

        # Distribute grid and implement core fusion work
        print("Starting fusion for all cells in output grid:")
        @ray.remote
        def fuse_grid_block(grid_block, bb_min, bb_max, per_view_transforms, output_path, overlap_strategy):
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
            map_fusion_instructions = GenerateFusionInstructions(per_view_transforms, grid_block, bb_min, bb_max, overlap_strategy, overlapping_views)
            image_instructions, blocks = map_fusion_instructions.run()

            # Fuse blocks into cell
            fuse_cell = FuseCell(
                image_instructions, blocks, per_view_transforms, output_path, grid_block, bb_min, bb_max, overlap_strategy
            )
            fuse_cell.run()

        #  jobs with progress prints
        with_progress = FusionProgress(grid, fuse_grid_block, bb_min, bb_max, per_view_transforms, self.output_path, self.overlap_strategy)
        with_progress.run()
        print("Fusion is done")

    def run(self):
        self.affine_fusion()


# ITERATIVE APPROACH
# for grid_block in grid:        
#     # DEBUG: run only this specific fused grid block ----
#     # TARGET_OFFSET = (65024, 512, 0)

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
#     map_fusion_instructions = GenerateFusionInstructions(per_view_transforms, grid_block, bb_min, bb_max, self.overlap_strategy, overlapping_views)
#     image_instructions, blocks = map_fusion_instructions.run()

#     # Fuse blocks into cell
#     fuse_cell = FuseCell(
#         image_instructions, blocks, per_view_transforms, self.output_path, grid_block, bb_min, bb_max, self.overlap_strategy
#     )
#     fuse_cell.run()