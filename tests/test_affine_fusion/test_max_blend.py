import unittest

import numpy as np

from Rhapso.affine_fusion.fuse_cell import FuseCell
from Rhapso.affine_fusion.generate_fusion_instructions import GenerateFusionInstructions


class TestMaxBlend(unittest.TestCase):
    @staticmethod
    def masking_instruction(size_xyz, source_offset_xyz=(0, 0, 0)):
        inv_t = np.eye(4, dtype=np.float32)
        inv_t[:3, 3] = np.asarray(source_offset_xyz, dtype=np.float32)
        size_xyz = np.asarray(size_xyz, dtype=np.float32)
        return {
            "inv_t": inv_t,
            "b0": np.zeros(3, dtype=np.float32),
            "b3": size_xyz - 1,
        }

    @staticmethod
    def make_cell(view_ids, per_view_transforms=None):
        if per_view_transforms is None:
            per_view_transforms = {view_id: {} for view_id in view_ids}

        return FuseCell(
            image_instructions={},
            blocks={},
            per_view_transforms=per_view_transforms,
            output_path="unused",
            grid_block=(np.zeros(3, dtype=np.int64), np.ones(3, dtype=np.int64)),
            fusion_min_global=np.zeros(3, dtype=np.int64),
            fusion_max_global=np.zeros(3, dtype=np.int64),
            overlap_strategy="max_blend",
        )

    @staticmethod
    def full_block(max_xyz):
        max_xyz = np.asarray(max_xyz, dtype=np.int64)
        return np.zeros(3, dtype=np.int64), max_xyz

    def test_max_blend_binary_union_is_order_independent(self):
        arrays = {
            0: np.array([[[0, 1, 0], [1, 0, 0]]], dtype=np.float32),
            1: np.array([[[1, 0, 0], [0, 0, 1]]], dtype=np.float32),
        }
        expected = np.array([[[1, 1, 0], [1, 0, 1]]], dtype=np.float32)
        instructions = {
            view_id: self.masking_instruction((3, 2, 1))
            for view_id in arrays
        }
        cell = self.make_cell(arrays.keys())
        cell.load_source_chunk_for_view = lambda view_id, _src_min, _src_max: (
            arrays[view_id],
            np.zeros(3, dtype=np.int64),
        )
        block_min, block_max = self.full_block((2, 1, 0))

        forward = cell.render_fused_block(
            instructions,
            {0: None, 1: None},
            block_min,
            block_max,
        )
        reverse = cell.render_fused_block(
            instructions,
            {1: None, 0: None},
            block_min,
            block_max,
        )

        np.testing.assert_array_equal(forward, expected)
        np.testing.assert_array_equal(reverse, expected)

    def test_max_blend_preserves_non_binary_maximum_values(self):
        arrays = {
            0: np.array([[[2, 9]]], dtype=np.float32),
            1: np.array([[[7, 3]]], dtype=np.float32),
        }
        cell = self.make_cell(arrays.keys())
        cell.load_source_chunk_for_view = lambda view_id, _src_min, _src_max: (
            arrays[view_id],
            np.zeros(3, dtype=np.int64),
        )
        instructions = {
            view_id: self.masking_instruction((2, 1, 1))
            for view_id in arrays
        }
        block_min, block_max = self.full_block((1, 0, 0))

        fused = cell.render_fused_block(
            instructions,
            {0: None, 1: None},
            block_min,
            block_max,
        )

        np.testing.assert_array_equal(fused, np.array([[[7, 9]]], dtype=np.float32))

    def test_max_blend_uses_nearest_neighbor_sampling(self):
        source = np.array([[[0, 10]]], dtype=np.float32)
        cell = self.make_cell([0])
        cell.load_source_chunk_for_view = lambda _view_id, _src_min, _src_max: (
            source,
            np.zeros(3, dtype=np.int64),
        )
        instructions = {
            0: self.masking_instruction((2, 1, 1), source_offset_xyz=(0.6, 0, 0))
        }
        block_min, block_max = self.full_block((0, 0, 0))

        fused = cell.render_fused_block(
            instructions,
            {0: None},
            block_min,
            block_max,
        )

        np.testing.assert_array_equal(fused, np.array([[[10]]], dtype=np.float32))

    def test_max_blend_masks_nearest_edge_extrapolation(self):
        source = np.array([[[4, 8]]], dtype=np.float32)
        cell = self.make_cell([0])
        cell.load_source_chunk_for_view = lambda _view_id, _src_min, _src_max: (
            source,
            np.zeros(3, dtype=np.int64),
        )
        instructions = {0: self.masking_instruction((2, 1, 1))}
        block_min, block_max = self.full_block((2, 0, 0))

        fused = cell.render_fused_block(
            instructions,
            {0: None},
            block_min,
            block_max,
        )

        np.testing.assert_array_equal(
            fused,
            np.array([[[4, 8, 0]]], dtype=np.float32),
        )

    def test_max_blend_reads_split_views_from_original_tile_offset(self):
        source = np.array([[[0, 0, 4, 8, 0]]], dtype=np.float32)
        per_view_transforms = {
            0: {
                "split_def": {
                    "split_min_xyz": np.array([2, 0, 0], dtype=np.int64),
                },
            },
        }
        cell = self.make_cell([0], per_view_transforms=per_view_transforms)
        requested_bounds = []

        def load_source(_view_id, src_min_xyz, src_max_xyz):
            src_min_xyz = np.asarray(src_min_xyz, dtype=np.int64)
            src_max_xyz = np.asarray(src_max_xyz, dtype=np.int64)
            requested_bounds.append((src_min_xyz.copy(), src_max_xyz.copy()))
            x0, y0, z0 = src_min_xyz
            x1, y1, z1 = src_max_xyz
            return (
                source[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1],
                src_min_xyz,
            )

        cell.load_source_chunk_for_view = load_source
        instructions = {0: self.masking_instruction((2, 1, 1))}
        block_min, block_max = self.full_block((1, 0, 0))

        fused = cell.render_fused_block(
            instructions,
            {0: None},
            block_min,
            block_max,
        )

        np.testing.assert_array_equal(fused, np.array([[[4, 8]]], dtype=np.float32))
        np.testing.assert_array_equal(requested_bounds[0][0], np.array([2, 0, 0]))
        np.testing.assert_array_equal(requested_bounds[0][1], np.array([3, 0, 0]))


class TestMaxBlendInstructions(unittest.TestCase):
    @staticmethod
    def make_generator(overlap_strategy):
        return GenerateFusionInstructions(
            per_view_transforms={
                0: {
                    "transform": np.eye(4, dtype=np.float64),
                    "size": np.array([2, 2, 2], dtype=np.int64),
                },
            },
            grid_block=(
                np.zeros(3, dtype=np.int64),
                np.array([2, 2, 2], dtype=np.int64),
            ),
            fusion_min_global=np.zeros(3, dtype=np.int64),
            fusion_max_global=np.ones(3, dtype=np.int64),
            overlap_strategy=overlap_strategy,
            overlapping_views=[0],
        )

    def test_max_blend_uses_masking_instructions(self):
        instructions, blocks = self.make_generator("max_blend").run()

        self.assertIn(0, instructions)
        self.assertIn(0, blocks)
        self.assertNotIn("b1", instructions[0])
        np.testing.assert_array_equal(instructions[0]["b0"], np.zeros(3))
        np.testing.assert_array_equal(instructions[0]["b3"], np.ones(3))

    def test_invalid_strategy_lists_all_supported_values(self):
        with self.assertRaisesRegex(
            ValueError,
            "'avg_blend'.*'lowest_view_wins'.*'max_blend'",
        ):
            self.make_generator("not_a_strategy").run()


if __name__ == "__main__":
    unittest.main()
