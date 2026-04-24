import unittest

import numpy as np

from Rhapso.matching.load_and_transform_points import (
    SPLIT_TILE_TRANSFORM_NAME,
    LoadAndTransformPoints,
)


def _translation_transform(name, tx, ty, tz):
    """Build a view_registrations entry (name + 3x4 affine string) for a
    pure translation (tx, ty, tz)."""
    affine = f"1.0 0.0 0.0 {tx} 0.0 1.0 0.0 {ty} 0.0 0.0 1.0 {tz}"
    return {"type": "affine", "name": name, "affine": affine}


def _scale_transform(name, sx, sy, sz):
    affine = f"{sx} 0.0 0.0 0.0 0.0 {sy} 0.0 0.0 0.0 0.0 {sz} 0.0"
    return {"type": "affine", "name": name, "affine": affine}


class TestImageSplittingSkip(unittest.TestCase):
    """The 'Image Splitting' ViewTransform must NOT be composed by
    ``get_transformation_matrix``. Detection has already baked the
    split-tile's world translation into the stored N5 IP coords.
    """

    def setUp(self):
        self.loader = LoadAndTransformPoints(
            data_global={},
            xml_input_path="/dev/null",
            n5_output_path="",
            match_type="split_affine",
        )

    def test_image_splitting_transform_is_skipped(self):
        view_id = (0, 1)
        view_registrations = {
            view_id: [
                _scale_transform("calibration", 1.0, 1.0, 3.866),
                _translation_transform(
                    SPLIT_TILE_TRANSFORM_NAME, 384.0, 0.0, 0.0
                ),
            ]
        }
        M = self.loader.get_transformation_matrix(view_id, view_registrations)

        # Only calibration should apply; translation from Image Splitting
        # must be skipped. Check translation column = 0.
        np.testing.assert_array_almost_equal(M[:3, 3], [0.0, 0.0, 0.0])
        # Scale column reflects calibration.
        np.testing.assert_array_almost_equal(
            np.diag(M[:3, :3]), [1.0, 1.0, 3.866]
        )

    def test_other_named_translation_still_applied(self):
        """Non-'Image Splitting' transforms must still compose. This
        guards against over-broad filtering."""
        view_id = (0, 1)
        view_registrations = {
            view_id: [
                _scale_transform("calibration", 1.0, 1.0, 3.866),
                _translation_transform("Stitching Solver", 10.0, 20.0, 30.0),
            ]
        }
        M = self.loader.get_transformation_matrix(view_id, view_registrations)
        np.testing.assert_array_almost_equal(
            M[:3, 3], [10.0, 20.0, 30.0 * 3.866]
        )

    def test_correspondence_delta_invariant_to_split_translation(self):
        """End-to-end: a moving point at world (920, 234, 139.1) and a
        fixed point at the identical world position must produce a
        zero delta after transform composition, regardless of whether
        the moving view has an 'Image Splitting' translation in its
        XML chain. Without the skip, the delta would equal the split
        translation.
        """
        mov_view = (0, 1)
        fix_view = (0, 16)
        view_registrations = {
            mov_view: [
                _scale_transform("calibration", 1.0, 1.0, 1.0),
                _translation_transform(
                    SPLIT_TILE_TRANSFORM_NAME, 384.0, 0.0, 0.0
                ),
            ],
            fix_view: [
                _scale_transform("calibration", 1.0, 1.0, 1.0),
            ],
        }

        mov_pts_world = np.array([[920.0, 234.0, 139.1]])
        fix_pts_world = np.array([[920.0, 234.0, 139.1]])

        M_mov = self.loader.get_transformation_matrix(mov_view, view_registrations)
        M_fix = self.loader.get_transformation_matrix(fix_view, view_registrations)

        mov_t = self.loader.transform_interest_points(mov_pts_world, M_mov)
        fix_t = self.loader.transform_interest_points(fix_pts_world, M_fix)

        delta = mov_t[0] - fix_t[0]
        np.testing.assert_array_almost_equal(delta, [0.0, 0.0, 0.0])

    def test_case_sensitive_name_match(self):
        """The reserved name is an exact string match; a subtly
        different name (e.g. lowercase) should NOT be skipped."""
        view_id = (0, 1)
        view_registrations = {
            view_id: [
                _translation_transform("image splitting", 384.0, 0.0, 0.0),
            ]
        }
        M = self.loader.get_transformation_matrix(view_id, view_registrations)
        np.testing.assert_array_almost_equal(M[:3, 3], [384.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
