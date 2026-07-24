import os
from Rhapso.r2r.xml_prep import XMLPrep
from Rhapso.r2r.data_prep import DataPrep
from Rhapso.r2r.prewarp_moving_inputs import PrewarpMovingInputs
from Rhapso.r2r.registration_and_alignment import RegistrationAndAlignment
from Rhapso.r2r.create_displacement_field import CreateDisplacementField

class R2R:
    def __init__(self, fixed_root, moving_root, seg_root, min_level, output_dir, min_block_size, alignment_config, r2r_res_levels):
        self.fixed_root = fixed_root
        self.moving_root = moving_root
        self.seg_root = seg_root
        self.min_level = min_level
        self.output_dir = output_dir
        self.min_block_size = min_block_size
        self.alignment_config = alignment_config
        self.alignment_levels = r2r_res_levels
        self.block_shape_zyx = (32, 128, 128)

    def r2r(self):
        xml_prep = XMLPrep(self.fixed_root, self.moving_root, self.seg_root, self.min_block_size, self.min_level)
        data_prep = DataPrep(self.min_level)
        registration = RegistrationAndAlignment(self.fixed_root, self.moving_root, self.alignment_config)

        original_moving_root = self.moving_root
        original_seg_root = self.seg_root

        block_grids = xml_prep.create_tile_schedule(original_moving_root)

        if len(block_grids) > 4:
            raise ValueError(f"R2R supports 4 registration rounds, got {len(block_grids)}")

        field_paths = []

        try:
            for loop_index, block_grid_zyx in enumerate(block_grids):
                round_number = loop_index + 1
                alignment_level = self.alignment_levels[loop_index]
                loop_dir = os.path.join(self.output_dir, f"loop_{loop_index}")

                if not loop_dir.startswith("s3://"):
                    os.makedirs(loop_dir, exist_ok=True)

                if loop_index == 0:
                    round_moving_root = original_moving_root
                    round_seg_root = original_seg_root
                else:
                    prewarp_dir = os.path.join(self.output_dir, f"prewarp_{loop_index}")

                    prewarp = PrewarpMovingInputs(
                        self.fixed_root, original_moving_root, original_seg_root, field_paths,
                        prewarp_dir, loop_index, self.block_shape_zyx, alignment_level
                    )
                    round_moving_root, round_seg_root = prewarp.run()

                self.moving_root = round_moving_root
                self.seg_root = round_seg_root

                xml_prep.moving_image_multiscale_root = round_moving_root
                xml_prep.moving_segmentation_zarr_path = round_seg_root
                registration.moving_image_multiscale_root = round_moving_root

                fixed_xml = xml_prep.create_dataset_xml(self.fixed_root, "fixed.dataset.xml", loop_dir)
                moving_xml = xml_prep.create_dataset_xml(round_moving_root, "moving.dataset.xml", loop_dir)

                split_xml = os.path.join(loop_dir, "moving.split.dataset.xml")
                split_xml, tile_size_zyx, overlap_zyx = registration.split_moving_dataset(
                    moving_xml, split_xml, block_grid_zyx
                )

                tile_bboxes_zyx = xml_prep.remove_empty_moving_tiles(split_xml, alignment_level, 0.25)
                moving_ids = xml_prep.get_split_ids(split_xml)

                if not moving_ids:
                    raise RuntimeError(f"No moving tiles remain after filtering in round {round_number}")

                fixed_id = max(moving_ids) + 1
                xml_prep.renumber_fixed_setup(fixed_xml, fixed_id)

                fixed_points = os.path.join(loop_dir, "fixed_interestpoints")
                moving_points = os.path.join(loop_dir, "moving_interestpoints")
                combined_points = os.path.join(loop_dir, "interestpoints.n5")

                fixed_detected_xml = registration.detect_interest_points(
                    fixed_xml, self.fixed_root, os.path.join(loop_dir, "fixed.detected.xml"),
                    fixed_points, "fixed", round_number
                )

                moving_detected_xml = registration.detect_interest_points(
                    split_xml, round_moving_root, os.path.join(loop_dir, "moving.detected.xml"),
                    moving_points, "moving", round_number
                )

                if loop_index == 0:
                    xml_prep.normalize_loop0_calibration(fixed_detected_xml, moving_detected_xml)
                else:
                    fixed_spacing = data_prep.get_level0_spacing_zyx(self.fixed_root)
                    shared_calibration_xyz = (1.0, 1.0, fixed_spacing[0] / fixed_spacing[1])

                    xml_prep.set_xml_calibration(fixed_detected_xml, shared_calibration_xyz)
                    xml_prep.set_xml_calibration(moving_detected_xml, shared_calibration_xyz)

                    print(f"Round {round_number} shared calibration XYZ: {shared_calibration_xyz}")

                combined_xml = xml_prep.combine_detected_xmls(
                    moving_detected_xml, fixed_detected_xml, os.path.join(loop_dir, "combined.detected.xml")
                )

                data_prep.combine_interest_point_stores(moving_points, fixed_points, combined_points)

                registered_xml = registration.align_interest_points(
                    combined_xml, fixed_id, loop_dir, combined_points, round_number
                )

                field_path = os.path.join(loop_dir, "displacement_field.zarr")

                field = CreateDisplacementField(
                    self.fixed_root, round_moving_root, registered_xml, fixed_id, block_grid_zyx,
                    tile_size_zyx, overlap_zyx, tile_bboxes_zyx, field_path, self.output_dir, alignment_level
                )
                field.run()

                field_paths.append(field_path)

                print(f"Round {round_number} registered XML: {registered_xml}")
                print(f"Round {round_number} displacement field: {field_path}")
            
            final_dir = os.path.join(self.output_dir, "final")

            final_prewarp = PrewarpMovingInputs(
                self.fixed_root, original_moving_root, original_seg_root, field_paths,
                final_dir, len(field_paths), self.block_shape_zyx, self.min_level
            )
            final_moving_root, final_seg_root = final_prewarp.run()

            print(f"Final warped moving image: {final_moving_root}")
            print(f"Final warped segmentation: {final_seg_root}")

        finally:
            self.moving_root = original_moving_root
            self.seg_root = original_seg_root
            registration.stop_cluster()

        print(f"Completed {len(field_paths)} registration rounds")
        return final_moving_root, final_seg_root, field_paths

    def run(self):
        return self.r2r()
