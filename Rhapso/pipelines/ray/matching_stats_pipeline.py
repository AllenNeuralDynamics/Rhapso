import numpy as np
from Rhapso.evaluation.alignment_threshold import AlignmentThreshold
from Rhapso.evaluation.match_retrieval import MatchProcessor
from Rhapso.evaluation.matching_KDE import MatchingKDE
from Rhapso.evaluation.matching_descriptors import DescriptiveStatsMatching
from Rhapso.evaluation.matching_voxel_vis import VoxelVis
from Rhapso.evaluation.matching_voxelization import Voxelizer
from Rhapso.evaluation.save_metrics import JSONFileHandler
from Rhapso.evaluation.threshold import Threshold
from Rhapso.evaluation.total_ips import DetectionOutput

class StatsPipeline:
    def __init__(self, args, xml_file, base_path, metrics_output_path, file_source, xml_bucket_name, KDE_type, bandwidth, view_id, pair, plot,
                 thresholding, min_alignment, max_alignment, minimum_points, maximum_points, minimum_total_matches, maximum_total_matches, max_kde, min_kde, max_cv, min_cv ):
        self.args = args
        self.xml_file = xml_file
        self.base_path = base_path
        self.metrics_output_path = metrics_output_path
        self.file_source = file_source
        self.xml_bucket_name = xml_bucket_name
        self.thresholding = thresholding

        # KDE parameters
        self.KDE_type = KDE_type
        self.bandwidth = bandwidth
        self.view_id = view_id
        self.pair = pair
        self.plot = plot

        self.min_alignment = min_alignment
        self.max_alignment = max_alignment
        self.minimum_points = minimum_points
        self.maximum_points = maximum_points
        self.minimum_total_matches = minimum_total_matches
        self.maximum_total_matches = maximum_total_matches
        self.max_kde = max_kde
        self.min_kde = min_kde
        self.max_cv = max_cv
        self.min_cv = min_cv

    def run(self):
        # Detection Output
        detection_output = DetectionOutput(self.base_path, self.xml_file, self.metrics_output_path)
        detection_output.run()
        print("Detection output complete")

        # Match Processing
        processor = MatchProcessor(self.base_path, self.xml_file)
        matches, total_matches = processor.run()

        # Matching Descriptive Statistics
        descriptive_stats = DescriptiveStatsMatching(matches, total_matches)
        saveJSON = JSONFileHandler(self.metrics_output_path)
        points = descriptive_stats.get_matches()
        results = descriptive_stats.results()
        saveJSON.update("Descriptive stats", results)
        print("Descriptive statistics complete")

        # Voxelization
        if self.args["voxel"]:
            voxelization = Voxelizer(points, 10)
            voxel_info = voxelization.compute_statistics()
            saveJSON.update("Voxelization stats", voxel_info)
            print("Voxel statistics complete")

        # Voxel Visualization
        if self.args["voxel_vis"]:
            voxel_vis = VoxelVis(("30", "0"), matches)
            voxel_vis.run_voxel_vis()
            print("Voxel visualization complete")

        # KDE Analysis
        if self.args["KDE"]:
            kde = MatchingKDE(matches, self.KDE_type, self.bandwidth, self.view_id, self.pair, self.plot)
            kde_result = kde.get_data()
            saveJSON.update("KDE", kde_result)
            print("KDE computation complete")

        # Thresholding (Optional)
        if self.thresholding:
            threshold = Threshold(
                    self.minimum_points,
                    self.maximum_points,
                    self.minimum_total_matches,
                    self.maximum_total_matches,
                    self.max_kde,
                    self.min_kde,
                    self.max_cv,
                    self.min_cv,
                    self.metrics_output_path,
                )

            threshold.get_metric_json()
            threshold.run_threshold_checks()
            # Will error out if solve has not already been ran
            # alignmentThreshold = AlignmentThreshold(
            #     min_alignment, max_alignment, metrics_output_path
            # )
            # alignmentThreshold.check_alignment()
            print("Thresholding complete")

        print("All requested metrics are complete")


