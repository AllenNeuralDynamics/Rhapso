import numpy as np

from Rhapso.accuracy_metrics.match_retrieval import MatchProcessor
from Rhapso.accuracy_metrics.matching_KDE import MatchingKDE
from Rhapso.accuracy_metrics.matching_descriptors import DescriptiveStatsMatching
from Rhapso.accuracy_metrics.matching_voxel_vis import VoxelVis
from Rhapso.accuracy_metrics.matching_voxelization import Voxelizer

file_source = "local"
xml_file_path_output = "IP_TIFF_XML/dataset9.xml"
xml_bucket_name = "rhapso-tif-sample"
base_path = "/Users/ai/Downloads/IP_TIFF_XML/interestpoints.n5"
xml_file = "/Users/ai/Downloads/IP_TIFF_XML/dataset.xml~1"

processor = MatchProcessor(base_path, xml_file, file_source, xml_bucket_name)
# Matches = a dictionary with matches split by view_id, total matches is an int
matches, total_matches = processor.run(processor)

descriptive_stats = DescriptiveStatsMatching(matches, total_matches)

points = descriptive_stats.get_matches()
descriptive_stats.results()

voxelization = Voxelizer(points, 10)

voxel_info = voxelization.compute_statistics()

# voxel_vis = VoxelVis(('30', '0'), matches)
# voxel_vis.run_voxel_vis()

# bandwidth can be a float, if not inputted it defaults to Scott's value.
kde = MatchingKDE(matches, "all", None, None, None, False)
kde.get_data()
