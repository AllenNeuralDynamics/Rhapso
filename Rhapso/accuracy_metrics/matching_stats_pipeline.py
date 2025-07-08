import numpy as np

from Rhapso.accuracy_metrics.match_retrieval import MatchProcessor
from Rhapso.accuracy_metrics.matching_KDE import MatchingKDE
from Rhapso.accuracy_metrics.matching_KDE import MatchingKDE
from Rhapso.accuracy_metrics.matching_descriptors import DescriptiveStatsMatching
from Rhapso.accuracy_metrics.matching_voxel_vis import VoxelVis
from Rhapso.accuracy_metrics.matching_voxelization import Voxelizer
from Rhapso.accuracy_metrics.save_metrics import JSONFileHandler

file_source = "local"
xml_file_path_output = "IP_TIFF_XML/dataset.xml"
xml_bucket_name = "rhapso-tif-sample"
base_path = "/Users/ai/Downloads/IP_TIFF_XML/interestpoints.n5"
xml_file = "/Users/ai/Downloads/IP_TIFF_XML/dataset.xml~1"
metrics_output_path = "/Users/ai/Downloads/IP_TIFF_XML/metrics.json"
args = {"voxel": True, "voxel_vis": False, "KDE": True}

def MatchingStatsPipeline(args, xml_file, base_path, output_path):
    # KDE type is "all", "pair", or "tile" in order to define how large of data set to run.
    KDE_type = None
    bandwidth = None
    view_id = None
    pair = None
    plot = False
    processor = MatchProcessor(base_path, xml_file, file_source, xml_bucket_name)
    matches, total_matches = processor.run(processor)
    descriptive_stats = DescriptiveStatsMatching(matches, total_matches)
    saveJSON = JSONFileHandler(output_path)
    points = descriptive_stats.get_matches()

    results = descriptive_stats.results()
    saveJSON.update("descriptive_stats", results)
    print("Descriptive Stats are done")

    if args["voxel"]:
        voxelization = Voxelizer(points, 10)
        voxel_info = voxelization.compute_statistics()
        saveJSON.update("voxelization stats", voxel_info)
        print("Voxel Statistics are done")

    if args["voxel_vis"]:
        voxel_vis = VoxelVis(("30", "0"), matches)
        voxel_vis.run_voxel_vis()
        # Python pop up window must be closed before the rest of the program can finish.

    if args["KDE"]:
        kde = MatchingKDE(matches, KDE_type, bandwidth, view_id, pair, plot)
        kde_result = kde.get_data()
        saveJSON.update("KDE", kde_result)
        # Python pop up window must be closed before the rest of the program can finish.
        print("KDE Computation Complete")
    
    print("All requested metrics are complete")

MatchingStatsPipeline(args, xml_file, base_path, metrics_output_path)

