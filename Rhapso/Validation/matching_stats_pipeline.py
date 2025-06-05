import numpy as np
from Rhapso.Validation.Match_retrieval import MatchProcessor
from Rhapso.Validation.matching_descriptors import DescriptiveStatsMatching

file_source = 'local'
xml_file_path_output = "IP_TIFF_XML/dataset9.xml"
xml_bucket_name = "rhapso-tif-sample"
base_path = "/Users/ai/Downloads/IP_TIFF_XML/interestpoints.n5"
xml_file = "/Users/ai/Downloads/IP_TIFF_XML/dataset.xml~1"

processor = MatchProcessor(base_path, xml_file, file_source)
matches, total_matches = processor.run(processor)

descriptive_stats = DescriptiveStatsMatching(matches, total_matches)

results = descriptive_stats.results()

descriptive_stats.get_matches()
results = descriptive_stats.results()




