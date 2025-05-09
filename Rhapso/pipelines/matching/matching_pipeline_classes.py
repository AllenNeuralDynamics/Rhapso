from Rhapso.matching.xml_parser import XMLParser
from Rhapso.matching.data_loader import DataLoader
from Rhapso.matching.matcher import Matcher
from Rhapso.matching.result_saver import ResultSaver
from Rhapso.matching.ransac import RANSAC

class MatchingPipeline:
    def __init__(self, xml_file, n5_folder_base, output_path):
        self.xml_file = xml_file
        self.n5_folder_base = n5_folder_base
        self.output_path = output_path
        self.parser = XMLParser(xml_file)
        self.data_loader = DataLoader(n5_folder_base)
        
        # Create matcher with explicit model parameters using nested enums
        self.matcher = Matcher(
            transform_model=Matcher.TransformationModel.AFFINE,
            reg_model=Matcher.RegularizationModel.RIGID,
            lambda_val=0.1
        )
        self.ransac = RANSAC()
        self.saver = ResultSaver(output_path)

    def run(self):
        # Load complete dataset information
        data_global = self.parser.parse()
        
        # Use data_global for all subsequent operations
        view_ids_global = data_global['viewsInterestPoints']
        view_registrations = data_global['viewRegistrations']
        
        # Set up view groups using complete dataset info
        setup = self.parser.setup_groups(view_registrations)
        
        # Build label map using view IDs only
        label_map_global = self.data_loader.build_label_map(view_ids_global)
        
        # TODO: in parallel
        all_results = []
        for pair in setup['pairs']:
            task_result = self._process_matching_task(pair, label_map_global)
            all_results.extend(task_result if task_result else [])

        # TODO: in parallel
        for view_id in view_ids_global:
            corresponding_matches = [r for r in all_results if r[0] == view_id or r[1] == view_id]
            if corresponding_matches:
                self.saver.save_correspondences_for_view(view_id, corresponding_matches, data_global)

    def _process_matching_task(self, pair, label_map):
        """Process a single matching task"""
        viewA, viewB = pair
        # Load and transform interest points
        pointsA = self.data_loader.get_transformed_interest_points(viewA)
        pointsB = self.data_loader.get_transformed_interest_points(viewB)
        
        # Get candidates using matcher
        candidates = self.matcher._get_candidates(pointsA, pointsB)
        
        # Compute matches using geometric matcher
        matches = self.matcher._compute_match(candidates, pointsA, pointsB)
        
        # Filter matches using RANSAC
        filtered_matches = self.ransac.filter_matches(pointsA, pointsB, matches)
        
        return [(viewA, viewB, m[0], m[1]) for m in filtered_matches]

def main(xml_file, n5_folder_base, output_path):
    """
    Runs the matching pipeline.

    Parameters
    ----------
    xml_file : str
        Path to the dataset XML file.
    n5_folder_base : str
        Base path to the N5 image data folder.
    output_path : str
        Directory to save the results.
    """
    pipeline = MatchingPipeline(xml_file, n5_folder_base, output_path)
    pipeline.run()

if __name__ == "__main__":
    xml_file = "/home/martin/Documents/Allen/Data/IP_TIFF_XML/dataset.xml"
    n5_folder_base = "/home/martin/Documents/Allen/Data/IP_TIFF_XML/interestpoints.n5"
    output_path = "/home/martin/Documents/Allen/Data/IP_TIFF_XML/matchingOutput.n5"
    main(xml_file, n5_folder_base, output_path)
