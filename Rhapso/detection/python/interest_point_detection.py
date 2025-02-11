from data_preparation.tiff_image_reader import TiffImageReader

# This component detects interest points within overlapping areas of images

class PythonInterestPointDetection:
    def __init__(self, dataframes, overlapping_area, dsxy, dsz, prefix):
        self.localization = 1
        self.downsample_z = 2
        self.downsample_xy = 4
        self.image_sigma_x = 0.5
        self.image_sigma_y = 0.5
        self.image_sigma_z = 0.5
        self.min_intensity = 0.0
        self.max_intensity = 2048.0
        self.sigma = 1.8
        self.threshold = 0.008
        self.find_min = False
        self.find_max = True

        self.dataframes = dataframes
        self.image_loader_df = dataframes['image_loader']
        self.overlapping_area = overlapping_area
        self.dsxy = dsxy
        self.dsz = dsz
        self.prefix = prefix

        self.image_data = None
        self.interest_points = []

    # takes images and finds interest points
    def compute_difference_of_gaussian(self, images):
        initial_sigma = self.sigma
        min_peak_value = self.threshold
        min_initial_peak_value = min_peak_value / 3.0
        min = self.min_intensity
        max = self.max_intensity
        input_float = 0               # first big process
    
    def load_image_data(self, process_intervals, file_path):
        image_reader = TiffImageReader(self.dsxy, self.dsz, self.overlapping_area, process_intervals, file_path)                              
        return image_reader.run()

    def interest_point_detection(self):
        for _, row in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['view_setup']}"
            process_intervals = self.overlapping_area[view_id]
            file_path = self.prefix + row['file_path'] 
            images = self.load_image_data(process_intervals, file_path)
            self.compute_difference_of_gaussian(images)

    def run(self):
        self.interest_point_detection()

