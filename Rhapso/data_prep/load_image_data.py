from ..data_prep.tiff_image_reader import TiffImageReader 
from ..data_prep.zarr_image_reader import ZarrImageReader  

# This class interfaces the loading of Zarr or TIFF image data

class LoadImageData:
    def __init__(self, dataframes, overlapping_area, dsxy, dsz, prefix, file_type):
        self.dataframes = dataframes
        self.image_loader_df = dataframes['image_loader']
        self.overlapping_area = overlapping_area
        self.dsxy = dsxy
        self.dsz = dsz
        self.prefix = prefix
        self.file_type = file_type

    def load_image_data(self, process_intervals, file_path, view_id):
        """
        Loads image data based on the file type specified and processes it using the 
        corresponding image reader.
        """
        if self.file_type == 'zarr':
            image_reader = ZarrImageReader(self.dsxy, self.dsz, process_intervals, file_path, view_id)
        elif self.file_type == 'tiff':
            image_reader = TiffImageReader(self.dsxy, self.dsz, process_intervals, file_path, view_id) 
                                    
        return image_reader.run()

    def load_and_process_images(self):
        """
        Iterates through all entries in a DataFrame, interfacing the loading and processing
        of image data for each view configuration based on the file type.
        """
        images = []
        for _, row in self.image_loader_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['view_setup']}"
            process_intervals = self.overlapping_area[view_id]
            if self.file_type == 'zarr':
                file_path = self.prefix + row['file_path'] + f'/{0}'
            elif self.file_type == 'tiff':
                file_path = self.prefix + row['file_path'] 
            images.extend(self.load_image_data(process_intervals, file_path, view_id))

        return images
    
    def run(self):
        """
        Executes the entry point of the script.
        """
        return self.load_and_process_images()
