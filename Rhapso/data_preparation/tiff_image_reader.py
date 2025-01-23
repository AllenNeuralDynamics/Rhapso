import os
from PIL import Image

# This component parses Tiff image data and creates a data catalog of image metadata for optimized image access

class TiffImageReader:
    def __init__(self):
        print("hi")
    
    def load_image_metadata(self):
        file_name = "/Users/seanfite/Desktop/AllenInstitute/Rhapso/Data/IP_TIFF_XML/spim_TL18_Angle0.tif"
        try:
            with Image.open(file_name) as img:
                img.seek(0)
                firstIFD = {tag: img.tag[tag] for tag in img.tag_v2}
                print(firstIFD)
        except IOError as e:
            print(f"Error reading TIFF file: {e}")

    def run(self):
        self.load_image_metadata()