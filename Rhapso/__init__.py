__version__ = "0.1.0"

from Rhapso.data_preparation.xml_to_dataframe import XMLToDataFrame
from Rhapso.detection.overlap_detection import OverlapDetection
from Rhapso.utils import xmlToDataframe as xmlToDataframeUtil

def xmlToDataframe(file_location):
    return xmlToDataframeUtil(file_location, XMLToDataFrame)

def runOverlapDetection(dataframes):
    overlap_detection = OverlapDetection()
    overlap_detection.run()
    return "Overlap detection completed"
