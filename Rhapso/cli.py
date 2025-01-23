import argparse
from Rhapso.utils import xmlToDataframe
from Rhapso.data_preparation.xml_to_dataframe import XMLToDataFrame
from Rhapso.detection.overlap_detection import OverlapDetection
from Rhapso import __version__

def main():
    parser = argparse.ArgumentParser(
        description="Rhapso CLI - Image processing tool for detection, matching, solving, and fusion."
    )

    # Add flags directly
    parser.add_argument('--xmlToDataframe', type=str, help='Convert XML to DataFrame and print a portion of the results')
    parser.add_argument('--runOverlapDetection', action='store_true', help='Run overlap detection.')
    parser.add_argument('-v', '--version', action='version', version=f'Rhapso {__version__}', help='Show the version of Rhapso')

    # Parse arguments and call the corresponding function
    args = parser.parse_args()
    if args.xmlToDataframe:
        dataframes = xmlToDataframe(args.xmlToDataframe, XMLToDataFrame)
        for name, df in dataframes.items():
            print(f"DataFrame: {name}")
            print(df.info())
            print(df.head())
    if args.runOverlapDetection:
        overlap_detection = OverlapDetection()
        overlap_detection.run()

if __name__ == "__main__":
    main()
