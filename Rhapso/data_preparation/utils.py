import boto3
import os
from urllib.parse import urlparse
from .xml_to_dataframe import XMLToDataFrame

def xml_to_dataframe(file_location):
    """Print Hello, World!"""
    print("Hello, World!")

if __name__ == "__main__":
    file_location = input("Enter the file location: ")
    xml_to_dataframe(file_location)
