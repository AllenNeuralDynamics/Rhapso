from setuptools import setup, find_packages

'''
To install certain dependencies, use the following commands:

- For matching-related libraries:
  pip install .[matching]

- For n5 reader libraries:
  pip install .[n5_reader]

'''

setup(
    name="Rhapso",
    version="0.1.8",
    packages=find_packages(),
    extras_require={
        "matching": ["scikit-learn", "bioio-tifffile", "bioio", "pandas", "boto3"],
        "n5_reader": [
            "zarr==2.18.7", 
            "s3fs==0.4.2", 
            "numpy==2.2.6", 
            "h5py==3.13.0", 
            "dask==2025.5.1", 
            "tensorstore==0.1.75"
        ],
    },
    entry_points={
        "console_scripts": [
            "Rhapso = Rhapso.cli:main",
            "featureMatchingHomeography_tiff = Rhapso.matching:featureMatchingHomeography_tiff",
        ],
    },
)
