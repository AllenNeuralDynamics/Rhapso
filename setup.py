from setuptools import setup, find_packages

# The following dependencies are not needed as they are included in AWS Glue 5.0:
# - numpy
# - boto3
# - pandas
# - matplotlib
# - s3fs
# - pyarrow

setup(
    name="Rhapso",
    version="0.0.3",
    packages=find_packages(),
    install_requires=[
        'dask[array]',  
        'zarr',
        'scipy',
        'scikit-image',
        'bioio',
        'bioio-tifffile',
        'tifffile==2025.1.10',
        'opencv-python',
    ],
    entry_points={
        "console_scripts": [
            "Rhapso = Rhapso.cli:main",
            "featureMatchingHomeography_tiff = Rhapso.matching:featureMatchingHomeography_tiff",
        ],
    },
)
