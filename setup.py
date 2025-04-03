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
    version="0.1.8",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "Rhapso = Rhapso.cli:main",
            "featureMatchingHomeography_tiff = Rhapso.matching:featureMatchingHomeography_tiff",
        ],
    },
)
