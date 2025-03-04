from setuptools import setup, find_packages

setup(
    name="Rhapso",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'boto3',
        'pandas',
        'matplotlib',
        'dask[array]',  
        'zarr',
        's3fs',
        'scipy',
        'scikit-image',
        'bioio',
        'bioio-tifffile',
        'tifffile==2025.1.10',
        'opencv-python',  
        'pyarrow',
    ],
    entry_points={
        "console_scripts": [
            "Rhapso = Rhapso.cli:main",
            "featureMatchingHomeography_tiff = Rhapso.matching:featureMatchingHomeography_tiff",
        ],
    },
)
