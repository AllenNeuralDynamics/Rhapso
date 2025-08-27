from setuptools import setup, find_packages

'''
To install certain dependencies, use the following commands:

- For matching-related libraries:
  pip install .[matching]

- For n5 reader libraries:
  pip install .[n5_reader]

- For detection-related libraries:
  pip install .[detection]

'''

setup(
    name='Rhapso',
    version='0.1.9',
    author='Team OCTO',
    author_email='alleninstitute.org',
    description='A python package for stitching light sheet fluorescence microscopy images together',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'dask[array]==2024.12.1',  
        'zarr==2.18.3',
        'scipy==1.13.1',
        'scikit-image', 
        'bioio==1.3.0',
        'bioio-tifffile==1.0.0',
        'tifffile==2025.1.10',  
        'dask-image==2024.5.3',
        'boto3==1.35.92',
        'numcodecs==0.13.1',
        'matplotlib==3.10.0',
        'memory-profiler==0.61.0',
        's3fs==2024.12.0',
        'scikit-learn'
    ],
    python_requires='>=3.7', 
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
)
