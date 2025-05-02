# Interest Point Detection

1.	**XML to DataFrame** -	Converts XML metadata into structured DataFrames to facilitate data manipulation.
2.	**View Transform Models** -	Generates transformation matrices from XML data to align multiple views in a dataset.
3.	**Overlap Detection** -	Identifies overlapping areas between different views using the transformation matrices.
4.	**Load Image Data** -	Loads and preprocesses image data based on detected overlaps, preparing it for feature detection.
5.	**Difference of Gaussian** - Applies the Difference of Gaussian (DoG) method to identify potential interest points in the image data.
6.	**Advanced Refinement** -	Refines detected points using a KD-tree structure to ensure accuracy and relevance of features.
7.	**Save Interest Points** - Saves the refined interest points and associated metadata for further analysis or usage.

## Running Interest Point Detection

You can run the Matching pipeline by replacing the local file variables in the pipeline file with your local file paths. 

 [Pipeline File](../pipelines/detection/python_pipeline.py)
