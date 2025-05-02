# Interest Point Matching

1.	**XML Parsing** -	Extracts necessary metadata such as view IDs and setup information from an XML file, crucial for correlating different datasets.
2.	**Data Retrieval** - Fetches data from specified sources (local or cloud storage) based on the XML configuration, ensuring that all relevant image data is accessible for processing.
3.	**Interest Points Loading** -	Loads interest points data, which contains critical features extracted from images. This step is essential for subsequent matching procedures.
4.	**Interest Points Filtering** -	Filters out irrelevant or less significant points based on predefined criteria, refining the dataset for more accurate matching.
5.	**View Grouping** -	Organizes views into logical groups, facilitating efficient and systematic pairing for the matching process.
6.	**Pairwise Matching Setup** -	Prepares and configures the conditions and parameters for pairwise matching between the grouped views.
7.	**RANSAC for Matching** -	Applies the RANSAC algorithm to find the best match between pairs, using geometric constraints to validate the correspondences.
8.	**Match Refinement** - Refines the matches to ensure high accuracy, discarding outliers and confirming valid correspondences based on robust statistical methods.
9.	**Results Compilation and Storage** -	Aggregates all matching results and stores them in a designated format and location for further analysis or use in downstream processes.

## Running Matching

You can run the Matching pipeline by replacing the local file variables in the pipeline file with your local file paths. 
<!-- TODO -->
 [Pipeline File](../pipelines/matching/pairwise_matching_imports.py)
