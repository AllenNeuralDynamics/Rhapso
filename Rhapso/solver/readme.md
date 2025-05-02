# Solver

1.	**XML to DataFrame** - Converts XML metadata into structured DataFrames to facilitate data manipulation and subsequent operations.
2.	**View Transform Models** -	Generates affine transformation matrices from the DataFrames, essential for aligning multiple views in a coherent manner.
3.	**Data Preparation** - Prepares and organizes data retrieved from different sources, setting the stage for effective model generation and tile setup.
4.	**Model and Tile Setup** - Creates models and configures tiles based on the prepared data and the transformation matrices, crucial for the optimization process.
5.	**Align Tiles** -	Applies transformation models to tiles, aligning them according to the specified parameters and conditions.
6.	**Global Optimization** -	Performs a comprehensive optimization over all tiles to refine the alignment based on a global perspective, ensuring consistency and accuracy across the dataset.
7.	**Save Results** - Saves the optimized results back to XML, documenting the new affine transformations for each view, thereby finalizing the process.

## Running Solver

You can run the solver pipeline by replacing the local file variables in the pipeline file with your local file paths. 

[Pipeline File](./Rhapso/Rhapso/pipelines/solver/python_pipeline.py)