from Rhapso.pipelines.ray.fusion import AffineFusion
from Rhapso.pipelines.ray.multiscale import MultiScale
from Rhapso.util.ng_link_gen_z1 import NeuroglancerLinkGeneratorZ1
import yaml

# Point to param file
with open("Rhapso/pipelines/ray/param/fusion/exaSPIM_720164.yml", "r") as file:
    config = yaml.safe_load(file)

# FUSION
fusion = AffineFusion(
    aligned_xml_path=config["aligned_xml_path"],
    zarr_input_prefix=config["zarr_input_prefix"],
    output_path=config["output_path"],
    block_size=config["block_size"],
    intensity_range=config["intensity_range"],
    block_scale=config["block_scale"],
)

# MULTISCALE
multiscale = MultiScale(
    zarr_path=config["multiscale_zarr_path"],
    chunk_size=config["multiscale_chunk_size"],
    voxel_size=config["voxel_size"],                           
    n_lvls=config["n_lvls"],
    scale_factor=config["scale_factor"],
    target_block_size_mb=config["target_block_size_mb"],
    base_level=config["base_level"],
)

ng_link_gen = NeuroglancerLinkGeneratorZ1(      
    zarr_paths=config["zarr_paths"],
    json_upload_path=config["json_upload_path"],
    vmin=config["vmin"],
    vmax=config["vmax"],
)

# Run pipeline
fusion.execute_job()
multiscale.run()
# ng_link_gen.run()