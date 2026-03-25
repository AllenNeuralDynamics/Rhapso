from Rhapso.pipelines.ray.translation_fusion import TranslationFusion
from Rhapso.pipelines.ray.multiscale import MultiScale
import yaml

# Point to param file
with open("Rhapso/pipelines/ray/param/fusion/HCR_823476.yml", "r") as file:
    config = yaml.safe_load(file)

# FUSION
fusion = TranslationFusion(
    xml_path=config["xml_path_translation_fusion"],
    input_path=config["input_path_translation_fusion"],
    output_s3_path=config["output_s3_path_translation_fusion"],
    channel=config["channel"],
    default_chunk_size=config["default_chunk_size"],
    cpu_cell_size=config["cpu_cell_size"],
)

# MULTISCALE
multiscale = MultiScale(
    xml_path=config["xml_path_multiscale"],
    zarr_path=config["multiscale_zarr_path"],
    chunk_size=config["multiscale_chunk_size"],
    n_lvls=config["n_lvls"],
    scale_factor=config["scale_factor"],
    target_block_size_mb=config["target_block_size_mb"],
    base_level=config["base_level"],
)

# Run pipeline
# fusion.execute_job()
multiscale.run()