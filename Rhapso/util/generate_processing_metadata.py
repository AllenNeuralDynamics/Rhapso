#!/usr/bin/env python3
"""Generate AIND processing metadata for Rhapso affine fusion."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3
import yaml

from aind_data_schema.base import _GenericModel
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage
from aind_data_schema_models.process_names import ProcessName


# --------------------------------------------------
# Hardcoded metadata
# --------------------------------------------------

REPO_URL = "https://github.com/AllenNeuralDynamics/Rhapso"
GIT_URL = f"{REPO_URL}.git"

CODE_NAME = "rhapso-affine-fusion"
RUN_SCRIPT = "/code/run"
LANGUAGE = "Python"
EXPERIMENTERS = ["Sean Fite"]

CONFIG_YML = Path("/code/config/config.yml")
LOCAL_PROCESSING_JSON = Path("/results/processing.json")

S3_XML_PATH = "s3://aind-open-data/exaSPIM_720164_2025-07-07_17-55-45_processed_2025-07-15_16-22-02/tile_alignment/rhapso/rhapso-solver-split-affine.xml"
ZARR_LOCATION = "s3://aind-open-data/exaSPIM_720164_2025-07-07_17-55-45_processed_2025-07-15_16-22-02/fusion/fused.zarr"
PROCESSING_JSON_S3 = "s3://aind-scratch-data/sean.fite/processing-json-test/exaSPIM-fusion/processing.json"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def fetch_main_commit_sha() -> str:
    result = subprocess.run(
        ["git", "ls-remote", GIT_URL, "refs/heads/main"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    return result.stdout.split()[0]


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def upload_file_to_s3(local_path: Path, s3_uri: str) -> None:
    parsed = urlparse(s3_uri)

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)

    print(f"[processing-metadata] Uploaded {local_path} -> {s3_uri}")


def build_parameters(config: dict) -> _GenericModel:
    parameters = {
        "s3_xml_path": S3_XML_PATH,
        "zarr_location": ZARR_LOCATION,

        # AFFINE FUSION
        "s3_xml_path": S3_XML_PATH,
        "zarr_location": ZARR_LOCATION,
        "block_size_xyz": config["block_size"],
        "intensity_range": config["intensity_range"],
        "block_scale_xyz": config["block_scale"],
        "overlap_strategy": config["overlap_strategy"],

        # MULTISCALE
        "zarr_location": ZARR_LOCATION,
        "multiscale_chunk_size": config["multiscale_chunk_size"],
        "n_lvls": config["n_lvls"],
        "scale_factor_zyx": config["scale_factor"],
        "target_block_size_mb": config["target_block_size_mb"],
        "base_level": config["base_level"],
    }

    return _GenericModel(**parameters)


def build_processing_metadata(config: dict) -> Processing:
    timestamp = datetime.now(timezone.utc)

    return Processing(
        data_processes=[
            DataProcess(
                process_type=ProcessName.IMAGE_TILE_FUSING,
                stage=ProcessStage.PROCESSING,
                code=Code(
                    url=REPO_URL,
                    name=CODE_NAME,
                    version=fetch_main_commit_sha(),
                    run_script=RUN_SCRIPT,
                    language=LANGUAGE,
                    parameters=build_parameters(config),
                ),
                experimenters=EXPERIMENTERS,
                start_date_time=timestamp,
                end_date_time=timestamp,
            )
        ]
    )


def main() -> int:
    config = load_config(CONFIG_YML)

    processing = build_processing_metadata(config)

    LOCAL_PROCESSING_JSON.parent.mkdir(parents=True, exist_ok=True)

    LOCAL_PROCESSING_JSON.write_text(
        processing.model_dump_json(indent=3),
        encoding="utf-8",
    )

    print(f"[processing-metadata] Wrote local file: {LOCAL_PROCESSING_JSON}")

    upload_file_to_s3(LOCAL_PROCESSING_JSON, PROCESSING_JSON_S3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())