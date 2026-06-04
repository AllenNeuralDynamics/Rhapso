#!/usr/bin/env python3
"""Generate AIND processing metadata for the fusion capsule run."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aind_data_schema.base import _GenericModel
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage
from aind_data_schema_models.process_names import ProcessName


REPO_URL = "https://github.com/AllenNeuralDynamics/exaspim-bigstitcher-fusion-capsule"
GIT_URL = f"{REPO_URL}.git"
GITHUB_API_URL = (
    "https://api.github.com/repos/"
    "AllenNeuralDynamics/exaspim-bigstitcher-fusion-capsule/commits/main"
)
EXPERIMENTERS = ["Cameron Arshadi"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write AIND processing.json metadata for an ExaSPIM fusion run."
    )
    parser.add_argument("--s3-xml-path", required=True)
    parser.add_argument("--zarr-location", required=True)
    parser.add_argument("--spark-jar", required=True)
    parser.add_argument("--local-spark-jar", required=True)
    parser.add_argument("--block-size", required=True)
    parser.add_argument("--compression", required=True)
    parser.add_argument("--data-type", required=True)
    parser.add_argument("--storage-format", required=True)
    parser.add_argument("--s3-region", required=True)
    parser.add_argument("--downsample", action="append", default=[])
    parser.add_argument("--emr-arg", action="append", default=[])
    parser.add_argument("--start-date-time", required=True)
    parser.add_argument("--end-date-time", required=True)
    parser.add_argument(
        "--output-file",
        default="/results/processing.json",
        type=Path,
        help="Path where processing.json should be written.",
    )
    return parser.parse_args()


def fetch_main_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "ls-remote", GIT_URL, "refs/heads/main"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        sha = result.stdout.split()[0]
        if sha:
            return sha
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        pass

    request = Request(
        GITHUB_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "exaspim-bigstitcher-fusion-capsule",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to fetch code version from {GITHUB_API_URL}") from exc

    sha = payload.get("sha")
    if not isinstance(sha, str) or not sha:
        raise RuntimeError(f"GitHub response did not include a commit SHA: {payload}")
    return sha


def parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def build_parameters(args: argparse.Namespace) -> _GenericModel:
    parameters: dict[str, Any] = {
        "s3_xml_path": args.s3_xml_path,
        "zarr_location": args.zarr_location,
        "spark_jar": args.spark_jar,
        "local_spark_jar": args.local_spark_jar,
        "block_size": args.block_size,
        "compression": args.compression,
        "data_type": args.data_type,
        "storage_format": args.storage_format,
        "s3_region": args.s3_region,
        "downsamples": args.downsample,
        "emr_args": args.emr_arg,
    }
    return _GenericModel(**parameters)


def build_processing_metadata(args: argparse.Namespace) -> Processing:
    return Processing(
        data_processes=[
            DataProcess(
                process_type=ProcessName.IMAGE_TILE_FUSING,
                stage=ProcessStage.PROCESSING,
                code=Code(
                    url=REPO_URL,
                    name="exaspim-bigstitcher-fusion-capsule",
                    version=fetch_main_commit_sha(),
                    run_script="/code/run",
                    language="Python",
                    parameters=build_parameters(args),
                ),
                experimenters=EXPERIMENTERS,
                start_date_time=parse_datetime(args.start_date_time),
                end_date_time=parse_datetime(args.end_date_time),
            )
        ]
    )


def main() -> int:
    args = parse_args()
    processing = build_processing_metadata(args)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(processing.model_dump_json(indent=3), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())