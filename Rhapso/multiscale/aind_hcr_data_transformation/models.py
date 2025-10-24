"""Helpful models used in the compression job"""

from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from aind_data_transformation.core import BasicJobSettings
from dask import array as da
from numcodecs import Blosc
from pydantic import Field

ArrayLike = Union[da.Array, np.ndarray]
PathLike = Union[str, Path]


class CompressorName(str, Enum):
    """Enum for compression algorithms a user can select"""

    BLOSC = Blosc.codec_id


class ZeissJobSettings(BasicJobSettings):
    """ZeissCompressionJob settings."""

    input_source: PathLike = Field(
        ...,
        description=("Source of the Zeiss stack data."),
    )
    output_directory: PathLike = Field(
        ...,
        description=("Where to write the data to locally."),
    )
    s3_location: Optional[str] = None
    num_of_partitions: int = Field(
        ...,
        description=(
            "This script will generate a list of individual stacks, "
            "and then partition the list into this number of partitions."
        ),
    )
    partition_to_process: int = Field(
        ...,
        description=("Which partition of stacks to process. "),
    )
    compressor_name: CompressorName = Field(
        default=CompressorName.BLOSC,
        description="Type of compressor to use.",
        title="Compressor Name.",
    )
    # It will be safer if these kwargs fields were objects with known schemas
    compressor_kwargs: dict = Field(
        default={"cname": "zstd", "clevel": 3, "shuffle": Blosc.SHUFFLE},
        description="Arguments to be used for the compressor.",
        title="Compressor Kwargs",
    )
    compress_job_save_kwargs: dict = Field(
        default={"n_jobs": -1},  # -1 to use all available cpu cores.
        description="Arguments for recording save method.",
        title="Compress Job Save Kwargs",
    )
    chunk_size: List[int] = Field(
        default=[128, 128, 128],  # Default list with three integers
        description="Chunk size in axis, a list of three integers",
        title="Chunk Size",
    )
    scale_factor: List[int] = Field(
        default=[2, 2, 2],  # Default list with three integers
        description="Scale factors in axis, a list of three integers",
        title="Scale Factors",
    )
    downsample_levels: int = Field(
        default=4,
        description="The number of levels of the image pyramid",
        title="Downsample Levels",
    )
    target_size_mb: int = Field(
        default=19200,
        description="Target size to pull from the CZI file to zarr",
        title="Target Size",
    )
