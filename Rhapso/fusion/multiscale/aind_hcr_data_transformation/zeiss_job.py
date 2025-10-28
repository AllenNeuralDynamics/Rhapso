"""Module to handle zeiss data compression"""

import logging
import os
import shutil
import sys
from pathlib import Path
from time import time
from typing import Any, List, Optional

from aind_data_transformation.core import GenericEtl, JobResponse, get_parser
from numcodecs.blosc import Blosc

from .compress.czi_to_zarr import (
    czi_stack_zarr_writer,
)
from .models import (
    CompressorName,
    ZeissJobSettings,
)
from .utils import utils

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))


class ZeissCompressionJob(GenericEtl[ZeissJobSettings]):
    """Job to handle compressing and uploading Zeiss data."""

    @staticmethod
    def partition_list(
        lst: List[Any], num_of_partitions: int
    ) -> List[List[Any]]:
        """Partitions a list"""
        accumulated_list = []
        for _ in range(num_of_partitions):
            accumulated_list.append([])
        for list_item_index, list_item in enumerate(lst):
            a_index = list_item_index % num_of_partitions
            accumulated_list[a_index].append(list_item)
        return accumulated_list

    def _get_partitioned_list_of_stack_paths(self) -> List[List[Path]]:
        """
        Scans through the input source and partitions a list of stack
        paths that it finds there.
        """
        all_stack_paths = []
        total_counter = 0
        for p in (
            Path(self.job_settings.input_source).joinpath("SPIM").glob("*.czi")
        ):
            if p.is_file():
                total_counter += 1
                all_stack_paths.append(p)

        # Important to sort paths so every node computes the same list
        all_stack_paths.sort(key=lambda x: str(x))
        return self.partition_list(
            all_stack_paths, self.job_settings.num_of_partitions
        )

    @staticmethod
    def _get_voxel_resolution(acquisition_path: Path) -> List[float]:
        """Get the voxel resolution from an acquisition.json file."""

        if not acquisition_path.is_file():
            raise FileNotFoundError(
                f"acquisition.json file not found at: {acquisition_path}"
            )

        acquisition_config = utils.read_json_as_dict(acquisition_path)

        # Grabbing a tile with metadata from acquisition - we assume all
        # dataset was acquired with the same resolution
        tile_coord_transforms = acquisition_config["tiles"][0][
            "coordinate_transformations"
        ]

        scale_transform = [
            x["scale"] for x in tile_coord_transforms if x["type"] == "scale"
        ][0]

        x = float(scale_transform[0])
        y = float(scale_transform[1])
        z = float(scale_transform[2])

        return [z, y, x]

    def _get_compressor(self) -> Optional[Blosc]:
        """
        Utility method to construct a compressor class.
        Returns
        -------
        Blosc | None
          An instantiated Blosc compressor. Return None if not set in configs.

        """
        if self.job_settings.compressor_name == CompressorName.BLOSC:
            return Blosc(**self.job_settings.compressor_kwargs)
        else:
            return None

    def _write_stacks(self, stacks_to_process: List) -> None:
        """
        Write a list of stacks.
        Parameters
        ----------
        stacks_to_process : List

        Returns
        -------
        None

        """

        if not len(stacks_to_process):
            logging.info("No stacks to process!")
            return

        compressor = self._get_compressor()

        # Acquisition path in root folder
        acquisition_path = self.job_settings.input_source.joinpath(
            "acquisition.json"
        )

        # Getting voxel resolution
        voxel_size_zyx = self._get_voxel_resolution(
            acquisition_path=acquisition_path
        )

        # Converting CZI tiles to Multiscale OMEZarr
        for stack in stacks_to_process:
            logging.info(f"Converting {stack}")
            stack_name = stack.stem

            output_path = Path(self.job_settings.output_directory)

            msg = (
                f"Voxel resolution ZYX {voxel_size_zyx} for {stack} "
                f"with name {stack_name} - output: {output_path}"
            )
            logging.info(msg)

            czi_stack_zarr_writer(
                czi_path=str(stack),
                output_path=output_path,
                voxel_size=voxel_size_zyx,
                final_chunksize=self.job_settings.chunk_size,
                scale_factor=self.job_settings.scale_factor,
                n_lvls=self.job_settings.downsample_levels,
                channel_name=stack_name,
                stack_name=f"{stack_name}.ome.zarr",
                logger=logging,
                writing_options=compressor,
                target_size_mb=self.job_settings.target_size_mb,
            )

            if self.job_settings.s3_location is not None:
                channel_zgroup_file = output_path / ".zgroup"
                s3_channel_zgroup_file = (
                    f"{self.job_settings.s3_location}/.zgroup"
                )
                logging.info(
                    f"Uploading {channel_zgroup_file} to "
                    f"{s3_channel_zgroup_file}"
                )
                utils.copy_file_to_s3(
                    channel_zgroup_file, s3_channel_zgroup_file
                )
                ome_zarr_stack_name = f"{stack_name}.ome.zarr"
                ome_zarr_stack_path = output_path.joinpath(ome_zarr_stack_name)
                s3_stack_dir = (
                    f"{self.job_settings.s3_location}/"
                    f"{ome_zarr_stack_name}"
                )
                logging.info(
                    f"Uploading {ome_zarr_stack_path} to {s3_stack_dir}"
                )
                utils.sync_dir_to_s3(ome_zarr_stack_path, s3_stack_dir)
                logging.info(f"Removing: {ome_zarr_stack_path}")
                # Remove stack if uploaded to s3. We can potentially do all
                # the stacks in the partition in parallel using dask to speed
                # this up
                shutil.rmtree(ome_zarr_stack_path)

    def _upload_derivatives_folder(self):
        """
        Uploads the derivatives folder inside of
        the SPIM folder in the cloud.
        """
        s3_derivatives_dir = f"{self.job_settings.s3_location}/derivatives"
        derivatives_path = Path(self.job_settings.input_source).joinpath(
            "derivatives"
        )

        if not derivatives_path.exists():
            raise FileNotFoundError(f"{derivatives_path} does not exist.")

        if self.job_settings.s3_location is not None:
            logging.info(
                f"Uploading {derivatives_path} to {s3_derivatives_dir}"
            )
            utils.sync_dir_to_s3(derivatives_path, s3_derivatives_dir)
            logging.info(f"{derivatives_path} uploaded to s3.")

    def run_job(self):
        """Main entrypoint to run the job."""
        job_start_time = time()

        # Reading data within the SPIM folder
        partitioned_list = self._get_partitioned_list_of_stack_paths()

        # Upload derivatives folder
        if self.job_settings.partition_to_process == 0:
            self._upload_derivatives_folder()

        stacks_to_process = partitioned_list[
            self.job_settings.partition_to_process
        ]

        self._write_stacks(stacks_to_process=stacks_to_process)
        total_job_duration = time() - job_start_time
        return JobResponse(
            status_code=200, message=f"Job finished in {total_job_duration}"
        )


# TODO: Add this to core aind_data_transformation class
def job_entrypoint(sys_args: list):
    """Main function"""
    parser = get_parser()
    cli_args = parser.parse_args(sys_args)
    if cli_args.job_settings is not None:
        job_settings = ZeissJobSettings.model_validate_json(
            cli_args.job_settings
        )
    elif cli_args.config_file is not None:
        job_settings = ZeissJobSettings.from_config_file(cli_args.config_file)
    else:
        # Construct settings from env vars
        job_settings = ZeissJobSettings()
    job = ZeissCompressionJob(job_settings=job_settings)
    job_response = job.run_job()
    logging.info(job_response.model_dump_json())


if __name__ == "__main__":
    job_entrypoint(sys.argv[1:])
