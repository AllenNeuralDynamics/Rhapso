#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:59:07 2023

@author: nicholas.lusk
"""
import json
import os
from pathlib import Path
from typing import List, Union

import dask
import dask.array as da
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster, performance_report
from scipy.ndimage import zoom

# IO types
PathLike = Union[str, Path]


class ng_compressed_segmentation:
    """
    Class for creating a precomputed format for a compressed segmentation
    """

    def __init__(
        self,
        save_path: PathLike,
        resolution: List[int],
        dimensions: List[int],
        levels: List[int],
        chunk_size: int,
        compressed_encoding_size: int,
    ) -> None:
        """
        Class constructor

        Parameters
        ------------------------
        save_path: Pathlike
            Location for saving the precomputed format
        resolution: List[int]
            Resolution of each axis in the highest resolution volume (nm):
        dimentions: List[int]
            Dimentions of the tissue volume in voxels
        levels: List[int]
            factors by which to up or downsample resolutions for segmentation
            pyramid relative to image. Values greater than 1 upsample and
            values less than 1 downsample (e.g. 2 doubles resolution and 0.5
            halves resolution)
        chunk_size: int
            Chunk size (int**3) of each individual chunk. Common to use zarr
            chunk size
        compressed_encoding_size: int
            Chunk size (int**3) for each compressed sub-chunk. Common value is
            cunk_size/2
        """

        self.save_path = save_path
        self.resolution = resolution
        self.dimensions = dimensions
        self.levels = levels
        self.chunk_size = chunk_size
        self.compressed_encoding_size = compressed_encoding_size

        self.seg_path = os.path.join(save_path, "segment_properties")

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            os.mkdir(self.seg_path)

    def write_info_file(self) -> None:
        """
        Creates an info file in the main precomputed folder for a
        multi-scale compressed segmentation
        """
        scales = []

        for level in range(self.levels):
            res = [int(r / level) for r in self.resolution]
            current_scale = {
                "chunk_sizes": [
                    self.chunk_size,
                    self.chunk_size,
                    self.chunk_size,
                ],
                "encoding": "compressed_segmentation",
                "compressed_encoding_block_size": [
                    self.compressed_encoding_size,
                    self.compressed_encoding_size,
                    self.compressed_encoding_size,
                ],
                "key": "_".join([str(r) for r in res]),
                "resolution": res,
                "size": [int(d * level) for d in self.dimensions],
            }

        scales.append(current_scale)

        data = {
            "type": "segmentation",
            "segmentation_properties": "segment_properties",
            "data_type": "uint32",
            "num_channels": 1,
            "scales": scales,
        }

        with open(os.path.join(self.save_path, "info"), "w") as outfile:
            json.dump(data, outfile, indent=2)

    def write_seg_info(self, ccf_reference_path: PathLike) -> None:
        """
        Creates a segmentation info file and places it in a
        precomputed subfolder

        Parameters
        ----------
        ccf_reference_path : PathLike
            location of ccf_ref.csv. A file containing a mapping of region IDs
            to their corresponding acronyms

        Returns
        -------
        None.

        """

        df_ccf = pd.read_csv(ccf_reference_path)

        # build json for segmentation properties
        data = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": df_ccf["id"].to_list(),
                "properties": [
                    {
                        "id": "label",
                        "type": "label",
                        "values": df_ccf["struct"].to_list(),
                    }
                ],
            },
        }

        with open(os.path.join(self.seg_path, "info"), "w") as outfile:
            json.dump(data, outfile, indent=2)

    @dask.delayed
    def compress_array(
        self, sub_array: np.array, factor: float, blocksize: int
    ) -> list:
        """

        Parameters
        ----------
        sub_array : array
            Numpy array corresponding to the size of a block prior to scaling
            t0 chunk size associated with the compressed segmentation
        factor : float
            How to scale input array. Numbers larger than 1 upsample,
            smaller than 1 downsample
        blocksize int:
            Size of block for compressed encoding

        Raises
        ------
        RuntimeError
            Compressed segmentations must be chuncked in cubes, therefore
            if shape is not a cube, throw error.

        Returns
        -------
        encoding_list: list
            delayed output for the encoding of a given sub array into
            precomputed format

        """
        resample = zoom(
            np.asarray(sub_array), factor, order=0
        )  # order 0 gives nearest neighbor interpolation

        nx, ny, nz = resample.shape
        encoding_list = []
        for z0 in range(0, nz, blocksize):
            z1 = min(z0 + blocksize, nz)
            for y0 in range(0, ny, blocksize):
                y1 = min(y0 + blocksize, ny)
                for x0 in range(0, nx, blocksize):
                    x1 = min(x0 + blocksize, nx)

                    # check block shape and pad if needed.
                    block = self.get_block(
                        resample[x0:x1, y0:y1, z0:z1], blocksize=blocksize
                    )

                    if block.shape != (blocksize, blocksize, blocksize):
                        raise RuntimeError(
                            f"block.shape {block.shape} but "
                            f"blocksize = {blocksize}"
                        )

                    encoding_list.append(self.encode_block(block))

        return encoding_list

    def build_compression(self, img: np.array, level: float) -> list:
        """
        Takes numpy array of segmenations and converts to byte stream
        compatable with neuroglancer precomputed compressed segmentation

        Parameters
        ----------
        img: np.array
            3D array of segmenation values
        level: float
            the factor by which the current image needs to be scaled

        Returns
        -------
        List:
            list containing a list of file names and a list of delayed
            functions

        """

        b_size = int(self.chunk_size / level)
        data = da.from_array(img, chunks=(b_size, b_size, b_size))

        level_dir = os.path.join(
            self.save_path, "_".join([str(r / level) for r in self.resolution])
        )

        os.mkdir(level_dir)

        output = []
        fpaths = []
        data_dims = [len(x) for x in data.chunks]

        print("Building delayed output list")
        for x in range(data_dims[0]):
            x0, x1 = x * self.chunk_size, (x + 1) * self.chunk_size
            for y in range(data_dims[1]):
                y0, y1 = y * self.chunk_size, (y + 1) * self.chunk_size
                for z in range(data_dims[2]):
                    z0, z1 = z * self.chunk_size, (z + 1) * self.chunk_size
                    fname = f"{x0}-{x1}_{y0}-{y1}_{z0}-{z1}"
                    fpaths.append(os.path.join(level_dir, fname))
                    output.append(
                        self.compress_array(
                            np.array(data.blocks[x, y, z]),
                            level,
                            self.compressed_encoding_size,
                        )
                    )
        return [fpaths, output]

    def write_encoding(self, encoding_list: list, file_path: PathLike) -> None:
        """
        Take encoding list from compress ccf delayed output and write file

        data will be orderd as recommended
        * headers
        * data for block0
        * lookup for block0
        * data for block1
        * lookup for block1
        """

        n_blocks = len(encoding_list)
        header_offset = n_blocks * 2
        header_list = []
        running_offset = header_offset

        for i_block in range(n_blocks):
            this_encoding = encoding_list[i_block]

            n_bits = this_encoding["n_bits"]

            n_lookup_bytes = len(this_encoding["lookup_table"])
            assert n_lookup_bytes % 4 == 0
            n_lookup = n_lookup_bytes // 4

            n_data_bytes = len(this_encoding["encoded_data"])
            assert n_data_bytes % 4 == 0
            n_data = n_data_bytes // 4

            data_offset = running_offset
            assert data_offset < 2 ** 32
            running_offset += n_data

            lookup_offset = running_offset
            assert lookup_offset < 2 ** 24
            running_offset += n_lookup

            this_header = b""
            this_header += lookup_offset.to_bytes(3, byteorder="little")
            this_header += n_bits.to_bytes(1, byteorder="little")
            this_header += data_offset.to_bytes(4, byteorder="little")
            if len(this_header) != 8:
                raise RuntimeError(
                    f"header\n{this_header}\nlen {len(this_header)}"
                )
            header_list.append(this_header)

        print(f"Writing {file_path}")
        with open(file_path, "wb") as out_file:
            # specify that this is just one channel
            out_file.write((1).to_bytes(4, byteorder="little"))
            for header in header_list:
                out_file.write(header)
            for encoding in encoding_list:
                out_file.write(encoding["encoded_data"])
                out_file.write(encoding["lookup_table"])

    def get_block(self, block: np.array, blocksize: int):
        """
        Get and return a block of data from all_data (a np.ndarray).

        blocksize is the desired size (the block will be a cube)
        of the datablock. if a block is smaller than
        a (blocksize, blocksize, blocksize) cube, use np.pad to fill out
        the block.
        """

        pad_x = 0
        if block.shape[0] != blocksize:
            pad_x = blocksize - block.shape[0]
        pad_y = 0
        if block.shape[1] != blocksize:
            pad_y = blocksize - block.shape[1]
        pad_z = 0
        if block.shape[2] != blocksize:
            pad_z = blocksize - block.shape[2]

        if pad_x + pad_y + pad_z > 0:
            val = block[0, 0, 0]
            block = np.pad(
                block,
                pad_width=[[0, pad_x], [0, pad_y], [0, pad_z]],
                mode="constant",
                constant_values=val,
            )
        return block

    def encode_block(self, data: np.array) -> dict:
        """
        Returns dict containing

        the byte stream that is the encoded data

        the byte stream that is the lookup table of values

        the number of bits used to encode the values in the
        lookup table
        """
        # ensure that data is of type int
        data = data.astype("uint32")

        encoding = self.get_block_lookup_table(data)

        n_bits = encoding["n_bits_to_encode"]
        encoder = encoding["dict"]

        nx = data.shape[0]
        ny = data.shape[1]
        nz = data.shape[2]

        byte_stream = b""
        ct = data.size
        if n_bits > 0:
            bit_stream = self.block_to_bits(
                block=data, encoder_dict=encoder, n_bits=n_bits
            )

            byte_stream = self.bits_to_bytes(bit_stream)

        expected_len = np.ceil(ct * n_bits / 8).astype(int)

        # because compression must be in integer multiples
        # of 32 bits (I think)
        if expected_len % 4 > 0:
            expected_len += 4 - (expected_len % 4)

        if len(byte_stream) != expected_len:
            raise RuntimeError(
                f"len bytes {len(byte_stream)}\n"
                f"expected {expected_len}\n"
                f"{(nx, ny, nz)}\n"
                f"{n_bits}\n"
                f"len(bit_stream) {len(bit_stream)}"
            )

        return {
            "encoded_data": byte_stream,
            "lookup_table": encoding["bytes"],
            "n_bits": n_bits,
        }

    def block_to_bits(
        self, block: np.array, encoder_dict: dict, n_bits: int
    ) -> np.array:
        """
        Convert block into a string of bits encoded according
        to encoder_dict.

        Parameters
        ----------
        block: np.ndarray
            Data to encode

        encoder_dict: dict
            Dict mapping values in block to encoded values
            (smaller ints)

        n_bits: int
            Number of bits used to store each value in the
            returned bit stream

        Returns
        -------
        bit_stream: np.ndarray
           Booleans representing the bits of the encoded
           values. Should be block.size*n_bits long.
           Values will be little-endian (least significatn
           bit first).
        """
        n_total_bits = block.size * n_bits
        if n_total_bits % 32 > 0:
            n_total_bits += 32 - (n_total_bits % 32)
        assert n_total_bits % 32 == 0

        bit_stream = np.zeros(n_total_bits, dtype=bool)
        bit_masks = np.array([2 ** ii for ii in range(n_bits)]).astype(int)
        block = np.array([encoder_dict[val] for val in block.flatten("F")])

        for i_bit in range(n_bits):
            detections = block & bit_masks[i_bit]
            # flake8: noqa: E203
            bit_stream[i_bit : detections.size * n_bits : n_bits] = (
                detections > 0
            )

        return bit_stream

    def bits_to_bytes(self, bit_stream: np.array) -> bytes:
        """
        Convert the bit stream produced  to a byte
        stream that can be written out to the
        compressed data file

        Parameters
        ----------
        bit_stream: np.ndarray
            Data from block_to_bits to convert into
            bytes

        Returns
        -------
        byte_stream: bytes
            Bit data converted to a byte stream

        """

        n_bits = len(bit_stream)
        assert n_bits % 32 == 0

        # Convert the bit stream into a series of little-endian
        # 32 bit unsigned integers. These values will ultimately
        # get converted to bytes and stored in the output byte
        # stream.
        bit_grid = np.array(bit_stream).reshape(n_bits // 32, 32)
        values = np.zeros(bit_grid.shape[0], dtype=np.uint32)
        pwr = 1
        for icol in range(bit_grid.shape[1]):
            these_bits = bit_grid[:, icol]
            values[these_bits] += pwr
            pwr *= 2

        # initialize empty byte stream
        byte_stream = bytearray(n_bits // 8)

        # transcribe values in byte stream
        for i_val, val in enumerate(values):
            # flake8: noqa: E203
            byte_stream[i_val * 4 : (i_val + 1) * 4] = int(val).to_bytes(
                4, byteorder="little", signed=False
            )

        return bytes(byte_stream)

    def get_block_lookup_table(self, data: np.array) -> dict:
        """
        Get the lookup table for encoded values in data.

        Parameters
        ----------
        data: np.array
            cubic block containing segment IDs

        Returns
        -------
        table: dict
            Mapping between raw values to encoded values

        byte stream representing the lookup table of raw values
        (this is just a sequence of values; the value's position
        in bytestream represents its encoded value, i.e. the 5th
        raw value in byte stream gets encoded to the value 5)

        number of bits used to encode each value
        """
        max_val = data.max()
        if data.max() >= 2 ** 32:
            raise RuntimeError(f"max_val {max_val} >= 2**32")

        unq_values = np.unique(data).astype(np.uint32)
        n_unq = len(unq_values)
        raw_n_bits_to_encode = np.ceil(np.log(n_unq) / np.log(2)).astype(int)

        if raw_n_bits_to_encode == 0:
            n_bits_to_encode = 0
        else:
            n_bits_to_encode = 1
            while n_bits_to_encode < raw_n_bits_to_encode:
                n_bits_to_encode *= 2

        if n_bits_to_encode >= 32:
            raise RuntimeError(
                f"n_bits_to_encode {n_bits_to_encode}\n" f"n_unq {n_unq}"
            )

        val_to_encoded = dict()
        byte_stream = b""

        for ii, val in enumerate(unq_values):
            val_to_encoded[val] = ii

            # bytestream will be encoded_to_val
            # since that is used for *decoding* the data
            val_bytes = int(val).to_bytes(4, byteorder="little")

            # encoded_bytes = int(ii).to_bytes(4, byteorder='little')
            # byte_stream += encoded_bytes

            byte_stream += val_bytes

        assert len(byte_stream) == 4 * len(unq_values)

        return {
            "bytes": byte_stream,
            "dict": val_to_encoded,
            "n_bits_to_encode": n_bits_to_encode,
        }

    def initialize_dask(self) -> None:
        dask_folder = Path(os.path.abspath("/scratch/tmp"))
        # Setting dask configuration
        dask.config.set(
            {
                "temporary-directory": dask_folder,
                "local_directory": dask_folder,
                "tcp-timeout": "300s",
                "array.chunk-size": "384MiB",
                "distributed.comm.timeouts": {
                    "connect": "300s",
                    "tcp": "300s",
                },
                "distributed.scheduler.bandwidth": 100000000,
                "distributed.worker.memory.rebalance.measure": "optimistic",
                "distributed.worker.memory.target": False,
                "distributed.worker.memory.spill": 0.92,
                "distributed.worker.memory.pause": 0.95,
                "distributed.worker.memory.terminate": 0.98,
            }
        )


def main(params: dict) -> None:
    ng_compressed = ng_compressed_segmentation(
        params["save_path"],
        params["resolution"],
        params["dimensions"],
        params["levels"],
        params["chunk_size"],
        params["compressed_encoding_size"],
    )

    ccf_reference_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "ccf_ref.csv"
    )

    ng_compressed.write_info_file()
    ng_compressed.write_seg_info(ccf_reference_path)
    ng_compressed.initialize_dask()

    # start client
    cluster = LocalCluster(n_workers=16, processes=True, threads_per_worker=1,)

    client = Client(cluster)

    for c, level in enumerate(ng_compressed.levels):
        outputs = ng_compressed.build_compression(params["img"], level)

        graph = len(outputs[1])

        print(f"Computing delayed for {graph} objects.")
        dask_report_file = "/results/dask_profile_loop_{0}.html".format(c)
        with performance_report(filename=dask_report_file):
            encodings = dask.delayed(outputs[1]).compute()

        print("Writing segmentation compression files")
        for encoding, fpath in zip(encodings, outputs[0]):
            ng_compressed.write_encoding(encoding, fpath)

    client.close()


if __name__ == "__main__":
    params = {
        "img": "/data/img",
        "save_path": "/results/seg_precompute",
        "resolution": [2000, 1800, 1800],
        "dimensions": [4192, 1024, 7400],
        "levels": [1, 2.5, 5, 10],
        "chunk_size": 128,
        "compressed_encoding_size": 64,
    }

    main(params)
