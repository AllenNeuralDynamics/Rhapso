"""
Algorithm geometry primitives and utilities.
"""

import numpy as np
import torch
from nptyping import NDArray, Shape

Matrix = NDArray[Shape["3, 4"], np.float64]
AABB = tuple[int, int, int, int, int, int]


class Transform:
    """
    Registration Transform implemented in PyTorch.
    forward/backward transforms preserve the shape of the data.
    """

    def forward(
        self, data: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        raise NotImplementedError("Please implement in Transform subclass.")

    def backward(
        self, data: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        raise NotImplementedError("Please implement in Transform subclass.")


class Affine(Transform):
    """
    Rotation + Translation Registration.
    """

    def __init__(self, matrix: Matrix):
        super().__init__()
        assert matrix.shape == (
            3,
            4,
        ), "Matrix shape is {matrix.shape}, must be (3, 4)"

        self.matrix = torch.Tensor(matrix)
        self.matrix_3x3 = self.matrix[:, :3]
        self.translation = self.matrix[:, 3]

        self.backward_matrix_3x3 = torch.linalg.inv(self.matrix_3x3)
        self.backward_translation = -self.translation

    def forward(
        self, data: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        data: (dims) + (3,)
        data is a list/tensor of zyx vectors.

        device: {cuda:n, 'cpu'}
        device to perform computation on.

        Returns:
        --------
        transformed_data: (dims) + (3,)
        transformed_data is identical shape to the input.
        transformed_data lives on the device specified

        """
        assert (
            data.shape[-1] == 3
        ), "Data shape is {data.shape}, last dimension of input data must be 3d."

        # Ensure the matrix and translation are on the same device as data
        matrix = self.matrix_3x3.to(data.device)
        translation = self.translation.to(data.device)

        # Reshape translation if necessary
        translation = translation.reshape(3)

        # Apply matrix transformation
        # We use einsum for the matrix multiplication
        data = torch.einsum('ij,zyxj->zyxi', matrix, data)

        # Apply translation
        data += translation

        return data


    def backward(
        self, data: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        data: (dims) + (3,)
        data is a list/tensor of zyx vectors.

        device: {cuda:n, 'cpu'}
        device to perform computation on.

        Returns:
        --------
        transformed_data: (dims) + (3,)
        transformed_data is identical shape to the input.
        transformed_data lives on the device specified
        """

        assert (
            data.shape[-1] == 3
        ), "Data shape is {data.shape}, last dimension of input data must be 3d."

        # Ensure the matrix and translation are on the same device as data
        matrix = self.backward_matrix_3x3.to(data.device)
        translation = self.backward_translation.to(data.device)

        # Reshape translation if necessary
        translation = translation.reshape(3)

        # Apply matrix transformation
        # We use einsum for the matrix multiplication
        data = torch.einsum('ij,zyxj->zyxi', matrix, data)

        # Apply translation
        data += translation

        return data


def aabb_3d(data) -> AABB:
    """
    Parameters:
    -----------
    data: (dims) + (3,)
    data is a list/tensor of zyx vectors.

    Returns:
    --------
    aabb: Ranges ordered in same order as components in input buffer.
    """

    assert (
        data.shape[-1] == 3
    ), "Data shape is {data.shape}, last dimension of input data must be 3d."
    dims = len(data.shape)

    output = []
    for i in range(3):
        # Slice syntax:
        # (slice(None, None, None)) => arr[:]
        # (i) => arr[i]
        dim_slice = [slice(None, None, None)] * (dims - 1)
        dim_slice = tuple(dim_slice + [i])

        dim_min = torch.min(data[dim_slice]).item()
        dim_max = torch.max(data[dim_slice]).item()
        output.append(dim_min)
        output.append(dim_max)

    return tuple(output)
