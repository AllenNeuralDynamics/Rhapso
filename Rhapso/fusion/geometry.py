import numpy as np
from nptyping import NDArray, Shape

"""
Algorithm geometry primitives and utilities.
"""

Matrix = NDArray[Shape["3, 4"], np.float32]
AABB = tuple[int, int, int, int, int, int]

class Transform:
    """
    Registration Transform implemented in PyTorch.
    forward/backward transforms preserve the shape of the data.
    """

    def forward_np(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Please implement in Transform subclass.")

    def backward_np(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Please implement in Transform subclass.")

    # def forward(
    #     self, data: torch.Tensor, device: torch.device
    # ) -> torch.Tensor:
    #     raise NotImplementedError("Please implement in Transform subclass.")

    # def backward(
    #     self, data: torch.Tensor, device: torch.device
    # ) -> torch.Tensor:
    #     raise NotImplementedError("Please implement in Transform subclass.")

class Affine(Transform):
    def __init__(self, matrix: Matrix):
        super().__init__()

        # keep precision
        mat = np.asarray(matrix, dtype=np.float32)

        assert mat.shape == (3, 4), f"Matrix shape is {mat.shape}, must be (3, 4)"

        self.matrix = mat
        self.matrix_3x3 = mat[:, :3]
        self.translation = mat[:, 3].reshape(3)

        # Backward transform: inverse affine (float32)
        M = self.matrix_3x3
        t = self.translation

        M_inv = np.linalg.inv(M)
        t_inv = -M_inv @ t

        self.backward_matrix_3x3 = M_inv
        self.backward_translation = t_inv

    def forward_np(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"forward_np expects np.ndarray, got {type(data)}")
        if data.shape[-1] != 3:
            raise ValueError(f"Expected (...,3), got {data.shape}")

        d = np.asarray(data, dtype=np.float32)
        return d @ self.matrix_3x3.T + self.translation

    def backward_np(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"backward_np expects np.ndarray, got {type(data)}")
        if data.shape[-1] != 3:
            raise ValueError(f"Expected last dim == 3, got {data.shape}")

        d = np.asarray(data, dtype=np.float32)
        return d @ self.backward_matrix_3x3.T + self.backward_translation

# class Affine(Transform):
#     """
#     Rotation + Translation Registration.
#     """

#     def __init__(self, matrix: Matrix):
#         super().__init__()

#         mat = np.asarray(matrix, dtype=np.float32)

#         assert matrix.shape == (
#             3,
#             4,
#         ), "Matrix shape is {matrix.shape}, must be (3, 4)"

#         # self.matrix = torch.Tensor(matrix)
#         # self.matrix_3x3 = self.matrix[:, :3]
#         # self.translation = self.matrix[:, 3]

#         # self.backward_matrix_3x3 = torch.linalg.inv(self.matrix_3x3)
#         # self.backward_translation = -self.translation

#         self.matrix = mat                      # (3,4)
#         self.matrix_3x3 = mat[:, :3]           # (3,3)
#         self.translation = mat[:, 3].reshape(3)  # (3,)

#         # Backward transform: inverse affine
#         M = self.matrix_3x3
#         t = self.translation

#         M_inv = np.linalg.inv(M).astype(np.float32, copy=False)
#         t_inv = (-M_inv @ t).astype(np.float32, copy=False)

#         self.backward_matrix_3x3 = M_inv
#         self.backward_translation = t_inv

#     def forward_np(self, data: np.ndarray) -> np.ndarray:
#         if not isinstance(data, np.ndarray):
#             raise TypeError(f"forward_np expects np.ndarray, got {type(data)}")
#         if data.shape[-1] != 3:
#             raise ValueError(f"Expected (...,3), got {data.shape}")

#         data = data.astype(np.float32, copy=False)
#         # self.matrix_3x3: (3,3) numpy
#         # self.translation: (3,) numpy
#         return data @ self.matrix_3x3.T + self.translation

#     def backward_np(self, data: np.ndarray) -> np.ndarray:
#         """
#         NumPy-only backward transform.
#         data: (...,3) zyx float/any -> (...,3) float32
#         """
#         if not isinstance(data, np.ndarray):
#             raise TypeError(f"backward_np expects np.ndarray, got {type(data)}")
#         if data.shape[-1] != 3:
#             raise ValueError(f"Expected last dim == 3, got {data.shape}")

#         d = data.astype(np.float32, copy=False)
#         # self.backward_matrix_3x3 and self.backward_translation are already np.float32
#         return d @ self.backward_matrix_3x3.T + self.backward_translation
    
     # def forward(
    #     self, data: torch.Tensor, device: torch.device
    # ) -> torch.Tensor:
    #     """
    #     Parameters:
    #     -----------
    #     data: (dims) + (3,)
    #     data is a list/tensor of zyx vectors.

    #     device: {cuda:n, 'cpu'}
    #     device to perform computation on.

    #     Returns:
    #     --------
    #     transformed_data: (dims) + (3,)
    #     transformed_data is identical shape to the input.
    #     transformed_data lives on the device specified

    #     """
    #     assert (
    #         data.shape[-1] == 3
    #     ), "Data shape is {data.shape}, last dimension of input data must be 3d."

    #     # Ensure the matrix and translation are on the same device as data
    #     matrix = self.matrix_3x3.to(data.device)
    #     translation = self.translation.to(data.device)

    #     # Reshape translation if necessary
    #     translation = translation.reshape(3)

    #     # Apply matrix transformation
    #     # We use einsum for the matrix multiplication
    #     data = torch.einsum('ij,zyxj->zyxi', matrix, data)

    #     # Apply translation
    #     data += translation

    #     return data

    # def backward_np(self, data: np.ndarray) -> np.ndarray:
    #     """
    #     NumPy-only backward transform.

    #     Parameters
    #     ----------
    #     data : np.ndarray
    #         Shape (..., 3) of zyx vectors.

    #     Returns
    #     -------
    #     np.ndarray
    #         Shape (..., 3), float32.
    #     """
    #     if not isinstance(data, np.ndarray):
    #         raise TypeError(f"backward_np expects np.ndarray, got {type(data)}")
    #     if data.shape[-1] != 3:
    #         raise ValueError(f"Expected last dim == 3, got {data.shape}")

    #     data = data.astype(np.float32, copy=False)

    #     # --- Get matrix/translation as numpy ---
    #     M = self.backward_matrix_3x3
    #     t = self.backward_translation

    #     # If you haven't converted transforms yet, these may be torch.Tensors.
    #     if hasattr(M, "detach"):  # torch tensor
    #         M = M.detach().cpu().numpy()
    #     if hasattr(t, "detach"):
    #         t = t.detach().cpu().numpy()

    #     M = np.asarray(M, dtype=np.float32)
    #     t = np.asarray(t, dtype=np.float32).reshape(3)

    #     # Apply: (...,3) -> (...,3)
    #     out = data @ M.T
    #     out += t
    #     return out

    # def backward(
    #     self, data: torch.Tensor, device: torch.device
    # ) -> torch.Tensor:
    #     """
    #     Parameters:
    #     -----------
    #     data: (dims) + (3,)
    #     data is a list/tensor of zyx vectors.

    #     device: {cuda:n, 'cpu'}
    #     device to perform computation on.

    #     Returns:
    #     --------
    #     transformed_data: (dims) + (3,)
    #     transformed_data is identical shape to the input.
    #     transformed_data lives on the device specified
    #     """

    #     assert (
    #         data.shape[-1] == 3
    #     ), "Data shape is {data.shape}, last dimension of input data must be 3d."

    #     # Ensure the matrix and translation are on the same device as data
    #     matrix = self.backward_matrix_3x3.to(data.device)
    #     translation = self.backward_translation.to(data.device)

    #     # Reshape translation if necessary
    #     translation = translation.reshape(3)

    #     # Apply matrix transformation
    #     # We use einsum for the matrix multiplication
    #     data = torch.einsum('ij,zyxj->zyxi', matrix, data)

    #     # Apply translation
    #     data += translation

    #     return data

def aabb_3d_np(data) -> AABB:
    """
    NumPy AABB.
    data: np.ndarray shape (..., 3) in zyx order
    returns: (zmin, zmax, ymin, ymax, xmin, xmax)
    """
    data = np.asarray(data)
    if data.shape[-1] != 3:
        raise ValueError(f"Expected last dim == 3, got {data.shape}")

    pts = data.reshape(-1, 3)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    return (
        float(mins[0]), float(maxs[0]),
        float(mins[1]), float(maxs[1]),
        float(mins[2]), float(maxs[2]),
    )

# def aabb_3d(data) -> AABB:
#     """
#     Parameters:
#     -----------
#     data: (dims) + (3,)
#     data is a list/tensor of zyx vectors.

#     Returns:
#     --------
#     aabb: Ranges ordered in same order as components in input buffer.
#     """

#     assert (
#         data.shape[-1] == 3
#     ), "Data shape is {data.shape}, last dimension of input data must be 3d."
#     dims = len(data.shape)

#     output = []
#     for i in range(3):
#         # Slice syntax:
#         # (slice(None, None, None)) => arr[:]
#         # (i) => arr[i]
#         dim_slice = [slice(None, None, None)] * (dims - 1)
#         dim_slice = tuple(dim_slice + [i])

#         dim_min = torch.min(data[dim_slice]).item()
#         dim_max = torch.max(data[dim_slice]).item()
#         output.append(dim_min)
#         output.append(dim_max)

#     return tuple(output)
