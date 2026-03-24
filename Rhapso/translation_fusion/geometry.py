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

