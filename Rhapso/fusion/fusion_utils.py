from collections import defaultdict
import re
import numpy as np
import xmltodict
import Rhapso.fusion.geometry as geometry
import boto3
from io import BytesIO
import xmltodict

def check_collision(
    cell_box: geometry.AABB,
    t_aabb: geometry.AABB
) -> bool:
    """
    Check collision between two boxes.
    """
    return ((cell_box[1] > t_aabb[0] and cell_box[0] < t_aabb[1])
            and (cell_box[3] > t_aabb[2] and cell_box[2] < t_aabb[3])
            and (cell_box[5] > t_aabb[4] and cell_box[4] < t_aabb[5]))

def _cell_corners_zyx(cell_box):
    z0, z1, y0, y1, x0, x1 = cell_box
    zs = np.array([z0 + 0.5, z1 - 0.5], dtype=np.float32)
    ys = np.array([y0 + 0.5, y1 - 0.5], dtype=np.float32)
    xs = np.array([x0 + 0.5, x1 - 0.5], dtype=np.float32)
    # 8 corners in zyx order
    corners = np.array([[z, y, x] for z in zs for y in ys for x in xs], dtype=np.float32)
    return corners  # (8,3)

def _aabb_from_points_zyx(pts_zyx: np.ndarray):
    # pts_zyx shape (N,3) in (z,y,x)
    mins = pts_zyx.min(axis=0)
    maxs = pts_zyx.max(axis=0)
    return (float(mins[0]), float(maxs[0]),
            float(mins[1]), float(maxs[1]),
            float(mins[2]), float(maxs[2]))

def calculate_image_crop(
    cell_box,
    output_volume_origin,
    transform_list,
    src_vol_shape_zyx,
):
    """
    Torch-free version.
    Returns a 5D slice tuple: (0,0, z-slice, y-slice, x-slice)
    """
    # 8 corner points in zyx basis
    pts = _cell_corners_zyx(cell_box)  # (8,3)

    # add origin (broadcast)
    origin = np.asarray(output_volume_origin, dtype=np.float32)  # (3,)
    pts = pts + origin

    # apply inverse transforms in reverse order
    # IMPORTANT: tfm.backward must accept numpy input or we wrap it (see note below).
    for tfm in reversed(transform_list):
        pts = tfm.backward_np(pts)  # expect shape (N,3) back

    # AABB in source space
    z_min, z_max, y_min, y_max, x_min, x_max = _aabb_from_points_zyx(pts)

    sv_z, sv_y, sv_x = src_vol_shape_zyx

    # collision check (same logic as before)
    aabb_src = (0.0, float(sv_z), 0.0, float(sv_y), 0.0, float(sv_x))
    if check_collision((z_min, z_max, y_min, y_max, x_min, x_max), aabb_src) is False:
        raise ValueError(
            "Provided cell_box does not transform into the provided source_volume."
        )

    # clip to bounds
    crop_min_z = max(0.0, z_min)
    crop_max_z = min(float(sv_z), z_max)
    crop_min_y = max(0.0, y_min)
    crop_max_y = min(float(sv_y), y_max)
    crop_min_x = max(0.0, x_min)
    crop_max_x = min(float(sv_x), x_max)

    # floor/ceil to integer indices
    crop_min_z = int(np.floor(crop_min_z))
    crop_min_y = int(np.floor(crop_min_y))
    crop_min_x = int(np.floor(crop_min_x))
    crop_max_z = int(np.ceil(crop_max_z))
    crop_max_y = int(np.ceil(crop_max_y))
    crop_max_x = int(np.ceil(crop_max_x))

    return (
        0,
        0,
        slice(crop_min_z, crop_max_z),
        slice(crop_min_y, crop_max_y),
        slice(crop_min_x, crop_max_x),
    )

# def calculate_image_crop(cell_box: geometry.AABB,
#                          output_volume_origin: tuple[float, float, float],
#                          transform_list: list[geometry.Transform],
#                          src_vol_shape_zyx: tuple[int, int, int],
#                          device='cpu') -> tuple[slice]:
#     """
#     Inverse transforms cell_box boundary into source_vol space
#     and returns an index-aligned, boundary-clipped crop in source_vol coordinates.
#     Outputs a 5d-tuple of slices.

#     IMPORTANT NOTE ON INPUTS:
#     - cell_box is a AABB of index ranges, not absolute coordinates.

#     - The expected order of transform_list is feedforward.
#     We will apply these matrices in reverse!
#     """

#     # Derive cell_box extrema
#     z_min, z_max, y_min, y_max, x_min, x_max = cell_box
#     z_grid, y_grid, x_grid = torch.meshgrid(
#         torch.Tensor([z_min + 0.5, z_max - 0.5]),
#         torch.Tensor([y_min + 0.5, y_max - 0.5]),
#         torch.Tensor([x_min + 0.5, x_max - 0.5]),
#         indexing='ij'
#     )
#     cell_box_pts = torch.stack([z_grid, y_grid, x_grid], dim=-1)

#     # Apply inverse transform
#     cell_box_pts = cell_box_pts + torch.Tensor(output_volume_origin).to(
#         device
#     )

#     for tfm in reversed(transform_list):
#         cell_box_pts = tfm.backward(cell_box_pts, device=device)

#     # Derive axis-aligned, boundary-clipped image
#     cell_box_src: geometry.AABB = geometry.aabb_3d(cell_box_pts)
#     z_min, z_max, y_min, y_max, x_min, x_max = cell_box_src

#     sv_z, sv_y, sv_x = src_vol_shape_zyx
#     aabb_src: geometry.AABB = (0, sv_z, 0, sv_y, 0, sv_x)

#     # Validation
#     if check_collision(cell_box_src, aabb_src) is False:
#         raise ValueError(f"""Provided cell_box does not transform
#                          into the provided source_volume.
#                          Please check all inputs.""")

#     # Calculate overlapping region between transformed coords and image boundary
#     # For intervals (A, B):
#     # The lower bound of overlapping region = max(A_min, B_min)
#     # The upper bound of overlapping region = min(A_max, B_max)
#     crop_min_z = torch.max(torch.Tensor([0, z_min]))
#     crop_max_z = torch.min(torch.Tensor([sv_z, z_max]))

#     crop_min_y = torch.max(torch.Tensor([0, y_min]))
#     crop_max_y = torch.min(torch.Tensor([sv_y, y_max]))

#     crop_min_x = torch.max(torch.Tensor([0, x_min]))
#     crop_max_x = torch.min(torch.Tensor([sv_x, x_max]))

#     # Round to integers for indexing
#     # Minimum values are rounded down to nearest integer.
#     # Maximum values are rounded up to nearest integer.
#     crop_min_z = int(torch.floor(crop_min_z))
#     crop_min_y = int(torch.floor(crop_min_y))
#     crop_min_x = int(torch.floor(crop_min_x))

#     crop_max_z = int(torch.ceil(crop_max_z))
#     crop_max_y = int(torch.ceil(crop_max_y))
#     crop_max_x = int(torch.ceil(crop_max_x))

#     image_crop_slice = (
#         0,
#         0,
#         slice(crop_min_z, crop_max_z),
#         slice(crop_min_y, crop_max_y),
#         slice(crop_min_x, crop_max_x),
#     )

#     return image_crop_slice

def calculate_sample_field_np(
    cell_box: tuple[int, int, int, int, int, int],
    output_volume_origin: tuple[float, float, float],
    transform_list,  # list of Affine-like transforms
    src_vol_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """
    Returns grid_xyz_norm float32, shape (z, y, x, 3),
    xyz basis in last dim, normalized [-1, 1] (align_corners=False style).

    Main improvements (still only editing this one function):
      - Fuse transform_list into one affine (A,b) once per call
      - Use 8-corner transform for crop bounds
      - Avoid meshgrid+stack (broadcast instead)
      - Avoid calling backward_np on a (Z,Y,X,3) array
    """
    z0, z1, y0, y1, x0, x1 = cell_box
    sv_z, sv_y, sv_x = src_vol_shape_zyx

    origin = np.asarray(output_volume_origin, dtype=np.float32)

    # ------------------------------------------------------------
    # 1) Fuse backward affines inside this call: p' = p @ A.T + b
    # ------------------------------------------------------------
    A = np.eye(3, dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)

    # Your previous code applied reversed(transform_list) of backward_np.
    # For Affine.backward_np: p' = p @ M.T + t
    for tfm in reversed(transform_list):
        M = np.asarray(tfm.backward_matrix_3x3, dtype=np.float32)
        t = np.asarray(tfm.backward_translation, dtype=np.float32).reshape(3)

        # Compose: (p @ A.T + b) then ( @ M.T + t)
        A = (M @ A).astype(np.float32, copy=False)
        b = (b @ M.T + t).astype(np.float32, copy=False)

    # ------------------------------------------------------------
    # 2) Compute crop bounds using only 8 corners (cheap)
    # ------------------------------------------------------------
    zs = np.array([z0 + 0.5, z1 - 0.5], dtype=np.float32)
    ys = np.array([y0 + 0.5, y1 - 0.5], dtype=np.float32)
    xs = np.array([x0 + 0.5, x1 - 0.5], dtype=np.float32)

    corners = np.array([[z, y, x] for z in zs for y in ys for x in xs], dtype=np.float32)
    corners += origin  # global output coords

    tc = corners @ A.T + b  # transformed into source coords (zyx)
    z_min, y_min, x_min = tc.min(axis=0)
    z_max, y_max, x_max = tc.max(axis=0)

    # Clamp to source bounds and integer crop
    crop_min_z = int(np.floor(max(0.0, z_min)))
    crop_max_z = int(np.ceil(min(float(sv_z), z_max)))
    crop_min_y = int(np.floor(max(0.0, y_min)))
    crop_max_y = int(np.ceil(min(float(sv_y), y_max)))
    crop_min_x = int(np.floor(max(0.0, x_min)))
    crop_max_x = int(np.ceil(min(float(sv_x), x_max)))

    crop_z_len = float(crop_max_z - crop_min_z)
    crop_y_len = float(crop_max_y - crop_min_y)
    crop_x_len = float(crop_max_x - crop_min_x)
    if crop_x_len <= 0 or crop_y_len <= 0 or crop_z_len <= 0:
        raise ValueError("Degenerate crop sizes computed in calculate_sample_field_np.")

    # ------------------------------------------------------------
    # 3) Build transformed coords for all voxels using broadcasting
    #    (z,y,x) centers in output space -> transform -> crop-local
    # ------------------------------------------------------------
    z = (np.arange(z0, z1, dtype=np.float32) + 0.5)[:, None, None]
    y = (np.arange(y0, y1, dtype=np.float32) + 0.5)[None, :, None]
    x = (np.arange(x0, x1, dtype=np.float32) + 0.5)[None, None, :]

    oz = z + origin[0]
    oy = y + origin[1]
    ox = x + origin[2]

    # Apply p' = p @ A.T + b in components:
    # Note: p is (oz,oy,ox) as zyx. A is (3,3) mapping zyx->zyx.
    tz = oz * A[0,0] + oy * A[0,1] + ox * A[0,2] + b[0]
    ty = oz * A[1,0] + oy * A[1,1] + ox * A[1,2] + b[1]
    tx = oz * A[2,0] + oy * A[2,1] + ox * A[2,2] + b[2]

    # Crop-local (still zyx)
    tz = tz - np.float32(crop_min_z)
    ty = ty - np.float32(crop_min_y)
    tx = tx - np.float32(crop_min_x)

    # ------------------------------------------------------------
    # 4) Change basis zyx -> xyz and normalize to [-1,1]
    #    (keep identical behavior expectation for interpolate_np)
    # ------------------------------------------------------------
    # grid_xyz[...,0]=x,1=y,2=z (grid_sample xyz convention)
    # Normalize (same as your current code)
    gx = (tx - (crop_x_len / 2.0)) / (crop_x_len / 2.0)
    gy = (ty - (crop_y_len / 2.0)) / (crop_y_len / 2.0)
    gz = (tz - (crop_z_len / 2.0)) / (crop_z_len / 2.0)

    grid_xyz = np.empty((tz.shape[0], ty.shape[1], tx.shape[2], 3), dtype=np.float32)
    grid_xyz[..., 0] = gx
    grid_xyz[..., 1] = gy
    grid_xyz[..., 2] = gz

    return grid_xyz

# def calculate_sample_field_np(
#     cell_box: tuple[int, int, int, int, int, int],
#     output_volume_origin: tuple[float, float, float],
#     transform_list,  # list of transforms providing backward_np(...)
#     src_vol_shape_zyx: tuple[int, int, int],
# ) -> np.ndarray:
#     """
#     Torch-free version.

#     Returns:
#         grid_xyz_norm: np.ndarray float32, shape (z, y, x, 3)
#         - xyz basis (x,y,z) in last dim (as grid_sample expects)
#         - normalized to [-1, 1]
#     """

#     z0, z1, y0, y1, x0, x1 = cell_box
#     sv_z, sv_y, sv_x = src_vol_shape_zyx

#     # Indices + 0.5
#     z = np.arange(z0, z1, dtype=np.float32) + 0.5
#     y = np.arange(y0, y1, dtype=np.float32) + 0.5
#     x = np.arange(x0, x1, dtype=np.float32) + 0.5

#     # Meshgrid -> (z,y,x)
#     z_grid, y_grid, x_grid = np.meshgrid(z, y, x, indexing="ij")

#     # Stack into zyx coords: (z,y,x,3) with last dim [z,y,x]
#     tile_coords = np.stack([z_grid, y_grid, x_grid], axis=-1).astype(np.float32, copy=False)

#     # Add origin
#     origin = np.asarray(output_volume_origin, dtype=np.float32).reshape(1, 1, 1, 3)
#     tile_coords = tile_coords + origin

#     # Apply inverse transforms in reverse order (NumPy)
#     for tfm in reversed(transform_list):
#         tile_coords = tfm.backward_np(tile_coords)  # must accept (...,3) numpy and return same

#     # AABB of transformed coords (still in zyx)
#     z_min = float(tile_coords[..., 0].min())
#     z_max = float(tile_coords[..., 0].max())
#     y_min = float(tile_coords[..., 1].min())
#     y_max = float(tile_coords[..., 1].max())
#     x_min = float(tile_coords[..., 2].min())
#     x_max = float(tile_coords[..., 2].max())
#     transformed_cell_box = (z_min, z_max, y_min, y_max, x_min, x_max)

#     source_aabb = (0.0, float(sv_z), 0.0, float(sv_y), 0.0, float(sv_x))
#     if check_collision(transformed_cell_box, source_aabb) is False:
#         raise ValueError(
#             "Provided cell_box does not transform into the provided source_volume."
#         )

#     # Clip to bounds then floor/ceil to integer crop box
#     crop_min_z = int(np.floor(max(0.0, z_min)))
#     crop_max_z = int(np.ceil(min(float(sv_z), z_max)))

#     crop_min_y = int(np.floor(max(0.0, y_min)))
#     crop_max_y = int(np.ceil(min(float(sv_y), y_max)))

#     crop_min_x = int(np.floor(max(0.0, x_min)))
#     crop_max_x = int(np.ceil(min(float(sv_x), x_max)))

#     crop_z_len = float(crop_max_z - crop_min_z)
#     crop_y_len = float(crop_max_y - crop_min_y)
#     crop_x_len = float(crop_max_x - crop_min_x)
#     if crop_x_len <= 0 or crop_y_len <= 0 or crop_z_len <= 0:
#         raise ValueError("Degenerate crop sizes computed in calculate_sample_field_np.")

#     # Offset into crop-local coords (still zyx)
#     offset = np.array([crop_min_z, crop_min_y, crop_min_x], dtype=np.float32).reshape(1, 1, 1, 3)
#     tile_coords = tile_coords - offset

#     # Change of basis: zyx -> xyz (grid_sample expects xyz)
#     # Your torch CoB effectively maps [z,y,x] -> [x,y,z]
#     grid_xyz = np.empty_like(tile_coords, dtype=np.float32)
#     grid_xyz[..., 0] = tile_coords[..., 2]  # x
#     grid_xyz[..., 1] = tile_coords[..., 1]  # y
#     grid_xyz[..., 2] = tile_coords[..., 0]  # z

#     # Normalize each dimension to [-1, 1]
#     grid_xyz[..., 0] = (grid_xyz[..., 0] - (crop_x_len / 2.0)) / (crop_x_len / 2.0)
#     grid_xyz[..., 1] = (grid_xyz[..., 1] - (crop_y_len / 2.0)) / (crop_y_len / 2.0)
#     grid_xyz[..., 2] = (grid_xyz[..., 2] - (crop_z_len / 2.0)) / (crop_z_len / 2.0)

#     return grid_xyz

# def calculate_sample_field(cell_box: geometry.AABB,
#                            output_volume_origin: tuple[float, float, float],
#                            transform_list: list[geometry.Transform],
#                            src_vol_shape_zyx: tuple[int, int, int],
#                            device='cuda:0') -> torch.Tensor:
#     """
#     Inverse transforms indices in cell_box boundary into source_vol space
#     and returns index-aligned, boundary-clipped samples in source_vol coordinates.

#     Returned samples are properly normalized and in the pytorch basis,
#     ready to input into grid_sample. Shape: (z, y, x, 3)

#     NOTE on inputs:
#     - cell_box is a AABB of index ranges, not absolute coordinates.

#     - The expected order of transform_list is feedforward.
#     We will apply these matrices in reverse!
#     """

#     z_indices = torch.arange(cell_box[0], cell_box[1], step=1, device=device) + 0.5
#     y_indices = torch.arange(cell_box[2], cell_box[3], step=1, device=device) + 0.5
#     x_indices = torch.arange(cell_box[4], cell_box[5], step=1, device=device) + 0.5

#     z_grid, y_grid, x_grid = torch.meshgrid(z_indices, y_indices, x_indices, indexing="ij")
#     tile_coords = torch.stack([z_grid, y_grid, x_grid], dim=-1)

#     # Define tile coords wrt output vol origin
#     tile_coords = tile_coords + torch.Tensor(output_volume_origin).to(
#         device
#     )

#     # Send tile_coords through inverse transforms
#     # NOTE: tile_transforms list must be iterated thru in reverse
#     # (z, y, x, 3) -> (z, y, x, 3)
#     for tfm in reversed(transform_list):
#         tile_coords = tfm.backward(tile_coords, device=device)

#     # Calculate AABB of transformed coords
#     transformed_cell_box: geometry.AABB = geometry.aabb_3d(
#         tile_coords
#     )
#     z_min, z_max, y_min, y_max, x_min, x_max = transformed_cell_box

#     sv_z, sv_y, sv_x = src_vol_shape_zyx
#     source_aabb: geometry.AABB = (0,
#                                   sv_z,
#                                   0,
#                                   sv_y,
#                                   0,
#                                   sv_x)

#     if check_collision(transformed_cell_box, source_aabb) is False:
#         raise ValueError("""Provided cell_box does not transform
#                          into the provided source_volume.
#                          Please check all inputs.""")

#     # Calculate overlapping region between transformed coords and image boundary
#     # For intervals (A, B):
#     # The lower bound of overlapping region = max(A_min, B_min)
#     # The upper bound of overlapping region = min(A_max, B_max)
#     crop_min_z = torch.max(torch.Tensor([0, z_min]))
#     crop_max_z = torch.min(torch.Tensor([sv_z, z_max]))

#     crop_min_y = torch.max(torch.Tensor([0, y_min]))
#     crop_max_y = torch.min(torch.Tensor([sv_y, y_max]))

#     crop_min_x = torch.max(torch.Tensor([0, x_min]))
#     crop_max_x = torch.min(torch.Tensor([sv_x, x_max]))

#     # Make sure crop_{min, max}_{z, y, x} are integers to be used as indices.
#     # Minimum values are rounded down to nearest integer.
#     # Maximum values are rounded up to nearest integer.
#     crop_min_z = int(torch.floor(crop_min_z))
#     crop_min_y = int(torch.floor(crop_min_y))
#     crop_min_x = int(torch.floor(crop_min_x))

#     crop_max_z = int(torch.ceil(crop_max_z))
#     crop_max_y = int(torch.ceil(crop_max_y))
#     crop_max_x = int(torch.ceil(crop_max_x))

#     # Define tile coords wrt base image crop coordinates
#     image_crop_offset = torch.Tensor(
#         [crop_min_z, crop_min_y, crop_min_x]
#     ).to(device)
#     tile_coords = tile_coords - image_crop_offset

#     # Pytorch flow field follows a different basis than the image numpy basis.
#     # Change of basis to interpolation basis, which preserves relative distances/angles/positions.
#     # (z, y, x, 3) -> (z, y, x, 3)
#     interp_cob_matrix = torch.Tensor(
#         [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
#     )
#     interp_cob = geometry.Affine(interp_cob_matrix)
#     tile_coords = interp_cob.forward(tile_coords, device=device)

#     # Interpolation expects 'grid' parameter/sample locations to be normalized [-1, 1].
#     # Very specific per-dimension normalization according to CoB
#     crop_z_length = crop_max_z - crop_min_z
#     crop_y_length = crop_max_y - crop_min_y
#     crop_x_length = crop_max_x - crop_min_x
#     tile_coords[:, :, :, 0] = (
#         tile_coords[:, :, :, 0] - (crop_x_length / 2)
#     ) / (crop_x_length / 2)
#     tile_coords[:, :, :, 1] = (
#         tile_coords[:, :, :, 1] - (crop_y_length / 2)
#     ) / (crop_y_length / 2)
#     tile_coords[:, :, :, 2] = (
#         tile_coords[:, :, :, 2] - (crop_z_length / 2)
#     ) / (crop_z_length / 2)

#     return tile_coords

# def interpolate(image_crop: torch.Tensor,
#                 sample_field_xyz: torch.Tensor,
#                 device="cuda:0") -> torch.Tensor:
#     """
#     Execute grid_sample and returns 5d tile_sample.

#     Notes:
#     - image_crop shape is (z_img, y_img, x_img)
#     - sample_field shape is (z, y, x, 3)
#     - tile_sample shape is (1, 1, z, y, x)

#     NOTE: sample_field is expected in xyz basis!
#     """

#     # Final reshaping
#     # image_crop: (z_in, y_in, x_in) -> (1, 1, z_in, y_in, x_in)
#     # tile_coords: (z_out, y_out, x_out, 3) -> (1, z_out, y_out, x_out, 3)
#     # => tile_contribution: (1, 1, z_out, y_out, x_out)
#     image_crop = image_crop.unsqueeze(0).unsqueeze(0)
#     sample_field_xyz = sample_field_xyz.unsqueeze(0)

#     # Interpolate
#     image_crop = image_crop.to(device)
#     sample_field_xyz = sample_field_xyz.to(device)
#     tile_sample = torch.nn.functional.grid_sample(
#         image_crop,
#         sample_field_xyz,
#         padding_mode="zeros",
#         mode="nearest",
#         align_corners=False,
#     )

#     return tile_sample

def interpolate_np(
    image_crop_zyx: np.ndarray,
    sample_field_xyz_norm: np.ndarray,
) -> np.ndarray:
    if image_crop_zyx.ndim != 3:
        raise ValueError(f"image_crop_zyx must be (z,y,x), got {image_crop_zyx.shape}")
    if sample_field_xyz_norm.shape[-1] != 3:
        raise ValueError(f"sample_field_xyz_norm last dim must be 3, got {sample_field_xyz_norm.shape}")

    z_in, y_in, x_in = image_crop_zyx.shape
    z_out, y_out, x_out, _ = sample_field_xyz_norm.shape
    n = z_out * y_out * x_out

    # Flatten grids early (views when contiguous)
    g = sample_field_xyz_norm.reshape(-1, 3)
    gx = g[:, 0].astype(np.float32, copy=False)
    gy = g[:, 1].astype(np.float32, copy=False)
    gz = g[:, 2].astype(np.float32, copy=False)

    # align_corners=False: pix = g*(size/2) + (size-1)/2
    # nearest: floor(pix + 0.5)
    xi = np.floor(gx * (x_in * 0.5) + ((x_in - 1) * 0.5) + 0.5).astype(np.int32)
    yi = np.floor(gy * (y_in * 0.5) + ((y_in - 1) * 0.5) + 0.5).astype(np.int32)
    zi = np.floor(gz * (z_in * 0.5) + ((z_in - 1) * 0.5) + 0.5).astype(np.int32)

    valid = (
        (xi >= 0) & (xi < x_in) &
        (yi >= 0) & (yi < y_in) &
        (zi >= 0) & (zi < z_in)
    )

    out_flat = np.zeros(n, dtype=image_crop_zyx.dtype)

    # One linear gather
    img_flat = image_crop_zyx.reshape(-1)
    stride_yx = y_in * x_in
    lin = zi[valid] * stride_yx + yi[valid] * x_in + xi[valid]
    out_flat[valid] = img_flat[lin]

    return out_flat.reshape(1, 1, z_out, y_out, x_out)

# def interpolate_np(
#     image_crop_zyx: np.ndarray,          # (z_in, y_in, x_in)
#     sample_field_xyz_norm: np.ndarray,   # (z_out, y_out, x_out, 3) in xyz, normalized [-1,1]
# ) -> np.ndarray:
#     """
#     Torch-free replacement for torch.grid_sample(..., mode='nearest', padding_mode='zeros', align_corners=False).

#     Returns:
#         tile_sample: np.ndarray float32 (or same dtype policy), shape (1, 1, z_out, y_out, x_out)
#     """

#     if image_crop_zyx.ndim != 3:
#         raise ValueError(f"image_crop_zyx must be (z,y,x), got {image_crop_zyx.shape}")
#     if sample_field_xyz_norm.shape[-1] != 3:
#         raise ValueError(f"sample_field_xyz_norm last dim must be 3, got {sample_field_xyz_norm.shape}")

#     z_in, y_in, x_in = image_crop_zyx.shape
#     z_out, y_out, x_out, _ = sample_field_xyz_norm.shape

#     # Split normalized coords (xyz)
#     gx = sample_field_xyz_norm[..., 0].astype(np.float32, copy=False)
#     gy = sample_field_xyz_norm[..., 1].astype(np.float32, copy=False)
#     gz = sample_field_xyz_norm[..., 2].astype(np.float32, copy=False)

#     # align_corners=False mapping from grid [-1,1] to pixel indices [0, size-1]:
#     # x = ((g + 1) * size - 1) / 2
#     # This is the standard PyTorch mapping for align_corners=False.
#     x = ((gx + 1.0) * x_in - 1.0) * 0.5
#     y = ((gy + 1.0) * y_in - 1.0) * 0.5
#     z = ((gz + 1.0) * z_in - 1.0) * 0.5

#     # nearest mode: round to nearest integer index
#     xi = np.rint(x).astype(np.int32)
#     yi = np.rint(y).astype(np.int32)
#     zi = np.rint(z).astype(np.int32)

#     # output (zeros padding)
#     out = np.zeros((z_out, y_out, x_out), dtype=image_crop_zyx.dtype)

#     valid = (
#         (xi >= 0) & (xi < x_in) &
#         (yi >= 0) & (yi < y_in) &
#         (zi >= 0) & (zi < z_in)
#     )
#     out[valid] = image_crop_zyx[zi[valid], yi[valid], xi[valid]]

#     # match torch output shape: (1,1,z,y,x)
#     return out[np.newaxis, np.newaxis, :, :, :]

def get_overlap_regions(
    tile_layout: list[list[int]],
    tile_aabbs: dict[int, geometry.AABB],
    include_diagonals: bool = False
) -> tuple[dict[int, list[int]], dict[int, geometry.AABB]]:
    """
    Input:
    tile_layout: array of tile ids arranged corresponding to stage coordinates
    tile_aabbs: dict of tile_id -> AABB, defined in fusion initalization.

    Output:
    tile_to_overlap_ids: Maps tile_id to associated overlap region id
    overlaps: Maps overlap_id to actual overlap region AABB

    Access pattern:
    tile_id -> overlap_id -> overlaps
    """

    def _get_overlap_aabb(aabb_1: geometry.AABB, aabb_2: geometry.AABB):
        """
        Utility for finding overlapping regions between tiles and chunks.
        """

        # Check AABB's are colliding, meaning they colllide in all 3 axes
        assert (
            (aabb_1[1] > aabb_2[0] and aabb_1[0] < aabb_2[1])
            and (aabb_1[3] > aabb_2[2] and aabb_1[2] < aabb_2[3])
            and (aabb_1[5] > aabb_2[4] and aabb_1[4] < aabb_2[5])
        ), f"Input AABBs are not colliding: {aabb_1=}, {aabb_2=}"

        # Between two colliding intervals A and B,
        # the overlap interval is the maximum of (A_min, B_min)
        # and the minimum of (A_max, B_max).
        overlap_aabb = (
            np.max([aabb_1[0], aabb_2[0]]),
            np.min([aabb_1[1], aabb_2[1]]),
            np.max([aabb_1[2], aabb_2[2]]),
            np.min([aabb_1[3], aabb_2[3]]),
            np.max([aabb_1[4], aabb_2[4]]),
            np.min([aabb_1[5], aabb_2[5]]),
        )

        return overlap_aabb

    # Output Data Structures
    tile_to_overlap_ids: dict[int, list[int]] = defaultdict(list)
    overlaps: dict[int, geometry.AABB] = {}

    # 1) Find all unique edges
    edges: list[tuple[int, int]] = []
    x_length = len(tile_layout)
    y_length = len(tile_layout[0])
    directions = [
        (-1, 0), (0, -1), (0, 1), (1, 0)
    ]
    if include_diagonals:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    for x in range(x_length):
        for y in range(y_length):
            for dx, dy in directions:
                nx = x + dx
                ny = y + dy
                # Boundary conditions and spacer conditions
                if (
                    0 <= nx
                    and nx < x_length
                    and 0 <= ny
                    and ny < y_length
                    and tile_layout[x][y] != -1
                    and tile_layout[nx][ny] != -1
                ):

                    id_1 = tile_layout[x][y]
                    id_2 = tile_layout[nx][ny]
                    e = tuple(sorted([id_1, id_2]))
                    edges.append(e)
    edges = sorted(list(set(edges)), key=lambda x: (x[0], x[1]))

    # 2) Find overlap regions
    overlap_id = 0
    for id_1, id_2 in edges:
        aabb_1 = tile_aabbs[id_1]
        aabb_2 = tile_aabbs[id_2]

        try:
            o_aabb = _get_overlap_aabb(aabb_1, aabb_2)
        except:  # noqa: E722
            continue

        overlaps[overlap_id] = o_aabb
        tile_to_overlap_ids[id_1].append(overlap_id)
        tile_to_overlap_ids[id_2].append(overlap_id)

        overlap_id += 1

    return tile_to_overlap_ids, overlaps

def parse_yx_tile_layout(xml_path: str, channel: int) -> list[list[int]]:
    """
    Utility for parsing tile layout from a bigstitcher xml
    requested by some blending modules.

    tile_layout follows axis convention:
    +--- +x
    |
    |
    +y

    Tile ids in output tile_layout uses the same tile ids
    defined in the xml file. Spaces denoted with tile id '-1'.
    """

    # --- Load XML ---
    if xml_path.startswith("s3://"):
        # Handle S3 path
        s3 = boto3.client("s3")
        bucket_name, key = xml_path[5:].split("/", 1)
        response = s3.get_object(Bucket=bucket_name, Key=key)
        file_stream = BytesIO(response["Body"].read())
        data = xmltodict.parse(file_stream.read().decode("utf-8"))
    else:
        with open(xml_path, "r") as file:
            data = xmltodict.parse(file.read())

    # --- Get channel tiles from zgroups ---
    channel_tile_ids: list[str] = []

    try:
        xml_key = data["SpimData"]["SequenceDescription"]["ImageLoader"]["ImageLoader"]["zgroups"]["zgroup"]
    except KeyError:
        xml_key = data["SpimData"]["SequenceDescription"]["ImageLoader"]["zgroups"]["zgroup"]

    # xmltodict returns a dict when there is only one zgroup
    if isinstance(xml_key, dict):
        zgroups_iter = [xml_key]
    else:
        zgroups_iter = xml_key

    for zgroup in zgroups_iter:
        tile_id = zgroup["@setup"]

        try:
            tile_name = zgroup["path"]
        except KeyError:
            tile_name = zgroup["@path"]

        match = re.search(r"ch_(\d+)", tile_name)
        if not match:
            continue
        ch = int(match.group(1))

        if ch == channel:
            channel_tile_ids.append(tile_id)

    # --- Get channel tile stage positions (Translation to Nominal Grid) ---
    stage_positions_xyz: dict[str, tuple[float, float, float]] = {}

    view_regs = data["SpimData"]["ViewRegistrations"]["ViewRegistration"]
    if isinstance(view_regs, dict):
        view_regs_iter = [view_regs]
    else:
        view_regs_iter = view_regs

    for d in view_regs_iter:
        tile_id = d["@setup"]
        if tile_id not in channel_tile_ids:
            continue

        view_transform = d["ViewTransform"]
        # Use the last transform (usually "Translation to Nominal Grid")
        if isinstance(view_transform, list):
            view_transform = view_transform[-1]

        nums = [float(val) for val in view_transform["affine"].split(" ")]
        # (x, y, z) from affine translation terms
        stage_positions_xyz[tile_id] = tuple(nums[3::4])

    if not stage_positions_xyz:
        return []

    # --- Quantize / cluster X and Y to avoid tiny float drift ---
    # Tile spacing is ~10^4 microns, drift is ~10^-3, so rounding to 1 decimal is safe.
    ROUND_DECIMALS = 1

    positions = list(stage_positions_xyz.values())
    x_pos_quant = sorted(
        set(round(pos[0], ROUND_DECIMALS) for pos in positions)
    )
    y_pos_quant = sorted(
        set(round(pos[1], ROUND_DECIMALS) for pos in positions)
    )

    # Initialize layout with -1
    tile_layout = np.ones((len(y_pos_quant), len(x_pos_quant)), dtype=int) * -1

    # Fill layout by quantized rank of positions
    for tile_id, (x, y, _z) in stage_positions_xyz.items():
        qx = round(x, ROUND_DECIMALS)
        qy = round(y, ROUND_DECIMALS)

        try:
            ix = x_pos_quant.index(qx)
            iy = y_pos_quant.index(qy)
        except ValueError:
            # Shouldn't happen because x_pos_quant / y_pos_quant are built from these values
            print(
                f"[parse_yx_tile_layout] Could not find indices for "
                f"tile_id={tile_id}, pos=({x}, {y}) -> quant=({qx}, {qy})"
            )
            continue

        tile_layout[iy, ix] = int(tile_id)

    return tile_layout.tolist()

# Version 2 but not working for exaSPIM
# def parse_yx_tile_layout(xml_path: str, channel: int) -> list[list[int]]:
#     """
#     Utility for parsing tile layout from a bigstitcher xml
#     requested by some blending modules.

#     tile_layout follows axis convention:
#     +--- +x
#     |
#     |
#     +y

#     Tile ids in output tile_layout uses the same tile ids
#     defined in the xml file. Spaces denoted with tile id '-1'.
#     """

#     if xml_path.startswith("s3://"):
#         # Handle S3 path
#         s3 = boto3.client("s3")
#         bucket_name, key = xml_path[5:].split("/", 1)
#         response = s3.get_object(Bucket=bucket_name, Key=key)
#         file_stream = BytesIO(response["Body"].read())
#         data = xmltodict.parse(file_stream.read().decode("utf-8"))
#     else:
#         with open(xml_path, "r") as file:
#             data = xmltodict.parse(file.read())

#     # Get channel tiles
#     channel_tile_ids: list[int] = []

#     try:
#         xml_key = data["SpimData"]["SequenceDescription"]["ImageLoader"]["ImageLoader"]["zgroups"]["zgroup"]
#     except KeyError:
#         xml_key = data["SpimData"]["SequenceDescription"]["ImageLoader"]["zgroups"]["zgroup"]

#     for zgroup in xml_key:
#         tile_id = zgroup["@setup"]

#         try:
#             tile_name = zgroup["path"]
#         except KeyError:
#             tile_name = zgroup["@path"]

#         match = re.search(r"ch_(\d+)", tile_name)
#         ch = int(match.group(1))

#         if ch == channel:
#             channel_tile_ids.append(tile_id)

#     # Get channel tile stage positions
#     stage_positions_xyz: dict[int, tuple[float, float, float]] = {}
#     for d in data["SpimData"]["ViewRegistrations"]["ViewRegistration"]:
#         tile_id = d["@setup"]
#         if tile_id in channel_tile_ids:
#             view_transform = d["ViewTransform"]
#             if isinstance(view_transform, list):
#                 view_transform = view_transform[-1]

#             nums = [float(val) for val in view_transform["affine"].split(" ")]
#             # (x, y, z) from affine
#             stage_positions_xyz[tile_id] = tuple(nums[3::4])

#     if not stage_positions_xyz:
#         return []

#     # Collect and sort unique x and y positions
#     positions_arr_xyz = np.array(list(stage_positions_xyz.values()))
#     x_pos = sorted(set(positions_arr_xyz[:, 0]))
#     y_pos = sorted(set(positions_arr_xyz[:, 1]))

#     # Initialize layout with -1
#     tile_layout = np.ones((len(y_pos), len(x_pos)), dtype=float) * -1

#     # Fill layout by rank of positions (no delta-based scaling)
#     for tile_id, (x, y, _z) in stage_positions_xyz.items():
#         try:
#             ix = x_pos.index(x)
#             iy = y_pos.index(y)
#         except ValueError:
#             # Shouldn't happen because x_pos/y_pos are built from these values
#             print(
#                 f"[parse_yx_tile_layout] Could not find indices for "
#                 f"tile_id={tile_id}, pos=({x}, {y})"
#             )
#             continue

#         tile_layout[iy, ix] = int(tile_id)

#     return tile_layout.astype(int).tolist()


# Original
# def parse_yx_tile_layout(xml_path: str, channel: int) -> list[list[int]]:
#     """
#     Utility for parsing tile layout from a bigstitcher xml
#     requested by some blending modules.

#     tile_layout follows axis convention:
#     +--- +x
#     |
#     |
#     +y

#     Tile ids in output tile_layout uses the same tile ids
#     defined in the xml file. Spaces denoted with tile id '-1'.
#     """

#     if xml_path.startswith('s3://'):
#         # Handle S3 path
#         s3 = boto3.client('s3')
#         bucket_name, key = xml_path[5:].split('/', 1)
#         response = s3.get_object(Bucket=bucket_name, Key=key)
#         file_stream = BytesIO(response['Body'].read())
#         data = xmltodict.parse(file_stream.read().decode('utf-8'))
#     else:
#         with open(xml_path, "r") as file:
#             data = xmltodict.parse(file.read())

#     # Get channel tiles    
#     channel_tile_ids: list[int] = [] 
#     for zgroup in data["SpimData"]["SequenceDescription"]["ImageLoader"][
#         "zgroups"
#     ]["zgroup"]:
#         tile_id = zgroup["@setup"]
#         # tile_name = zgroup["@path"]
#         tile_name = zgroup["path"]

#         match = re.search(r'ch_(\d+)', tile_name)
#         ch = int(match.group(1))

#         if ch == channel:
#             channel_tile_ids.append(tile_id)
    
#     # Get channel tile stage positions
#     stage_positions_xyz: dict[int, tuple[float, float, float]] = {}
#     for d in data["SpimData"]["ViewRegistrations"]["ViewRegistration"]:
#         tile_id = d["@setup"]
#         if tile_id in channel_tile_ids:
#             view_transform = d["ViewTransform"]
#             if isinstance(view_transform, list):
#                 view_transform = view_transform[-1]

#             nums = [float(val) for val in view_transform["affine"].split(" ")]
#             stage_positions_xyz[tile_id] = tuple(nums[3::4])

#     # Calculate delta_x and delta_y
#     positions_arr_xyz = np.array([pos for pos in stage_positions_xyz.values()])
#     x_pos = list(set(positions_arr_xyz[:, 0]))
#     x_pos = sorted(x_pos)
#     delta_x = x_pos[1] - x_pos[0]
#     y_pos = list(set(positions_arr_xyz[:, 1]))
#     y_pos = sorted(y_pos)
#     delta_y = y_pos[1] - y_pos[0]

#     M = np.array([[(1. / delta_x), 0.],
#                   [0., (1. / delta_y)]])
#     tile_layout = np.ones((len(y_pos), len(x_pos))) * -1
#     for tile_id, s_pos in stage_positions_xyz.items():
#         index = M @ np.array([s_pos[0] - x_pos[0],
#                               s_pos[1] - y_pos[0]])

#         # Round to nearest integer
#         iy = int(round(index[1]))
#         ix = int(round(index[0]))

#         tile_layout[iy, ix] = tile_id

#     tile_layout = tile_layout.astype(int)

#     return tile_layout