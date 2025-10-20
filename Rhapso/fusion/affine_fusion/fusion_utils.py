from collections import defaultdict
import re
import numpy as np
import torch
import xmltodict
from . import geometry as geometry
import boto3
from io import BytesIO
import xmltodict
import numpy as np

from botocore.client import Config
from botocore import UNSIGNED

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

def calculate_image_crop(cell_box: geometry.AABB,
                         output_volume_origin: tuple[float, float, float],
                         transform_list: list[geometry.Transform],
                         src_vol_shape_zyx: tuple[int, int, int],
                         device='cpu') -> tuple[slice]:
    """
    Inverse transforms cell_box boundary into source_vol space
    and returns an index-aligned, boundary-clipped crop in source_vol coordinates.
    Outputs a 5d-tuple of slices.

    IMPORTANT NOTE ON INPUTS:
    - cell_box is a AABB of index ranges, not absolute coordinates.

    - The expected order of transform_list is feedforward.
    We will apply these matrices in reverse!
    """

    # Derive cell_box extrema
    z_min, z_max, y_min, y_max, x_min, x_max = cell_box
    z_grid, y_grid, x_grid = torch.meshgrid(
        torch.Tensor([z_min + 0.5, z_max - 0.5]),
        torch.Tensor([y_min + 0.5, y_max - 0.5]),
        torch.Tensor([x_min + 0.5, x_max - 0.5]),
        indexing='ij'
    )
    cell_box_pts = torch.stack([z_grid, y_grid, x_grid], dim=-1)

    # Apply inverse transform
    cell_box_pts = cell_box_pts + torch.Tensor(output_volume_origin).to(
        device
    )

    for tfm in reversed(transform_list):
        cell_box_pts = tfm.backward(cell_box_pts, device=device)

    # Derive axis-aligned, boundary-clipped image
    cell_box_src: geometry.AABB = geometry.aabb_3d(cell_box_pts)
    z_min, z_max, y_min, y_max, x_min, x_max = cell_box_src

    sv_z, sv_y, sv_x = src_vol_shape_zyx
    aabb_src: geometry.AABB = (0, sv_z, 0, sv_y, 0, sv_x)

    # Validation
    if check_collision(cell_box_src, aabb_src) is False:
        raise ValueError(f"""Provided cell_box does not transform
                         into the provided source_volume.
                         Please check all inputs.""")

    # Calculate overlapping region between transformed coords and image boundary
    # For intervals (A, B):
    # The lower bound of overlapping region = max(A_min, B_min)
    # The upper bound of overlapping region = min(A_max, B_max)
    crop_min_z = torch.max(torch.Tensor([0, z_min]))
    crop_max_z = torch.min(torch.Tensor([sv_z, z_max]))

    crop_min_y = torch.max(torch.Tensor([0, y_min]))
    crop_max_y = torch.min(torch.Tensor([sv_y, y_max]))

    crop_min_x = torch.max(torch.Tensor([0, x_min]))
    crop_max_x = torch.min(torch.Tensor([sv_x, x_max]))

    # Round to integers for indexing
    # Minimum values are rounded down to nearest integer.
    # Maximum values are rounded up to nearest integer.
    crop_min_z = int(torch.floor(crop_min_z))
    crop_min_y = int(torch.floor(crop_min_y))
    crop_min_x = int(torch.floor(crop_min_x))

    crop_max_z = int(torch.ceil(crop_max_z))
    crop_max_y = int(torch.ceil(crop_max_y))
    crop_max_x = int(torch.ceil(crop_max_x))

    image_crop_slice = (
        0,
        0,
        slice(crop_min_z, crop_max_z),
        slice(crop_min_y, crop_max_y),
        slice(crop_min_x, crop_max_x),
    )

    return image_crop_slice

def calculate_sample_field(cell_box: geometry.AABB,
                           output_volume_origin: tuple[float, float, float],
                           transform_list: list[geometry.Transform],
                           src_vol_shape_zyx: tuple[int, int, int],
                           device='cuda:0') -> torch.Tensor:
    """
    Inverse transforms indices in cell_box boundary into source_vol space
    and returns index-aligned, boundary-clipped samples in source_vol coordinates.

    Returned samples are properly normalized and in the pytorch basis,
    ready to input into grid_sample. Shape: (z, y, x, 3)

    NOTE on inputs:
    - cell_box is a AABB of index ranges, not absolute coordinates.

    - The expected order of transform_list is feedforward.
    We will apply these matrices in reverse!
    """

    z_indices = torch.arange(cell_box[0], cell_box[1], step=1, device=device) + 0.5
    y_indices = torch.arange(cell_box[2], cell_box[3], step=1, device=device) + 0.5
    x_indices = torch.arange(cell_box[4], cell_box[5], step=1, device=device) + 0.5

    z_grid, y_grid, x_grid = torch.meshgrid(z_indices, y_indices, x_indices, indexing="ij")
    tile_coords = torch.stack([z_grid, y_grid, x_grid], dim=-1)

    # Define tile coords wrt output vol origin
    tile_coords = tile_coords + torch.Tensor(output_volume_origin).to(
        device
    )

    # Send tile_coords through inverse transforms
    # NOTE: tile_transforms list must be iterated thru in reverse
    # (z, y, x, 3) -> (z, y, x, 3)
    for tfm in reversed(transform_list):
        tile_coords = tfm.backward(tile_coords, device=device)

    # Calculate AABB of transformed coords
    transformed_cell_box: geometry.AABB = geometry.aabb_3d(
        tile_coords
    )
    z_min, z_max, y_min, y_max, x_min, x_max = transformed_cell_box

    sv_z, sv_y, sv_x = src_vol_shape_zyx
    source_aabb: geometry.AABB = (0,
                                  sv_z,
                                  0,
                                  sv_y,
                                  0,
                                  sv_x)

    if check_collision(transformed_cell_box, source_aabb) is False:
        raise ValueError("""Provided cell_box does not transform
                         into the provided source_volume.
                         Please check all inputs.""")

    # Calculate overlapping region between transformed coords and image boundary
    # For intervals (A, B):
    # The lower bound of overlapping region = max(A_min, B_min)
    # The upper bound of overlapping region = min(A_max, B_max)
    crop_min_z = torch.max(torch.Tensor([0, z_min]))
    crop_max_z = torch.min(torch.Tensor([sv_z, z_max]))

    crop_min_y = torch.max(torch.Tensor([0, y_min]))
    crop_max_y = torch.min(torch.Tensor([sv_y, y_max]))

    crop_min_x = torch.max(torch.Tensor([0, x_min]))
    crop_max_x = torch.min(torch.Tensor([sv_x, x_max]))

    # Make sure crop_{min, max}_{z, y, x} are integers to be used as indices.
    # Minimum values are rounded down to nearest integer.
    # Maximum values are rounded up to nearest integer.
    crop_min_z = int(torch.floor(crop_min_z))
    crop_min_y = int(torch.floor(crop_min_y))
    crop_min_x = int(torch.floor(crop_min_x))

    crop_max_z = int(torch.ceil(crop_max_z))
    crop_max_y = int(torch.ceil(crop_max_y))
    crop_max_x = int(torch.ceil(crop_max_x))

    # Define tile coords wrt base image crop coordinates
    image_crop_offset = torch.Tensor(
        [crop_min_z, crop_min_y, crop_min_x]
    ).to(device)
    tile_coords = tile_coords - image_crop_offset

    # Pytorch flow field follows a different basis than the image numpy basis.
    # Change of basis to interpolation basis, which preserves relative distances/angles/positions.
    # (z, y, x, 3) -> (z, y, x, 3)
    interp_cob_matrix = torch.Tensor(
        [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    )
    interp_cob = geometry.Affine(interp_cob_matrix)
    tile_coords = interp_cob.forward(tile_coords, device=device)

    # Interpolation expects 'grid' parameter/sample locations to be normalized [-1, 1].
    # Very specific per-dimension normalization according to CoB
    crop_z_length = crop_max_z - crop_min_z
    crop_y_length = crop_max_y - crop_min_y
    crop_x_length = crop_max_x - crop_min_x
    tile_coords[:, :, :, 0] = (
        tile_coords[:, :, :, 0] - (crop_x_length / 2)
    ) / (crop_x_length / 2)
    tile_coords[:, :, :, 1] = (
        tile_coords[:, :, :, 1] - (crop_y_length / 2)
    ) / (crop_y_length / 2)
    tile_coords[:, :, :, 2] = (
        tile_coords[:, :, :, 2] - (crop_z_length / 2)
    ) / (crop_z_length / 2)

    return tile_coords

def interpolate(image_crop: torch.Tensor,
                sample_field_xyz: torch.Tensor,
                device="cuda:0") -> torch.Tensor:
    """
    Execute grid_sample and returns 5d tile_sample.

    Notes:
    - image_crop shape is (z_img, y_img, x_img)
    - sample_field shape is (z, y, x, 3)
    - tile_sample shape is (1, 1, z, y, x)

    NOTE: sample_field is expected in xyz basis!
    """

    # Final reshaping
    # image_crop: (z_in, y_in, x_in) -> (1, 1, z_in, y_in, x_in)
    # tile_coords: (z_out, y_out, x_out, 3) -> (1, z_out, y_out, x_out, 3)
    # => tile_contribution: (1, 1, z_out, y_out, x_out)
    image_crop = image_crop.unsqueeze(0).unsqueeze(0)
    sample_field_xyz = sample_field_xyz.unsqueeze(0)

    # Interpolate
    image_crop = image_crop.to(device)
    sample_field_xyz = sample_field_xyz.to(device)
    tile_sample = torch.nn.functional.grid_sample(
        image_crop,
        sample_field_xyz,
        padding_mode="zeros",
        mode="nearest",
        align_corners=False,
    )

    return tile_sample

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
    Utility for parsing tile layout from a BigStitcher XML.

    tile_layout follows axis convention:
        +--- +x
        |
        |
        +y

    Tile ids in output tile_layout use the same tile ids defined in the XML.
    Spaces are denoted with tile id '-1'.
    """
    # --- load XML (S3 or local) ---
    if xml_path.startswith('s3://'):
        s3 = boto3.client('s3')
        # s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        bucket_name, key = xml_path[5:].split('/', 1)
        response = s3.get_object(Bucket=bucket_name, Key=key)
        file_stream = BytesIO(response['Body'].read())
        data = xmltodict.parse(file_stream.read().decode('utf-8'))
    else:
        with open(xml_path, "r") as file:
            data = xmltodict.parse(file.read())

    # --- gather tile ids for this channel ---
    channel_tile_ids: list[str] = []
    for zgroup in data["SpimData"]["SequenceDescription"]["ImageLoader"]["zgroups"]["zgroup"]:
        tile_id = zgroup["@setup"]  # NOTE: XML ids are strings
        tile_name = zgroup["path"]
        match = re.search(r'ch_(\d+)', tile_name)
        ch = int(match.group(1))
        if ch == channel:
            channel_tile_ids.append(tile_id)

    # --- collect stage positions for those tiles ---
    stage_positions_xyz: dict[str, tuple[float, float, float]] = {}
    for d in data["SpimData"]["ViewRegistrations"]["ViewRegistration"]:
        tile_id = d["@setup"]
        if tile_id in channel_tile_ids:
            view_transform = d["ViewTransform"]
            if isinstance(view_transform, list):
                view_transform = view_transform[-1]
            nums = [float(val) for val in view_transform["affine"].split()]
            # affine is 3x4 row-major; take translations at indices 3, 7, 11
            stage_positions_xyz[tile_id] = (nums[3], nums[7], nums[11])

    # --- build robust grid (collapse float jitter; no division by tiny deltas) ---
    positions_arr_xyz = np.array([pos for pos in stage_positions_xyz.values()]) if stage_positions_xyz else np.zeros((0, 3))
    x_raw = np.sort(positions_arr_xyz[:, 0]) if positions_arr_xyz.size else np.array([])
    y_raw = np.sort(positions_arr_xyz[:, 1]) if positions_arr_xyz.size else np.array([])

    # collapse near-duplicates by rounding (adjust decimals if your jitter differs)
    _DEC = 2
    x_unique = np.unique(np.round(x_raw, _DEC)) if x_raw.size else np.array([])
    y_unique = np.unique(np.round(y_raw, _DEC)) if y_raw.size else np.array([])

    # grid size from unique coords (avoids inf when diffs are ~0)
    W = int(x_unique.size)
    H = int(y_unique.size)
    tile_layout = np.full((H, W), -1, dtype=int)

    if H == 0 or W == 0:
        return tile_layout  # nothing to place

    # --- place tiles by nearest unique coordinate (no M, no 1/dx) ---
    for tile_id, (sx, sy, _sz) in stage_positions_xyz.items():
        sxr = round(sx, _DEC)
        syr = round(sy, _DEC)

        # X index via searchsorted, then pick the closest neighbor
        ix = int(np.searchsorted(x_unique, sxr, side="left"))
        if ix == 0:
            ix = 0
        elif ix >= W:
            ix = W - 1
        else:
            # choose nearer of left/right
            left = x_unique[ix - 1]
            right = x_unique[ix]
            ix = ix - 1 if abs(sx - left) <= abs(right - sx) else ix

        # Y index via searchsorted, then pick the closest neighbor
        iy = int(np.searchsorted(y_unique, syr, side="left"))
        if iy == 0:
            iy = 0
        elif iy >= H:
            iy = H - 1
        else:
            left = y_unique[iy - 1]
            right = y_unique[iy]
            iy = iy - 1 if abs(sy - left) <= abs(right - sy) else iy

        tile_layout[iy, ix] = int(tile_id)  # keep your int ids

    return tile_layout

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