import zarr
import numpy as np
from bioio import BioImage
import bioio_tifffile
import dask.array as da
import s3fs

"""
Image Reader loads and downsamples Zarr and TIFF OME data
"""


def _per_axis_pyramid_ds_xyz(file_path: str, level: int):
    """Return (ds_x, ds_y, ds_z) downsample factors for ``level`` vs L0.

    Source of truth: OME-zarr v0.4 ``coordinateTransformations.scale``
    metadata in the parent group's ``.zattrs`` — i.e. the pyramid
    writer's declared per-axis sampling-density ratio. This is the
    correct primitive (the metadata explicitly encodes whatever
    anisotropy the pyramid has) and avoids integer-flooring slack from
    array-shape ratios (e.g. dataset A L0_z=220 / L4_z=13 = 16.92,
    while the metadata correctly says scale_z(L4)/scale_z(L0) = 16.0).

    Returns ``(None, None, None)`` when the metadata cannot be read —
    caller is expected to fall back to legacy ``2 ** level`` behavior.

    ``file_path`` is the full path to the level-N array
    (e.g. ``s3://…/channel_488.zarr/2``); the parent group is the
    OME-zarr root carrying the multiscales metadata.
    """
    try:
        root_path = file_path.rstrip('/').rsplit('/', 1)[0]
        root = zarr.open(root_path, mode='r')
        scale_l0 = _ome_zarr_scale_zyx(root, "0")
        scale_ln = _ome_zarr_scale_zyx(root, str(level))
    except Exception:
        return None, None, None
    if scale_l0 is None or scale_ln is None:
        return None, None, None
    # Per-axis ds = scale(L) / scale(L0). Round to int (≥ 1) since the
    # caller ultimately uses these as integer divisors for voxel bounds.
    sz0, sy0, sx0 = scale_l0
    szn, syn, sxn = scale_ln
    ds_z = max(1, int(round(szn / max(sz0, 1e-12))))
    ds_y = max(1, int(round(syn / max(sy0, 1e-12))))
    ds_x = max(1, int(round(sxn / max(sx0, 1e-12))))
    return ds_x, ds_y, ds_z


def _ome_zarr_scale_zyx(root_group, level_name: str):
    """Return ``(scale_z, scale_y, scale_x)`` from OME-zarr multiscales.

    Reads ``coordinateTransformations[type==scale]`` for the given
    level path. Slices the trailing ZYX entries from a 3- or 5-axis
    declaration. Returns ``None`` if the metadata is missing or
    malformed — caller should treat that as "metadata unreadable" and
    fall back to a legacy heuristic.
    """
    try:
        attrs = root_group.attrs.asdict()
        multiscales = attrs.get("multiscales", [])
        if not multiscales:
            return None
        for d in multiscales[0].get("datasets", []):
            if str(d.get("path")) != str(level_name):
                continue
            for ct in d.get("coordinateTransformations", []):
                if ct.get("type") == "scale":
                    s = ct.get("scale", [])
                    if len(s) == 5:
                        return float(s[2]), float(s[3]), float(s[4])
                    if len(s) == 3:
                        return float(s[0]), float(s[1]), float(s[2])
                    return None
    except Exception:
        return None
    return None

class CustomBioImage(BioImage):
    def standard_metadata(self):
        pass
    
    def scale(self):
        pass
    
    def time_interval(self):
        pass

class ImageReader:
    def __init__(self, file_type):
        self.file_type = file_type

    def downsample(self, arr, axis):
        """
        Reduce size by 2 along `axis` by averaging adjacent elements
        """
        s0 = [slice(None)] * arr.ndim
        s1 = [slice(None)] * arr.ndim
        s0[axis] = slice(0, None, 2)
        s1[axis] = slice(1, None, 2)

        a0 = arr[tuple(s0)]
        a1 = arr[tuple(s1)]

        len1 = a1.shape[axis]
        s0c = [slice(None)] * a0.ndim
        s0c[axis] = slice(0, len1)
        a0 = a0[tuple(s0c)]

        return (a0 + a1) * 0.5

    def interface_downsampling(self, data, dsxy, dsz):
        """
        Downsample a 3D volume by powers of two by repeatedly halving along each axis
        """
        # Process X dimension
        f = dsxy
        while f > 1:
            data = self.downsample(data, axis=0)  
            f //= 2
        
        # Process Y dimension
        f = dsxy
        while f > 1:
            data = self.downsample(data, axis=1)  
            f //= 2
        
        # Process Z dimension
        f = dsz
        while f > 1:
            data = self.downsample(data, axis=2)  
            f //= 2
        
        return data

    def fetch_image_data(self, record, dsxy, dsz):
        """
        Loads image chunk, downsamples it, and sub_chunks based on predefined intervals.
        """
        view_id = record['view_id']
        file_path = record['file_path']
        interval_key = record['interval_key']
        offset = record['offset']
        lower_bound = record['lb']
        
        # Create image pathways using Dask
        if self.file_type == "tiff":
            img = CustomBioImage(file_path, reader=bioio_tifffile.Reader)
            dask_array = img.get_dask_stack()[0, 0, 0, :, :, :]
        
        elif self.file_type == "zarr":
            full_path = f"{file_path}"
            is_local = not full_path.startswith("s3://")
            try:
                if is_local:
                    zarr_array = zarr.open(full_path, mode='r')
                else:
                    s3 = s3fs.S3FileSystem(anon=False)
                    store = s3fs.S3Map(root=full_path, s3=s3)
                    zarr_array = zarr.open(store, mode='r')
                if zarr_array.ndim == 5:
                    dask_array = da.from_zarr(zarr_array)[0, 0, :, :, :]
                elif zarr_array.ndim == 3:
                    dask_array = da.from_zarr(zarr_array)
                else:
                    raise ValueError(f"Expected 3D or 5D zarr, got {zarr_array.ndim}D with shape {zarr_array.shape}")
            except Exception as e:
                print(f"[ImageReader] ERROR opening zarr at {full_path}: {e}")
                # Try to inspect root to show available multiscales
                try:
                    root_path = full_path.rsplit('/', 1)[0]
                    print(f"[ImageReader] Attempting to inspect root zarr at: {root_path}")
                    if is_local:
                        root_zarr = zarr.open(root_path, mode='r')
                    else:
                        root_store = s3fs.S3Map(root=root_path, s3=s3)
                        root_zarr = zarr.open(root_store, mode='r')
                    available_levels = list(root_zarr.keys()) if hasattr(root_zarr, 'keys') else 'unknown'
                    print(f"[ImageReader] Available multiscale levels at root: {available_levels}")
                except Exception as e2:
                    print(f"[ImageReader] Could not inspect root zarr: {e2}")
                raise

        dask_array = dask_array.astype(np.float32)
        dask_array = dask_array.transpose()

        # Store original crop bounds (in level-0 coordinates) for later application
        crop_min = record.get('crop_min')
        crop_max = record.get('crop_max')

        # Downsample Dask array
        downsampled_stack = self.interface_downsampling(dask_array, dsxy, dsz)

        # Get lower and upper bounds
        lb = list(interval_key[0])
        ub = list(interval_key[1])

        # Bounds are in full-resolution (level 0) coordinates.
        # We loaded from a potentially downsampled multiscale level,
        # so we need to scale the bounds down to that level's voxel
        # space. Anisotropic-pyramid-safe: compute per-axis ds from
        # actual ``shape(L0) / shape(level)`` rather than the legacy
        # isotropic ``2 ** level`` (which broke on pyramids that
        # preserve Z at coarse levels — e.g. HCR_823476_s5 keeps Z
        # full-res at L1/L2 while halving XY, causing the legacy code
        # to read only the top quarter of Z; see
        # new_reports/11_ANISOTROPIC_PYRAMID_BUG.md).
        #
        # ``lb``/``ub`` are XYZ-ordered, but zarr ``shape`` is ZYX —
        # the indexing below is explicit on that axis swap.
        try:
            level_str = file_path.rstrip('/').split('/')[-1]
            level = int(level_str)
            print(f"[ImageReader] file_path={file_path}, extracted level={level}")
            print(f"[ImageReader] Before scaling: lb={lb}, ub={ub}, downsampled_stack.shape={downsampled_stack.shape}")
            if level > 0:
                ds_x, ds_y, ds_z = _per_axis_pyramid_ds_xyz(file_path, level)
                if ds_x is not None:
                    lb = [lb[0] // ds_x, lb[1] // ds_y, lb[2] // ds_z]
                    ub = [ub[0] // ds_x, ub[1] // ds_y, ub[2] // ds_z]
                    print(
                        f"[ImageReader] After per-axis scaling "
                        f"(ds_xyz=({ds_x},{ds_y},{ds_z})): lb={lb}, ub={ub}"
                    )
                else:
                    # Fallback: legacy isotropic behavior (e.g. when the
                    # parent pyramid metadata isn't accessible).
                    scale = 2 ** level
                    lb = [x // scale for x in lb]
                    ub = [x // scale for x in ub]
                    print(
                        f"[ImageReader] After scaling by 2^{level}={scale} "
                        f"(fallback, parent pyramid not readable): "
                        f"lb={lb}, ub={ub}"
                    )
        except (ValueError, IndexError) as e:
            print(f"[ImageReader] Level extraction failed ({e}); using bounds as-is")
            pass  # Level extraction failed; use bounds as-is

        # Now apply split tile crop if present (using scaled crop bounds)
        if crop_min is not None and crop_max is not None:
            if len(crop_min) != 3 or len(crop_max) != 3:
                raise ValueError(
                    f"crop_min and crop_max must both be length 3 for 3D cropping; "
                    f"got crop_min={crop_min}, crop_max={crop_max}"
                )

            # Scale crop bounds from level-0 coordinates to downsampled array
            # coordinates. The array has been downsampled by the pyramid
            # level (anisotropic-safe per-axis ds — see comment above)
            # AND by dsxy/dsz (interface_downsampling), so crop bounds
            # must be divided by the total per-axis factor.
            try:
                level_str = file_path.rstrip('/').split('/')[-1]
                level = int(level_str)
                if level > 0:
                    ds_x_p, ds_y_p, ds_z_p = _per_axis_pyramid_ds_xyz(
                        file_path, level
                    )
                    if ds_x_p is None:
                        # Same legacy fallback as the lb/ub block above.
                        ds_x_p = ds_y_p = ds_z_p = 2 ** level
                else:
                    ds_x_p = ds_y_p = ds_z_p = 1
                total_scale_x = ds_x_p * dsxy
                total_scale_y = ds_y_p * dsxy
                total_scale_z = ds_z_p * dsz
            except (ValueError, IndexError):
                total_scale_x = total_scale_y = dsxy
                total_scale_z = dsz

            # crop bounds are in XYZ order: [0]=X, [1]=Y, [2]=Z.
            scales = [total_scale_x, total_scale_y, total_scale_z]
            crop_min_scaled = [int(x // s) for x, s in zip(crop_min, scales)]
            crop_max_scaled = [int(np.ceil((x + 1) / s) - 1) for x, s in zip(crop_max, scales)]

            # Validate and clamp crop bounds to downsampled array dimensions
            array_shape = downsampled_stack.shape
            for i in range(3):
                if crop_min_scaled[i] < 0:
                    raise ValueError(
                        f"crop_min_scaled[{i}]={crop_min_scaled[i]} is negative"
                    )
                # Clamp crop_max to valid range
                crop_max_scaled[i] = min(crop_max_scaled[i], array_shape[i] - 1)
                if crop_min_scaled[i] > crop_max_scaled[i]:
                    raise ValueError(
                        f"crop_min_scaled[{i}]={crop_min_scaled[i]} > crop_max_scaled[{i}]={crop_max_scaled[i]}"
                    )

            print(f"[ImageReader] Applying crop: crop_min_scaled={crop_min_scaled}, crop_max_scaled={crop_max_scaled}")
            downsampled_stack = downsampled_stack[
                crop_min_scaled[0]:crop_max_scaled[0] + 1,
                crop_min_scaled[1]:crop_max_scaled[1] + 1,
                crop_min_scaled[2]:crop_max_scaled[2] + 1
            ]
            # After cropping, detected peaks land at coords relative to
            # the CROPPED chunk's origin (0..chunk_size). The subsequent
            # ``DoG.apply_lower_bounds(peaks, lb)`` step adds ``lb``
            # elementwise — that addition happens BEFORE
            # ``upsample_coordinates`` scales up to L0, so ``lb`` must
            # be in the same array-voxel unit system as the peaks.
            # Without this correction, ``lower_bound`` is (0,0,0) in
            # split-tile mode (it comes from the tile-local
            # ``_split_tile_shape`` bounds which always start at zero),
            # so every tile's peaks re-origin to (0,0,0) and all tiles'
            # IPs collapse into the global frame's top-left corner —
            # visible in ``03-tile_edge_filter/moving.png`` as IPs
            # clustered in the 0..tile_size region regardless of which
            # grid cell the tile is supposed to cover. Add
            # ``crop_min_scaled`` so the crop offset propagates into
            # the peak coordinate transform.
            #
            # COORDINATE-FRAME CONTRACT (split-tile IPs):
            # The stored N5 IPs produced downstream of this adjustment
            # are in L0 WORLD voxel coords (tile grid position baked
            # in). Matching MUST skip the "Image Splitting"
            # ViewTransform when composing per-view transforms — see
            # ``Rhapso/matching/load_and_transform_points.py``
            # (``SPLIT_TILE_TRANSFORM_NAME``). Applying it again would
            # double-translate each split tile's IPs and produce a
            # residual gradient of k × tile_step across the grid.
            lower_bound = [
                int(lower_bound[i]) + int(crop_min_scaled[i])
                for i in range(3)
            ]
            print(f"[ImageReader] crop offset applied → lower_bound={lower_bound}")

        # Load image chunk into mem
        downsampled_image_chunk = downsampled_stack[lb[0]:ub[0]+1, lb[1]:ub[1]+1, lb[2]:ub[2]+1].compute()

        interval_key = (
            tuple(lb),
            tuple(ub),
            tuple((ub[0] - lb[0]+1, ub[1] - lb[1]+1, ub[2] - lb[2]+1))
        )

        return view_id, interval_key, downsampled_image_chunk, offset, lower_bound

    def run(self, metadata_df, dsxy, dsz):
        """
        Executes the entry point of the script.
        """
        return self.fetch_image_data(metadata_df, dsxy, dsz)

