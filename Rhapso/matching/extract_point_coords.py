import zarr
import numpy as np

def get_interest_point_coords(n5_path, timepoint, view_setup, point_index):
    """Get coordinates for a specific interest point."""
    # Construct the full path to the locations dataset
    loc_path = f"{n5_path}/tpId_{timepoint}_viewSetupId_{view_setup}/beads/interestpoints/loc"
    
    # Open the N5 store and get the locations dataset
    store = zarr.N5Store(n5_path)
    root = zarr.open(store)
    locations = root[f"tpId_{timepoint}_viewSetupId_{view_setup}/beads/interestpoints/loc"]
    
    # Get the coordinates for the specified point index
    if point_index < locations.shape[0]:
        coords = locations[point_index]
        return coords
    else:
        raise IndexError(f"Point index {point_index} is out of bounds. Dataset has {locations.shape[0]} points.")

if __name__ == "__main__":
    n5_path = "/mnt/c/Users/marti/Documents/allen/data/April Dataset for Interest Points As TIFF XML (unaligned)/BSS detection, rhapso matching/interestpoints.n5"
    coords = get_interest_point_coords(n5_path, 18, 6, 3865)
    print(f"Coordinates for point 3865: x={coords[0]}, y={coords[1]}, z={coords[2]}")

    target_view_setup = 5  # Replace with the suspected target view setup
    target_locations = root[f"tpId_18_viewSetupId_{target_view_setup}/beads/interestpoints/loc"]
    print(f"Target dataset shape: {target_locations.shape}")
