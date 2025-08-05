import math
import numpy as np
from xml.etree import ElementTree as ET
from collections import defaultdict

def gcd(a, b):
    """Calculate the greatest common divisor of two numbers."""
    while b:
        a, b = b, a % b
    return a

def lowest_common_multiplier(a, b):
    """Calculate the lowest common multiplier of two numbers."""
    return abs(a * b) // gcd(int(a), int(b)) if a and b else max(a, b)

def closest_larger_long_divisable_by(a, b):
    """Finds the closest multiple of b that is greater than or equal to a."""
    if a % b == 0:
        return a
    return (a + b) - (a % b)

def collect_image_sizes(root):
    """
    Collects image sizes from all ViewSetups in the XML.

    Returns a tuple containing:
    - A dictionary mapping size strings to their count.
    - A numpy array of the minimum dimensions found.
    """
    sizes = defaultdict(int)
    min_size = None
    view_setups = root.find('SequenceDescription/ViewSetups')

    for vs in view_setups.findall('ViewSetup'):
        size_str = vs.find('size').text
        size_dims = np.array([int(d) for d in size_str.split()])
        
        size_key = "x".join(map(str, size_dims))
        sizes[size_key] += 1

        if min_size is None:
            min_size = size_dims
        else:
            min_size = np.minimum(min_size, size_dims)
            
    return dict(sizes), min_size

def find_min_step_size(root):
    """
    Determines the minimal step size for splitting based on the structure
    of the XML data, matching the behavior of Java's findMinStepSize.
    """
    print("\n🔍 [find_min_step_size] Starting to determine minimal step size for splitting...")
    
    # Initialize with a minimum step size of 1 for each dimension
    min_step_size = np.array([1, 1, 1], dtype=np.int64)
    
    try:
        # Check for multi-resolution image loader
        img_loader = root.find('SequenceDescription/ImageLoader')
        if img_loader is None:
            print("🔍 [find_min_step_size] No ImageLoader found, using default step size of [1, 1, 1]")
            return min_step_size

        # Check if it's a multi-resolution loader
        is_multi_res = False
        if img_loader.get('format') is not None:
            is_multi_res = 'multi' in img_loader.get('format').lower() or 'split' in img_loader.get('format').lower() or 'zarr' in img_loader.get('format').lower()
        
        # In the example XML, there's a nested ImageLoader
        nested_loader = img_loader.find('ImageLoader')
        if nested_loader is not None and nested_loader.get('format') is not None:
            is_multi_res = is_multi_res or 'multi' in nested_loader.get('format').lower() or 'zarr' in nested_loader.get('format').lower()

        if is_multi_res:
            print("🔍 [find_min_step_size] Multi-resolution image loader detected")
            print("🔍 [find_min_step_size] Searching for resolution steps in each view setup...")
            
            # Find all view setups
            view_setups = root.findall('SequenceDescription/ViewSetups/ViewSetup')
            
            for vs in view_setups:
                vs_id = vs.find('id')
                vs_name = vs.find('name')
                
                vs_id_text = vs_id.text if vs_id is not None else "unknown"
                vs_name_text = vs_name.text if vs_name is not None else "unknown"
                
                # Try to find mipmap resolutions for this view setup
                # In Zarr/N5 formats, this is often stored in attributes or in a special metadata file
                # For this example, we'll extract it from the XML if possible, otherwise use default values
                mipmapResolutions = []
                
                # Look for resolutions in various possible locations
                # 1. Direct child of the view setup
                resolutions = vs.find('mipmapResolutions')
                if resolutions is not None:
                    try:
                        # Parse the resolutions if they're stored as text
                        mipmapResolutions = eval(resolutions.text)
                    except:
                        pass
                
                # 2. Inside a nested structure
                if not mipmapResolutions and nested_loader is not None:
                    # Try to find setup-specific resolutions
                    setup_loader = nested_loader.find(f'SetupImgLoader[@id="{vs_id_text}"]')
                    if setup_loader is not None:
                        resolutions = setup_loader.find('mipmapResolutions')
                        if resolutions is not None:
                            try:
                                mipmapResolutions = eval(resolutions.text)
                            except:
                                pass
                
                # 3. If still not found, check for resolutions in the main image loader
                if not mipmapResolutions:
                    resolutions = img_loader.find('mipmapResolutions')
                    if resolutions is not None:
                        try:
                            mipmapResolutions = eval(resolutions.text)
                        except:
                            pass
                
                # 4. If all else fails, use default resolutions based on common patterns
                # For Zarr/N5 multi-resolution formats, 2x downsampling per level is common
                if not mipmapResolutions:
                    # Use common mipmap resolution pattern [1, 2, 4, 8, 16, 32, 64] for each dimension
                    mipmapResolutions = [
                        [1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                        [4.0, 4.0, 4.0],
                        [8.0, 8.0, 8.0],
                        [16.0, 16.0, 16.0],
                        [32.0, 32.0, 32.0],
                        [64.0, 64.0, 64.0]
                    ]
                
                print(f"🔍 [find_min_step_size] ViewSetup: {vs_name_text} (id={vs_id_text}): {mipmapResolutions}")
                
                # Get the lowest resolution (last in the array)
                if mipmapResolutions:
                    lowest_resolution = mipmapResolutions[-1]
                    print(f"🔍 [find_min_step_size] Lowest resolution for this view setup: {lowest_resolution}")
                    
                    # Update the min step size with the LCM for each dimension
                    for d in range(len(min_step_size)):
                        if d < len(lowest_resolution):
                            # Check for non-integer values (with tolerance)
                            res_value = lowest_resolution[d]
                            if abs(res_value % 1) > 0.001 and (1.0 - abs(res_value % 1)) > 0.001:
                                raise RuntimeError("Downsampling has a fraction > 0.001, cannot split dataset since it does not seem to be a rounding error.")
                            
                            rounded_res = round(res_value)
                            print(f"🔍 [find_min_step_size] Updating minStepSize for dimension {d}: previous={min_step_size[d]}, new={rounded_res}")
                            min_step_size[d] = lowest_common_multiplier(min_step_size[d], rounded_res)
                    
                    print(f"🔍 [find_min_step_size] Updated min step size after this view setup: {min_step_size}")
        else:
            print("🔍 [find_min_step_size] Not a multi-resolution image loader, all data splits are possible. Using default step size of 1 for each dimension.")
        
        print(f"🔍 [find_min_step_size] Final minimal step size per dimension: {min_step_size}")
        print("\n")
        return min_step_size
        
    except Exception as e:
        print(f"Error: Failed to find minimum step size. {e}")
        # Return the default value in case of error
        return np.array([1, 1, 1], dtype=np.int64)