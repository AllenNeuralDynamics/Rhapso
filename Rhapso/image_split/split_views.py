import numpy as np
from collections import defaultdict

# ============================================================================
# Mathematical Utilities
# ============================================================================ 

def gcd(a, b):
    """Calculate the greatest common divisor of two numbers."""
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """Calculate the least common multiple of two numbers."""
    return abs(a * b) // gcd(int(a), int(b)) if a and b else max(a, b)


def next_multiple(a, b):
    """Find the smallest multiple of b that is greater than or equal to a."""
    if a % b == 0:
        return a
    return (a + b) - (a % b)


# ============================================================================
# XML Processing Utilities
# ============================================================================

def _find_one(root, name):
    """Namespaced-safe find for existing elements."""
    return root.find(f'.//{{*}}{name}') or root.find(name)


def _find_all(root, name):
    """Namespaced-safe findall for existing elements."""
    return root.findall(f'.//{{*}}{name}') or root.findall(name)


def _get_text_safe(elem, default=""):
    """Safely extract text from XML element."""
    if elem is not None and elem.text:
        return elem.text.strip()
    return default


def _safe_eval(text, default=None):
    """Safely evaluate text with fallback."""
    try:
        return eval(text) if text else default
    except (SyntaxError, NameError):
        return default


# ============================================================================
# Core Functions
# ============================================================================

def collect_image_sizes(root):
    """
    Collect image sizes from all ViewSetups in the XML.
    
    Returns:
        tuple: (sizes_dict, min_dimensions)
            - sizes_dict: Dictionary mapping size strings to their count
            - min_dimensions: Numpy array of the minimum dimensions found
    """
    sizes = defaultdict(int)
    min_size = None
    view_setups = _find_one(root, 'SequenceDescription/ViewSetups')
    
    if view_setups is None:
        return dict(sizes), min_size

    for vs in _find_all(view_setups, 'ViewSetup'):
        size_elem = _find_one(vs, 'size')
        if not size_elem or not size_elem.text:
            continue
            
        size_dims = np.array([int(d) for d in size_elem.text.split()])
        size_key = "x".join(map(str, size_dims))
        sizes[size_key] += 1

        if min_size is None:
            min_size = size_dims
        else:
            min_size = np.minimum(min_size, size_dims)
            
    return dict(sizes), min_size


def _is_multi_resolution_loader(img_loader):
    """Check if the image loader supports multi-resolution formats."""
    if img_loader is None:
        return False
        
    format_attr = img_loader.get('format', '').lower()
    multi_res_keywords = ['multi', 'split', 'zarr', 'n5']
    return any(keyword in format_attr for keyword in multi_res_keywords)


def _get_mipmap_resolutions(vs, nested_loader, img_loader, vs_id):
    """Extract mipmap resolutions from various possible locations in the XML."""
    # Try direct child of view setup
    resolutions = _find_one(vs, 'mipmapResolutions')
    if resolutions:
        result = _safe_eval(_get_text_safe(resolutions))
        if result:
            return result
    
    # Try nested structure
    if nested_loader is not None:
        setup_loader = nested_loader.find(f'SetupImgLoader[@id="{vs_id}"]')
        if setup_loader is not None:
            resolutions = _find_one(setup_loader, 'mipmapResolutions')
            if resolutions:
                result = _safe_eval(_get_text_safe(resolutions))
                if result:
                    return result
    
    # Try main image loader
    resolutions = _find_one(img_loader, 'mipmapResolutions')
    if resolutions:
        result = _safe_eval(_get_text_safe(resolutions))
        if result:
            return result
    
    # Default mipmap resolutions for common multi-resolution formats
    return [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [4.0, 4.0, 4.0],
        [8.0, 8.0, 8.0],
        [16.0, 16.0, 16.0],
        [32.0, 32.0, 32.0],
        [64.0, 64.0, 64.0]
    ]


def find_min_step_size(root):
    """
    Determine the minimal step size for splitting based on XML structure.
    
    Args:
        root: XML root element
        
    Returns:
        numpy.ndarray: Minimal step size for each dimension
        
    Raises:
        RuntimeError: If downsampling has non-integer values
    """
    min_step_size = np.array([1, 1, 1], dtype=np.int64)
    
    try:
        img_loader = _find_one(root, 'SequenceDescription/ImageLoader')
        if img_loader is None:
            return min_step_size

        # Check for multi-resolution support
        is_multi_res = _is_multi_resolution_loader(img_loader)
        nested_loader = _find_one(img_loader, 'ImageLoader')
        if nested_loader is not None:
            is_multi_res = is_multi_res or _is_multi_resolution_loader(nested_loader)

        if not is_multi_res:
            return min_step_size

        # Process multi-resolution view setups
        view_setups = _find_all(root, 'SequenceDescription/ViewSetups/ViewSetup')
        
        for vs in view_setups:
            vs_id_elem = _find_one(vs, 'id')
            vs_id_text = _get_text_safe(vs_id_elem, "unknown")
            
            mipmap_resolutions = _get_mipmap_resolutions(vs, nested_loader, img_loader, vs_id_text)
            
            if mipmap_resolutions:
                lowest_resolution = mipmap_resolutions[-1]
                
                # Update min step size with LCM for each dimension
                for d in range(len(min_step_size)):
                    if d < len(lowest_resolution):
                        res_value = lowest_resolution[d]
                        
                        # Validate resolution values
                        if abs(res_value % 1) > 0.001 and (1.0 - abs(res_value % 1)) > 0.001:
                            raise RuntimeError(
                                "Downsampling has fraction > 0.001, cannot split dataset"
                            )
                        
                        rounded_res = round(res_value)
                        min_step_size[d] = lcm(min_step_size[d], rounded_res)
        
        return min_step_size
        
    except Exception as e:
        print(f"Warning: Failed to determine minimum step size: {e}")
        return min_step_size