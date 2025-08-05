import math
import numpy as np
from xml.etree import ElementTree as ET
from collections import defaultdict

def greatest_common_divisor(a, b):
    """Computes the greatest common divisor of a and b."""
    return math.gcd(int(a), int(b))

def lowest_common_multiplier(a, b):
    """Computes the lowest common multiplier of a and b."""
    if a == 0 or b == 0:
        return 0
    return abs(int(a) * int(b)) // greatest_common_divisor(a, b)

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
    Determines the minimal step size for splitting based on mipmap resolutions.
    """
    img_loader = root.find('SequenceDescription/ImgLoader')
    min_step_size = np.array([1, 1, 1], dtype=np.int64)

    if 'class' in img_loader.attrib and 'MultiResolution' in img_loader.attrib['class']:
        view_setups = root.find('SequenceDescription/ViewSetups')
        
        for vs in view_setups.findall('ViewSetup'):
            setup_id_elem = vs.find('id')
            if setup_id_elem is None:
                continue
            setup_id = setup_id_elem.text
            
            # Find the corresponding mipmap setup
            mipmap_setup_xpath = f".//setup[@id='{setup_id}']"
            mipmap_setup = img_loader.find(mipmap_setup_xpath)
            
            if mipmap_setup is not None:
                resolutions_elem = mipmap_setup.find('resolutions')
                if resolutions_elem is not None:
                    # Get the last resolution (lowest)
                    all_res = resolutions_elem.findall('resolution')
                    if all_res:
                        lowest_res_str = all_res[-1].text
                        lowest_resolution = np.round(np.fromstring(lowest_res_str, sep=' ')).astype(np.int64)
                        
                        for d in range(len(min_step_size)):
                            min_step_size[d] = lowest_common_multiplier(min_step_size[d], lowest_resolution[d])
    
    return min_step_size