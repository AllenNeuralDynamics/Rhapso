"""
Image splitting module for Rhapso.

This module provides functionality for splitting large images into smaller tiles
while maintaining metadata and creating fake interest points for alignment.
"""

from .split_datasets import main as split_images_main
from .split_views import collect_image_sizes, find_min_step_size, next_multiple
from .splitting_tools import split_images

__all__ = [
    'split_images_main',
    'collect_image_sizes', 
    'find_min_step_size',
    'next_multiple',
    'split_images'
]
