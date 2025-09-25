"""Module for Myna applications that are based on the pytorch package"""

from .patches import (
    extract_patches_as_strided,
    recombine_patches_view,
    get_patch_coordinates_from_index,
)
