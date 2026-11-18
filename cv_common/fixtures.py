"""Deterministic synthetic fixtures, never represented as real mission data."""
import cv2
import numpy as np
from PIL import Image
from .geometry import intrinsics

def depth_fixture(width=480, height=320):
    v, u = np.mgrid[:height, :width]
    depth = (6 + 12 * (1 - v / height)).astype(np.float32)
    image = np.full((height, width, 3), [55, 68, 84], np.uint8)
    image[v > height // 2] = [114, 122, 130]
    for x, z, color in [(int(width * 0.2), 8, (191, 121, 67)), (int(width * 0.6), 11, (74, 142, 165))]:
        mask = (u >= x) & (u < x + width // 5) & (v > height // 5) & (v < height * 0.8)
        depth[mask] = z
        image[mask] = color
    grid = (u % 32 < 2) | (v % 32 < 2)
    image[grid] = image[grid] // 2
    return (Image.fromarray(image), 1 / depth)
