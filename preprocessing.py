import cv2

import numpy as np


def load_image(image_path):
    """
    Reads an image from disk.

    Parameters:
        image_path (str): Path of the uploaded image.

    Returns:
        image (numpy.ndarray): Loaded image.
    """

    image = cv2.imread(image_path)

    return image