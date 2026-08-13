import os
import cv2

from yolo_detector import detect_objects


# Project root
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


image_path = os.path.join(
    BASE_DIR,
    "models",
    "waternet",
    "output",
    "2",
    "fish3.jpg"
)


print("Image path:", image_path)


image = cv2.imread(image_path)


if image is None:
    raise FileNotFoundError(
        "WaterNet output image not found"
    )


print("Image shape:", image.shape)


result = detect_objects(image)


print("\nDetection Result:")
print(result)