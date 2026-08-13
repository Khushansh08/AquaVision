import os
import cv2

from quality_metrics import compare_metrics


# -----------------------------
# Project paths
# -----------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


original_path = os.path.join(
    BASE_DIR,
    "uploads",
    "fish3.jpg"
)


enhanced_path = os.path.join(
    BASE_DIR,
    "models",
    "waternet",
    "output",
    "2",
    "fish3.jpg"
)


# -----------------------------
# Load Images
# -----------------------------

original = cv2.imread(original_path)

enhanced = cv2.imread(enhanced_path)


# -----------------------------
# Check loading
# -----------------------------

print("Original path:", original_path)
print("Enhanced path:", enhanced_path)


if original is None:
    raise FileNotFoundError(
        "Original image not found"
    )

if enhanced is None:
    raise FileNotFoundError(
        "Enhanced image not found"
    )


print("Original shape:", original.shape)
print("Enhanced shape:", enhanced.shape)


# -----------------------------
# Calculate Metrics
# -----------------------------

result = compare_metrics(
    original,
    enhanced
)


print("\n========== IMAGE QUALITY REPORT ==========\n")

print(result)

print("\n==========================================")