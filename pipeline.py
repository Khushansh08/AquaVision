import time

from preprocessing import load_image
from enhancement import enhance_image
from metrics.quality_metrics import compare_metrics
from detection.yolo_detector import detect_objects


def process_image(image_path):

    start = time.time()

    original = load_image(image_path)

    enhanced = enhance_image(original)

    metrics = compare_metrics(
        original,
        enhanced
    )

    detections, detected_image = detect_objects(
        enhanced
    )

    processing_time = round(
        time.time() - start,
        2
    )

    return {

        "original": original,

        "enhanced": enhanced,

        "detected_image": detected_image,

        "metrics": metrics,

        "detections": detections,

        "processing_time": processing_time

    }