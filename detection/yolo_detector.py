import os
import cv2
import urllib.request
from ultralytics import YOLO

MODEL_URL = "https://huggingface.co/khushansh08/aquavision-yolo/resolve/main/trash_mbari_09072023_640imgsz_50epochs_yolov8.pt"
MODEL_PATH = os.path.join(
    "models",
    "yolo",
    "trash_mbari_09072023_640imgsz_50epochs_yolov8.pt"
)


def download_model():
    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )


download_model()

model = YOLO(MODEL_PATH)


def remove_duplicate_boxes(
    detections,
    distance_threshold=15
):
    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True
    )

    final = []

    for det in detections:
        x1, y1, x2, y2 = det["box"]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        keep = True

        for saved in final:
            sx1, sy1, sx2, sy2 = saved["box"]

            scx = (sx1 + sx2) // 2
            scy = (sy1 + sy2) // 2

            if (
                abs(cx - scx) < distance_threshold
                and abs(cy - scy) < distance_threshold
            ):
                keep = False
                break

        if keep:
            final.append(det)

    return final


def draw_boxes(
    image,
    detections
):
    output = image.copy()

    for obj in detections:
        x1, y1, x2, y2 = obj["box"]

        label = (
            f'{obj["class"]} '
            f'{obj["confidence"]:.2f}'
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        text_y = max(
            y1 - 10,
            20
        )

        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return output


def detect_objects(image):
    results = model(
        image,
        conf=0.11,
        iou=0.45
    )

    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "class": model.names[cls_id],
                    "confidence": round(
                        confidence,
                        3
                    ),
                    "box": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ]
                }
            )

    detections = remove_duplicate_boxes(
        detections
    )

    detected_image = draw_boxes(
        image,
        detections
    )

    return (
        detections,
        detected_image
    )