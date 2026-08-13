import cv2
import numpy as np


def calculate_brightness(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(np.mean(gray))


def calculate_contrast(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(np.std(gray))


def calculate_sharpness(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    return float(laplacian.var())


def calculate_color_balance(image):

    b, g, r = cv2.split(image)

    return {
        "blue": float(np.mean(b)),
        "green": float(np.mean(g)),
        "red": float(np.mean(r))
    }


def calculate_metrics(image):

    return {
        "brightness": calculate_brightness(image),
        "contrast": calculate_contrast(image),
        "sharpness": calculate_sharpness(image),
        "color_balance": calculate_color_balance(image)
    }


def calculate_improvement(before, after):

    if before == 0:
        return 0

    return round(((after - before) / before) * 100, 2)


def compare_metrics(original, enhanced):

    before = calculate_metrics(original)
    after = calculate_metrics(enhanced)

    return {
        "before": before,
        "after": after,
        "improvement_percent": {
            "brightness": calculate_improvement(
                before["brightness"],
                after["brightness"]
            ),
            "contrast": calculate_improvement(
                before["contrast"],
                after["contrast"]
            ),
            "sharpness": calculate_improvement(
                before["sharpness"],
                after["sharpness"]
            )
        }
    }