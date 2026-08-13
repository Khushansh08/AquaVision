import os
import cv2
import torch
import numpy as np

from models.waternet.waternet.net import WaterNet
from models.waternet.waternet.data import transform

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

WEIGHTS_PATH = os.path.join(
    "models",
    "waternet",
    "waternet_exported_state_dict-daa0ee.pt"
)

model = WaterNet()

checkpoint = torch.load(
    WEIGHTS_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)
model.to(device)
model.eval()


def arr2ten(arr):
    ten = torch.from_numpy(arr) / 255.0

    if len(ten.shape) == 3:
        ten = torch.permute(ten, (2, 0, 1))
        ten = torch.unsqueeze(ten, dim=0)

    elif len(ten.shape) == 4:
        ten = torch.permute(ten, (0, 3, 1, 2))

    return ten.float()


def ten2arr(ten):
    arr = ten.cpu().detach().numpy()

    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)

    arr = np.transpose(arr, (0, 2, 3, 1))

    return arr[0]


def resize_image(image, max_size=640):
    h, w = image.shape[:2]

    scale = max_size / max(h, w)

    if scale < 1:
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    return image


def aqua_enhance(image):
    # Mild color correction
    img = image.astype(np.float32)

    b, g, r = cv2.split(img)

    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)

    mean_all = (
        mean_b +
        mean_g +
        mean_r
    ) / 3.0

    b = b * np.clip(
        mean_all / (mean_b + 1e-6),
        0.90,
        1.10
    )

    g = g * np.clip(
        mean_all / (mean_g + 1e-6),
        0.90,
        1.10
    )

    r = r * np.clip(
        mean_all / (mean_r + 1e-6),
        0.90,
        1.10
    )

    img = cv2.merge((b, g, r))

    img = np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

    # CLAHE contrast enhancement
    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    current_contrast = np.std(
        l.astype(np.float32)
    )

    if current_contrast < 35:
        clip_limit = 3.0
    elif current_contrast < 50:
        clip_limit = 2.5
    else:
        clip_limit = 2.0

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    img = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    # Mild sharpening
    blurred = cv2.GaussianBlur(
        img,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        img,
        1.08,
        blurred,
        -0.15,
        0
    )

    sharpened = np.clip(
        sharpened,
        0,
        255
    ).astype(np.uint8)

    return sharpened


def enhance_image(image):
    image = resize_image(image)

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    rgb = rgb.astype(np.uint8)

    wb, gc, he = transform(rgb)

    rgb_ten = arr2ten(rgb).to(device)
    wb_ten = arr2ten(wb).to(device)
    gc_ten = arr2ten(gc).to(device)
    he_ten = arr2ten(he).to(device)

    with torch.no_grad():
        output = model(
            rgb_ten,
            wb_ten,
            he_ten,
            gc_ten
        )

    enhanced = ten2arr(output)

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_RGB2BGR
    )

    enhanced = aqua_enhance(enhanced)

    return enhanced