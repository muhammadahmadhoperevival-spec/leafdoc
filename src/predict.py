"""Inference: prediction + Grad-CAM + 'not sure' logic."""
import json

import numpy as np
import timm
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import eval_tf, IMG_SIZE

THRESHOLD = 0.6      # evaluate.py se chuna gaya
MIN_GREEN = 0.08     # leaf check: image ka kam az kam 8% hissa hara/peela-hara

classes = json.load(open("classes.json"))
model = timm.create_model("resnet18", pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load("tl_model.pth", map_location="cpu"))
model.eval()
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])


def looks_like_leaf(img: Image.Image) -> bool:
    """Simple colour check: leaf photos mein hara/peela-hara hissa hota hai."""
    hsv = np.array(img.convert("RGB").resize((128, 128)).convert("HSV"), dtype=float)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h > 25) & (h < 110) & (s > 40) & (v > 40)   # PIL HSV: hue 0-255
    return mask.mean() >= MIN_GREEN


def predict(img: Image.Image):
    img = img.convert("RGB")
    if not looks_like_leaf(img):
        return {"status": "not_leaf", "label": None, "confidence": 0.0, "heatmap": None}

    x = eval_tf(img).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    conf, idx = p.max(0)
    idx, conf = int(idx), float(conf)

    rgb = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255
    g = cam(input_tensor=x, targets=[ClassifierOutputTarget(idx)])[0]
    heat = show_cam_on_image(rgb, g, use_rgb=True)

    status = "ok" if conf >= THRESHOLD else "not_sure"
    return {"status": status, "label": classes[idx], "confidence": conf, "heatmap": heat}


if __name__ == "__main__":
    import pandas as pd

    # Test set ki 6 images par Grad-CAM save karo (README screenshots ke liye)
    test = pd.read_csv("data/splits/test.csv").sample(6, random_state=1)
    for i, row in enumerate(test.itertuples()):
        r = predict(Image.open(row.path))
        print(i, "true:", row.label, "| pred:", r["label"], f"{r['confidence']:.2f}", r["status"])
        if r["heatmap"] is not None:
            Image.fromarray(r["heatmap"]).save(f"gradcam_{i}.png")

    # Python 3.14 par exit ke waqt aane wale __del__ error se bachne ke liye
    try:
        cam.activations_and_grads.release()
    except Exception:
        pass
    cam.__class__.__del__ = lambda self: None