"""LeafDoc Gradio app: upload leaf photo -> disease, confidence, Grad-CAM, advice (EN + UR)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from PIL import Image

from predict import predict
from advice import ADVICE

NICE = lambda c: c.replace("___", " - ").replace("__", " - ").replace("_", " ")


def big(heat):
    return Image.fromarray(heat).resize((384, 384), Image.BICUBIC)


def run(img):
    if img is None:
        return None, "Please upload a leaf photo. / براہ کرم پتے کی تصویر اپ لوڈ کریں۔", ""
    r = predict(Image.fromarray(img))

    if r["status"] == "not_leaf":
        msg = ("**This does not look like a leaf photo.** Please take a clear, close photo of a single leaf.\n\n"
               "<div dir='rtl'>**یہ پتے کی تصویر نہیں لگتی۔** براہ کرم ایک پتے کی صاف اور قریب سے تصویر لیں۔</div>")
        return None, msg, ""

    if r["status"] == "not_sure":
        c = r["confidence"] * 100
        msg = (f"**Not sure** (confidence {c:.0f}%). Try a clearer, closer photo in daylight, "
               "or ask a local agriculture expert.\n\n"
               f"<div dir='rtl'>**یقین نہیں** (اعتماد {c:.0f}%)۔ دن کی روشنی میں صاف اور قریب سے تصویر لیں "
               "یا کسی زرعی ماہر سے مشورہ کریں۔</div>")
        return big(r["heatmap"]), msg, ""

    a = ADVICE[r["label"]]
    head = f"## {NICE(r['label'])}\nConfidence: **{r['confidence']*100:.0f}%**"
    adv = f"### English\n{a['en']}\n\n<div dir='rtl'>\n\n### اردو\n{a['ur']}\n\n</div>"
    return big(r["heatmap"]), head, adv


demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="numpy", label="Leaf photo / پتے کی تصویر"),
    outputs=[
        gr.Image(label="Grad-CAM (where the model looked)"),
        gr.Markdown(label="Result"),
        gr.Markdown(label="Advice"),
    ],
    title="LeafDoc: Plant Disease Detection",
    description="Upload or capture a tomato, potato or pepper leaf photo. "
                "This is a decision aid, not a replacement for an agriculture expert.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()