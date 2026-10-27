"""LeafDoc (Streamlit): leaf photo -> disease, confidence, Grad-CAM, advice (EN + UR)."""
import json

import numpy as np
import streamlit as st
import timm
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms as T

IMG_SIZE = 128
THRESHOLD = 0.6
MIN_GREEN = 0.08

eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

HEALTHY_EN = "The leaf looks healthy. Keep regular watering, balanced fertiliser and check plants weekly."
HEALTHY_UR = "پتا صحت مند لگتا ہے۔ باقاعدہ پانی دیں، متوازن کھاد استعمال کریں اور ہفتہ وار پودوں کا معائنہ کریں۔"

ADVICE = {
    "Pepper__bell___healthy": (HEALTHY_EN, HEALTHY_UR),
    "Potato___healthy": (HEALTHY_EN, HEALTHY_UR),
    "Tomato_healthy": (HEALTHY_EN, HEALTHY_UR),
    "Pepper__bell___Bacterial_spot": (
        "Remove infected leaves, avoid overhead watering, use certified disease-free seed, and apply copper-based spray as advised locally. Rotate crops.",
        "متاثرہ پتے توڑ دیں، اوپر سے پانی دینے سے گریز کریں، صحت مند تصدیق شدہ بیج استعمال کریں اور مقامی ماہر کے مشورے سے کاپر والا سپرے کریں۔ فصل کی ادل بدل کریں۔"),
    "Potato___Early_blight": (
        "Remove infected leaves, keep plants well spaced, avoid wet foliage, and use a recommended fungicide. Rotate crops yearly.",
        "متاثرہ پتے ہٹا دیں، پودوں میں مناسب فاصلہ رکھیں، پتوں کو گیلا نہ رہنے دیں اور تجویز کردہ فنگس کش دوا استعمال کریں۔ ہر سال فصل بدلیں۔"),
    "Potato___Late_blight": (
        "Act quickly: remove and destroy infected plants, avoid irrigation late in the day, and apply a recommended fungicide. Do not store infected tubers.",
        "فوری اقدام کریں: متاثرہ پودے نکال کر تلف کریں، شام کو آبپاشی نہ کریں اور تجویز کردہ فنگس کش دوا لگائیں۔ متاثرہ آلو ذخیرہ نہ کریں۔"),
    "Tomato_Bacterial_spot": (
        "Remove infected leaves, avoid overhead watering, use clean seed and tools, and apply copper-based spray as advised locally.",
        "متاثرہ پتے ہٹا دیں، اوپر سے پانی نہ دیں، صاف بیج اور اوزار استعمال کریں اور مقامی ماہر کے مشورے سے کاپر والا سپرے کریں۔"),
    "Tomato_Early_blight": (
        "Remove lower infected leaves, mulch the soil, avoid wet foliage, and use a recommended fungicide. Rotate crops.",
        "نچلے متاثرہ پتے ہٹا دیں، زمین پر ملچ کریں، پتے گیلے نہ رکھیں اور تجویز کردہ فنگس کش دوا استعمال کریں۔ فصل کی ادل بدل کریں۔"),
    "Tomato_Late_blight": (
        "Remove and destroy infected plants quickly, improve air flow, avoid evening watering, and apply a recommended fungicide.",
        "متاثرہ پودے جلدی نکال کر تلف کریں، ہوا کا گزر بہتر بنائیں، شام کو پانی نہ دیں اور تجویز کردہ فنگس کش دوا لگائیں۔"),
    "Tomato_Leaf_Mold": (
        "Improve ventilation, reduce humidity, avoid wetting leaves, remove infected leaves, and use a recommended fungicide.",
        "ہوا کا گزر بہتر کریں، نمی کم رکھیں، پتے گیلے نہ کریں، متاثرہ پتے ہٹا دیں اور تجویز کردہ فنگس کش دوا استعمال کریں۔"),
    "Tomato_Septoria_leaf_spot": (
        "Remove infected leaves, avoid overhead watering, mulch soil, rotate crops and use a recommended fungicide.",
        "متاثرہ پتے ہٹا دیں، اوپر سے پانی نہ دیں، زمین پر ملچ کریں، فصل کی ادل بدل کریں اور تجویز کردہ فنگس کش دوا استعمال کریں۔"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": (
        "Spray plants with water to wash off mites, use insecticidal soap or a recommended miticide, and avoid dusty, dry conditions.",
        "پودوں پر پانی کا سپرے کر کے مائیٹس دھو دیں، کیڑے مار صابن یا تجویز کردہ مائیٹ کش دوا استعمال کریں اور گرد آلود خشک ماحول سے بچائیں۔"),
    "Tomato__Target_Spot": (
        "Remove infected leaves, improve air flow, avoid overhead watering, and use a recommended fungicide.",
        "متاثرہ پتے ہٹا دیں، ہوا کا گزر بہتر کریں، اوپر سے پانی نہ دیں اور تجویز کردہ فنگس کش دوا استعمال کریں۔"),
    "Tomato__Tomato_mosaic_virus": (
        "No cure. Remove and destroy infected plants, wash hands and tools, control weeds, and use resistant varieties next season.",
        "اس کا کوئی علاج نہیں۔ متاثرہ پودے نکال کر تلف کریں، ہاتھ اور اوزار دھوئیں، جڑی بوٹیاں کنٹرول کریں اور اگلے سیزن میں مزاحم اقسام لگائیں۔"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": (
        "No cure. Control whiteflies (the carrier) with traps or recommended insecticide, remove infected plants, and use resistant varieties.",
        "اس کا کوئی علاج نہیں۔ سفید مکھی (وائرس پھیلانے والی) کو ٹریپ یا تجویز کردہ دوا سے کنٹرول کریں، متاثرہ پودے ہٹا دیں اور مزاحم اقسام لگائیں۔"),
}

NICE = lambda c: c.replace("___", " - ").replace("__", " - ").replace("_", " ")


@st.cache_resource
def load():
    classes = json.load(open("classes.json"))
    model = timm.create_model("resnet18", pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load("tl_model.pth", map_location="cpu"))
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    return classes, model, cam


def looks_like_leaf(img):
    hsv = np.array(img.convert("RGB").resize((128, 128)).convert("HSV"), dtype=float)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h > 25) & (h < 110) & (s > 40) & (v > 40)).mean() >= MIN_GREEN


def predict(img, classes, model, cam):
    img = img.convert("RGB")
    if not looks_like_leaf(img):
        return "not_leaf", None, 0.0, None
    x = eval_tf(img).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    conf, idx = p.max(0)
    idx, conf = int(idx), float(conf)
    rgb = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255
    g = cam(input_tensor=x, targets=[ClassifierOutputTarget(idx)])[0]
    heat = Image.fromarray(show_cam_on_image(rgb, g, use_rgb=True)).resize((384, 384), Image.BICUBIC)
    return ("ok" if conf >= THRESHOLD else "not_sure"), classes[idx], conf, heat


def rtl(text):
    st.markdown(f"<div dir='rtl' style='text-align:right'>{text}</div>", unsafe_allow_html=True)


st.set_page_config(page_title="LeafDoc", page_icon="🌿")
st.title("🌿 LeafDoc: Plant Disease Detection")
st.caption("Upload or capture a tomato, potato or pepper leaf photo. "
           "This is a decision aid, not a replacement for an agriculture expert.")

classes, model, cam = load()

src = st.radio("Photo source / تصویر", ["Upload", "Camera"], horizontal=True)
file = st.file_uploader("Leaf photo / پتے کی تصویر", type=["jpg", "jpeg", "png"]) if src == "Upload" \
    else st.camera_input("Take a leaf photo")

if file is not None:
    img = Image.open(file)
    st.image(img, caption="Your photo", use_container_width=True)
    status, label, conf, heat = predict(img, classes, model, cam)

    if status == "not_leaf":
        st.warning("This does not look like a leaf photo. Please take a clear, close photo of a single leaf.")
        rtl("یہ پتے کی تصویر نہیں لگتی۔ براہ کرم ایک پتے کی صاف اور قریب سے تصویر لیں۔")
    elif status == "not_sure":
        st.warning(f"Not sure (confidence {conf*100:.0f}%). Try a clearer, closer photo in daylight, "
                   "or ask a local agriculture expert.")
        rtl(f"یقین نہیں (اعتماد {conf*100:.0f}%)۔ دن کی روشنی میں صاف اور قریب سے تصویر لیں یا کسی زرعی ماہر سے مشورہ کریں۔")
        st.image(heat, caption="Grad-CAM (where the model looked)")
    else:
        en, ur = ADVICE[label]
        st.subheader(NICE(label))
        st.write(f"Confidence: **{conf*100:.0f}%**")
        st.image(heat, caption="Grad-CAM (where the model looked)")
        st.markdown("### English")
        st.write(en)
        st.markdown("### اردو")
        rtl(ur)