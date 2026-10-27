"""Treatment / prevention advice, English + Urdu, for all 15 classes."""

def _a(en, ur):
    return {"en": en, "ur": ur}

HEALTHY_EN = "The leaf looks healthy. Keep regular watering, balanced fertiliser and check plants weekly."
HEALTHY_UR = "پتا صحت مند لگتا ہے۔ باقاعدہ پانی دیں، متوازن کھاد استعمال کریں اور ہفتہ وار پودوں کا معائنہ کریں۔"

ADVICE = {
    "Pepper__bell___healthy": _a(HEALTHY_EN, HEALTHY_UR),
    "Potato___healthy": _a(HEALTHY_EN, HEALTHY_UR),
    "Tomato_healthy": _a(HEALTHY_EN, HEALTHY_UR),
    "Pepper__bell___Bacterial_spot": _a(
        "Remove infected leaves, avoid overhead watering, use certified disease-free seed, and apply copper-based spray as advised locally. Rotate crops.",
        "متاثرہ پتے توڑ دیں، اوپر سے پانی دینے سے گریز کریں، صحت مند تصدیق شدہ بیج استعمال کریں اور مقامی ماہر کے مشورے سے کاپر والا سپرے کریں۔ فصل کی ادل بدل کریں۔"),
    "Potato___Early_blight": _a(
        "Remove infected leaves, keep plants well spaced, avoid wet foliage, and use a recommended fungicide. Rotate crops yearly.",
        "متاثرہ پتے ہٹا دیں، پودوں میں مناسب فاصلہ رکھیں، پتوں کو گیلا نہ رہنے دیں اور تجویز کردہ فنگس کش دوا استعمال کریں۔ ہر سال فصل بدلیں۔"),
    "Potato___Late_blight": _a(
        "Act quickly: remove and destroy infected plants, avoid irrigation late in the day, and apply a recommended fungicide. Do not store infected tubers.",
        "فوری اقدام کریں: متاثرہ پودے نکال کر تلف کریں، شام کو آبپاشی نہ کریں اور تجویز کردہ فنگس کش دوا لگائیں۔ متاثرہ آلو ذخیرہ نہ کریں۔"),
    "Tomato_Bacterial_spot": _a(
        "Remove infected leaves, avoid overhead watering, use clean seed and tools, and apply copper-based spray as advised locally.",
        "متاثرہ پتے ہٹا دیں، اوپر سے پانی نہ دیں، صاف بیج اور اوزار استعمال کریں اور مقامی ماہر کے مشورے سے کاپر والا سپرے کریں۔"),
    "Tomato_Early_blight": _a(
        "Remove lower infected leaves, mulch the soil, avoid wet foliage, and use a recommended fungicide. Rotate crops.",
        "نچلے متاثرہ پتے ہٹا دیں، زمین پر ملچ کریں، پتے گیلے نہ رکھیں اور تجویز کردہ فنگس کش دوا استعمال کریں۔ فصل کی ادل بدل کریں۔"),
    "Tomato_Late_blight": _a(
        "Remove and destroy infected plants quickly, improve air flow, avoid evening watering, and apply a recommended fungicide.",
        "متاثرہ پودے جلدی نکال کر تلف کریں، ہوا کا گزر بہتر بنائیں، شام کو پانی نہ دیں اور تجویز کردہ فنگس کش دوا لگائیں۔"),
    "Tomato_Leaf_Mold": _a(
        "Improve ventilation, reduce humidity, avoid wetting leaves, remove infected leaves, and use a recommended fungicide.",
        "ہوا کا گزر بہتر کریں، نمی کم رکھیں، پتے گیلے نہ کریں، متاثرہ پتے ہٹا دیں اور تجویز کردہ فنگس کش دوا استعمال کریں۔"),
    "Tomato_Septoria_leaf_spot": _a(
        "Remove infected leaves, avoid overhead watering, mulch soil, rotate crops and use a recommended fungicide.",
        "متاثرہ پتے ہٹا دیں، اوپر سے پانی نہ دیں، زمین پر ملچ کریں، فصل کی ادل بدل کریں اور تجویز کردہ فنگس کش دوا استعمال کریں۔"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": _a(
        "Spray plants with water to wash off mites, use insecticidal soap or a recommended miticide, and avoid dusty, dry conditions.",
        "پودوں پر پانی کا سپرے کر کے مائیٹس دھو دیں، کیڑے مار صابن یا تجویز کردہ مائیٹ کش دوا استعمال کریں اور گرد آلود خشک ماحول سے بچائیں۔"),
    "Tomato__Target_Spot": _a(
        "Remove infected leaves, improve air flow, avoid overhead watering, and use a recommended fungicide.",
        "متاثرہ پتے ہٹا دیں، ہوا کا گزر بہتر کریں، اوپر سے پانی نہ دیں اور تجویز کردہ فنگس کش دوا استعمال کریں۔"),
    "Tomato__Tomato_mosaic_virus": _a(
        "No cure. Remove and destroy infected plants, wash hands and tools, control weeds, and use resistant varieties next season.",
        "اس کا کوئی علاج نہیں۔ متاثرہ پودے نکال کر تلف کریں، ہاتھ اور اوزار دھوئیں، جڑی بوٹیاں کنٹرول کریں اور اگلے سیزن میں مزاحم اقسام لگائیں۔"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": _a(
        "No cure. Control whiteflies (the carrier) with traps or recommended insecticide, remove infected plants, and use resistant varieties.",
        "اس کا کوئی علاج نہیں۔ سفید مکھی (وائرس پھیلانے والی) کو ٹریپ یا تجویز کردہ دوا سے کنٹرول کریں، متاثرہ پودے ہٹا دیں اور مزاحم اقسام لگائیں۔"),
}