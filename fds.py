import os
import io
import uuid
import tempfile
import logging
from typing import List

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

# Optional imports
try:
    import easyocr
    from pdf2image import convert_from_path
    from google.cloud import documentai_v1 as documentai
    from google.cloud import storage
except ImportError:
    easyocr = None
    convert_from_path = None
    documentai = None
    storage = None

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- EasyOCR languages (all major Indian scripts supported) ---
EASYOCR_LANGS = [
    "en", "as", "bn", "gu", "hi", "kn", "ml", "mr",
    "ne", "or", "pa", "ta", "te", "ur"
]

# Initialize EasyOCR reader
easyocr_reader = None
if easyocr:
    try:
        easyocr_reader = easyocr.Reader(EASYOCR_LANGS, gpu=False)
        logging.info(f"EasyOCR initialized with languages: {', '.join(EASYOCR_LANGS)}")
    except Exception as e:
        logging.error(f"EasyOCR initialization error: {e}")
        easyocr_reader = None

# --- Tesseract language map (covers all 22 official languages) ---
TESSERACT_LANG_MAP = {
    "en": "eng", "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin",
    "kn": "kan", "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori",
    "pa": "pan", "ta": "tam", "te": "tel", "ur": "urd",
    "sa": "san", "mai": "mai", "sat": "sat", "sd": "snd"
}

# --- Helper functions ---
def preprocess(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def detect_forgery_rules(extracted_text: str, processed_images: List[str]) -> dict:
    confidence = 0.0
    reasons = []
    if "Times New Roman" in extracted_text and "Arial" in extracted_text:
        reasons.append("Font mismatch detected")
        confidence += 0.3
    if "Invoice Number" not in extracted_text:
        reasons.append("Missing expected field: Invoice Number")
        confidence += 0.2
    return {
        "overall_forgery_confidence": min(confidence, 1.0),
        "is_flagged_for_review": confidence > 0.5,
        "flagging_reasons": reasons
    }

# --- Streamlit UI ---
st.set_page_config(page_title="OCR + Forgery Detection", layout="wide")
st.title("OCR + Forgery Detection")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    pil = Image.open(uploaded_file).convert("RGB")
    img_cv = np.array(pil)[:, :, ::-1].copy()
    processed_img = preprocess(img_cv)

    st.image(processed_img, caption="Preprocessed Image")

    # OCR with EasyOCR first, fallback to Tesseract
    text = ""
    if easyocr_reader:
        try:
            result = easyocr_reader.readtext(processed_img, lang_list=EASYOCR_LANGS)
            text = " ".join([t[1] for t in result])
        except Exception as e:
            logging.error(f"EasyOCR error: {e}")
            # fallback to Tesseract
            t_langs = "+".join([TESSERACT_LANG_MAP.get(l, "eng") for l in EASYOCR_LANGS])
            text = pytesseract.image_to_string(processed_img, lang=t_langs)
    else:
        t_langs = "+".join([TESSERACT_LANG_MAP.get(l, "eng") for l in EASYOCR_LANGS])
        text = pytesseract.image_to_string(processed_img, lang=t_langs)

    st.text_area("Extracted Text", value=text, height=300)

    # Forgery detection
    report = detect_forgery_rules(text, [])
    st.json(report)
