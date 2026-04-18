import os
import io
import logging
from typing import List

import cv2
import numpy as np
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
    suspicious_sections = []
    if "Times New Roman" in extracted_text and "Arial" in extracted_text:
        reasons.append("Font mismatch detected")
        confidence += 0.3
        suspicious_sections.append({
            "type": "font_anomaly",
            "description": "Font mismatch detected"
        })
    if "Invoice Number" not in extracted_text:
        reasons.append("Missing expected field: Invoice Number")
        confidence += 0.2
    return {
        "overall_forgery_confidence": min(confidence, 1.0),
        "is_flagged_for_review": confidence > 0.5,
        "flagging_reasons": reasons,
        "suspicious_sections": suspicious_sections
    }

def detect_signature_forgery(processed_img: np.ndarray):
    h, w = processed_img.shape[:2]
    suspicious_sections = []
    reasons = []
    confidence = 0.0

    roi = processed_img[int(h*0.6):h, :]
    edges = cv2.Canny(roi, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stroke_count = len(contours)

    if stroke_count < 10:
        reasons.append("Signature appears too simple (possible fake)")
        confidence += 0.3
        suspicious_sections.append({
            "type": "signature_anomaly",
            "bbox": [0, int(h*0.6), w, h],
            "description": "Low stroke complexity"
        })

    areas = [cv2.contourArea(c) for c in contours]
    if areas and np.std(areas) < 5:
        reasons.append("Signature strokes too uniform (possible printed)")
        confidence += 0.3

    blur_score = cv2.Laplacian(roi, cv2.CV_64F).var()
    if blur_score < 30:
        reasons.append("Signature region is blurred (possible copy-paste)")
        confidence += 0.2

    return {
        "confidence": min(confidence, 1.0),
        "reasons": reasons,
        "sections": suspicious_sections
    }

def generate_forgery_report(extracted_text: str, processed_img: np.ndarray, processed_images: List[str]) -> dict:
    text_report = detect_forgery_rules(extracted_text, processed_images)
    signature_report = detect_signature_forgery(processed_img)

    combined_confidence = min(
        text_report["overall_forgery_confidence"] + signature_report["confidence"], 1.0
    )
    combined_reasons = text_report["flagging_reasons"] + signature_report["reasons"]
    combined_sections = text_report["suspicious_sections"] + signature_report["sections"]

    return {
        "overall_forgery_confidence": combined_confidence,
        "is_flagged_for_review": combined_confidence > 0.5,
        "flagging_reasons": combined_reasons,
        "suspicious_sections": combined_sections,
        "sub_reports": {
            "text_report": text_report,
            "signature_report": signature_report
        }
    }

# --- Streamlit UI ---
st.set_page_config(page_title="OCR + Forgery Detection", layout="wide")
st.title("OCR + Forgery Detection (EasyOCR only)")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    pil = Image.open(uploaded_file).convert("RGB")
    img_cv = np.array(pil)[:, :, ::-1].copy()
    processed_img = preprocess(img_cv)

    st.image(processed_img, caption="Preprocessed Image")

    text = ""
    if easyocr_reader:
        try:
            result = easyocr_reader.readtext(processed_img, lang_list=EASYOCR_LANGS)
            text = " ".join([t[1] for t in result])
        except Exception as e:
            logging.error(f"EasyOCR error: {e}")
            text = "(OCR failed)"
    else:
        text = "(EasyOCR not available)"

    st.text_area("Extracted Text", value=text, height=300)

    forgery_report = generate_forgery_report(text, processed_img, [])
    st.subheader("Unified Forgery Report")
    st.json(forgery_report)

    st.download_button(
        label="Download Forgery Report",
        data=io.BytesIO(str(forgery_report).encode("utf-8")),
        file_name=f"{uploaded_file.name}_forgery_report.json",
        mime="application/json"
    )

