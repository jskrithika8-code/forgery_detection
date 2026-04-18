import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from pdf2image import convert_from_path
import pytesseract
import easyocr
import logging
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
import uuid # For generating unique filenames in GCS
import random # For generating random confidence scores for demonstration
import streamlit as st
st.title("Forgery Detection App")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Configuration for Google Cloud Document AI ---
GCP_PROJECT_ID = "YOUR_GCP_PROJECT_ID"
GCP_PROCESSOR_LOCATION = "YOUR_PROCESSOR_LOCATION"
GCP_OCR_PROCESSOR_ID = "YOUR_ENTERPRISE_OCR_PROCESSOR_ID"
GCP_STORAGE_BUCKET_NAME = "YOUR_GCS_BUCKET_NAME"

# --- EasyOCR Language Setup ---
EASYOCR_LANGS = ['en','as','bn','gu','hi','kn','ml','mr','ne','or','pa','ta','te','ur']
try:
    easyocr_reader = easyocr.Reader(EASYOCR_LANGS)
    logging.info(f"EasyOCR reader initialized successfully with languages: {', '.join(EASYOCR_LANGS)}.")
except Exception as e:
    logging.error(f"Error initializing EasyOCR reader: {e}")
    easyocr_reader = None

# --- Tesseract Language Setup ---
TESSERACT_LANG_MAP = {
    'en':'eng','as':'asm','bn':'ben','gu':'guj','hi':'hin','kn':'kan','mai':'mai','mal':'mal',
    'mr':'mar','ne':'nep','or':'ori','pa':'pun','sa':'san','sat':'sat','sd':'snd','ta':'tam',
    'te':'tel','ur':'urd'
}
TESSERACT_COMBINED_LANGS = '+'.join(TESSERACT_LANG_MAP.values())

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {'png','jpg','jpeg','gif','pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Helper function ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# --- Preprocess Function ---
def preprocess(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,11,2)
    return thresh

# --- Forgery Detection ---
def analyze_document_for_forgery(page_number, original_img_cv, ocr_results, all_extracted_text):
    flagging_reasons = []
    suspicious_sections = []
    confidence = random.uniform(0.6,0.95)
    flagged = False

    if "invoice" in all_extracted_text.lower() and "urgent payment" in all_extracted_text.lower():
        flagged = True
        flagging_reasons.append("Suspicious combination of 'invoice' and 'urgent payment'.")
        for (bbox,text,conf) in ocr_results:
            if "urgent payment" in text.lower():
                suspicious_sections.append({
                    "page":page_number,
                    "type":"text_tampering",
                    "description":"Keyword 'urgent payment' detected.",
                    "bbox":[int(x) for x in bbox[0]+bbox[2]],
                    "reason_code":"SUSP_KEYWORDS",
                    "evidence_detail":f"Found '{text}' with confidence {conf:.2f}."
                })
                break

    if ocr_results and len(ocr_results)<5 and page_number>1:
        flagged = True
        flagging_reasons.append("Sparse text layout detected.")
        h,w = original_img_cv.shape[:2]
        suspicious_sections.append({
            "page":page_number,
            "type":"layout_anomaly",
            "description":"Page contains unusually few text elements.",
            "bbox":[0,0,w,h],
            "reason_code":"SPARSE_LAYOUT",
            "evidence_detail":f"{len(ocr_results)} text elements found."
        })

    if random.random()<0.1 and page_number==1:
        flagged = True
        flagging_reasons.append("Potential font inconsistency detected.")
        if ocr_results:
            bbox,text,conf = random.choice(ocr_results)
            suspicious_sections.append({
                "page":page_number,
                "type":"font_inconsistency",
                "description":"Font mismatch simulation.",
                "bbox":[int(x) for x in bbox[0]+bbox[2]],
                "reason_code":"FONT_MISMATCH",
                "evidence_detail":f"OCR identified '{text}' (simulation)."
            })

    return {
        "overall_forgery_confidence":round(confidence,2),
        "is_flagged_for_review":flagged,
        "flagging_reasons":flagging_reasons if flagged else ["No significant forgery indicators found."],
        "suspicious_sections":suspicious_sections
    }

# --- Visual Highlighting ---
def draw_suspicious_sections(img, suspicious_sections, page_number):
    annotated_img = img.copy()
    for section in suspicious_sections:
        if section["page"]==page_number:
            x1,y1,x2,y2 = section["bbox"]
            cv2.rectangle(annotated_img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(annotated_img,section.get("reason_code","Suspicious"),
                        (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    return annotated_img

# --- Flask routes (local OCR and GCP OCR) ---
# ... (upload-and-ocr-local and upload-and-ocr-gcp routes as in your original)
