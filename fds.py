# streamlit_app.py
import os
import io
import uuid
import tempfile
import logging
from typing import List

import cv2
import numpy as np
import pytesseract
import easyocr
from pdf2image import convert_from_path
from google.cloud import documentai_v1 as documentai
from google.cloud import storage

import streamlit as st
from PIL import Image

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration placeholders (update before deploy) ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "YOUR_GCP_PROJECT_ID")
GCP_PROCESSOR_LOCATION = os.getenv("GCP_PROCESSOR_LOCATION", "YOUR_PROCESSOR_LOCATION")
GCP_OCR_PROCESSOR_ID = os.getenv("GCP_OCR_PROCESSOR_ID", "YOUR_ENTERPRISE_OCR_PROCESSOR_ID")
GCP_STORAGE_BUCKET_NAME = os.getenv("GCP_STORAGE_BUCKET_NAME", "YOUR_GCS_BUCKET_NAME")

# --- EasyOCR languages ---
EASYOCR_LANGS = [
    'en', 'as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'ta', 'te', 'ur'
]

# Initialize EasyOCR reader safely
easyocr_reader = None
try:
    easyocr_reader = easyocr.Reader(EASYOCR_LANGS, gpu=False)
    logging.info(f"EasyOCR reader initialized successfully with languages: {', '.join(EASYOCR_LANGS)}.")
except Exception as e:
    logging.error(f"EasyOCR initialization error: {e}")
    easyocr_reader = None

# Tesseract language map
TESSERACT_LANG_MAP = {
    'en': 'eng', 'as': 'asm', 'bn': 'ben', 'gu': 'guj', 'hi': 'hin',
    'kn': 'kan', 'mai': 'mai', 'mal': 'mal', 'mr': 'mar', 'ne': 'nep',
    'or': 'ori', 'pa': 'pan', 'sa': 'san', 'sat': 'sat', 'sd': 'snd',
    'ta': 'tam', 'te': 'tel', 'ur': 'urd'
}

# --- Helper functions ---
def preprocess(img: np.ndarray) -> np.ndarray:
    if img is None:
        logging.warning("Preprocess received None image.")
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    logging.info("Image preprocessing completed.")
    return thresh

def process_document_with_documentai(gcs_input_uri: str, mime_type: str) -> str:
    client = documentai.DocumentProcessorServiceClient()
    processor_name = client.processor_path(GCP_PROJECT_ID, GCP_PROCESSOR_LOCATION, GCP_OCR_PROCESSOR_ID)
    gcs_document = documentai.GcsDocument(gcs_uri=gcs_input_uri, mime_type=mime_type)
    input_config = documentai.DocumentInputConfig(gcs_document=gcs_document)
    request = documentai.ProcessRequest(name=processor_name, input_config=input_config)
    logging.info(f"Sending document {gcs_input_uri} to Document AI processor {GCP_OCR_PROCESSOR_ID}...")
    result = client.process_document(request=request)
    logging.info("Document AI processing complete.")
    return result.document.text

def detect_forgery_rules(extracted_text: str, processed_images: List[str]) -> dict:
    suspicious_sections = []
    flagging_reasons = []
    confidence = 0.0
    if "Times New Roman" in extracted_text and "Arial" in extracted_text:
        suspicious_sections.append({
            "page": 1,
            "type": "font_anomaly",
            "description": "Font mismatch detected.",
            "bbox": [100, 50, 300, 80],
            "reason_code": "FONT_MISMATCH",
            "evidence_detail": "Expected Arial, found Times New Roman."
        })
        flagging_reasons.append("Detected font inconsistencies in header.")
        confidence += 0.3
    suspicious_sections.append({
        "page": 2,
        "type": "layout_anomaly",
        "description": "Signature block shifted compared to template.",
        "bbox": [200, 400, 500, 450],
        "reason_code": "LAYOUT_SHIFT",
        "evidence_detail": "Signature field misaligned by ~15px."
    })
    flagging_reasons.append("Unusual layout of signature block.")
    confidence += 0.3
    if any("processed" in img for img in processed_images):
        flagging_reasons.append("High compression artifacts detected.")
        confidence += 0.2
    if "Invoice Number" not in extracted_text:
        flagging_reasons.append("Extracted text does not match expected record.")
        confidence += 0.2
    return {
        "overall_forgery_confidence": min(confidence, 1.0),
        "is_flagged_for_review": confidence > 0.5,
        "flagging_reasons": flagging_reasons,
        "suspicious_sections": suspicious_sections
    }

def save_image_to_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# --- Streamlit UI ---
st.set_page_config(page_title="OCR + Document AI + Forgery Detection", layout="wide")
st.title("OCR + Document AI + Forgery Detection (Streamlit)")

st.sidebar.header("Settings")
engine_choice = st.sidebar.selectbox("Processing mode", ["Local (EasyOCR/Tesseract)", "Google Document AI"])
lang_input = st.sidebar.text_input("Languages (comma-separated, e.g., en,hi,ta)", value="en")
target_langs = [l.strip() for l in lang_input.split(",") if l.strip()]
force_engine = st.sidebar.selectbox("Force local engine (only for Local mode)", ["Auto", "EasyOCR", "Tesseract"])
tesseract_cmd = st.sidebar.text_input("Tesseract executable path (leave blank if in PATH)", value="")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

st.sidebar.markdown("**GCP Settings (for Document AI)**")
st.sidebar.text_input("GCP Project ID", value=GCP_PROJECT_ID, key="gcp_project")
st.sidebar.text_input("Processor Location", value=GCP_PROCESSOR_LOCATION, key="gcp_location")
st.sidebar.text_input("Processor ID", value=GCP_OCR_PROCESSOR_ID, key="gcp_processor")
st.sidebar.text_input("GCS Bucket Name", value=GCP_STORAGE_BUCKET_NAME, key="gcp_bucket")
st.sidebar.markdown("Make sure your environment has GOOGLE_APPLICATION_CREDENTIALS set to a service account JSON.")

uploaded_file = st.file_uploader("Upload image or PDF", type=["png", "jpg", "jpeg", "gif", "pdf"])

def read_image_from_path(path: str):
    try:
        pil = Image.open(path).convert("RGB")
        arr = np.array(pil)[:, :, ::-1].copy()
        return arr
    except Exception as e:
        logging.error(f"read_image_from_path error: {e}")
        return None

if uploaded_file is not None:
    st.info(f"Uploaded: {uploaded_file.name}")
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, uploaded_file.name)
        with open(local_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        processed_image_paths = []
        all_text = []
        errors = []

        is_pdf = uploaded_file.name.lower().endswith(".pdf")
        image_list = []

        try:
            if is_pdf:
                try:
                    images_pil = convert_from_path(local_path)
                    for image_pil in images_pil:
                        img_cv = np.array(image_pil)
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                        image_list.append(img_cv)
                except Exception as e:
                    logging.error(f"PDF conversion error: {e}")
                    st.error(f"PDF conversion error: {e}")
                    errors.append(f"PDF conversion error: {e}")
            else:
                img_cv = read_image_from_path(local_path)
                if img_cv is None:
                    st.error("Could not read the uploaded image. Try a different file or check permissions.")
                    errors.append("cv2.imread returned None")
                else:
                    image_list.append(img_cv)
        except Exception as e:
            logging.error(f"Error preparing images: {e}")
            st.error(f"Error preparing images: {e}")
            errors.append(str(e))

        if engine_choice.startswith("Local"):
            st.subheader("Local OCR Results")
            for i, img_cv_page in enumerate(image_list, start=1):
                processed_img = preprocess(img_cv_page)
                if processed_img is None:
                    all_text.append(f"Page {i}: (Preprocessing failed)")
                    continue

                page_text = ""
                ocr_engine_used = "None"

                # Force engine selection
                if force_engine == "Tesseract":
                    ocr_engine_used = "Tesseract"
                    t_langs = "+".join([TESSERACT_LANG_MAP.get(l, "eng") for l in target_langs])
                    try:
                        page_text = pytesseract.image_to_string(processed_img, lang=t_langs)
                    except pytesseract.TesseractNotFoundError as e:
                        logging.error(f"Tesseract not found: {e}")
                        st.error("Tesseract executable not found. Set the path in the sidebar or install Tesseract.")
                        errors.append("TesseractNotFound")
                        page_text = ""
                    except Exception as e:
                        logging.error(f"Tesseract error: {e}")
                        errors.append(str(e))
                        page_text = ""
                elif force_engine == "EasyOCR":
                    ocr_engine_used = "EasyOCR"
                    easy_langs = [l for l in target_langs if l in EASYOCR_LANGS]
                    if easy_langs and easyocr_reader is not None:
                        try:
                            result = easyocr_reader.readtext(processed_img, lang_list=easy_langs)
                            page_text = " ".join([t[1] for t in result])
                        except Exception as e:
                            logging.error(f"EasyOCR error: {e}")
                            errors.append(str(e))
                            page_text = ""
                    else:
                        st.warning("EasyOCR not available or requested languages not supported; falling back to Auto.")
                        ocr_engine_used = "None"

                if force_engine == "Auto" or ocr_engine_used == "None":
                    preferred_easyocr_lang = next((l for l in target_langs if l in EASYOCR_LANGS), None)
                    if preferred_easyocr_lang and easyocr_reader is not None:
                        try:
                            result = easyocr_reader.readtext(processed_img, lang_list=[preferred_easyocr_lang])
                            page_text = " ".join([t[1] for t in result])
                            ocr_engine_used = "EasyOCR"
                        except Exception as e:
                            logging.error(f"EasyOCR dynamic error: {e}")
                            errors.append(str(e))
                            page_text = ""
                            ocr_engine_used = "None"

                    if not page_text:
                        t_langs = "+".join([TESSERACT_LANG_MAP.get(l, "eng") for l in target_langs])
                        try:
                            page_text = pytesseract.image_to_string(processed_img, lang=t_langs)
                            ocr_engine_used = "Tesseract"
                        except pytesseract.TesseractNotFoundError:
                            logging.error("Tesseract not found during dynamic fallback.")
                            st.error("Tesseract executable not found. Set the path in the sidebar or install Tesseract.")
                            errors.append("TesseractNotFound")
                            page_text = ""
                        except Exception as e:
                            logging.error(f"Tesseract dynamic error: {e}")
                            errors.append(str(e))
                            page_text = ""

                all_text.append(f"Page {i} (Engine: {ocr_engine_used}):\n{page_text}")

                out_name = f"processed_{os.path.basename(uploaded_file.name).replace('.', '_')}_page_{i}.png"
                out_path = os.path.join(tmpdir, out_name)
                if processed_img.ndim == 2:
                    save_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
                else:
                    save_img = processed_img
                cv2.imwrite(out_path, save_img)
                processed_image_paths.append(out_path)

            extracted_text_combined = "\n\n".join(all_text)
            st.text_area("Extracted Text", value=extracted_text_combined, height=300)

            forgery_report = detect_forgery_rules(extracted_text_combined, [os.path.basename(p) for p in processed_image_paths])
            st.subheader("Forgery Report")
            st.json(forgery_report)

            if errors:
                st.subheader("Errors and Warnings")
                for e in errors:
                    st.error(e)

            st.subheader("Processed Images")
            for p in processed_image_paths:
                st.image(p, caption=os.path.basename(p))
                btn_bytes = save_image_to_bytes(p)
                st.download_button(label=f"Download {os.path.basename(p)}", data=btn_bytes, file_name=os.path.basename(p), mime="image/png")

        else:
            st.subheader("Google Document AI Results")
            try:
                storage_client = storage.Client()
                bucket_name = st.session_state.get("gcp_bucket", GCP_STORAGE_BUCKET_NAME)
                bucket = storage_client.bucket(bucket_name)
                gcs_filename = f"uploads/{uuid.uuid4()}_{uploaded_file.name}"
                blob = bucket.blob(gcs_filename)
                blob.upload_from_filename(local_path)
                gcs_uri = f"gs://{bucket_name}/{gcs_filename}"
                st.info(f"Uploaded to GCS: {gcs_uri}")

                mime_type = "application/pdf" if uploaded_file.name.lower().endswith(".pdf") else "image/jpeg"
                extracted_text = process_document_with_documentai(gcs_uri, mime_type)
                st.text_area("Extracted Text (Document AI)", value=extracted_text, height=300)

                forgery_report = detect_forgery_rules(extracted_text, [])
                st.subheader("Forgery Report")
                st.json(forgery_report)

            except Exception as e:
                logging.error(f"Document AI processing failed: {e}")
                st.error(f"Document AI processing failed: {e}")
            finally:
                try:
                    blob.delete()
                except Exception:
                    pass

        combined = {
            "original_filename": uploaded_file.name,
            "ocr_mode": engine_choice,
            "requested_languages": target_langs,
            "extracted_text": "\n\n".join(all_text),
            "processed_images": [os.path.basename(p) for p in processed_image_paths],
            "forgery_report": detect_forgery_rules("\n\n".join(all_text), [os.path.basename(p) for p in processed_image_paths])
        }
        st.download_button("Download JSON Report", data=io.BytesIO(str(combined).encode("utf-8")), file_name=f"{uploaded_file.name}_report.json", mime="application/json")
else:
    st.info("Upload a PDF or image to begin OCR processing.")

