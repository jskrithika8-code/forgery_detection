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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Configuration for Google Cloud Document AI ---
GCP_PROJECT_ID = "YOUR_GCP_PROJECT_ID" # e.g., "my-document-analysis-12345"
GCP_PROCESSOR_LOCATION = "YOUR_PROCESSOR_LOCATION" # e.g., "us"
GCP_OCR_PROCESSOR_ID = "YOUR_ENTERPRISE_OCR_PROCESSOR_ID" # e.g., "abcdefg12345"
GCP_STORAGE_BUCKET_NAME = "YOUR_GCS_BUCKET_NAME" # e.g., "my-document-uploads"


# --- EasyOCR Language Setup ---
# EasyOCR codes for supported official Indian languages + English
EASYOCR_LANGS = [
    'en', 'as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'ta', 'te', 'ur'
]
try:
    easyocr_reader = easyocr.Reader(EASYOCR_LANGS)
    logging.info(f"EasyOCR reader initialized successfully with languages: {', '.join(EASYOCR_LANGS)}.")
except Exception as e:
    logging.error(f"Error initializing EasyOCR reader with languages {', '.join(EASYOCR_LANGS)}: {e}")
    easyocr_reader = None

# --- Tesseract Language Setup ---
# IMPORTANT: pytesseract.pytesseract.tesseract_cmd must be set if Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # Adjust path as needed

# Mapping of official Indian languages to their Tesseract codes
# Install corresponding Tesseract data packs (e.g., 'hin.traineddata' for Hindi)
TESSERACT_LANG_MAP = {
    'en': 'eng', 'as': 'asm', 'bn': 'ben', 'gu': 'guj', 'hi': 'hin',
    'kn': 'kan', 'mai': 'mai', 'mal': 'mal', 'mr': 'mar', 'ne': 'nep',
    'or': 'ori', 'pa': 'pun', 'sa': 'san', 'sat': 'sat', 'sd': 'snd',
    'ta': 'tam', 'te': 'tel', 'ur': 'urd'
}
# Combined Tesseract languages string for pytesseract (for default or multi-lang)
TESSERACT_COMBINED_LANGS = '+'.join(TESSERACT_LANG_MAP.values())

# --- Configuration for Flask app ---
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logging.info(f"Uploads directory: {UPLOAD_FOLDER}, Outputs directory: {OUTPUT_FOLDER}")

# --- Helper function for allowed file types ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Preprocess Function ---
def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Applies image preprocessing steps to enhance OCR accuracy.
    """
    if img is None:
        logging.warning("Preprocessing received a None image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    logging.info("Image preprocessing completed.")
    return thresh

# --- Function to process a document with Document AI ---
def process_document_with_documentai(gcs_input_uri: str, mime_type: str) -> str:
    """
    Sends a document to Google Cloud Document AI for OCR and returns the full text.
    """
    client = documentai.DocumentProcessorServiceClient()
    processor_name = client.processor_path(GCP_PROJECT_ID, GCP_PROCESSOR_LOCATION, GCP_OCR_PROCESSOR_ID)

    gcs_document = documentai.GcsDocument(gcs_uri=gcs_input_uri, mime_type=mime_type)
    input_config = documentai.DocumentInputConfig(gcs_document=gcs_document)

    request = documentai.ProcessRequest(
        name=processor_name,
        input_config=input_config
    )

    logging.info(f"Sending document {gcs_input_uri} to Document AI processor {GCP_OCR_PROCESSOR_ID}...")
    result = client.process_document(request=request)
    logging.info("Document AI processing complete.")

    return result.document.text


# --- Flask Route for Document Upload and OCR using LOCAL ENGINES (EasyOCR/Tesseract) ---
@app.route('/upload-and-ocr-local', methods=['POST'])
def upload_file_local():
    # User can specify 'lang' (e.g., 'hi', 'mai', 'en,ta') or 'engine' ('easyocr', 'tesseract')
    target_langs_str = request.args.get('lang', 'en') # Default to English
    target_langs = [l.strip() for l in target_langs_str.split(',') if l.strip()]

    # Optionally allow user to force an engine
    force_engine = request.args.get('engine', '').lower()

    if 'file' not in request.files:
        logging.warning("No 'file' part in request.")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logging.warning("No selected file name.")
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        original_filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(filepath)
        logging.info(f"File '{original_filename}' saved to {filepath}")

        all_text = []
        processed_image_paths = []
        
        # Determine if PDF or image
        image_list = []
        is_pdf = original_filename.lower().endswith('.pdf')
        if is_pdf:
            try:
                images_pil = convert_from_path(filepath)
                logging.info(f"Converted PDF '{original_filename}' to {len(images_pil)} images.")
                for i, image_pil in enumerate(images_pil):
                    # Convert PIL Image to OpenCV format (numpy array)
                    img_cv = np.array(image_pil)
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR) # Convert RGB to BGR
                    image_list.append(img_cv)
            except Exception as e:
                logging.error(f"Error converting PDF '{original_filename}': {e}")
                return jsonify({"error": f"Error converting PDF: {e}"}), 500
        else:
            img_cv = cv2.imread(filepath)
            if img_cv is None:
                logging.error(f"Could not read image file {filepath}.")
                return jsonify({"error": "Could not read image file"}), 400
            image_list.append(img_cv)

        for i, img_cv_page in enumerate(image_list):
            processed_img = preprocess(img_cv_page)
            if processed_img is None:
                logging.warning(f"Preprocessing returned None for page {i+1}.")
                all_text.append(f"Page {i+1}: (Preprocessing failed)")
                continue

            ocr_engine_used = "None"
            page_text = ""

            # Try to use specified engine, or choose best available
            if force_engine == 'tesseract':
                ocr_engine_used = "Tesseract"
                tesseract_langs_for_page = "+".join([TESSERACT_LANG_MAP.get(lang, 'eng') for lang in target_langs])
                try:
                    page_text = pytesseract.image_to_string(processed_img, lang=tesseract_langs_for_page)
                    logging.info(f"Tesseract OCR completed for page {i+1}, langs: {tesseract_langs_for_page}.")
                except Exception as e:
                    logging.error(f"Tesseract OCR failed for page {i+1}, langs: {tesseract_langs_for_page}: {e}")
            elif force_engine == 'easyocr':
                ocr_engine_used = "EasyOCR"
                easyocr_langs_for_page = [lang for lang in target_langs if lang in EASYOCR_LANGS]
                if easyocr_langs_for_page and easyocr_reader is not None:
                    try:
                        result = easyocr_reader.readtext(processed_img, lang_list=easyocr_langs_for_page)
                        page_text = " ".join([text[1] for text in result])
                        logging.info(f"EasyOCR completed for page {i+1}, langs: {', '.join(easyocr_langs_for_page)}.")
                    except Exception as e:
                        logging.error(f"EasyOCR failed for page {i+1}, langs: {', '.join(easyocr_langs_for_page)}: {e}")
                else:
                    logging.warning(f"EasyOCR not available or no requested language supported by EasyOCR for page {i+1}.")
            else: # Dynamic selection based on language support
                # Prioritize EasyOCR if it supports the primary requested language
                preferred_easyocr_lang = next((lang for lang in target_langs if lang in EASYOCR_LANGS), None)
                if preferred_easyocr_lang and easyocr_reader is not None:
                    ocr_engine_used = "EasyOCR"
                    try:
                        result = easyocr_reader.readtext(processed_img, lang_list=[preferred_easyocr_lang])
                        page_text = " ".join([text[1] for text in result])
                        logging.info(f"EasyOCR (dynamic) completed for page {i+1}, lang: {preferred_easyocr_lang}.")
                    except Exception as e:
                        logging.error(f"EasyOCR (dynamic) failed for page {i+1}, lang: {preferred_easyocr_lang}: {e}")
                        ocr_engine_used = "None" # Reset if failed, try Tesseract next

                # If EasyOCR didn't run or failed, try Tesseract
                if not page_text or ocr_engine_used == "None":
                    tesseract_langs_for_page = "+".join([TESSERACT_LANG_MAP.get(lang, 'eng') for lang in target_langs])
                    if tesseract_langs_for_page:
                        ocr_engine_used = "Tesseract"
                        try:
                            page_text = pytesseract.image_to_string(processed_img, lang=tesseract_langs_for_page)
                            logging.info(f"Tesseract (dynamic) completed for page {i+1}, langs: {tesseract_langs_for_page}.")
                        except Exception as e:
                            logging.error(f"Tesseract (dynamic) failed for page {i+1}, langs: {tesseract_langs_for_page}: {e}")
                            ocr_engine_used = "None"
            
            all_text.append(f"Page {i+1} (Engine: {ocr_engine_used}):\n{page_text}")

            # Save the processed image for inspection
            output_img_name = f"processed_{os.path.basename(original_filename).replace('.', '_')}_page_{i}.png" if is_pdf else f"processed_{os.path.basename(original_filename)}"
            output_img_path = os.path.join(OUTPUT_FOLDER, output_img_name)
            cv2.imwrite(output_img_path, processed_img)
            processed_image_paths.append(output_img_name)
            logging.info(f"Processed image saved to {output_img_path}")


        os.remove(filepath)
        logging.info(f"Original file '{filepath}' removed.")

        return jsonify({
            "original_filename": original_filename,
            "ocr_engine_strategy": "Hybrid Local (EasyOCR/Tesseract)",
            "requested_languages": target_langs,
            "extracted_text": "\n\n".join(all_text),
            "processed_images_available": processed_image_paths
        }), 200
    else:
        logging.warning(f"File '{file.filename}' is not allowed.")
        return jsonify({"error": "File type not allowed"}), 400


# --- Flask Route for Document Upload and OCR using GCP Document AI ---
@app.route('/upload-and-ocr-gcp', methods=['POST'])
def upload_file_gcp():
    if 'file' not in request.files:
        logging.warning("No 'file' part in request.")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logging.warning("No selected file name.")
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        original_filename = file.filename
        
        mime_type = file.mimetype if file.mimetype else 'application/octet-stream'
        if original_filename.lower().endswith('.pdf'):
            mime_type = 'application/pdf'
        elif original_filename.lower().endswith(('.png', '.jpeg', '.jpg')):
            mime_type = 'image/jpeg' # Or image/png, etc.

        storage_client = storage.Client()
        bucket = storage_client.bucket(GCP_STORAGE_BUCKET_NAME)
        gcs_filename = f"uploads/{uuid.uuid4()}_{original_filename}" 
        blob = bucket.blob(gcs_filename)
        
        temp_filepath_local = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(temp_filepath_local)
        
        blob.upload_from_filename(temp_filepath_local, content_type=mime_type)
        gcs_uri = f"gs://{GCP_STORAGE_BUCKET_NAME}/{gcs_filename}"
        logging.info(f"File '{original_filename}' uploaded to GCS: {gcs_uri}")

        extracted_text = ""
        try:
            extracted_text = process_document_with_documentai(gcs_uri, mime_type)
            logging.info(f"Successfully extracted text for '{original_filename}' using Document AI.")
        except Exception as e:
            logging.error(f"Error processing '{original_filename}' with Document AI: {e}")
            return jsonify({"error": f"Document AI processing failed: {e}"}), 500
        finally:
            blob.delete()
            os.remove(temp_filepath_local)
            logging.info(f"Cleaned up temporary GCS object '{gcs_uri}' and local file '{temp_filepath_local}'.")
        
        return jsonify({
            "original_filename": original_filename,
            "ocr_engine": "Google Cloud Document AI",
            "extracted_text": extracted_text,
            "gcp_processor_id": GCP_OCR_PROCESSOR_ID
        }), 200
    else:
        logging.warning(f"File '{file.filename}' is not allowed.")
        return jsonify({"error": "File type not allowed"}), 400

# --- Route to serve processed images (for inspection) ---
@app.route('/outputs/<filename>')
def download_processed_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    # IMPORTANT: UNCOMMENT AND ADJUST THE PATH TO YOUR TESSERACT EXECUTABLE
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # Example for macOS/Linux
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows
    
    # Run the Flask app
    logging.info("Starting Flask application...")
    app.run(debug=True, port=5000)

{
  "original_filename": "example.pdf",
  "overall_forgery_confidence": 0.85,  // e.g., 0-1 score
  "is_flagged_for_review": true,
  "flagging_reasons": [
    "Detected font inconsistencies in header on Page 1.",
    "Unusual layout of signature block on Page 2.",
    "High level of compression artifacts on Page 1 (potentially re-saved/edited).",
    "Extracted text does not match expected database record for 'Invoice Number'."
  ],
  "suspicious_sections": [
    {
      "page": 1,
      "type": "font_anomaly",
      "description": "Font mismatch detected. Expected 'Arial', found 'Times New Roman' in bounding box [x1, y1, x2, y2].",
      "bbox": [100, 50, 300, 80], // Bounding box for visual highlighting
      "reason_code": "FONT_MISMATCH",
      "evidence_detail": "OCR identified 'The Document Title' with font 'Times New Roman', whereas template expects 'Arial'."
    },
    {
      "page": 2,
      "type": "layout_anomaly",
      "description": "Signature block shifted by 15 pixels to the right compared to template.",
      "bbox": [200, 400, 500, 450],
      "reason_code": "LAYOUT_SHIFT",
      "evidence_detail": "Bounding box for 'Signature' field is [200,400,500,450], template expects [185,400,485,450]."
    },
    // ... more suspicious sections
  ],
  "extracted_text": "...", // The full extracted text
  "processed_images_available": ["processed_example_page_1.png", "..."] // For visual output
}

def detect_forgery_rules(extracted_text: str, processed_images: list) -> dict:
    """
    Simple rule-based forgery detection.
    Returns a JSON-like dictionary with confidence score, reasons, and suspicious sections.
    """

    suspicious_sections = []
    flagging_reasons = []
    confidence = 0.0

    # Rule 1: Font mismatch (simulate check)
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

    # Rule 2: Layout anomaly (simulate bounding box shift)
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

    # Rule 3: Compression artifacts (simulate check)
    if any("processed" in img for img in processed_images):
        flagging_reasons.append("High compression artifacts detected.")
        confidence += 0.2

    # Rule 4: Text mismatch (simulate database check)
    if "Invoice Number" not in extracted_text:
        flagging_reasons.append("Extracted text does not match expected record.")
        confidence += 0.2

    return {
        "overall_forgery_confidence": min(confidence, 1.0),
        "is_flagged_for_review": confidence > 0.5,
        "flagging_reasons": flagging_reasons,
        "suspicious_sections": suspicious_sections
    }

forgery_report = detect_forgery_rules(
    extracted_text="\n\n".join(all_text),
    processed_images=processed_image_paths
)

return jsonify({
    "original_filename": original_filename,
    "ocr_engine_strategy": "Hybrid Local (EasyOCR/Tesseract)",
    "requested_languages": target_langs,
    "extracted_text": "\n\n".join(all_text),
    "processed_images_available": processed_image_paths,
    "forgery_report": forgery_report
}), 200

