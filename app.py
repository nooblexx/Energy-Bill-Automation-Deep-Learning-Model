from __future__ import annotations

import sys, json, subprocess, os
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForTokenClassification

from utils.preprocessing import find_stop_line, score_page
from utils.ocr_utils import pdf2png, extract_text_lines, convert_file
from utils.labeling import group_lines_by_row, label_row, flatten_labels

RAW_PDFS = "data/raw_pdfs"
PAGES_DIR = "data/bill_png"
OCR_DIR = "data/ocr_json"
MODEL_DIR = "models/layoutlmv3-bills"

# Handle caching to not create function everytime
@st.cache_resource
def get_ocr_runner():
    def run(pages_dir, ocr_dir, pdf_stem):
        result = subprocess.run([sys.executable,"scripts/ocr_runner.py", pages_dir, ocr_dir,pdf_stem],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    return run

# Load model once to use for reruns
@st.cache_resource
def load_model(model_dir):
    processor = AutoProcessor.from_pretrained(model_dir, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    return processor, model

# Model inference function
def run_inference(ocr_path, img_path, processor, model):
    # Load image dimensions
    with Image.open(img_path) as img:
        page_width, page_height = img.size
    # Convert to word-level tokens
    entries = convert_file(ocr_path)
    # Group by row
    rows  = group_lines_by_row(entries)
    labeled_rows = [label_row(row) for row in rows]
    page_entry = flatten_labels(
        page_id      = ocr_path.stem,
        image_path   = img_path.as_posix(),
        labeled_rows = labeled_rows,
        page_width   = page_width,
        page_height  = page_height
    )
    # Run inference
    image = Image.open(img_path).convert("RGB")
    encoding = processor(
        images = image,
        text = page_entry["tokens"],
        boxes = page_entry["bboxes"],
        truncation = True,
        max_length = 512,
        padding = "max_length",
        return_tensors = "pt"
    )
    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    id2label    = model.config.id2label
    # Consider extra tokens added for seperation
    word_predictions = predictions[1:len(page_entry["tokens"]) + 1]
    # Extract results
    results = {}
    print("\n{'TOKEN':<20} {'LABEL'}")
    print("-" * 35)
    for token, pred in zip(page_entry["tokens"], word_predictions):
        label = id2label[pred]
        print(f"{token:<20} {label}")
        if label != "O":
            if label not in results:
                results[label] = []
            results[label].append(token)
    debug_path = Path("data") / "debug_page_entry.json"
    with open(debug_path, "w") as f:
        json.dump(page_entry, f, indent=2)
    return results
# Load model on page start up to avoid reloading everytime
processor, model = load_model(MODEL_DIR)

with st.sidebar:
    if model is not None:
        st.success("Model loaded ✓")
    else:
        st.error("Model failed to load")
def main() -> None:
    # Hold session state to avoid reloading
    if "best_page" not in st.session_state:
        st.session_state.best_page = None
    if "img_path" not in st.session_state:
        st.session_state.img_path = None
    if "ocr_path" not in st.session_state:
        st.session_state.ocr_path = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "run_extraction" not in st.session_state:
        st.session_state.run_extraction = False
    # Page Title
    st.title("BEAM - Billing Extraction & Analysis Model")
    # Project Pipeline Description
    with st.sidebar:
        st.header("Pipeline")
        st.markdown(
            """
            1. PDF to image
            
            :gray[PaddleOCR is able to accept PDFs to perform recognition on but we convert to png to get 200 dpi on an image]

            2. PaddleOCR to text and boxes

            :gray[Reads in PNG and converts to json file holding bounding box, confidence scores, and text]

            3. LayoutLM token classification

            :gray[Multi-model transformer for document AI using word embeddings and a linear embedding containing image patches to match tokens with vision AI]

            4. Excel-ready output :red[Future Work]

            :gray[Use provided json from LayoutLM to feed into numerous.ai, a generative AI model built to work with spreadsheets]
            """
        )
        st.info("Current mode: UI prototype with mock extraction output.")
    # Hold file to submit into model
    uploaded_file = st.file_uploader(
        "Upload a utility bill PDF",
        type=["pdf"],
        help="Image upload can be added later using the same extraction interface.",
    )
    # When file is not uploaded yet
    if uploaded_file is None:
        st.info("Upload a PDF to get started!")
        return
    # Return file size inforation
    file_bytes = uploaded_file.getvalue()
    pdf_stem = Path(uploaded_file.name).stem
    st.success(f"Loaded `{uploaded_file.name}`")
    # Optional preview that website works
    with st.expander("Uploaded File Details", expanded=False):
        st.json(
            {
                "name": uploaded_file.name,
                "type": uploaded_file.type,
                "size_bytes": len(file_bytes),
            }
        )
    # Run Model Button
    if st.button("Run Extraction", type="primary"):
        st.session_state.run_extraction = True
    
    if st.session_state.run_extraction:
        # Step 1: Save PDF to PDF path
        pdf_path = Path("data/raw_pdfs") / uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)
        st.success(f"Saved PDF: {pdf_path}")

        # Step 2: Convert pdf to images
        with st.spinner("Converting PDF to images..."):
            saved_images = pdf2png(pdf_path, PAGES_DIR, dpi=200)
        st.success(f"Converted {len(saved_images)} pages")

        # Step 3 - Load existing OCR output
        ocr_runner = get_ocr_runner()
        with st.spinner("Running OCR..."):
          ocr_output = ocr_runner(PAGES_DIR, OCR_DIR, pdf_stem)
        st.success(f"OCR complete ✓")

        # Step 4: Score pages and pick best
        with st.spinner("Selecting best page..."):
            page_scores = []

            for fname in sorted(os.listdir(OCR_DIR)):
                if not fname.startswith(pdf_stem) or not fname.endswith("_res.json"):
                    continue
                path = Path(OCR_DIR) / fname
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Repeat find stop index
                lines = extract_text_lines(data)
                stop_idx = find_stop_line(lines)
                lines = lines[:stop_idx]

                keyword_density, numeric_density = score_page(lines)
                page_scores.append({
                    "fname":            fname,
                    "keyword_density":  keyword_density,
                    "numeric_density":  numeric_density,
                })
            best_page = max(page_scores, key=lambda x: x["keyword_density"])
        # After Best page has been selected show the bill image to the user
        base = best_page['fname'].replace("_res.json", "")
        img_path = Path(PAGES_DIR) / (base + ".png")
        st.subheader("Step 4: Best Page Selected")
        # Show image
        if img_path.exists():
            st.image(str(img_path), caption=f"Page selected for model inference: {base}")
        else:
            st.warning("Page image not found in pages/ folder")
        st.info("Please confirm this is the correct billing summary page before extraction.")
        ocr_path = Path(OCR_DIR) / best_page['fname']
        # Store session states
        st.session_state.img_path = img_path
        st.session_state.ocr_path = ocr_path

        # Provide fallback incase of mistakes in OCR page selection
        col1, col2 = st.columns(2)
        with col1:
          confirm = st.button("✓ Confirm Page")
        with col2:
            if st.button("Choose Another Page"):
                st.session_state.best_page = None
                st.session_state.img_path = None
                st.session_state.ocr_path = None
                st.rerun()
        if confirm:
            # Step 5 - Run model inference
            with st.spinner("Running model inference..."):
                try:
                    results = run_inference(st.session_state.ocr_path, st.session_state.img_path, processor, model)
                    st.session_state.results = results
                except Exception as e:
                    st.error(f"Model failed: {e}")
                    st.stop()
            st.success("Extraction complete ✓")
            # Join tokens per label
            display_data = {"Field": [], "Value": []}
            label_map = {
                "B-KWH_USAGE": "kWh Usage",
                "B-KWH_COST":  "kWh Cost",
                "B-KW_USAGE":  "kW Usage",
                "B-KW_COST":   "kW Cost",
                "B-TOTAL_COST": "Total Cost"
            }
            for label, friendly_name in label_map.items():
                value = " ".join(results.get(label, ["N/A"]))
                display_data["Field"].append(friendly_name)
                display_data["Value"].append(value)
            df = pd.DataFrame(display_data)
            st.subheader("Extracted Values")
            st.table(df)
        else:
            st.caption("Please choose another PDF instead")

if __name__ == "__main__":
    main()
