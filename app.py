from __future__ import annotations

import sys, json, subprocess, os
import pandas as pd
import streamlit as st
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))
from extractor import extract_utility_bill_data
from preprocessing import find_stop_line, score_page
from ocr_utils import pdf2png, extract_text_lines

RAW_PDFS = "raw_pdfs"
PAGES_DIR = "pages"
OCR_DIR = "ocr_output"
TRAIN_ANN = "training_data/annotations_raw"
MODEL_DIR = "models/layoutlmv3-bills"

def main() -> None:
    # Page Title
    st.title("BEAM - Billing Extraction & Analysis Model")
    # Project Pipeline Description
    with st.sidebar:
        st.header("Pipeline")
        st.markdown(
            """
            1. PDF to image
            2. PaddleOCR to text and boxes (Currently debugging specific DLL error)
            3. LayoutLM token classification
            4. Optional row aggregation
            5. Excel-ready output
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
    run_extraction = st.button("Run Extraction", type="primary")
    if not run_extraction:
        st.caption("Click `Run Extraction` to process the bill.")
        return
    
    if run_extraction:
        # Step 1: Save PDF to PDF path
        pdf_path = Path("raw_pdfs") / uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)
        st.success(f"Saved PDF: {pdf_path}")
        # Step 2: Convert pdf to images
        with st.spinner("Converting PDF to images..."):
            saved_images = pdf2png(pdf_path, PAGES_DIR, dpi=200)
        st.success(f"Converted {len(saved_images)} pages")
        # Step 3 - Load existing OCR output
        with st.spinner("Loading OCR output..."):
            existing_files = list(Path(OCR_DIR).glob(f"{pdf_stem}_*_res.json"))
        if not existing_files:
            st.error(f"No OCR output found for {pdf_stem}. Please preprocess this bill first.")
            st.stop()
        st.success(f"OCR complete ✓ ({len(existing_files)} pages found)")
        # Step 3: Run OCR on each page
        # with st.spinner("Running OCR..."):
        #     result = subprocess.run([sys.executable, "ocr_runner.py", PAGES_DIR, OCR_DIR, pdf_stem],
        #                             capture_output=True,
        #                             text=True
        #     )
        #     if result.returncode != 0:
        #         st.error(f"OCR failed: {result.stderr}")
        #         st.stop
        # st.success("OCR complete")
        # Step 4: Score pages and pick best
        with st.spinner("Selecting best page..."):
            page_scores = []
            for fname in sorted(os.listdir(OCR_DIR)):
                if not fname.startswith(pdf_stem) or not fname.endswith("_res.json"):
                    continue
                path = Path(OCR_DIR) / fname
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
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
             # Show bill image to user
            base = best_page['fname'].replace("_res.json", "")
            img_path = Path(PAGES_DIR) / (base + ".png")
            if img_path.exists():
                st.image(str(img_path), caption=f"Best page: {base}", width=True)
            else:
                st.warning("Page image not found in pages/ folder")
            st.success(f"Best page selected ✓ ({best_page['fname']})")
        # # Step 5 - Run model inference
        # ann_path    = Path(TRAIN_ANN) / best_page['fname']
        # output_path = Path("output") / (base + "_results.json")
        # with st.spinner("Running model inference..."):
        #     result = subprocess.run(
        #         [sys.executable, "model_runner.py",
        #         str(ann_path),
        #         str(img_path),
        #         MODEL_DIR,
        #         str(output_path)],
        #         capture_output=True,
        #         text=True
        #     )
        #     if result.returncode != 0:
        #         st.error(f"Model failed: {result.stderr}")
        #         st.stop()
        # st.success("Extraction complete ✓")
        
        # Step 5 - Load mock extraction results
        with st.spinner("Running model inference..."):
            # Mock extracted values for demo
            extracted = {
                "B-KWH_USAGE": "58,200",
                "B-KWH_COST":  "$3,271.83",
                "B-KW_USAGE":  "239.0",
                "B-KW_COST":   "$608.73",
                "B-TOTAL_COST": "$2,841.38"
            }
        st.success("Extraction complete ✓")
        # Display results
        st.subheader("Extracted Values")
        display_data = {
            "Field": ["kWh Usage", "kWh Cost", "kW Usage", "kW Cost", "Total Charge"],
            "Value": [
                extracted["B-KWH_USAGE"],
                extracted["B-KWH_COST"],
                extracted["B-KW_USAGE"],
                extracted["B-KW_COST"],
                extracted["B-TOTAL_COST"]
            ]
        }
        df = pd.DataFrame(display_data)
        st.table(df, width=True)

    # with st.spinner("Running extraction pipeline..."):
    #     result = extract_utility_bill_data
    #         file_name=uploaded_file.name,
    #         file_bytes=io.BytesIO(file_bytes).getvalue(),
    #     )

    # render_dataframe_tab(result)

if __name__ == "__main__":
    main()
