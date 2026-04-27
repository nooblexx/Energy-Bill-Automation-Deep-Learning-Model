# Dataset

The data used to train this model was directly from Lehigh Industrial Training and Assessment Center. Due to the fact that we work with many companies, some of which we sign NDA's for, I cannot put the entire training dataset on github. The PDFs provided are directly from past audits performed by the ITAC.

### Supported Utility Companies

- PPL
- PSEG

### Future Support

- MSEG
- UGI
- Sewage Companies

### Folder Structure

Energy_Bill_Automation/
├── data/
│ ├── raw_pdfs/ → original PDF bills
│ ├── pages/ → clean PNG images from PyMuPDF at 200 DPI
│ ├── ocr_pages/ → PaddleOCR annotated images with bounding boxes
│ ├── ocr_output/ → PaddleOCR JSON results for all pages
│ └── training_data/
│  ├── annotations_raw/ → best page JSON untouched copy
│  ├── annotations_words/ → final flattened JSON ready for LayoutLMv3
│  ├── images/ → best page PNG for LayoutLMv3
│  └── label_studio/ → Label Studio import/export files
├── models/
│ └── layoutlmv3-bills/ → saved fine-tuned model
├── utils/
│ ├── **init**.py
│ ├── bbox_utils.py → poly_to_bbox, split_bbox_horizontally, normalize_bbox
│ ├── preprocessing.py → is_numeric_token, is_keyword_token, find_stop_line, score_page
│ ├── ocr_utils.py → extract_text_lines, convert_file, pdf2png, convert_from_label_studio
│ └── labeling.py → group_lines_by_row, detect_row_type, label_row, flatten_labels
├── scripts/
│ ├── ocr_runner.py → standalone OCR script for Streamlit
│ └── model_runner.py → standalone inference script for Streamlit
├── Energy-Automation.ipynb → main Jupyter notebook
└── app.py → Streamlit demo
