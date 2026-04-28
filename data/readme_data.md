# Dataset

The data used to train this model was directly from Lehigh Industrial Training and Assessment Center. Due to the fact that we work with many companies, some of which we sign NDA's for, I cannot put the entire training dataset on github. The PDFs provided are directly from past audits performed by the ITAC.

## Data Instrunctions

Folder structure is provided at the end. Folder structure is dense to help with debugging and labeling tools, most of it not required for model to work. For training and inference only three folders are required to use.
- raw_pdfs contain PDFs where user would insert data into
- training_data/word_level_tokens contains json files ready for training/testing
- training_data/images contains images to pass into LayoutLM


### Supported Utility Companies

- PPL
- PSEG
- conEd

### Future Support

- MSEG
- UGI
- Sewage Companies


### Folder Structure

Energy_Bill_Automation/
├── data/
│ ├── raw_pdfs/             → original PDF bills
│ ├── bill_png/             → clean PNG images from PyMuPDF at 200 DPI
│ ├── ocr_img/              → PaddleOCR annotated images with bounding boxes
│ ├── ocr_json/             → PaddleOCR JSON results for all pages
│ └── training_data/
│  ├── line_level_tokens/   → best page JSON untouched copy
│  ├── word_level_tokens/   → final flattened JSON ready for LayoutLMv3
│  ├── images/              → best page PNG for LayoutLMv3
│  └── label_studio/        → Label Studio import/export files
├── utils/
│ ├── **init**.py
│ ├── bbox_utils.py         → poly_to_bbox, split_bbox_horizontally, normalize_bbox
│ ├── preprocessing.py      → is_numeric_token, is_keyword_token, find_stop_line, score_page
│ ├── ocr_utils.py          → extract_text_lines, convert_file, pdf2png, convert_from_label_studio
│ └── labeling.py           → group_lines_by_row, detect_row_type, label_row, flatten_labels
├── scripts/
│ └── ocr_runner.py         → standalone OCR script for Streamlit
├── Energy-Automation.ipynb → main Jupyter notebook
└── app.py                  → Streamlit demo
