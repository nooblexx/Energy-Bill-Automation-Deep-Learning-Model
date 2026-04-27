import sys
from pathlib import Path
from paddleocr import PaddleOCR

pages_dir = sys.argv[1]
ocr_dir   = sys.argv[2]
pdf_stem  = sys.argv[3]

ocr = PaddleOCR(use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False)
for img_path in sorted(Path(pages_dir).glob(f"{pdf_stem}_*.png")):
    result = ocr.predict(input=str(img_path))
    for page in result:
        page.save_to_json(ocr_dir)
print("OCR complete")