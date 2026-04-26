import re, json
import fitz
from pathlib import Path
from preprocessing import find_stop_line
from bbox_utils import poly_to_bbox, split_bbox_horizontally

split_re = re.compile(r"\S+")

# Function to extract text from each line
def extract_text_lines(data):
    if isinstance(data, dict):
        return data.get("rec_texts") or data.get("texts") or []
    if isinstance(data, list):
        # maybe list of [box, text] or [[x1,y1,...], "text"]
        texts = []
        for it in data:
            if isinstance(it, list) and len(it) >= 2 and isinstance(it[-1], str):
                texts.append(it[-1])
        return texts
    return []

# Convert from line-level tokenization to word-level tokenization
def convert_file(input_path):
    entries = []
    obj = json.loads(input_path.read_text())
    rec_texts = obj.get("rec_texts", [])
    # Find stop index again to be safe
    stop_idx = find_stop_line(rec_texts)
    rec_texts = rec_texts[:stop_idx]
    rec_polys = obj.get("rec_polys", [])[:stop_idx]
    rec_scores = obj.get("rec_scores", [])[:stop_idx]
    
    for i, text in enumerate(rec_texts):
        poly = rec_polys[i] if i < len(rec_polys) else None
        score = rec_scores[i] if i < len(rec_scores) else None
        if text is None:
            continue
        words = split_re.findall(text)
        if not words:
            continue
        if poly:
            bbox = poly_to_bbox(poly)
            word_bboxes = split_bbox_horizontally(bbox, words)
        else:
            word_bboxes = [[0,0,0,0] for _ in words]

        entries.append({
            "id": i,
            "tokens": words,
            "bboxes": word_bboxes,
            "line_score": score,
        })
    return entries

def pdf2png(pdf_path, output_dir, dpi=200):
    doc = fitz.open(pdf_path)
    pdf_stem = Path(pdf_path).stem
    saved = []

    for i in range(len(doc)):
      page = doc[i]
      # Create transformation matrix for DPI scaling
      mat = fitz.Matrix(dpi/72, dpi/72)
      pix = page.get_pixmap(matrix=mat)
      out_path = Path(output_dir) / f"{pdf_stem}_{i}.png"
      pix.save(str(out_path))
      saved.append(out_path)
      print(f"Saved page image: {out_path}")
  
    doc.close()
    return saved
# USED CHATGPT TO ASSIST WITH FUNCTION
def convert_from_label_studio(export_path, original_path, output_path):
    with open(export_path, "r") as f:
        tasks = json.load(f)
    with open(original_path, "r") as f:
        page_entry = json.load(f)
    # Handle both predictions (import) and annotations (export)
    task = tasks[0]
    if "annotations" in task and task["annotations"]:
        results = task["annotations"][0]["result"]
    elif "predictions" in task and task["predictions"]:
        results = task["predictions"][0]["result"]
    else:
        print("No annotations or predictions found")
        return None
    # Extract corrected labels in order
    labels = []
    for result in results:
        label = result["value"]["rectanglelabels"][0]
        labels.append(label)
    page_entry["labels"] = labels
    with open(output_path, "w") as fh:
        fh.write(json.dumps(page_entry, ensure_ascii=False, indent=2) + "\n")
    print(f"Updated {len(labels)} labels from Label Studio")
    return page_entry