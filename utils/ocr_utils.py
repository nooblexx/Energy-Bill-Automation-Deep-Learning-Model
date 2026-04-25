import re, json
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
