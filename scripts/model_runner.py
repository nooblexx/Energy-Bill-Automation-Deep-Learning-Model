import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "utils"))

from ocr_utils import convert_file
from labeling import group_lines_by_row, label_row, flatten_labels
from bbox_utils import normalize_bbox
from preprocessing import find_stop_line
from PIL import Image
from transformers import AutoProcessor, AutoModelForTokenClassification
import torch

# Args from command line
ann_path   = Path(sys.argv[1])
img_path   = Path(sys.argv[2])
model_dir  = sys.argv[3]
output_path = Path(sys.argv[4])

# Load model
processor = AutoProcessor.from_pretrained(model_dir, apply_ocr=False)
model     = AutoModelForTokenClassification.from_pretrained(model_dir)
model.eval()
# Load image dimensions
with Image.open(img_path) as img:
    page_width, page_height = img.size
# Convert file to entries
entries      = convert_file(ann_path)
rows         = group_lines_by_row(entries)
labeled_rows = [label_row(row) for row in rows]
page_entry   = flatten_labels(
    page_id      = ann_path.stem,
    image_path   = img_path.as_posix(),
    labeled_rows = labeled_rows,
    page_width   = page_width,
    page_height  = page_height
)
# Run inference
image    = Image.open(img_path).convert("RGB")
encoding = processor(
    images         = image,
    text           = page_entry["tokens"],
    boxes          = page_entry["bboxes"],
    truncation     = True,
    max_length     = 512,
    padding        = "max_length",
    return_tensors = "pt"
)
with torch.no_grad():
    outputs = model(**encoding)

predictions = outputs.logits.argmax(-1).squeeze().tolist()
id2label    = model.config.id2label

# Extract results
results = {}
for token, pred, bbox in zip(page_entry["tokens"], predictions, page_entry["bboxes"]):
    label = id2label[pred]
    if label != "O":
        results[label] = token

# Save results to output
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print("Inference complete")