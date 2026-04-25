import re
from preprocessing import is_numeric_token, KEYWORDS, COMPILED_KEYWORDS
from bbox_utils import normalize_bbox

# Avoid incorrect tokenization of charges (usually far away from previous tokens)
def group_lines_by_row(entries, y_tolerance=7):
    rows = []
    used = set()
    # Calculate center to have one number determine placement on page
    for i, entry in enumerate(entries):
        if i in used:
            continue
        y_center_i = (entry["bboxes"][0][1] + entry["bboxes"][0][3]) / 2

        row = [entry]
        used.add(i)
        # Loop through remaining ungrouped entries to determine what should be added to row
        for j, other in enumerate(entries):
            if j in used:
                continue
            y_center_j = (other["bboxes"][0][1] + other["bboxes"][0][3]) / 2
            if abs(y_center_i - y_center_j) <= y_tolerance:
                row.append(other)
                used.add(j)
        # Sort left to right based on x coordinate
        row.sort(key=lambda e: e["bboxes"][0][0])
        rows.append(row)
    return rows

# Determines row type 
def detect_row_type(all_tokens):
    joined = " ".join(all_tokens)
    if COMPILED_KEYWORDS["energy"].search(joined):
        return "kwh"
    if COMPILED_KEYWORDS["demand"].search(joined):
        return "kw"
    if COMPILED_KEYWORDS["total"].search(joined):
        return "total"
    return None

def label_row(row_entries):
    # Flatten all tokens across the row to detect type
    all_tokens = [t for entry in row_entries for t in entry["tokens"]]
    row_type   = detect_row_type(all_tokens)

    if row_type is None:
        # No relevant keyword found, label everything O
        for entry in row_entries:
            entry["labels"] = ["O"] * len(entry["tokens"])
        return row_entries

    # Find the last numeric token index across the full row to determine charge
    last_numeric_idx = None
    for i, t in enumerate(all_tokens):
        if is_numeric_token(t):
            last_numeric_idx = i

    # Find the keyword index to anchor usage label
    keyword_idx = None
    for i, t in enumerate(all_tokens):
        if row_type == "kwh" and COMPILED_KEYWORDS["energy"].search(t):
            keyword_idx = i
            break
        if row_type == "kw" and COMPILED_KEYWORDS["demand"].search(t):
            keyword_idx = i
            break
    # Assign labels token by token across the full row
    flat_labels = []
    for i, t in enumerate(all_tokens):
        if not is_numeric_token(t):
            flat_labels.append("O")
            continue
        # Last numeric is always the cost
        if i == last_numeric_idx:
            if row_type == "kwh":
                flat_labels.append("B-KWH_COST")
            elif row_type == "kw":
                flat_labels.append("B-KW_COST")
            elif row_type == "total":
                flat_labels.append("B-TOTAL_COST")
        # Numeric before the keyword is the usage
        elif keyword_idx is not None and i < keyword_idx:
            if row_type == "kwh":
                flat_labels.append("B-KWH_USAGE")
            elif row_type == "kw":
                flat_labels.append("B-KW_USAGE")
            else:
                flat_labels.append("O")
        else:
            flat_labels.append("O")
    # Map flat labels back to each entry
    idx = 0
    for entry in row_entries:
        entry["labels"] = flat_labels[idx: idx + len(entry["tokens"])]
        idx += len(entry["tokens"])
    return row_entries

# Flattan labeled row-entries for LayoutLMv3
def flatten_labels(page_id, image_path, labeled_rows, page_width, page_height):
    all_tokens = []
    all_bboxes = []
    all_labels = []

    for row in labeled_rows:
        for entry in row:
            for token, bbox, label in zip(entry["tokens"], entry["bboxes"], entry["labels"]):
                all_tokens.append(token)
                all_bboxes.append(normalize_bbox(bbox, page_width, page_height))
                all_labels.append(label)

    return {
        "id":         page_id,
        "image":      image_path,
        "tokens":     all_tokens,
        "bboxes":     all_bboxes,
        "labels":     all_labels,
        "token_count": len(all_tokens)
    }
