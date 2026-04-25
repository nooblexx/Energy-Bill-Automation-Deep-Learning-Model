# Create word level token bounding box
def poly_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]

# Split line tokens into seperate word tokens
def split_bbox_horizontally(bbox, words):
    x0, y0, x1, y1 = bbox
    total_chars = sum(len(w) for w in words) if words else 1
    bboxes = []
    cur_x = x0
    width = x1 - x0
    for w in words:
        # allocate proportional width
        frac = len(w) / total_chars
        w_w = max(1, round(frac * width))
        x_start = cur_x
        x_end = x_start + w_w
        if x_end > x1:
            x_end = x1
        bboxes.append([int(x_start), int(y0), int(x_end), int(y1)])
        cur_x = x_end
    # fix last bbox to end exactly at x1
    if bboxes:
        bboxes[-1][2] = int(x1)
    return bboxes

# Required normalization for LayoutLMv3
def normalize_bbox(bbox, page_width, page_height):
    x1, y1, x2, y2 = bbox
    return [
        int(x1 / page_width  * 1000),
        int(y1 / page_height * 1000),
        int(x2 / page_width  * 1000),
        int(y2 / page_height * 1000),
    ]
