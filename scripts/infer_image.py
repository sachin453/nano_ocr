"""Run UltraOCR on a single image: PaddleOCR detection + UltraOCR recognition + annotation.

Usage:
    python scripts/infer_image.py --input path/to/image.jpg --config ocr_mobilenetv3
    python scripts/infer_image.py --input img.jpg --output annotated.jpg --checkpoint artifacts/aocr_best.pt
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

from ultraocr.config import Config
from ultraocr.model import OCR
from ultraocr.utils import ctc_decode


# ---------------------------------------------------------------------------
# Image preprocessing (matches OCRDataset._preprocess)
# ---------------------------------------------------------------------------

def preprocess_crop(crop, img_h=32, img_w=256):
    """Resize a crop to fit (img_h, img_w) maintaining aspect ratio, pad with black.

    Args:
        crop: HxWx3 BGR uint8 image
        img_h: target height
        img_w: target width

    Returns:
        CHW float32 tensor normalized to [0, 1]
    """
    h, w = crop.shape[:2]
    scale = min(img_w / w, img_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    x_off = (img_w - new_w) // 2
    y_off = (img_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # BGR -> RGB, normalize, HWC -> CHW
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    canvas = np.transpose(canvas, (2, 0, 1))
    return torch.tensor(canvas, dtype=torch.float32)


def split_wide_crop(crop, img_h=32, img_w=256, overlap_ratio=0.15):
    """If a crop is too wide for the model, split into overlapping vertical strips.

    The model expects 32x256 (aspect ratio 8:1). If the crop's aspect ratio
    exceeds this, we split horizontally into strips that each fit within 256px
    width at 32px height.

    Args:
        crop: HxWx3 BGR uint8 image
        img_h: target height
        img_w: target width
        overlap_ratio: fractional overlap between adjacent strips

    Returns:
        list of (strip_crop, x_offset) tuples — strip_crop is HxW_crop BGR uint8,
        x_offset is the x-coordinate of the strip's left edge in the original crop.
    """
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return []

    # Max width that maintains aspect ratio at img_h
    max_w = int(img_h * (img_w / img_h))  # = img_w (256) when img_h=32
    # Actually: at height img_h, the max width keeping aspect ratio is:
    # scale = img_h / h  →  max_w = w * scale = w * img_h / h
    # But we want to know if the crop, when scaled to img_h, exceeds img_w
    scale = img_h / h
    scaled_w = w * scale

    if scaled_w <= img_w:
        # No split needed
        return [(crop, 0)]

    # Need to split — compute strip width in original crop coordinates
    # Each strip, when scaled to img_h, should be at most img_w wide
    # strip_w_orig * (img_h / h) <= img_w  →  strip_w_orig <= img_w * h / img_h
    strip_w_orig = int(img_w * h / img_h)
    if strip_w_orig <= 0:
        strip_w_orig = 1

    overlap = int(strip_w_orig * overlap_ratio)
    step = strip_w_orig - overlap
    if step <= 0:
        step = 1

    strips = []
    x = 0
    while x < w:
        x_end = min(x + strip_w_orig, w)
        strip = crop[:, x:x_end]
        strips.append((strip, x))
        if x_end >= w:
            break
        x += step

    return strips


# ---------------------------------------------------------------------------
# PaddleOCR detection
# ---------------------------------------------------------------------------

def run_detection(det_predictor, image_path):
    """Run PaddleOCR text detection and return list of polygons.

    Uses the paddlex detection predictor directly (detection-only, no
    recognition model needed).

    Args:
        det_predictor: paddlex TextDetRunnerPredictor instance
        image_path: path to input image

    Returns:
        list of np.array polygons, each of shape (N, 2)
    """
    results = list(det_predictor.predict(image_path))
    if not results:
        return []

    result = results[0]

    # PaddleOCR 3.x result structure — try multiple access patterns
    polys = None

    # Pattern 1: .json['dt_polys'] (top-level)
    if hasattr(result, 'json') and 'dt_polys' in result.json:
        polys = result.json['dt_polys']

    # Pattern 2: .json['res']['dt_polys'] (nested, observed in PP-OCRv6)
    elif hasattr(result, 'json') and isinstance(result.json.get('res'), dict) and 'dt_polys' in result.json['res']:
        polys = result.json['res']['dt_polys']

    # Pattern 3: .polys attribute
    elif hasattr(result, 'polys'):
        polys = result.polys

    # Pattern 4: .json['polys']
    elif hasattr(result, 'json') and 'polys' in result.json:
        polys = result.json['polys']

    if polys is None:
        print("Warning: could not find detection polygons in result")
        print("Available keys:", list(result.json.keys()) if hasattr(result, 'json') else dir(result))
        return []

    # Convert to list of np arrays
    poly_list = []
    for p in polys:
        p = np.array(p, dtype=np.float32)
        poly_list.append(p)

    return poly_list


def poly_to_rect(poly):
    """Convert a polygon to an axis-aligned bounding rectangle (x, y, w, h)."""
    xs = poly[:, 0]
    ys = poly[:, 1]
    x_min, x_max = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    y_min, y_max = int(np.floor(ys.min())), int(np.ceil(ys.max()))
    return x_min, y_min, x_max - x_min, y_max - y_min


# ---------------------------------------------------------------------------
# UltraOCR recognition
# ---------------------------------------------------------------------------

def load_recognition_model(cfg, checkpoint_path, device):
    """Load the UltraOCR recognition model from a checkpoint."""
    model = OCR(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
    elif hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def recognize_crops(model, crops, cfg, device, img_h=32, img_w=256):
    """Run recognition on a list of crop images.

    Args:
        model: UltraOCR model
        crops: list of HxWx3 BGR uint8 numpy arrays
        cfg: Config object
        device: torch device
        img_h, img_w: model input dimensions

    Returns:
        list of recognized text starings
    """
    if not crops:
        return []

    # Preprocess all crops into a batch
    tensors = []
    strip_map = []  # (crop_idx, strip_idx) for each tensor

    for crop_idx, crop in enumerate(crops):
        strips = split_wide_crop(crop, img_h, img_w)
        for strip_idx, (strip, _) in enumerate(strips):
            tensor = preprocess_crop(strip, img_h, img_w)
            tensors.append(tensor)
            strip_map.append(crop_idx)

    if not tensors:
        return [""] * len(crops)

    batch = torch.stack(tensors).to(device)

    texts = [""] * len(crops)
    with torch.no_grad():
        logits = model(batch)  # (N, T, C)

        # Group results by crop (merge strips)
        crop_texts = {}
        for i, crop_idx in enumerate(strip_map):
            pred = ctc_decode(logits[i], cfg.idx_to_char)
            if crop_idx not in crop_texts:
                crop_texts[crop_idx] = pred
            else:
                # Merge strips — simple concatenation
                # (overlap means some chars may repeat, but for simplicity we concat)
                crop_texts[crop_idx] += pred

    for i in range(len(crops)):
        texts[i] = crop_texts.get(i, "")

    return texts


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate_image(image, polys, texts):
    """Draw detection boxes and recognized text on the image.

    Args:
        image: HxWx3 BGR uint8 image (copy, will be modified)
        polys: list of polygons (np arrays)
        texts: list of recognized text strings

    Returns:
        annotated image
    """
    for poly, text in zip(polys, texts):
        # Draw polygon outline in green
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw text label above the box
        x_min = int(poly[:, 0].min())
        y_min = int(poly[:, 1].min())

        label = text if text else "?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Background rectangle for text
        y_label = max(y_min - text_h - baseline - 4, 0)
        cv2.rectangle(image, (x_min, y_label), (x_min + text_w + 4, y_label + text_h + baseline + 2),
                      (0, 255, 0), -1)
        cv2.putText(image, label, (x_min + 2, y_label + text_h),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_path, config_name, checkpoint_path, output_path):
    # --- Config ---
    config_path = f"config/{config_name}.yaml"
    cfg = Config(config_path)
    print(f"Config: {config_path}")
    print(f"Charset size: {cfg.num_chars}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load image ---
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: could not read image: {input_path}")
        sys.exit(1)
    print(f"Input image: {input_path}  ({image.shape[1]}x{image.shape[0]})")

    # --- PaddleOCR detection (detection-only, no recognition model) ---
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddlex.inference.models import create_predictor

    print("Loading PaddleOCR text detection model...")
    det_predictor = create_predictor(
        model_name="PP-OCRv6_medium_det",
        model_dir="paddle_ocr_weights/PP-OCRv6_medium_det_infer",
    )

    print("Running text detection...")
    polys = run_detection(det_predictor, input_path)
    print(f"Detected {len(polys)} text regions")

    if not polys:
        print("No text detected. Saving original image as output.")
        cv2.imwrite(output_path, image)
        return

    # --- UltraOCR recognition ---
    checkpoint_path = checkpoint_path or cfg.best_path
    print(f"Loading UltraOCR recognition model: {checkpoint_path}")
    model = load_recognition_model(cfg, checkpoint_path, device)

    # --- Crop detected regions ---
    crops = []
    for poly in polys:
        x, y, w, h = poly_to_rect(poly)
        # Clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        if w <= 0 or h <= 0:
            crops.append(np.zeros((1, 1, 3), dtype=np.uint8))
            continue
        crop = image[y:y + h, x:x + w]
        crops.append(crop)

    # --- Recognize ---
    print("Running recognition...")
    img_h = cfg.model.img_h
    img_w = cfg.model.img_w
    texts = recognize_crops(model, crops, cfg, device, img_h, img_w)

    for i, (poly, text) in enumerate(zip(polys, texts)):
        print(f"  [{i}] {text}")

    # --- Annotate ---
    annotated = annotate_image(image.copy(), polys, texts)

    # --- Save ---
    cv2.imwrite(output_path, annotated)
    print(f"\nAnnotated image saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run UltraOCR on an image: PaddleOCR detection + UltraOCR recognition + annotation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--config", type=str, default="ocr_mobilenetv3",
        help="Config name (without .yaml extension)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint path (default: uses best_path from config)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output image path (default: annotated_<inputname>.jpg)",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        base = os.path.basename(args.input)
        name, _ = os.path.splitext(base)
        output_path = f"annotated_{name}.jpg"

    main(args.input, args.config, args.checkpoint, output_path)