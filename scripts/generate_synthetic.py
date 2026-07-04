"""Generate a pre-built synthetic OCR dataset with augmentations (multiprocessing).

Usage:
    python scripts/generate_synthetic.py --config ocr_mobilenetv3 --count 100000 --output data/synthetic
    python scripts/generate_synthetic.py --config ocr_mobilenetv3 --count 50000 --workers 8
"""

import argparse
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ultraocr.config import Config
from ultraocr.utils import (
    _init_from_config,
    IMG_W,
    IMG_H,
    FONT_PATHS,
    get_random_colors,
)


# ---------------------------------------------------------------------------
# Worker state (initialized per-process)
# ---------------------------------------------------------------------------

_WORKER_CFG = None
_WORKER_IMAGES_DIR = None


def _worker_init(config_name, images_dir):
    """Initialize worker process: load config, discover fonts, set globals."""
    global _WORKER_CFG, _WORKER_IMAGES_DIR
    _WORKER_CFG = Config(f"config/{config_name}.yaml")
    _WORKER_IMAGES_DIR = images_dir
    _init_from_config(_WORKER_CFG)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def _aug_noise(img):
    """Gaussian or salt-and-pepper noise."""
    if random.random() < 0.5:
        noise = np.random.randn(*img.shape) * random.randint(3, 12)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    else:
        prob = random.uniform(0.001, 0.01)
        mask = np.random.random(img.shape[:2]) < prob
        img[mask] = random.randint(0, 255)
    return img


def _aug_blur(img):
    """Gaussian or motion blur."""
    ksize = random.choice([3, 5])
    if random.random() < 0.6:
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
    kernel = kernel / ksize
    return cv2.filter2D(img, -1, kernel)


def _aug_brightness(img):
    """Brightness / contrast shift."""
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-20, 20)
    return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def _aug_rotation(img):
    """Slight rotation (±3°)."""
    angle = random.uniform(-3, 3)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _aug_perspective(img):
    """Perspective skew."""
    h, w = img.shape[:2]
    jitter = random.randint(1, 4)
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([
        [random.randint(0, jitter), random.randint(0, jitter)],
        [w - random.randint(0, jitter), random.randint(0, jitter)],
        [random.randint(0, jitter), h - random.randint(0, jitter)],
        [w - random.randint(0, jitter), h - random.randint(0, jitter)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _aug_jpeg(img):
    """JPEG compression artifacts (simulates screenshot compression)."""
    quality = random.randint(30, 85)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _aug_scaling(img):
    """Scaling artifacts (downscale → upscale, simulates resolution mismatch)."""
    h, w = img.shape[:2]
    scale = random.uniform(0.5, 0.85)
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=random.choice([
        cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST
    ]))


def _aug_gamma(img):
    """Gamma correction (display calibration differences)."""
    gamma = random.uniform(0.6, 1.8)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def _aug_hue(img):
    """Hue/Saturation shift (color profile differences)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-15, 15)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _aug_channel_swap(img):
    """Channel permutation (RGB/BGR rendering differences)."""
    perm = random.choice([(0, 1, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0)])
    return img[:, :, list(perm)]


def _aug_elastic(img):
    """Elastic distortion (subtle warping)."""
    return _elastic_distortion(img, alpha=random.uniform(2, 6),
                               sigma=random.uniform(0.5, 1.5))


# All available augmentations
_AUGMENTATIONS = [
    _aug_noise, _aug_blur, _aug_brightness, _aug_rotation,
    _aug_perspective, _aug_jpeg, _aug_scaling, _aug_gamma,
    _aug_hue, _aug_channel_swap, _aug_elastic,
]

MAX_AUGMENTATIONS = 3


def apply_augmentations(img):
    """Apply up to MAX_AUGMENTATIONS random distortions to a uint8 HxWxC image.

    Instead of rolling each augmentation independently (which could stack 5-6
    at once and destroy legibility), we pick a random subset of at most 3.
    """
    img = img.copy()
    n = random.randint(0, MAX_AUGMENTATIONS)
    chosen = random.sample(_AUGMENTATIONS, n)
    for aug in chosen:
        img = aug(img)
    return img


def _elastic_distortion(img, alpha=4, sigma=1.0):
    """Apply elastic distortion to simulate subtle display warping."""
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(np.random.uniform(-1, 1, (h, w)).astype(np.float32),
                          (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(np.random.uniform(-1, 1, (h, w)).astype(np.float32),
                          (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _make_gradient_bg(w, h, base_color):
    """Create a subtle gradient background (common in UI panels)."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    direction = random.choice(["horizontal", "vertical", "diagonal"])
    c = np.array(base_color, dtype=np.float32)

    if direction == "horizontal":
        for x in range(w):
            factor = 1.0 + (x / w - 0.5) * random.uniform(0.05, 0.15)
            bg[:, x] = np.clip(c * factor, 0, 255)
    elif direction == "vertical":
        for y in range(h):
            factor = 1.0 + (y / h - 0.5) * random.uniform(0.05, 0.15)
            bg[y, :] = np.clip(c * factor, 0, 255)
    else:  # diagonal
        for y in range(h):
            for x in range(w):
                factor = 1.0 + ((x / w + y / h) / 2 - 0.5) * random.uniform(0.05, 0.12)
                bg[y, x] = np.clip(c * factor, 0, 255)

    return bg


def _generate_one(cfg):
    """Generate one synthetic image (numpy HxWx3 RGB) + label string."""
    synth_cfg = getattr(cfg.dataset, "synthetic", None)

    tl_min, tl_max = (synth_cfg.text_length[0], synth_cfg.text_length[1]) if synth_cfg else (3, 12)
    fs_min, fs_max = (synth_cfg.font_size[0], synth_cfg.font_size[1]) if synth_cfg else (32, 48)
    pad_min, pad_max = (synth_cfg.pad[0], synth_cfg.pad[1]) if synth_cfg else (0, 3)
    jit_max_val = synth_cfg.jitter[1] if synth_cfg else 10

    chars = cfg.charset
    length = random.randint(tl_min, tl_max)
    text = "".join(random.choice(chars) for _ in range(length))
    bg_color, txt_color = get_random_colors()

    from PIL import Image, ImageDraw, ImageFont

    font_path = random.choice(FONT_PATHS)
    font_size = random.randint(fs_min, fs_max)
    font = ImageFont.truetype(font_path, font_size)

    dummy_img = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy_img).textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad_x = random.randint(pad_min, pad_max)
    pad_y = random.randint(pad_min, pad_max)
    crop_w = max(1, tw + pad_x * 2)
    crop_h = max(1, th + pad_y * 2)

    # --- Background: solid or gradient ---
    use_gradient = random.random() < 0.3
    if use_gradient:
        bg_array = _make_gradient_bg(crop_w, crop_h, bg_color)
        tight_img = Image.fromarray(bg_array)
    else:
        tight_img = Image.new("RGB", (crop_w, crop_h), color=bg_color)

    draw = ImageDraw.Draw(tight_img)

    # --- Text shadow (common in UI) ---
    if random.random() < 0.25:
        shadow_offset = random.randint(1, 2)
        shadow_color = tuple(max(0, c - 40) for c in bg_color)
        draw.text((pad_x - bbox[0] + shadow_offset, pad_y - bbox[1] + shadow_offset),
                  text, font=font, fill=shadow_color)

    # --- Text outline/stroke (common in UI) ---
    stroke_width = 0
    if random.random() < 0.15:
        stroke_width = 1

    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font,
              fill=txt_color, stroke_width=stroke_width,
              stroke_fill=tuple(max(0, c - 60) for c in txt_color))

    # --- Resize to target height ---
    scale = IMG_H / crop_h
    new_w = min(int(crop_w * scale), IMG_W)
    resized = tight_img.resize((new_w, IMG_H), Image.Resampling.BILINEAR)

    # --- Place on canvas ---
    if use_gradient:
        final_bg = _make_gradient_bg(IMG_W, IMG_H, bg_color)
        final_img = Image.fromarray(final_bg)
    else:
        final_img = Image.new("RGB", (IMG_W, IMG_H), color=bg_color)

    max_jitter = max(0, IMG_W - new_w)
    x_off = random.randint(0, min(jit_max_val, max_jitter))
    final_img.paste(resized, (x_off, 0))

    return np.array(final_img), text


def _generate_and_save(idx):
    """Worker function: generate one image, augment, save to disk, return label dict.

    Uses module-level _WORKER_CFG and _WORKER_IMAGES_DIR set by _worker_init.
    """
    img_np, text = _generate_one(_WORKER_CFG)
    img_aug = apply_augmentations(img_np)

    filename = f"{idx:08d}.png"
    filepath = os.path.join(_WORKER_IMAGES_DIR, filename)
    cv2.imwrite(filepath, cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))

    return {"path": os.path.abspath(filepath), "label": text}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_name, count, output_dir, workers):
    cfg = Config(f"config/{config_name}.yaml")
    _init_from_config(cfg)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Generating {count} images → {output_dir}")
    print(f"Charset: {cfg.charset} ({cfg.num_chars} chars)")
    print(f"Image size: {IMG_W}x{IMG_H}")
    print(f"Fonts: {len(FONT_PATHS)} available")
    print(f"Workers: {workers}")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(config_name, images_dir),
    ) as executor:
        # Use map with chunksize for efficient lazy submission
        # (avoids creating 500K futures upfront which hangs on large counts)
        chunksize = max(1, count // (workers * 50))
        results = executor.map(
            _generate_and_save, range(count), chunksize=chunksize
        )
        labels = list(tqdm(results, total=count, desc="Generating"))

    json_path = os.path.join(output_dir, "labels.json")
    with open(json_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nDone. {count} images saved to {images_dir}")
    print(f"Labels index: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a pre-built synthetic OCR dataset (multiprocessing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="ocr_mobilenetv3",
        help="Config name (without .yaml extension)",
    )
    parser.add_argument(
        "--count", type=int, default=1_000_000,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--output", type=str, default="data/synthetic",
        help="Output directory",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Number of parallel worker processes",
    )
    args = parser.parse_args()
    main(args.config, args.count, args.output, args.workers)