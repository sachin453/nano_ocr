"""Generate a pre-built synthetic OCR dataset with augmentations.

Usage:
    python scripts/generate_synthetic.py --config ocr --count 100000 --output data/synthetic
    python scripts/generate_synthetic.py --config ocr  # defaults to 1M, data/synthetic
"""

import argparse
import json
import os
import random
import shutil
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


def apply_augmentations(img):
    """Apply random distortions to a uint8 HxWxC image. Returns augmented image."""
    img = img.copy()

    # --- Noise ---
    if random.random() < 0.4:
        if random.random() < 0.5:
            # Gaussian noise
            noise = np.random.randn(*img.shape) * random.randint(3, 12)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            # Salt & pepper
            prob = random.uniform(0.001, 0.01)
            mask = np.random.random(img.shape[:2]) < prob
            img[mask] = random.randint(0, 255)

    # --- Blur ---
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        if random.random() < 0.6:
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        else:
            # Motion blur kernel
            kernel = np.zeros((ksize, ksize))
            kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
            kernel = kernel / ksize
            img = cv2.filter2D(img, -1, kernel)

    # --- Brightness / Contrast ---
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)  # contrast
        beta = random.randint(-20, 20)     # brightness
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # --- Slight rotation (±3°) ---
    if random.random() < 0.5:
        angle = random.uniform(-3, 3)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # --- Perspective skew ---
    if random.random() < 0.3:
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
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return img


def generate_one(cfg):
    """Generate one synthetic image tensor + label, with config support."""
    if cfg is not None:
        _init_from_config(cfg)
        synth_cfg = cfg.dataset.synthetic
    else:
        synth_cfg = None

    if not FONT_PATHS:
        for d in ("data/fonts_simple", "data/fonts_heavy"):
            for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
                FONT_PATHS.extend(str(p) for p in Path(d).rglob(ext))

    tl_min, tl_max = (synth_cfg.text_length[0], synth_cfg.text_length[1]) if synth_cfg else (3, 12)
    fs_min, fs_max = (synth_cfg.font_size[0], synth_cfg.font_size[1]) if synth_cfg else (32, 48)
    pad_min, pad_max = (synth_cfg.pad[0], synth_cfg.pad[1]) if synth_cfg else (0, 3)
    jit_max_val = synth_cfg.jitter[1] if synth_cfg else 10

    chars = cfg.charset if cfg else "0123456789"
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

    tight_img = Image.new("RGB", (crop_w, crop_h), color=bg_color)
    draw = ImageDraw.Draw(tight_img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=txt_color)

    scale = IMG_H / crop_h
    new_w = min(int(crop_w * scale), IMG_W)
    resized = tight_img.resize((new_w, IMG_H), Image.Resampling.BILINEAR)

    final_img = Image.new("RGB", (IMG_W, IMG_H), color=bg_color)
    max_jitter = max(0, IMG_W - new_w)
    x_off = random.randint(0, min(jit_max_val, max_jitter))
    final_img.paste(resized, (x_off, 0))

    return np.array(final_img), text


def main(config_name, count, output_dir):
    cfg = Config(f"config/{config_name}.yaml")
    _init_from_config(cfg)

    # Remove existing output directory to avoid mixing old/new data
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    labels = []

    print(f"Generating {count} images → {output_dir}")
    print(f"Charset: {cfg.charset} ({cfg.num_chars} chars)")
    print(f"Image size: {IMG_W}x{IMG_H}")
    print(f"Fonts: {len(FONT_PATHS)} available")

    for i in tqdm(range(count), desc="Generating"):
        img_pil, text = generate_one(cfg)

        # Apply augmentations (PIL → numpy uint8 → augment → save)
        img_np = np.array(img_pil)
        img_aug = apply_augmentations(img_np)

        filename = f"{i:08d}.png"
        filepath = os.path.join(images_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))

        labels.append({"path": os.path.abspath(filepath), "label": text})

    # Write labels.json
    json_path = os.path.join(output_dir, "labels.json")
    with open(json_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nDone. {count} images saved to {images_dir}")
    print(f"Labels index: {json_path}")
    print(f"Config to use:  dataset.data_json: \"{json_path}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a pre-built synthetic OCR dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ocr",
        help="Config name (without .yaml extension, looked up in config/ dir)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1_000_000,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic",
        help="Output directory (will contain images/ and labels.json)",
    )
    args = parser.parse_args()

    main(args.config, args.count, args.output)