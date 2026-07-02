"""Utility functions for synthetic data generation, CTC decoding, and evaluation."""

import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ----- Backward-compatible module-level globals (populated when config is loaded) -----
CHARS = "0123456789"
IMG_W = 512
IMG_H = 64
char_to_idx = {}
idx_to_char = {}
BLANK_IDX = 0
FONT_PATHS = []


def _init_from_config(cfg):
    """Initialize module-level globals from a Config object.

    Called automatically by Config-aware code paths but kept available
    so that existing scripts that import the globals directly still work.
    """
    global CHARS, IMG_W, IMG_H, char_to_idx, idx_to_char, BLANK_IDX, FONT_PATHS

    CHARS = cfg.charset
    IMG_W = cfg.image.width
    IMG_H = cfg.image.height
    char_to_idx = cfg.char_to_idx
    idx_to_char = cfg.idx_to_char
    BLANK_IDX = cfg.loss.blank_idx

    # Build font paths recursively from configured font directories
    font_dirs = cfg.image.font_dirs
    FONT_PATHS = []
    for d in font_dirs:
        for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
            FONT_PATHS.extend(str(p) for p in Path(d).rglob(ext))


def get_random_colors():
    """Generates contrasting random colors for background and text."""
    bg = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    if sum(bg) > 382:
        txt = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    else:
        txt = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
    return bg, txt


def generate_simple_synthetic(cfg=None):
    """Generates tightly cropped text scaled to IMG_H, simulating a detector pipeline.

    Args:
        cfg: Optional Config object. If None, uses module-level globals (must be initialized).
    """
    if cfg is not None:
        _init_from_config(cfg)
        synth_cfg = cfg.dataset.synthetic
    else:
        synth_cfg = None

    # Fallback: scan common font dirs recursively
    if not FONT_PATHS:
        for d in ("data/fonts_simple", "data/fonts_heavy"):
            for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
                FONT_PATHS.extend(str(p) for p in Path(d).rglob(ext))

    # Use config values or fall back to globals
    tl_min, tl_max = (synth_cfg.text_length[0], synth_cfg.text_length[1]) if synth_cfg else (3, 12)
    fs_min, fs_max = (synth_cfg.font_size[0], synth_cfg.font_size[1]) if synth_cfg else (32, 48)
    pad_min, pad_max = (synth_cfg.pad[0], synth_cfg.pad[1]) if synth_cfg else (0, 3)
    jit_max_val = synth_cfg.jitter[1] if synth_cfg else 10

    length = random.randint(tl_min, tl_max)
    text = "".join(random.choice(CHARS) for _ in range(length))
    bg_color, txt_color = get_random_colors()

    font_path = random.choice(FONT_PATHS)
    font_size = random.randint(fs_min, fs_max)
    font = ImageFont.truetype(font_path, font_size)

    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad_x = random.randint(pad_min, pad_max)
    pad_y = random.randint(pad_min, pad_max)
    crop_w = max(1, tw + (pad_x * 2))
    crop_h = max(1, th + (pad_y * 2))

    tight_img = Image.new("RGB", (crop_w, crop_h), color=bg_color)
    draw = ImageDraw.Draw(tight_img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=txt_color)

    scale = IMG_H / crop_h
    new_w = min(int(crop_w * scale), IMG_W)
    resized_img = tight_img.resize((new_w, IMG_H), Image.Resampling.BILINEAR)

    final_img = Image.new("RGB", (IMG_W, IMG_H), color=bg_color)
    max_jitter = max(0, IMG_W - new_w)
    x_off = random.randint(0, min(jit_max_val, max_jitter))
    final_img.paste(resized_img, (x_off, 0))

    img_np = np.array(final_img).astype(np.float32) / 255.0
    img_tensor = np.transpose(img_np, (2, 0, 1))
    return img_tensor, text


def ctc_decode(logits, idx_to_char_map=None):
    """CTC greedy decoder. Uses global idx_to_char by default."""
    mapping = idx_to_char_map if idx_to_char_map is not None else idx_to_char
    if not mapping:
        return ""
    pred = logits.argmax(-1)
    text = []
    prev = -1
    for p in pred:
        p = p.item()
        if p != prev and p != 0:
            if p in mapping:
                text.append(mapping[p])
        prev = p
    return "".join(text)


def levenshtein_distance(s1, s2):
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]