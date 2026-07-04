"""Shared utilities: CTC decoding, Levenshtein distance, collation, synthetic data helpers."""

import random
from pathlib import Path

import torch


# --- Image constants (used by synthetic generation) ---
IMG_H = 32
IMG_W = 256

# --- Font paths (auto-discovered) ---
FONT_PATHS = []


def _init_from_config(cfg):
    """Set module-level globals from a Config object."""
    global IMG_H, IMG_W, FONT_PATHS

    if hasattr(cfg, "model"):
        IMG_H = getattr(cfg.model, "img_h", IMG_H)
        IMG_W = getattr(cfg.model, "img_w", IMG_W)

    # Discover fonts if not already loaded
    if not FONT_PATHS:
        for d in ("data/fonts_simple", "data/fonts_heavy", "data/fonts"):
            for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
                FONT_PATHS.extend(str(p) for p in Path(d).rglob(ext))


def get_random_colors():
    """Return (bg_color, text_color) as RGB tuples.

    Generates UI-like color schemes: dark theme, light theme, or colored.
    Ensures sufficient contrast between background and text.
    """
    scheme = random.random()

    if scheme < 0.35:
        # Dark theme (dark bg, light text)
        bg = (
            random.randint(15, 50),
            random.randint(15, 50),
            random.randint(15, 60),
        )
        txt = (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255),
        )
    elif scheme < 0.70:
        # Light theme (light bg, dark text)
        bg = (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255),
        )
        txt = (
            random.randint(15, 80),
            random.randint(15, 80),
            random.randint(15, 80),
        )
    else:
        # Colored theme (colored bg, contrasting text)
        bg = (
            random.randint(40, 200),
            random.randint(40, 200),
            random.randint(40, 200),
        )
        # Pick text color with good contrast
        brightness = sum(bg) / 3
        if brightness > 128:
            txt = (random.randint(0, 60),) * 3
        else:
            txt = (random.randint(180, 255),) * 3

    return bg, txt


def ctc_decode(logits, idx_to_char):
    """Decode logits using CTC greedy decoding.

    Args:
        logits: (seq_len, num_classes) or (num_classes, seq_len) tensor
        idx_to_char: dict mapping class index → character

    Returns:
        decoded string
    """
    if logits.dim() == 2:
        # (seq_len, num_classes)
        pred_indices = torch.argmax(logits, dim=1).cpu().numpy()
    else:
        # (num_classes, seq_len)
        pred_indices = torch.argmax(logits, dim=0).cpu().numpy()

    pred_text = []
    prev_idx = None
    for idx in pred_indices:
        if idx != prev_idx and idx != 0:  # Skip repeated and blank tokens
            if idx in idx_to_char:
                pred_text.append(idx_to_char[idx])
        prev_idx = idx
    return "".join(pred_text)


def levenshtein_distance(s1, s2):
    """Compute the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def collate_fn(batch, char_to_idx, charset):
    """Collate function that encodes text labels and returns padded targets.

    Args:
        batch: list of (image, label) tuples
        char_to_idx: dict mapping char → class index
        charset: string of valid characters

    Returns:
        images: (B, C, H, W) tensor
        targets: concatenated encoded targets
        target_lengths: (B,) tensor of per-sample target lengths
        texts: original text labels
    """
    import warnings

    images, texts = zip(*batch)
    images = torch.stack(images)

    targets = []
    target_lengths = []

    for text in texts:
        encoded = [char_to_idx[c] for c in text if c in charset]
        if not encoded:
            warnings.warn(
                f"All characters in label '{text}' were dropped. "
                "Check that the config charset matches the dataset labels."
            )
        targets.extend(encoded)
        target_lengths.append(len(encoded))

    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, targets, target_lengths, texts