"""Disk-backed OCR dataset — reads pre-generated images via a JSON index."""

import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .utils import _init_from_config, char_to_idx, IMG_H, IMG_W


class OCRDataset(Dataset):
    """Reads images from disk as specified by a JSON file.

    JSON format:
        [{"path": "data/synthetic/images/000000.png", "label": "123"}, ...]

    Args:
        json_path: Path to the labels JSON file.
        shuffle: If True, shuffle data on init (for training).
        num_samples: If None, use all data. If <= available, randomly sample
                     without replacement. If > available, randomly oversample
                     with replacement.
    """

    def __init__(self, json_path, shuffle=False, num_samples=None):
        with open(json_path, "r") as f:
            all_data = json.load(f)

        if num_samples is not None and num_samples > 0:
            if num_samples <= len(all_data):
                self.data = random.sample(all_data, num_samples)
            else:
                self.data = random.choices(all_data, k=num_samples)
        else:
            self.data = all_data
            if shuffle:
                random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = cv2.imread(item["path"], cv2.IMREAD_COLOR)
        if img is None:
            # Return a blank image on read failure
            canvas = np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)
            return torch.tensor(np.transpose(canvas, (2, 0, 1)), dtype=torch.float32), ""
        return torch.tensor(self._preprocess(img), dtype=torch.float32), item["label"]

    def _preprocess(self, color_img):
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        h, w, _ = color_img.shape
        scale = min(IMG_W / w, IMG_H / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(color_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        x_off, y_off = (IMG_W - new_w) // 2, (IMG_H - new_h) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        canvas = canvas.astype(np.float32) / 255.0
        return np.transpose(canvas, (2, 0, 1))


def collate_fn(batch):
    """Collate function that encodes text labels and returns padded targets."""
    images, texts = zip(*batch)
    images = torch.stack(images)

    targets = []
    target_lengths = []

    for text in texts:
        encoded = [char_to_idx[c] for c in text if c in char_to_idx]
        targets.extend(encoded)
        target_lengths.append(len(encoded))

    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, targets, target_lengths, texts