"""Evaluate UltraOCR:  python scripts/eval.py --config ocr_mobilenetv3"""

import argparse
import torch
from torch.utils.data import DataLoader

from ultraocr.config import Config
from ultraocr.model import OCR
from ultraocr.dataset import OCRDataset
from ultraocr.utils import ctc_decode, levenshtein_distance, collate_fn


def main(config_name, checkpoint_path=None, num_samples=10):
    config_path = f"config/{config_name}.yaml"
    cfg = Config(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = checkpoint_path or cfg.best_path
    print(f"Config: {config_name}  |  Device: {device}  |  Checkpoint: {checkpoint_path}")

    # --- Model ---
    model = OCR(cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("Checkpoint loaded.\n")

    # --- Synthetic samples (random noise for quick sanity check) ---
    print("--- Synthetic Samples ---")
    for _ in range(num_samples):
        img = torch.randn(3, cfg.model.img_h, cfg.model.img_w, dtype=torch.float32)
        x = img.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0]
        pred = ctc_decode(logits, cfg.idx_to_char)
        print(f"  PRED: {pred}")

    # --- Validation set ---
    from functools import partial
    val_collate = partial(collate_fn, char_to_idx=cfg.char_to_idx, charset=cfg.charset)

    val_dataset = OCRDataset(
        cfg.dataset.json_path,
        shuffle=False,
        num_samples=cfg.dataset.num_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=val_collate,
    )

    total_edit_distance = 0
    total_gt_chars = 0

    with torch.no_grad():
        for val_images, _, _, val_texts in val_loader:
            val_images = val_images.to(device)
            val_logits = model(val_images)

            for idx in range(val_images.size(0)):
                pred_text = ctc_decode(val_logits[idx], cfg.idx_to_char)
                gt_text = val_texts[idx]
                total_edit_distance += levenshtein_distance(pred_text, gt_text)
                total_gt_chars += max(len(gt_text), 1)

    char_accuracy = (1.0 - (total_edit_distance / total_gt_chars)) * 100
    print(f"\nCharacter Accuracy: {char_accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate UltraOCR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ocr_mobilenetv3",
        help="Config name (without .yaml extension, looked up in config/ dir)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (default: uses best_path from config)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of synthetic samples to show",
    )
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.num_samples)