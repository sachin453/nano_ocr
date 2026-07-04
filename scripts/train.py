"""Train UltraOCR:  python scripts/train.py --config ocr_mobilenetv3"""

import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F

from ultraocr.config import Config
from ultraocr.model import OCR
from ultraocr.dataset import OCRDataset
from ultraocr.utils import ctc_decode, levenshtein_distance, collate_fn


def main(config_name):
    # --- Config ---
    config_path = f"config/{config_name}.yaml"
    cfg = Config(config_path)
    print(f"Loaded config: {config_path}")
    print(f"Backbone: {cfg.model.timm_name}, "
          f"num_blocks={cfg.model.num_blocks}, pretrained={cfg.model.pretrained}")
    print(f"Charset size: {cfg.num_chars}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Datasets ---
    full_dataset = OCRDataset(
        cfg.dataset.json_path,
        shuffle=cfg.dataset.shuffle,
        num_samples=cfg.dataset.num_samples,
    )
    train_size = int(cfg.dataset.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Use functools.partial to bind charset params to collate_fn
    from functools import partial
    train_collate = partial(collate_fn, char_to_idx=cfg.char_to_idx, charset=cfg.charset)
    val_collate = partial(collate_fn, char_to_idx=cfg.char_to_idx, charset=cfg.charset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        collate_fn=train_collate,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        collate_fn=val_collate,
    )

    # --- Model ---
    model = OCR(cfg).to(device)
    print(f"Model created.  Parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # Summary (optional — requires torchinfo)
    try:
        from torchinfo import summary
        summary(model, (1, 3, cfg.model.img_h, cfg.model.img_w))
    except ImportError:
        print("(torchinfo not installed, skipping summary)")

    # --- Loss & Optimizer ---
    ctc_loss = torch.nn.CTCLoss(
        blank=cfg.blank_token, reduction="mean", zero_infinity=True
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )

    epochs = cfg.training.epochs
    patience_limit = cfg.training.early_stopping_patience
    best_char_acc = 0.0
    patience = 0

    os.makedirs(cfg.checkpoint.dir, exist_ok=True)
    best_path = cfg.best_path
    latest_path = cfg.latest_path

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for images, targets, target_lengths, _ in pbar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            logits = model(images)  # (N, T, C)
            logits = logits.permute(1, 0, 2)  # (T, N, C) for CTC
            log_probs = F.log_softmax(logits, dim=2)

            T, B = log_probs.size(0), log_probs.size(1)
            input_lengths = torch.full(
                size=(B,), fill_value=T, dtype=torch.long, device=device
            )

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        total_edit_distance = 0
        total_gt_chars = 0

        with torch.no_grad():
            for val_images, _, _, val_texts in val_loader:
                val_images = val_images.to(device)
                val_logits = model(val_images)  # (N, T, C)

                for idx in range(val_images.size(0)):
                    pred_text = ctc_decode(val_logits[idx], cfg.idx_to_char)
                    gt_text = val_texts[idx]
                    total_edit_distance += levenshtein_distance(pred_text, gt_text)
                    total_gt_chars += max(len(gt_text), 1)

        char_accuracy = (1.0 - (total_edit_distance / total_gt_chars)) * 100
        print(
            f"Epoch {epoch} Summary: Loss={epoch_loss:.4f} | Char Accuracy={char_accuracy:.2f}%"
        )

        # --- Checkpointing ---
        torch.save(model, latest_path)

        if char_accuracy > best_char_acc:
            best_char_acc = char_accuracy
            torch.save(model, best_path)
            print(f"Saved best checkpoint (acc={best_char_acc:.2f}%)")
            patience = 0
        else:
            patience += 1

        if patience >= patience_limit:
            print(f"Early stopping at epoch {epoch}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train UltraOCR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ocr_mobilenetv3",
        help="Config name (without .yaml extension, looked up in config/ dir)",
    )
    args = parser.parse_args()
    main(args.config)