"""Train UltraOCR:  python scripts/train.py --config ocr"""

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from ultraocr.config import Config
from ultraocr.model import OCR
from ultraocr.dataset import OCRDataset, collate_fn
from ultraocr.loss import FocalCTCLoss
from ultraocr.utils import ctc_decode, levenshtein_distance, _init_from_config


def main(config_name):
    cfg = Config(f"config/{config_name}.yaml")
    _init_from_config(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Config: {config_name}  |  Device: {device}  |  Charset size: {cfg.num_chars}")

    # --- Datasets ---
    train_dataset = OCRDataset(cfg.dataset.data_json, shuffle=True, num_samples=cfg.dataset.get("num_samples"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collate_fn,
        prefetch_factor=cfg.dataset.prefetch_factor,
        persistent_workers=True,
    )

    val_dataset = OCRDataset(cfg.dataset.val_json, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=collate_fn,
        prefetch_factor=cfg.dataset.prefetch_factor,
        persistent_workers=True,
    )

    # --- Model (architecture is defined in model.py, not in config) ---
    model = OCR(num_of_chars=cfg.num_chars).to(device)

    ctc_loss = FocalCTCLoss(cfg=cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    epochs = cfg.training.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_char_acc = 0.0
    patience = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for images, targets, target_lengths, _ in pbar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            logits = model(images)
            logits = logits.permute(1, 0, 2)
            log_probs = F.log_softmax(logits, dim=2)

            T, B = log_probs.size(0), log_probs.size(1)
            input_lengths = torch.full(
                size=(B,), fill_value=T, dtype=torch.long, device=device
            )

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.training.grad_clip_norm
            )
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
                val_logits = model(val_images)

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
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_acc": best_char_acc,
            "config": cfg.to_dict(),
        }
        torch.save(checkpoint, cfg.checkpoint.latest_path)

        if char_accuracy > best_char_acc:
            best_char_acc = char_accuracy
            torch.save(checkpoint, cfg.checkpoint.best_path)
            print(f"Saved best checkpoint (acc={best_char_acc:.2f}%)")
            patience = 0
        else:
            patience += 1

        if patience >= cfg.training.early_stop_patience:
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
        default="ocr",
        help="Config name (without .yaml extension, looked up in config/ dir)",
    )
    args = parser.parse_args()
    main(args.config)