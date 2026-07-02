"""UltraOCR package.

Provides the core model, dataset, and utility modules for the OCR project.
"""

from .config import Config
from .model import OCR
from .dataset import OCRDataset, collate_fn
from .loss import FocalCTCLoss
from .utils import (
    generate_simple_synthetic,
    ctc_decode,
    levenshtein_distance,
    CHARS,
    char_to_idx,
    idx_to_char,
    BLANK_IDX,
    IMG_H,
    IMG_W,
)

__all__ = [
    "Config",
    "OCR",
    "OCRDataset",
    "collate_fn",
    "FocalCTCLoss",
    "generate_simple_synthetic",
    "ctc_decode",
    "levenshtein_distance",
    "CHARS",
    "char_to_idx",
    "idx_to_char",
    "BLANK_IDX",
    "IMG_H",
    "IMG_W",
]
