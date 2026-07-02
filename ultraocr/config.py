"""Configuration loader for UltraOCR.

Each model experiment has its own standalone YAML config file in config/.
Usage:
    cfg = Config("config/ocr.yaml")
    print(cfg.model.conv_channels)
"""

import os
import copy
import yaml


class Config:
    """Loads a single YAML config file and provides attribute-style access."""

    def __init__(self, path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, path)

        with open(full_path, "r") as f:
            data = yaml.safe_load(f)

        # Build char mappings from the literal charset string
        chars = data.get("charset", "")
        data["char_to_idx"] = {c: i + 1 for i, c in enumerate(chars)}
        data["idx_to_char"] = {i + 1: c for i, c in enumerate(chars)}
        data["num_chars"] = len(chars)

        self._data = data

        # Expose top-level keys and nested dicts as attributes
        _plain = {"char_to_idx", "idx_to_char"}
        for key, value in data.items():
            if isinstance(value, dict) and key not in _plain:
                setattr(self, key, _AttrDict(value))
            else:
                setattr(self, key, value)

    def to_dict(self):
        return copy.deepcopy(self._data)

    def __repr__(self):
        return f"Config({self._data})"


class _AttrDict:
    """Dict wrapper with attribute access for string keys."""

    def __init__(self, data):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                nested = _AttrDict(value)
                self._data[key] = nested
                if isinstance(key, str) and key.isidentifier():
                    setattr(self, key, nested)
            elif isinstance(key, str) and key.isidentifier():
                setattr(self, key, value)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __contains__(self, key):
        return key in self._data

    def __repr__(self):
        return repr(self._data)

    def to_dict(self):
        return copy.deepcopy(self._data)