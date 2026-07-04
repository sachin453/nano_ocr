"""YAML-based configuration loader for UltraOCR."""

import os
import yaml


class Config:
    """Loads a YAML config file and exposes sections via attribute access."""

    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self._raw = yaml.safe_load(f)

        # --- Charset ---
        self.charset = self._raw.get("charset", "")
        self.blank_token = self._raw.get("blank_token", 0)
        self.num_chars = len(self.charset)

        # Build mapping dicts
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.charset)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.charset)}

        # --- Model ---
        self.model = _DotDict(self._raw.get("model", {}))

        # --- Dataset ---
        self.dataset = _DotDict(self._raw.get("dataset", {}))

        # --- Training ---
        self.training = _DotDict(self._raw.get("training", {}))

        # --- Checkpoint ---
        self.checkpoint = _DotDict(self._raw.get("checkpoint", {}))

    @property
    def best_path(self):
        return os.path.join(self.checkpoint.dir, self.checkpoint.best_name)

    @property
    def latest_path(self):
        return os.path.join(self.checkpoint.dir, self.checkpoint.latest_name)


class _DotDict:
    """Thin wrapper that exposes dict keys as attributes (recursive, read-only)."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = _DotDict(value)
            elif isinstance(value, list):
                value = tuple(value)  # YAML lists → tuples (e.g. out_indices)
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        return repr(self.__dict__)
