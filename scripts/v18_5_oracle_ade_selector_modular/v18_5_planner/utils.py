from __future__ import annotations

import os
import json
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(label: Any) -> str:
    if label is None:
        return ""
    if isinstance(label, bytes):
        label = label.decode("utf-8", errors="ignore")
    return str(label).strip().upper()


def resize_bev_grid_np(g: np.ndarray, target_c: int, target_h: int, target_w: int) -> np.ndarray:
    """Resize/pad BEV cache grid to planner config.

    This makes v16.4 robust if a cache was accidentally generated at 80x60 or
    120x90. Spatial resizing is better than the old zero-padding behavior because
    object-aligned grid_sample uses metric coordinates and expects consistent size.
    """
    g = np.asarray(g, dtype=np.float32)
    if g.ndim != 3:
        fixed = np.zeros((target_c, target_h, target_w), dtype=np.float32)
        return fixed

    c_in, h_in, w_in = g.shape
    c_use = min(target_c, c_in)
    spatial = torch.from_numpy(g[:c_use]).unsqueeze(0)
    if (h_in, w_in) != (target_h, target_w):
        spatial = F.interpolate(spatial, size=(target_h, target_w), mode="bilinear", align_corners=False)
    spatial = spatial.squeeze(0).numpy().astype(np.float32)

    out = np.zeros((target_c, target_h, target_w), dtype=np.float32)
    out[:c_use] = spatial
    return out


class StandardScalerNP:
    def __init__(self, skip_indices=None):
        self.mean = None
        self.std = None
        self.skip_indices = set(skip_indices or [])

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0
        for idx in self.skip_indices:
            if 0 <= idx < self.mean.shape[1]:
                self.mean[0, idx] = 0.0
                self.std[0, idx] = 1.0
        return self

    def transform(self, x: np.ndarray):
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray):
        self.fit(x)
        return self.transform(x)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "skip_indices": list(self.skip_indices)}

