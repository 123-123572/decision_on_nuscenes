#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v16.4 High-Resolution Depth-Supervised Real Image LSS BEV Cache Builder

True BEV pipeline:
    nuScenes raw 6-camera images
    + camera intrinsics/extrinsics
    + learned depth bins
    + lift-splat geometry projection
    + sparse LiDAR pseudo-depth supervision
    + stronger boundary / thin-structure supervision
    -> ego-frame BEV cache [N, C=8, H=120, W=90] by default

Main upgrades over v16.3:
1) Higher input resolution and larger hidden dimension:
       IMAGE_H=320, IMAGE_W=512, LSS_HIDDEN_DIM=192
2) Higher BEV grid by default:
       BEV_H=120, BEV_W=90
   The annotation-BEV teacher is resized on the fly if its grid is still 80x60.
3) Stronger image backbone:
       BACKBONE=resnet50 by default.
   If torchvision is unavailable, the script falls back to a local Bottleneck-ResNet.
4) Better sparse depth supervision:
       LiDAR_TOP points are projected into each camera, nearest surface is kept,
       and sparse labels are densified with a small local neighbor kernel.
5) Boundary-sensitive BEV learning:
       BCE + focal + Dice + first-order edge + Sobel edge + local boundary band
       + thin/small-object foreground weighting.

Recommended 8GB sanity run:
    MAX_TRAIN_SAMPLES=2500 MAX_VAL_SAMPLES=400 EPOCHS=6 BATCH_SIZE=1 \
    IMAGE_H=320 IMAGE_W=512 LSS_HIDDEN_DIM=192 DEPTH_BINS=48 \
    BACKBONE=resnet50 BEV_H=120 BEV_W=90 \
    python build_v16_4_highres_depthsup_real_image_lss_bev_cache.py

If CUDA OOM:
    BACKBONE=local_resnet IMAGE_H=288 IMAGE_W=480 LSS_HIDDEN_DIM=160 DEPTH_BINS=32
"""

import os
import json
import math
import pickle
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def transform_matrix(translation, rotation, inverse: bool = False) -> np.ndarray:
    q = Quaternion(rotation)
    R = q.rotation_matrix.astype(np.float32)
    t = np.asarray(translation, dtype=np.float32)
    if inverse:
        R = R.T
        t = -R @ t
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@dataclass
class Config:
    seed: int = 42

    dataroot: str = "/media/ubuntu22/My Passport/myx/nuscenes"
    version: str = "v1.0-trainval"
    manifest_path: str = "/outputs_v8_map_agent/build_manifest_full_v8_map_agent_t6.json"
    shard_dir: str = "/outputs_v8_map_agent/shards_full_v8_map_agent_t6"
    teacher_bev_manifest: str = "/home/ubuntu22/decision_on_nuscenes/outputs_v11_0_annotation_bev_cache/annotation_bev_cache_manifest_v11_0.json"
    out_dir: str = "/home/ubuntu22/decision_on_nuscenes/outputs_v16_4_highres_depthsup_real_image_lss_bev_cache"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # v16.4 high-resolution image-LSS config. Batch size should stay 1 on 8GB GPUs.
    image_h: int = 320
    image_w: int = 512
    out_channels: int = 8
    hidden_dim: int = 192
    depth_bins: int = 48
    depth_min: float = 1.0
    depth_max: float = 60.0
    backbone: str = "resnet50"  # resnet50 | local_resnet | strong_cnn
    pretrained_backbone: bool = False

    # v16.4 actual BEV grid is higher than v16.3. Planner script must use same BEV_H/W.
    bev_h: int = 120
    bev_w: int = 90
    bev_x_min: float = -40.0
    bev_x_max: float = 40.0
    bev_y_min: float = -30.0
    bev_y_max: float = 30.0
    teacher_resize_mode: str = "nearest"  # nearest keeps lane/object labels sharper than bilinear.

    batch_size: int = 1
    num_workers: int = 2
    epochs: int = 8
    lr: float = 2.5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    cache_shard_size: int = 128

    max_train_samples: int = 3000
    max_val_samples: int = 500

    # Stronger foreground / boundary supervision.
    lambda_bce: float = 0.20
    lambda_focal: float = 0.35
    lambda_dice: float = 0.65
    lambda_edge: float = 0.80
    lambda_boundary: float = 0.55
    lambda_sobel: float = 0.30
    lambda_thin: float = 0.35
    lambda_smooth: float = 0.08
    focal_gamma: float = 2.0
    foreground_pos_weight: float = 8.0

    # v16.4: LiDAR sparse depth supervision, with local densification.
    lambda_depth: float = 0.55
    depth_ignore_index: int = -100
    min_depth_points: int = 8
    depth_dilate_kernel: int = 3
    depth_neighbor_decay: float = 0.65
    depth_label_smoothing: float = 0.02


def cfg_from_env() -> Config:
    c = Config()
    c.dataroot = os.getenv("NUSCENES_DATAROOT", c.dataroot)
    c.version = os.getenv("NUSCENES_VERSION", c.version)
    c.manifest_path = os.getenv("MANIFEST_PATH", c.manifest_path)
    c.shard_dir = os.getenv("SHARD_DIR", c.shard_dir)
    c.teacher_bev_manifest = os.getenv("TEACHER_BEV_MANIFEST", c.teacher_bev_manifest)
    c.out_dir = os.getenv("OUT_DIR", c.out_dir)
    c.device = os.getenv("DEVICE", c.device)
    c.image_h = int(os.getenv("IMAGE_H", c.image_h))
    c.image_w = int(os.getenv("IMAGE_W", c.image_w))
    c.hidden_dim = int(os.getenv("LSS_HIDDEN_DIM", c.hidden_dim))
    c.depth_bins = int(os.getenv("DEPTH_BINS", c.depth_bins))
    c.depth_min = float(os.getenv("DEPTH_MIN", c.depth_min))
    c.depth_max = float(os.getenv("DEPTH_MAX", c.depth_max))
    c.backbone = os.getenv("BACKBONE", c.backbone).lower().strip()
    c.pretrained_backbone = os.getenv("PRETRAINED_BACKBONE", "1" if c.pretrained_backbone else "0").lower() in {"1", "true", "yes", "y", "on"}
    c.bev_h = int(os.getenv("BEV_H", c.bev_h))
    c.bev_w = int(os.getenv("BEV_W", c.bev_w))
    c.bev_x_min = float(os.getenv("BEV_X_MIN", c.bev_x_min))
    c.bev_x_max = float(os.getenv("BEV_X_MAX", c.bev_x_max))
    c.bev_y_min = float(os.getenv("BEV_Y_MIN", c.bev_y_min))
    c.bev_y_max = float(os.getenv("BEV_Y_MAX", c.bev_y_max))
    c.teacher_resize_mode = os.getenv("TEACHER_RESIZE_MODE", c.teacher_resize_mode)
    c.batch_size = int(os.getenv("BATCH_SIZE", c.batch_size))
    c.num_workers = int(os.getenv("NUM_WORKERS", c.num_workers))
    c.epochs = int(os.getenv("EPOCHS", c.epochs))
    c.lr = float(os.getenv("LR", c.lr))
    c.weight_decay = float(os.getenv("WEIGHT_DECAY", c.weight_decay))
    c.max_train_samples = int(os.getenv("MAX_TRAIN_SAMPLES", c.max_train_samples))
    c.max_val_samples = int(os.getenv("MAX_VAL_SAMPLES", c.max_val_samples))
    c.cache_shard_size = int(os.getenv("CACHE_SHARD_SIZE", c.cache_shard_size))
    c.lambda_bce = float(os.getenv("LAMBDA_BCE", c.lambda_bce))
    c.lambda_focal = float(os.getenv("LAMBDA_FOCAL", c.lambda_focal))
    c.lambda_dice = float(os.getenv("LAMBDA_DICE", c.lambda_dice))
    c.lambda_edge = float(os.getenv("LAMBDA_EDGE", c.lambda_edge))
    c.lambda_boundary = float(os.getenv("LAMBDA_BOUNDARY", c.lambda_boundary))
    c.lambda_sobel = float(os.getenv("LAMBDA_SOBEL", c.lambda_sobel))
    c.lambda_thin = float(os.getenv("LAMBDA_THIN", c.lambda_thin))
    c.lambda_smooth = float(os.getenv("LAMBDA_SMOOTH", c.lambda_smooth))
    c.focal_gamma = float(os.getenv("FOCAL_GAMMA", c.focal_gamma))
    c.foreground_pos_weight = float(os.getenv("FOREGROUND_POS_WEIGHT", c.foreground_pos_weight))
    c.lambda_depth = float(os.getenv("LAMBDA_DEPTH", c.lambda_depth))
    c.depth_dilate_kernel = int(os.getenv("DEPTH_DILATE_KERNEL", c.depth_dilate_kernel))
    c.depth_neighbor_decay = float(os.getenv("DEPTH_NEIGHBOR_DECAY", c.depth_neighbor_decay))
    c.depth_label_smoothing = float(os.getenv("DEPTH_LABEL_SMOOTHING", c.depth_label_smoothing))
    c.min_depth_points = int(os.getenv("MIN_DEPTH_POINTS", c.min_depth_points))
    return c


def load_annotation_bev_cache(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    manifest = load_json(path)
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for p in manifest.get("cache_files", []):
        if not os.path.exists(p):
            print(f"[WARN] missing teacher cache shard: {p}")
            continue
        data = np.load(p, allow_pickle=True)
        tokens = [str(x) for x in data["sample_tokens"]]
        grid = data["bev_grid"].astype(np.float32)
        valid = data["bev_valid"].astype(np.float32)
        for tok, g, v in zip(tokens, grid, valid):
            cache[tok] = (g, v)
    return cache


def load_split_tokens(manifest_path: str, shard_dir: str, split: str) -> List[str]:
    manifest = load_json(manifest_path)
    shard_names = manifest[f"{split}_shards"]
    tokens: List[str] = []
    for name in shard_names:
        p = name if os.path.isabs(name) else os.path.join(shard_dir, name)
        data = load_pickle(p)
        for s in data:
            if not bool(s.get("history_valid", False)):
                continue
            tok = str(s.get("sample_token", ""))
            if tok:
                tokens.append(tok)
    return tokens

def resize_bev_teacher_np(bev: np.ndarray, cfg: Config) -> np.ndarray:
    """Resize teacher BEV [C,H,W] to cfg BEV grid.

    v11 annotation-BEV is usually 80x60. v16.4 can train a 120x90 image-BEV.
    Nearest keeps lane/object boundaries sharp; bilinear is allowed by env if needed.
    """
    bev = np.asarray(bev, dtype=np.float32)
    if bev.shape[-2:] == (cfg.bev_h, cfg.bev_w):
        return bev
    x = torch.from_numpy(bev).unsqueeze(0)
    mode = str(cfg.teacher_resize_mode).lower()
    if mode == "bilinear":
        y = F.interpolate(x, size=(cfg.bev_h, cfg.bev_w), mode="bilinear", align_corners=False)
    else:
        y = F.interpolate(x, size=(cfg.bev_h, cfg.bev_w), mode="nearest")
    return y.squeeze(0).numpy().astype(np.float32)


class NuScenesRawImageBEVDataset(Dataset):
    def __init__(self, cfg: Config, nusc: NuScenes, tokens: List[str], teacher_cache, max_samples: int = 0):
        self.cfg = cfg
        self.nusc = nusc
        self.tokens = [t for t in tokens if t in teacher_cache]
        if max_samples and max_samples > 0:
            self.tokens = self.tokens[:max_samples]
        self.teacher_cache = teacher_cache
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.tokens)

    def _load_cam(self, sample: Dict[str, Any], channel: str):
        sd = self.nusc.get("sample_data", sample["data"][channel])
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        img_path = os.path.join(self.cfg.dataroot, sd["filename"])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.cfg.image_w, self.cfg.image_h), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        arr = (arr - self.mean) / self.std

        K = np.asarray(cs["camera_intrinsic"], dtype=np.float32)
        K2 = K.copy()
        sx = self.cfg.image_w / float(orig_w)
        sy = self.cfg.image_h / float(orig_h)
        K2[0, :] *= sx
        K2[1, :] *= sy
        sensor2ego = transform_matrix(cs["translation"], cs["rotation"], inverse=False)
        return arr.astype(np.float32), K2.astype(np.float32), sensor2ego.astype(np.float32)


    def _load_lidar_points_ego(self, sample: Dict[str, Any]) -> np.ndarray:
        """Read LIDAR_TOP point cloud and transform points into ego frame.

        nuScenes .pcd.bin stores float32 [x,y,z,intensity,ring] in lidar sensor frame.
        """
        sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        lidar_path = os.path.join(self.cfg.dataroot, sd["filename"])
        pts = np.fromfile(lidar_path, dtype=np.float32)
        if pts.size % 5 != 0:
            pts = pts[: (pts.size // 5) * 5]
        pts = pts.reshape(-1, 5)[:, :3].astype(np.float32)
        if pts.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

        lidar2ego = transform_matrix(cs["translation"], cs["rotation"], inverse=False)
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        pts_ego = pts_h @ lidar2ego.T
        return pts_ego[:, :3].astype(np.float32)

    def _build_sparse_depth_targets(self, pts_ego: np.ndarray, Ks: List[np.ndarray], sensor2egos: List[np.ndarray]):
        """Project ego-frame LiDAR points into each resized camera image and create depth-bin labels.

        Returns:
            depth_target: [6,Hf,Wf] int64, ignore_index for empty cells.
            depth_mask  : [6,Hf,Wf] float32, 1 where depth supervision exists.
            depth_weight: [6,Hf,Wf] float32, confidence weight for sparse/densified labels.
        """
        cfg = self.cfg
        Hf = cfg.image_h // 8
        Wf = cfg.image_w // 8
        ignore = int(cfg.depth_ignore_index)
        target = np.full((len(CAMERA_CHANNELS), Hf, Wf), ignore, dtype=np.int64)
        mask = np.zeros((len(CAMERA_CHANNELS), Hf, Wf), dtype=np.float32)
        weight = np.zeros((len(CAMERA_CHANNELS), Hf, Wf), dtype=np.float32)
        nearest_depth = np.full((len(CAMERA_CHANNELS), Hf, Wf), np.inf, dtype=np.float32)
        depth_values = np.linspace(cfg.depth_min, cfg.depth_max, cfg.depth_bins).astype(np.float32)

        if pts_ego.shape[0] == 0:
            return target, mask, weight

        pts_h = np.concatenate([pts_ego, np.ones((pts_ego.shape[0], 1), dtype=np.float32)], axis=1)

        for cam_i, (K, s2e) in enumerate(zip(Ks, sensor2egos)):
            ego2cam = np.linalg.inv(s2e).astype(np.float32)
            pts_cam = pts_h @ ego2cam.T
            z = pts_cam[:, 2]
            valid = (z > cfg.depth_min) & (z < cfg.depth_max)
            if not np.any(valid):
                continue

            pc = pts_cam[valid, :3]
            z_valid = z[valid]
            uvw = pc @ K.T
            denom = np.clip(uvw[:, 2], 1e-6, None)
            u = uvw[:, 0] / denom
            v = uvw[:, 1] / denom

            inside = (u >= 0) & (u < cfg.image_w) & (v >= 0) & (v < cfg.image_h)
            if not np.any(inside):
                continue

            u = u[inside]
            v = v[inside]
            z_in = z_valid[inside]

            fx = np.floor(u / cfg.image_w * Wf).astype(np.int64)
            fy = np.floor(v / cfg.image_h * Hf).astype(np.int64)
            fx = np.clip(fx, 0, Wf - 1)
            fy = np.clip(fy, 0, Hf - 1)

            # Process nearer points first so duplicated feature cells keep the closest visible surface.
            order = np.argsort(z_in)
            for j in order:
                yy, xx = int(fy[j]), int(fx[j])
                zz = float(z_in[j])
                if zz < nearest_depth[cam_i, yy, xx]:
                    nearest_depth[cam_i, yy, xx] = zz
                    target[cam_i, yy, xx] = int(np.argmin(np.abs(depth_values - zz)))
                    mask[cam_i, yy, xx] = 1.0
                    weight[cam_i, yy, xx] = 1.0

                    # v16.4 pseudo-depth densification: fill a tiny local neighborhood
                    # around each LiDAR projection. This gives the depth head more useful
                    # gradients without pretending the whole image has dense ground truth.
                    k = max(1, int(cfg.depth_dilate_kernel))
                    if k > 1:
                        r = k // 2
                        for ddy in range(-r, r + 1):
                            for ddx in range(-r, r + 1):
                                ny, nx = yy + ddy, xx + ddx
                                if ny < 0 or ny >= Hf or nx < 0 or nx >= Wf:
                                    continue
                                dist = abs(ddy) + abs(ddx)
                                if dist == 0:
                                    continue
                                wgt = float(cfg.depth_neighbor_decay) ** float(dist)
                                # Only overwrite empty cells, or weaker densified labels.
                                if mask[cam_i, ny, nx] < 0.5 or weight[cam_i, ny, nx] < wgt:
                                    target[cam_i, ny, nx] = target[cam_i, yy, xx]
                                    mask[cam_i, ny, nx] = 1.0
                                    weight[cam_i, ny, nx] = wgt

        return target, mask, weight

    def __getitem__(self, idx):
        tok = self.tokens[idx]
        sample = self.nusc.get("sample", tok)
        imgs, Ks, s2es, valids = [], [], [], []
        for ch in CAMERA_CHANNELS:
            try:
                img, K, s2e = self._load_cam(sample, ch)
                valid = 1.0
            except Exception as e:
                print(f"[WARN] failed loading {tok}/{ch}: {e}")
                img = np.zeros((3, self.cfg.image_h, self.cfg.image_w), dtype=np.float32)
                K = np.eye(3, dtype=np.float32)
                s2e = np.eye(4, dtype=np.float32)
                valid = 0.0
            imgs.append(img)
            Ks.append(K)
            s2es.append(s2e)
            valids.append(valid)
        teacher, teacher_valid = self.teacher_cache[tok]
        teacher = resize_bev_teacher_np(teacher, self.cfg)
        try:
            pts_ego = self._load_lidar_points_ego(sample)
            depth_target, depth_mask, depth_weight = self._build_sparse_depth_targets(pts_ego, Ks, s2es)
        except Exception as e:
            print(f"[WARN] failed building pseudo-depth for {tok}: {e}")
            Hf, Wf = self.cfg.image_h // 8, self.cfg.image_w // 8
            depth_target = np.full((len(CAMERA_CHANNELS), Hf, Wf), int(self.cfg.depth_ignore_index), dtype=np.int64)
            depth_mask = np.zeros((len(CAMERA_CHANNELS), Hf, Wf), dtype=np.float32)
            depth_weight = np.zeros((len(CAMERA_CHANNELS), Hf, Wf), dtype=np.float32)

        return {
            "sample_token": tok,
            "images": torch.tensor(np.stack(imgs), dtype=torch.float32),
            "intrinsics": torch.tensor(np.stack(Ks), dtype=torch.float32),
            "sensor2ego": torch.tensor(np.stack(s2es), dtype=torch.float32),
            "camera_valid": torch.tensor(np.asarray(valids, dtype=np.float32), dtype=torch.float32),
            "teacher_bev": torch.tensor(teacher, dtype=torch.float32),
            "teacher_valid": torch.tensor(teacher_valid, dtype=torch.float32),
            "depth_target": torch.tensor(depth_target, dtype=torch.long),
            "depth_mask": torch.tensor(depth_mask, dtype=torch.float32),
            "depth_weight": torch.tensor(depth_weight, dtype=torch.float32),
        }


def collate_fn(batch):
    out = {"sample_tokens": [b["sample_token"] for b in batch]}
    for k in ["images", "intrinsics", "sensor2ego", "camera_valid", "teacher_bev", "teacher_valid", "depth_target", "depth_mask", "depth_weight"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, out_ch // 8)), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvGNAct(ch, ch, 3, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, ch // 8)), num_channels=ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class StrongBackbone(nn.Module):
    """A still-light backbone, but much less toy than v16.0 TinyBackbone.

    Downsample factor is 8, so IMAGE_H=256, IMAGE_W=448 gives feature map 32x56.
    """
    def __init__(self, hidden: int):
        super().__init__()
        c1 = max(32, hidden // 4)
        c2 = max(64, hidden // 2)
        self.stem = nn.Sequential(
            ConvGNAct(3, c1, 5, 2),
            ResidualBlock(c1),
            ConvGNAct(c1, c2, 3, 2),
            ResidualBlock(c2),
            ConvGNAct(c2, hidden, 3, 2),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )

    def forward(self, x):
        return self.stem(x)




class LocalBottleneckBlock(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, mid_ch, 1, 1)
        self.conv2 = ConvGNAct(mid_ch, mid_ch, 3, stride)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, out_ch // 8)), num_channels=out_ch),
        )
        self.short = None
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=max(1, min(8, out_ch // 8)), num_channels=out_ch),
            )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        s = x if self.short is None else self.short(x)
        return self.act(y + s)


class LocalResNetBackbone(nn.Module):
    """ResNet-like fallback with output stride 8 and GroupNorm.

    This is heavier than the old v16.3 StrongBackbone, but safer than requiring
    torchvision on every machine.
    """
    def __init__(self, hidden: int):
        super().__init__()
        c1 = max(48, hidden // 4)
        c2 = max(96, hidden // 2)
        c3 = hidden
        self.net = nn.Sequential(
            ConvGNAct(3, c1, 7, 2),
            LocalBottleneckBlock(c1, c1, c1, 1),
            LocalBottleneckBlock(c1, c1, c1, 1),
            LocalBottleneckBlock(c1, c2 // 2, c2, 2),
            LocalBottleneckBlock(c2, c2 // 2, c2, 1),
            LocalBottleneckBlock(c2, c3 // 2, c3, 2),
            LocalBottleneckBlock(c3, c3 // 2, c3, 1),
            LocalBottleneckBlock(c3, c3 // 2, c3, 1),
        )

    def forward(self, x):
        return self.net(x)


class TorchvisionResNet50Backbone(nn.Module):
    """ResNet50 feature extractor with output stride 8.

    Uses conv1+bn+relu+maxpool+layer1+layer2, then projects to hidden_dim.
    No internet is needed when PRETRAINED_BACKBONE=0, which is the default.
    """
    def __init__(self, hidden: int, pretrained: bool = False):
        super().__init__()
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            m = resnet50(weights=weights)
        except Exception as e:
            raise RuntimeError(f"torchvision resnet50 unavailable: {e}")
        self.body = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1,
            m.layer2,
        )
        self.proj = nn.Sequential(
            nn.Conv2d(512, hidden, 1, bias=False),
            nn.GroupNorm(num_groups=max(1, min(8, hidden // 8)), num_channels=hidden),
            nn.SiLU(inplace=True),
            ResidualBlock(hidden),
        )

    def forward(self, x):
        return self.proj(self.body(x))


def build_image_backbone(cfg: Config) -> nn.Module:
    name = str(cfg.backbone).lower()
    if name in {"resnet50", "torchvision_resnet50"}:
        try:
            return TorchvisionResNet50Backbone(cfg.hidden_dim, pretrained=cfg.pretrained_backbone)
        except Exception as e:
            print(f"[WARN] BACKBONE=resnet50 failed, falling back to local_resnet: {e}")
            return LocalResNetBackbone(cfg.hidden_dim)
    if name in {"local_resnet", "resnet", "bottleneck"}:
        return LocalResNetBackbone(cfg.hidden_dim)
    if name in {"strong_cnn", "cnn", "v16_3"}:
        return StrongBackbone(cfg.hidden_dim)
    raise ValueError(f"Unknown BACKBONE={cfg.backbone}. Use resnet50 | local_resnet | strong_cnn")

class RealImageLSSBEV(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_image_backbone(cfg)
        self.depth_head = nn.Conv2d(cfg.hidden_dim, cfg.depth_bins, 1)
        self.feat_head = nn.Conv2d(cfg.hidden_dim, cfg.out_channels, 1)
        self.post = nn.Sequential(
            nn.Conv2d(cfg.out_channels, cfg.out_channels * 2, 3, padding=1),
            nn.GroupNorm(4, cfg.out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.out_channels * 2, cfg.out_channels, 3, padding=1),
        )
        self.register_buffer("depth_values", torch.linspace(cfg.depth_min, cfg.depth_max, cfg.depth_bins))

    def _pixel_grid(self, Hf: int, Wf: int, device, dtype):
        ys = (torch.arange(Hf, device=device, dtype=dtype) + 0.5) * (self.cfg.image_h / Hf)
        xs = (torch.arange(Wf, device=device, dtype=dtype) + 0.5) * (self.cfg.image_w / Wf)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pix = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        return pix.reshape(-1, 3)

    def forward(self, images, intrinsics, sensor2ego, camera_valid, return_depth: bool = False):
        cfg = self.cfg
        B, N, _, H, W = images.shape
        feat = self.backbone(images.reshape(B * N, 3, H, W))
        _, _, Hf, Wf = feat.shape
        D, C = cfg.depth_bins, cfg.out_channels
        P = Hf * Wf
        dtype, device = images.dtype, images.device

        depth_logits_raw = self.depth_head(feat)  # [B*N,D,Hf,Wf]
        depth_prob = torch.softmax(depth_logits_raw, dim=1).reshape(B, N, D, P)
        depth_logits = depth_logits_raw.reshape(B, N, D, Hf, Wf)
        img_feat = torch.sigmoid(self.feat_head(feat)).reshape(B, N, C, P)
        pix = self._pixel_grid(Hf, Wf, device, dtype)
        depths = self.depth_values.to(device=device, dtype=dtype)

        bev = images.new_zeros((B, C, cfg.bev_h, cfg.bev_w))
        cnt = images.new_zeros((B, 1, cfg.bev_h, cfg.bev_w))

        for b in range(B):
            for n in range(N):
                if camera_valid[b, n] <= 0.5:
                    continue
                K_inv = torch.inverse(intrinsics[b, n])
                rays = pix @ K_inv.T
                pts_cam = rays.unsqueeze(0) * depths.view(D, 1, 1)
                ones = torch.ones((D, P, 1), device=device, dtype=dtype)
                pts_h = torch.cat([pts_cam, ones], dim=-1).reshape(D * P, 4)
                pts_ego = pts_h @ sensor2ego[b, n].T
                x = pts_ego[:, 0]
                y = pts_ego[:, 1]
                gx = ((x - cfg.bev_x_min) / (cfg.bev_x_max - cfg.bev_x_min) * cfg.bev_h).long()
                gy = ((y - cfg.bev_y_min) / (cfg.bev_y_max - cfg.bev_y_min) * cfg.bev_w).long()
                inside = (gx >= 0) & (gx < cfg.bev_h) & (gy >= 0) & (gy < cfg.bev_w)
                if inside.sum().item() == 0:
                    continue
                linear = gx * cfg.bev_w + gy
                w = depth_prob[b, n].reshape(D * P)
                f = img_feat[b, n].transpose(0, 1).unsqueeze(0).expand(D, P, C).reshape(D * P, C)
                vals = f * w.unsqueeze(-1)
                idx = linear[inside]
                vals = vals[inside]
                flat_bev = bev[b].reshape(C, -1)
                flat_cnt = cnt[b].reshape(1, -1)
                flat_bev.scatter_add_(1, idx.view(1, -1).expand(C, -1), vals.T)
                flat_cnt.scatter_add_(1, idx.view(1, -1), w[inside].view(1, -1))

        bev = bev / cnt.clamp_min(1e-4)
        bev_out = torch.sigmoid(self.post(bev))
        if return_depth:
            return bev_out, depth_logits
        return bev_out


def edge_map(x):
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    return F.pad(dx, (0, 1, 0, 0)) + F.pad(dy, (0, 0, 0, 1))

def sobel_edge_map(x):
    B, C, H, W = x.shape
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype).view(1, 1, 3, 3)
    kx = kx.expand(C, 1, 3, 3)
    ky = ky.expand(C, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def dilate2d(x, k: int = 3):
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def dice_loss(pred, target, mask, eps=1e-6):
    pred = pred * mask
    target = target * mask
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def weighted_bce_loss(pred, target, mask, cfg: Config):
    # Teacher BEV is sparse; without positive weighting, model can win by predicting background.
    weight = 1.0 + (cfg.foreground_pos_weight - 1.0) * target
    raw = F.binary_cross_entropy(pred, target, reduction="none")
    denom = (mask * weight).sum().clamp_min(1.0)
    return (raw * weight * mask).sum() / denom


def focal_bce_loss(pred, target, mask, cfg: Config):
    raw = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target > 0.5, pred, 1.0 - pred)
    focal = ((1.0 - pt).clamp_min(1e-4) ** cfg.focal_gamma) * raw
    # Extra foreground weighting because lane/object BEV cells are rare.
    weight = 1.0 + (cfg.foreground_pos_weight - 1.0) * target
    denom = (mask * weight).sum().clamp_min(1.0)
    return (focal * weight * mask).sum() / denom


def boundary_weighted_edge_loss(pred, target, mask, cfg: Config):
    pred_e = edge_map(pred)
    target_e = edge_map(target)
    # Dilate teacher edges so a one-cell geometric miss is punished, but not treated as total failure.
    edge_w = 1.0 + 4.0 * dilate2d((target_e > 0.05).float(), k=3)
    denom = (edge_w * mask).sum().clamp_min(1.0)
    return (F.smooth_l1_loss(pred_e, target_e, reduction="none") * edge_w * mask).sum() / denom


def local_boundary_loss(pred, target, mask, cfg: Config):
    # Focus directly around teacher foreground and teacher boundary.
    fg_band = dilate2d((target > 0.2).float(), k=5)
    ed_band = dilate2d((edge_map(target) > 0.05).float(), k=5)
    band = torch.clamp(fg_band + ed_band, 0, 1)
    weight = 1.0 + 5.0 * band
    denom = (weight * mask).sum().clamp_min(1.0)
    return (F.smooth_l1_loss(pred, target, reduction="none") * weight * mask).sum() / denom

def sobel_boundary_loss(pred, target, mask, cfg: Config):
    pred_s = sobel_edge_map(pred)
    target_s = sobel_edge_map(target)
    edge_band = dilate2d((target_s > 0.05).float(), k=3)
    weight = 1.0 + 6.0 * edge_band
    denom = (weight * mask).sum().clamp_min(1.0)
    return (F.smooth_l1_loss(pred_s, target_s, reduction="none") * weight * mask).sum() / denom


def thin_structure_loss(pred, target, mask, cfg: Config):
    """Give extra gradient to thin lane/object cells so the output does not become a gray blob."""
    fg = (target > 0.15).float()
    thin_band = torch.clamp(fg + dilate2d((edge_map(target) > 0.05).float(), k=3), 0, 1)
    weight = 1.0 + 8.0 * thin_band
    # Use BCE here because small targets disappear easily under SmoothL1.
    raw = F.binary_cross_entropy(pred.clamp(1e-4, 1 - 1e-4), target, reduction="none")
    denom = (weight * mask).sum().clamp_min(1.0)
    return (raw * weight * mask).sum() / denom


def compute_loss(pred, target, valid, cfg: Config):
    target = target.detach().clamp(0, 1)
    pred = pred.clamp(1e-4, 1 - 1e-4)
    mask = (valid > 0.5).view(-1, 1, 1, 1).to(pred.dtype)
    denom = (mask.sum() * pred.size(1) * pred.size(2) * pred.size(3)).clamp_min(1.0)

    bce = weighted_bce_loss(pred, target, mask, cfg)
    focal = focal_bce_loss(pred, target, mask, cfg)
    dice = dice_loss(pred, target, mask)
    smooth = F.smooth_l1_loss(pred * mask, target * mask, reduction="sum") / denom
    edge = boundary_weighted_edge_loss(pred, target, mask, cfg)
    boundary = local_boundary_loss(pred, target, mask, cfg)
    sobel = sobel_boundary_loss(pred, target, mask, cfg)
    thin = thin_structure_loss(pred, target, mask, cfg)

    loss = (
        cfg.lambda_bce * bce
        + cfg.lambda_focal * focal
        + cfg.lambda_dice * dice
        + cfg.lambda_edge * edge
        + cfg.lambda_boundary * boundary
        + cfg.lambda_sobel * sobel
        + cfg.lambda_thin * thin
        + cfg.lambda_smooth * smooth
    )
    return loss, {
        "bce": bce.item(),
        "focal": focal.item(),
        "dice": dice.item(),
        "edge": edge.item(),
        "boundary": boundary.item(),
        "sobel": sobel.item(),
        "thin": thin.item(),
        "smooth": smooth.item(),
    }



def pseudo_depth_loss(depth_logits, depth_target, depth_mask, depth_weight, cfg: Config):
    """Cross-entropy on sparse/densified LiDAR-projected camera depth bins.

    depth_logits: [B,6,D,Hf,Wf]
    depth_target: [B,6,Hf,Wf], ignore_index for empty cells
    depth_mask: [B,6,Hf,Wf]
    depth_weight: [B,6,Hf,Wf], 1.0 for real projected point, smaller for neighbors
    """
    B, N, D, Hf, Wf = depth_logits.shape
    valid = depth_mask > 0.5
    if valid.sum().item() < int(cfg.min_depth_points):
        return depth_logits.new_tensor(0.0), 0.0
    logits = depth_logits.reshape(B * N, D, Hf, Wf)
    target = depth_target.reshape(B * N, Hf, Wf)
    ce = F.cross_entropy(
        logits,
        target,
        ignore_index=int(cfg.depth_ignore_index),
        reduction="none",
        label_smoothing=float(cfg.depth_label_smoothing),
    )
    valid_flat = valid.reshape(B * N, Hf, Wf).to(ce.dtype)
    w = depth_weight.reshape(B * N, Hf, Wf).to(ce.dtype).clamp_min(0.0)
    w = torch.where(valid_flat > 0.5, w.clamp_min(0.05), torch.zeros_like(w))
    loss = (ce * w).sum() / w.sum().clamp_min(1.0)
    return loss, float(valid_flat.sum().item() / max(B, 1))


def make_loader(ds, cfg: Config, shuffle: bool):
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers,
                      pin_memory=(cfg.device == "cuda"), collate_fn=collate_fn)


def move_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_epoch(model, loader, opt, cfg: Config, epoch: int):
    model.train()
    total, n = 0.0, 0
    parts_sum = {"bce": 0.0, "focal": 0.0, "dice": 0.0, "edge": 0.0, "boundary": 0.0, "sobel": 0.0, "thin": 0.0, "smooth": 0.0, "depth": 0.0, "depth_pts": 0.0}
    for i, batch in enumerate(loader, 1):
        batch = move_batch(batch, cfg.device)
        opt.zero_grad(set_to_none=True)
        pred, depth_logits = model(batch["images"], batch["intrinsics"], batch["sensor2ego"], batch["camera_valid"], return_depth=True)
        bev_loss, parts = compute_loss(pred, batch["teacher_bev"], batch["teacher_valid"], cfg)
        dloss, dpts = pseudo_depth_loss(depth_logits, batch["depth_target"], batch["depth_mask"], batch["depth_weight"], cfg)
        loss = bev_loss + cfg.lambda_depth * dloss
        parts["depth"] = float(dloss.item())
        parts["depth_pts"] = float(dpts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        bs = pred.size(0)
        n += bs
        total += loss.item() * bs
        for k, v in parts.items():
            parts_sum[k] += v * bs
        if i % 20 == 0:
            print(f"[train e{epoch:03d} i{i:04d}] loss={loss.item():.5f} depth={parts['depth']:.5f} pts={parts['depth_pts']:.1f} bce={parts['bce']:.5f} focal={parts['focal']:.5f} dice={parts['dice']:.5f} edge={parts['edge']:.5f} boundary={parts['boundary']:.5f} sobel={parts['sobel']:.5f} thin={parts['thin']:.5f}")
    return {"loss": total / max(n, 1), **{k: v / max(n, 1) for k, v in parts_sum.items()}}


@torch.no_grad()
def eval_epoch(model, loader, cfg: Config):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, cfg.device)
        pred, depth_logits = model(batch["images"], batch["intrinsics"], batch["sensor2ego"], batch["camera_valid"], return_depth=True)
        bev_loss, _ = compute_loss(pred, batch["teacher_bev"], batch["teacher_valid"], cfg)
        dloss, _ = pseudo_depth_loss(depth_logits, batch["depth_target"], batch["depth_mask"], batch["depth_weight"], cfg)
        loss = bev_loss + cfg.lambda_depth * dloss
        bs = pred.size(0)
        n += bs
        total += loss.item() * bs
    return {"loss": total / max(n, 1)}


@torch.no_grad()
def save_preview(model, loader, cfg: Config, out_dir: str, max_samples: int = 12):
    import matplotlib.pyplot as plt
    ensure_dir(out_dir)
    model.eval()
    saved = 0
    for batch in loader:
        batch = move_batch(batch, cfg.device)
        pred = model(batch["images"], batch["intrinsics"], batch["sensor2ego"], batch["camera_valid"]).cpu().numpy()
        target = batch["teacher_bev"].cpu().numpy()
        for i in range(pred.shape[0]):
            if saved >= max_samples:
                return
            p = pred[i].max(axis=0)
            t = target[i].max(axis=0)
            e = np.abs(p - t)
            fig = plt.figure(figsize=(12, 4))
            for j, (arr, title) in enumerate([(t, "Teacher BEV"), (p, "Real-image LSS BEV"), (e, "Abs Error")]):
                ax = fig.add_subplot(1, 3, j + 1)
                im = ax.imshow(arr, origin="lower")
                ax.set_title(title)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"bev_preview_{saved:04d}.png"), dpi=160)
            plt.close(fig)
            saved += 1


@torch.no_grad()
def export_split(model, ds, cfg: Config, split: str, out_dir: str) -> List[str]:
    loader = make_loader(ds, cfg, shuffle=False)
    model.eval()
    files = []
    toks, bevs, valids = [], [], []
    shard = 0

    def flush():
        nonlocal toks, bevs, valids, shard
        if not toks:
            return
        path = os.path.join(out_dir, f"{split}_image_bev_cache_{shard:04d}.npz")
        np.savez_compressed(
            path,
            sample_tokens=np.asarray(toks, dtype=object),
            image_bev_feat=np.stack(bevs).astype(np.float32),
            image_bev_valid=np.stack(valids).astype(np.float32),
        )
        print(f"[export {split}] shard={shard:04d} samples={len(toks)} -> {path}")
        files.append(path)
        toks, bevs, valids = [], [], []
        shard += 1

    for batch in loader:
        batch_gpu = move_batch(batch, cfg.device)
        pred = model(batch_gpu["images"], batch_gpu["intrinsics"], batch_gpu["sensor2ego"], batch_gpu["camera_valid"]).cpu().numpy()
        valid = (batch["camera_valid"].sum(dim=1, keepdim=True).numpy() > 0).astype(np.float32)
        for tok, bev, v in zip(batch["sample_tokens"], pred, valid):
            toks.append(str(tok))
            bevs.append(bev)
            valids.append(v)
            if len(toks) >= cfg.cache_shard_size:
                flush()
    flush()
    return files


def main():
    cfg = cfg_from_env()
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)
    print("=" * 88)
    print("v16.4 HIGH-RES DEPTH-SUPERVISED raw-image LSS BEV cache builder")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    print("=" * 88)

    nusc = NuScenes(version=cfg.version, dataroot=cfg.dataroot, verbose=True)
    teacher_cache = load_annotation_bev_cache(cfg.teacher_bev_manifest)
    print(f"Teacher annotation-BEV samples: {len(teacher_cache)}")

    train_tokens = load_split_tokens(cfg.manifest_path, cfg.shard_dir, "train")
    val_tokens = load_split_tokens(cfg.manifest_path, cfg.shard_dir, "val")
    train_ds = NuScenesRawImageBEVDataset(cfg, nusc, train_tokens, teacher_cache, cfg.max_train_samples)
    val_ds = NuScenesRawImageBEVDataset(cfg, nusc, val_tokens, teacher_cache, cfg.max_val_samples)
    print(f"Train samples used: {len(train_ds)}")
    print(f"Val samples used  : {len(val_ds)}")

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)

    model = RealImageLSSBEV(cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_loss = math.inf
    best_path = os.path.join(cfg.out_dir, "best_v16_4_highres_depthsup_real_image_lss_bev.pt")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        tr = train_epoch(model, train_loader, opt, cfg, epoch)
        va = eval_epoch(model, val_loader, cfg)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(f"[epoch {epoch:03d}] train_loss={tr['loss']:.5f} val_loss={va['loss']:.5f} depth={tr['depth']:.5f} pts={tr['depth_pts']:.1f} bce={tr['bce']:.5f} focal={tr['focal']:.5f} dice={tr['dice']:.5f} edge={tr['edge']:.5f} boundary={tr['boundary']:.5f} sobel={tr['sobel']:.5f} thin={tr['thin']:.5f}")
        if va["loss"] < best_loss:
            best_loss = va["loss"]
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg), "history": history}, best_path)
            print(f"🔥 saved best LSS BEV model: {best_path}")

    ckpt = torch.load(best_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    preview_dir = os.path.join(cfg.out_dir, "bev_preview")
    save_preview(model, val_loader, cfg, preview_dir, max_samples=12)
    print(f"BEV previews saved to: {preview_dir}")

    train_files = export_split(model, train_ds, cfg, "train", cfg.out_dir)
    val_files = export_split(model, val_ds, cfg, "val", cfg.out_dir)
    manifest = {
        "version": "v16.4_highres_depthsup_real_image_lss_bev_cache",
        "description": "high-resolution depth-supervised raw image + calibration + densified LiDAR pseudo-depth + boundary-sensitive lift-splat geometry BEV cache; default 320x512, D=48, hidden=192, BEV=120x90",
        "config": asdict(cfg),
        "model_path": best_path,
        "train_cache_shards": train_files,
        "val_cache_shards": val_files,
        "cache_files": train_files + val_files,
        "bev_key": "image_bev_feat",
        "valid_key": "image_bev_valid",
        "sample_token_key": "sample_tokens",
    }
    manifest_path = os.path.join(cfg.out_dir, "image_bev_cache_manifest_v16_4.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(os.path.join(cfg.out_dir, "train_history_v16_4.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
