#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v16.4.1 Partial End-to-End Image-BEV Planner on nuScenes

What this script does:
    6 raw nuScenes camera images
      + camera intrinsics / extrinsics
      -> RealImageLSSBEV online BEV encoder
      -> MapAgentEgoPlanner
      -> behavior class + K-mode trajectories + score

This is NOT the old offline-cache planner. It runs the image-BEV encoder inside
forward(), so planning losses can optionally backpropagate into selected BEV
encoder layers.

Default training mode for 8GB GPU:
    - Load pretrained v16.4.1 BEV encoder checkpoint.
    - Load pretrained v16.4.1 planner base checkpoint if available.
    - Freeze image backbone.
    - Unfreeze only BEV feat_head + post, optionally depth_head.
    - Train planner + trainable BEV neck online.

Recommended first run on RTX 5070 Laptop 8GB:

    cd /home/ubuntu22/decision_on_nuscenes/scripts

    NUSCENES_DATAROOT="/media/ubuntu22/My Passport/myx/nuscenes" \
    MANIFEST_PATH=/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/build_manifest_full_v8_map_agent_t6.json \
    SHARD_DIR=/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/shards_full_v8_map_agent_t6 \
    BEV_ENCODER_CKPT=/home/ubuntu22/decision_on_nuscenes/outputs_v16_4_1_highres_depthsup_bgclean_real_image_lss_bev_cache/best_v16_4_1_highres_depthsup_bgclean_real_image_lss_bev.pt \
    TEACHER_BEV_MANIFEST=/home/ubuntu22/decision_on_nuscenes/outputs_v11_0_annotation_bev_cache/annotation_bev_cache_manifest_v11_0.json \
    BASE_CKPT_PATH=/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v16_4_1_bgclean_highres_depthsup_image_bev/v16_4_1_base/best_v16_4_1_base_model.pt \
    SAVE_DIR=/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v16_4_1_partial_end2end \
    IMAGE_H=224 IMAGE_W=384 LSS_HIDDEN_DIM=192 DEPTH_BINS=48 BACKBONE=resnet50 \
    BEV_H=80 BEV_W=60 BATCH_SIZE=1 GRAD_ACCUM_STEPS=8 USE_AMP=1 \
    E2E_MAX_TRAIN=2000 E2E_MAX_VAL=400 EPOCHS=12 \
    python train_v16_4_1_partial_end2end_image_bev_planner.py

For a stronger run after sanity pass, keep LSS_HIDDEN_DIM / DEPTH_BINS / BACKBONE
consistent with the pretrained BEV checkpoint; only raise image/BEV resolution:
    IMAGE_H=288 IMAGE_W=480 LSS_HIDDEN_DIM=192 DEPTH_BINS=48 BACKBONE=resnet50 BEV_H=120 BEV_W=90

If you change LSS_HIDDEN_DIM / DEPTH_BINS / BACKBONE, the pretrained BEV checkpoint
will not load cleanly. That is not “端到端”, that is 重新训练 BEV。

Key environment switches:
    FREEZE_BEV_BACKBONE=1          default, strongly recommended
    TRAIN_BEV_FEAT_HEAD=1          default
    TRAIN_BEV_POST=1               default
    TRAIN_BEV_DEPTH_HEAD=0         default, turn on after stable
    FREEZE_PLANNER=0               default
    LAMBDA_E2E_BEV=0.10            auxiliary teacher BEV loss
    LAMBDA_E2E_DEPTH=0.03          auxiliary sparse depth loss
    LAMBDA_E2E_TRAJ_GT=3.0         planning loss remains dominant
"""

import os
import sys
import json
import math
import pickle
import random
import importlib.util
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nuscenes.nuscenes import NuScenes


# ============================================================
# Dynamic imports: use the two existing v16.4.1 scripts directly.
# Put this script in the same folder as them, or pass env paths.
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

BEV_SCRIPT_PATH = os.getenv(
    "BEV_SCRIPT_PATH",
    os.path.join(THIS_DIR, "build_v16_4_1_bgclean_depthsup_real_image_lss_bev_cache.py"),
)
PLANNER_SCRIPT_PATH = os.getenv(
    "PLANNER_SCRIPT_PATH",
    os.path.join(THIS_DIR, "train_transformer_planning_v16_4_1_bgclean_highres_depthsup_image_bev_planner_full_pipeline.py"),
)


def import_by_path(module_name: str, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required script not found: {path}\n"
            f"Set {module_name.upper()}_PATH or put this file next to the original v16.4.1 scripts."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


bmod = import_by_path("v1641_bev_builder", BEV_SCRIPT_PATH)
pmod = import_by_path("v1641_planner", PLANNER_SCRIPT_PATH)


# ============================================================
# Small config helpers
# ============================================================

def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


class E2EConfig:
    seed: int = env_int("SEED", 42)
    save_dir: str = os.getenv("SAVE_DIR", "/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v16_4_1_partial_end2end")
    device: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    batch_size: int = env_int("BATCH_SIZE", 1)
    num_workers: int = env_int("NUM_WORKERS", 2)
    epochs: int = env_int("EPOCHS", 12)
    lr: float = env_float("LR", 2e-4)
    weight_decay: float = env_float("WEIGHT_DECAY", 1e-4)
    grad_clip: float = env_float("GRAD_CLIP", 5.0)
    grad_accum_steps: int = env_int("GRAD_ACCUM_STEPS", 8)
    use_amp: bool = env_bool("USE_AMP", True)
    early_stop_patience: int = env_int("EARLY_STOP_PATIENCE", 6)

    max_train: int = env_int("E2E_MAX_TRAIN", 2000)
    max_val: int = env_int("E2E_MAX_VAL", 400)

    bev_encoder_ckpt: str = os.getenv("BEV_ENCODER_CKPT", "")
    base_ckpt_path: str = os.getenv("BASE_CKPT_PATH", pmod.CFG.base_ckpt_path)

    freeze_bev_backbone: bool = env_bool("FREEZE_BEV_BACKBONE", True)
    train_bev_feat_head: bool = env_bool("TRAIN_BEV_FEAT_HEAD", True)
    train_bev_post: bool = env_bool("TRAIN_BEV_POST", True)
    train_bev_depth_head: bool = env_bool("TRAIN_BEV_DEPTH_HEAD", False)
    freeze_planner: bool = env_bool("FREEZE_PLANNER", False)

    # Planning-dominant loss. BEV/depth losses are auxiliary stabilizers.
    lambda_cls: float = env_float("LAMBDA_E2E_CLS", pmod.CFG.lambda_cls)
    lambda_stop: float = env_float("LAMBDA_E2E_STOP", pmod.CFG.lambda_stop_logit_penalty)
    lambda_traj_mode: float = env_float("LAMBDA_E2E_TRAJ_MODE", pmod.CFG.lambda_traj_mode)
    lambda_traj_gt: float = env_float("LAMBDA_E2E_TRAJ_GT", pmod.CFG.lambda_traj_gt)
    lambda_score: float = env_float("LAMBDA_E2E_SCORE", pmod.CFG.lambda_score)
    lambda_map: float = env_float("LAMBDA_E2E_MAP", pmod.CFG.lambda_map)
    lambda_diversity: float = env_float("LAMBDA_E2E_DIVERSITY", pmod.CFG.lambda_diversity)
    lambda_comfort: float = env_float("LAMBDA_E2E_COMFORT", pmod.CFG.lambda_comfort)
    lambda_collision: float = env_float("LAMBDA_E2E_COLLISION", pmod.CFG.lambda_collision)
    lambda_intent: float = env_float("LAMBDA_E2E_INTENT", pmod.CFG.lambda_intent)
    lambda_bev: float = env_float("LAMBDA_E2E_BEV", 0.10)
    lambda_depth: float = env_float("LAMBDA_E2E_DEPTH", 0.03)

    best_model_name: str = "best_partial_end2end_model.pt"
    metrics_name: str = "metrics_partial_end2end.json"


# ============================================================
# Utils
# ============================================================

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


def to_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


# ============================================================
# Planner record loading: same labels/features as old planner,
# but NO offline BEV grid is loaded. BEV comes from online images.
# ============================================================

def make_zero_projection_fields():
    return {
        "camera_feat": np.zeros((pmod.CFG.num_cameras, pmod.CFG.camera_feat_dim), dtype=np.float32),
        "camera_valid": np.zeros((pmod.CFG.num_cameras,), dtype=np.float32),
        "proj_agent_feat": np.zeros((pmod.CFG.max_agents, pmod.CFG.camera_feat_dim), dtype=np.float32),
        "proj_agent_valid": np.zeros((pmod.CFG.max_agents,), dtype=np.float32),
        "proj_map_feat": np.zeros((pmod.CFG.max_lanes, pmod.CFG.camera_feat_dim), dtype=np.float32),
        "proj_map_valid": np.zeros((pmod.CFG.max_lanes,), dtype=np.float32),
        "proj_corridor_feat": np.zeros((1, pmod.CFG.camera_feat_dim), dtype=np.float32),
        "proj_corridor_valid": np.zeros((1,), dtype=np.float32),
    }


def compute_intent_target_from_sample(s: Dict[str, Any], future_xy_local: np.ndarray) -> np.ndarray:
    future_t = np.asarray(
        s.get("future_t", np.arange(1, pmod.CFG.future_steps + 1, dtype=np.float32) * pmod.CFG.dt),
        dtype=np.float32,
    )
    if len(future_xy_local) >= 2 and len(future_t) >= 2:
        dt_last = float(future_t[min(len(future_t), pmod.CFG.future_steps) - 1] - future_t[min(len(future_t), pmod.CFG.future_steps) - 2])
        if dt_last <= 1e-6:
            dt_last = pmod.CFG.dt
        terminal_speed = float(np.linalg.norm(future_xy_local[-1] - future_xy_local[-2]) / dt_last)
    else:
        terminal_speed = float(s.get("future_terminal_speed", 0.0))

    total_disp = float(s.get("future_total_disp", np.linalg.norm(future_xy_local[-1])))
    lateral_disp = float(abs(future_xy_local[-1, 1]))
    future_yaw_local = s.get("future_yaw_local", None)
    if future_yaw_local is not None:
        fyaw = np.asarray(future_yaw_local, dtype=np.float32)
        yaw_delta = float(abs(fyaw[min(len(fyaw), pmod.CFG.future_steps) - 1])) if len(fyaw) > 0 else 0.0
    else:
        yaw_delta = 0.0

    return np.array([
        terminal_speed / pmod.CFG.intent_speed_scale,
        total_disp / pmod.CFG.intent_disp_scale,
        lateral_disp / pmod.CFG.intent_disp_scale,
        yaw_delta / pmod.CFG.intent_yaw_scale,
    ], dtype=np.float32)


def load_planning_records(
    manifest_path: str,
    shard_dir: str,
    split: str,
    teacher_cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
    agent3d_cache: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    projection_cache: Optional[Dict[str, Tuple[np.ndarray, ...]]] = None,
    max_samples: int = 0,
) -> List[Dict[str, Any]]:
    manifest = load_json(manifest_path)
    shard_names = manifest[f"{split}_shards"]
    records: List[Dict[str, Any]] = []

    for shard_name in shard_names:
        shard_path = shard_name if os.path.isabs(shard_name) else os.path.join(shard_dir, shard_name)
        data = load_pickle(shard_path)
        for s in data:
            if not bool(s.get("history_valid", False)):
                continue
            label = pmod.normalize_label(s.get("label_name", ""))
            if label not in pmod.CLASS_TO_ID:
                continue

            sample_token = str(s.get("sample_token", ""))
            if not sample_token:
                continue
            # Keep only samples that have teacher BEV. This also keeps BEV/depth aux loss defined.
            if sample_token not in teacher_cache:
                continue

            hist = np.asarray(s.get("history_features"), dtype=np.float32)
            if hist.ndim != 2 or hist.shape[0] < pmod.CFG.history_len or hist.shape[1] != pmod.CFG.history_dim:
                continue
            hist = hist[:pmod.CFG.history_len, :pmod.CFG.history_dim]

            future_xy_local = s.get("future_xy_local", None)
            if future_xy_local is None:
                continue
            future_xy_local = np.asarray(future_xy_local, dtype=np.float32)
            if future_xy_local.ndim != 2 or future_xy_local.shape[0] < pmod.CFG.future_steps or future_xy_local.shape[1] < 2:
                continue
            future_xy_local = future_xy_local[:pmod.CFG.future_steps, :2]
            y_traj = future_xy_local / pmod.CFG.traj_scale

            if agent3d_cache is not None and sample_token in agent3d_cache:
                agent3d_feat, agent3d_valid = agent3d_cache[sample_token]
                agent3d_feat = np.asarray(agent3d_feat, dtype=np.float32)
                agent3d_valid = np.asarray(agent3d_valid, dtype=np.float32)
            else:
                agent3d_feat = np.zeros((pmod.CFG.max_agent3d, pmod.CFG.agent3d_dim), dtype=np.float32)
                agent3d_valid = np.zeros((pmod.CFG.max_agent3d,), dtype=np.float32)

            if projection_cache is not None and sample_token in projection_cache:
                (camera_feat, camera_valid, proj_agent_feat, proj_agent_valid,
                 proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid) = projection_cache[sample_token]
                projection_fields = {
                    "camera_feat": np.asarray(camera_feat, dtype=np.float32),
                    "camera_valid": np.asarray(camera_valid, dtype=np.float32),
                    "proj_agent_feat": np.asarray(proj_agent_feat, dtype=np.float32),
                    "proj_agent_valid": np.asarray(proj_agent_valid, dtype=np.float32),
                    "proj_map_feat": np.asarray(proj_map_feat, dtype=np.float32),
                    "proj_map_valid": np.asarray(proj_map_valid, dtype=np.float32),
                    "proj_corridor_feat": np.asarray(proj_corridor_feat, dtype=np.float32),
                    "proj_corridor_valid": np.asarray(proj_corridor_valid, dtype=np.float32),
                }
            else:
                projection_fields = make_zero_projection_fields()

            rec = {
                "sample_token": sample_token,
                "hist": hist,
                "agents": pmod._fixed_array(s, "agent_features", (pmod.CFG.max_agents, pmod.CFG.agent_dim)),
                "agent_valid": pmod._fixed_array(s, "agent_valid", (pmod.CFG.max_agents,)),
                "agent3d": agent3d_feat,
                "agent3d_valid": agent3d_valid,
                "maps": pmod._fixed_array(s, "map_polylines", (pmod.CFG.max_lanes, pmod.CFG.lane_points, pmod.CFG.map_dim)),
                "map_valid": pmod._fixed_array(s, "map_polyline_valid", (pmod.CFG.max_lanes, pmod.CFG.lane_points)),
                "y_cls": pmod.CLASS_TO_ID[label],
                "y_traj": y_traj.astype(np.float32),
                "map_y_ref": pmod._fixed_array(s, "map_y_ref", (pmod.CFG.future_steps,)),
                "map_ref_valid": pmod._fixed_array(s, "map_ref_valid", (pmod.CFG.future_steps,)),
                "intent_target": compute_intent_target_from_sample(s, future_xy_local),
            }
            rec.update(projection_fields)
            records.append(rec)
            if max_samples and len(records) >= max_samples:
                return records
    return records


# ============================================================
# Joint dataset: planning shard fields + online raw image fields
# Reuses camera/LiDAR/depth methods from the BEV-builder dataset.
# ============================================================

class JointRawImagePlanningDataset(bmod.NuScenesRawImageBEVDataset):
    def __init__(
        self,
        bev_cfg,
        nusc: NuScenes,
        records: List[Dict[str, Any]],
        teacher_cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
        scaler: Optional[pmod.StandardScalerNP] = None,
    ):
        # Do not call parent __init__ because we already have records.
        self.cfg = bev_cfg
        self.nusc = nusc
        self.records = records
        self.tokens = [r["sample_token"] for r in records]
        self.teacher_cache = teacher_cache
        self.scaler = scaler
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        tok = rec["sample_token"]
        sample = self.nusc.get("sample", tok)

        imgs, Ks, s2es, cam_valids = [], [], [], []
        for ch in bmod.CAMERA_CHANNELS:
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
            cam_valids.append(valid)

        teacher, teacher_valid = self.teacher_cache[tok]
        teacher = bmod.resize_bev_teacher_np(teacher, self.cfg)
        try:
            pts_ego = self._load_lidar_points_ego(sample)
            depth_target, depth_mask, depth_weight = self._build_sparse_depth_targets(pts_ego, Ks, s2es)
        except Exception as e:
            print(f"[WARN] failed building pseudo-depth for {tok}: {e}")
            Hf, Wf = self.cfg.image_h // 8, self.cfg.image_w // 8
            depth_target = np.full((len(bmod.CAMERA_CHANNELS), Hf, Wf), int(self.cfg.depth_ignore_index), dtype=np.int64)
            depth_mask = np.zeros((len(bmod.CAMERA_CHANNELS), Hf, Wf), dtype=np.float32)
            depth_weight = np.zeros((len(bmod.CAMERA_CHANNELS), Hf, Wf), dtype=np.float32)

        hist = rec["hist"].astype(np.float32)
        if self.scaler is not None:
            hist = self.scaler.transform(hist).astype(np.float32)

        return {
            "sample_token": tok,
            "images": torch.tensor(np.stack(imgs), dtype=torch.float32),
            "intrinsics": torch.tensor(np.stack(Ks), dtype=torch.float32),
            "sensor2ego": torch.tensor(np.stack(s2es), dtype=torch.float32),
            "raw_camera_valid": torch.tensor(np.asarray(cam_valids, dtype=np.float32), dtype=torch.float32),
            "teacher_bev": torch.tensor(teacher, dtype=torch.float32),
            "teacher_valid": torch.tensor(np.asarray(teacher_valid, dtype=np.float32).reshape(-1)[:1], dtype=torch.float32),
            "depth_target": torch.tensor(depth_target, dtype=torch.long),
            "depth_mask": torch.tensor(depth_mask, dtype=torch.float32),
            "depth_weight": torch.tensor(depth_weight, dtype=torch.float32),

            "hist": torch.tensor(hist, dtype=torch.float32),
            "agents": torch.tensor(rec["agents"], dtype=torch.float32),
            "agent_valid": torch.tensor(rec["agent_valid"], dtype=torch.float32),
            "agent3d": torch.tensor(rec["agent3d"], dtype=torch.float32),
            "agent3d_valid": torch.tensor(rec["agent3d_valid"], dtype=torch.float32),
            "maps": torch.tensor(rec["maps"], dtype=torch.float32),
            "map_valid": torch.tensor(rec["map_valid"], dtype=torch.float32),
            "y_cls": torch.tensor(rec["y_cls"], dtype=torch.long),
            "y_traj": torch.tensor(rec["y_traj"], dtype=torch.float32),
            "map_y_ref": torch.tensor(rec["map_y_ref"], dtype=torch.float32),
            "map_ref_valid": torch.tensor(rec["map_ref_valid"], dtype=torch.float32),
            "intent_target": torch.tensor(rec["intent_target"], dtype=torch.float32),
            "camera_feat": torch.tensor(rec["camera_feat"], dtype=torch.float32),
            "camera_valid": torch.tensor(rec["camera_valid"], dtype=torch.float32),
            "proj_agent_feat": torch.tensor(rec["proj_agent_feat"], dtype=torch.float32),
            "proj_agent_valid": torch.tensor(rec["proj_agent_valid"], dtype=torch.float32),
            "proj_map_feat": torch.tensor(rec["proj_map_feat"], dtype=torch.float32),
            "proj_map_valid": torch.tensor(rec["proj_map_valid"], dtype=torch.float32),
            "proj_corridor_feat": torch.tensor(rec["proj_corridor_feat"], dtype=torch.float32),
            "proj_corridor_valid": torch.tensor(rec["proj_corridor_valid"], dtype=torch.float32),
        }


def collate_joint(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"sample_tokens": [b["sample_token"] for b in batch]}
    for k in batch[0].keys():
        if k == "sample_token":
            continue
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def move_batch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ============================================================
# Partial end-to-end model
# ============================================================

class PartialEndToEndImageBEVPlanner(nn.Module):
    def __init__(self, bev_encoder: nn.Module, planner: nn.Module):
        super().__init__()
        self.bev_encoder = bev_encoder
        self.planner = planner

    def forward(self, batch: Dict[str, torch.Tensor], return_depth: bool = True):
        if return_depth:
            bev_grid, depth_logits = self.bev_encoder(
                batch["images"],
                batch["intrinsics"],
                batch["sensor2ego"],
                batch["raw_camera_valid"],
                return_depth=True,
            )
        else:
            bev_grid = self.bev_encoder(
                batch["images"],
                batch["intrinsics"],
                batch["sensor2ego"],
                batch["raw_camera_valid"],
                return_depth=False,
            )
            depth_logits = None

        bev_valid = (batch["raw_camera_valid"].sum(dim=1, keepdim=True) > 0).float()

        cls_logits, traj_pred, scores, intent_pred = self.planner(
            batch["hist"],
            batch["agents"],
            batch["agent_valid"],
            batch["agent3d"],
            batch["agent3d_valid"],
            bev_grid,
            bev_valid,
            batch["maps"],
            batch["map_valid"],
            batch["camera_feat"],
            batch["camera_valid"],
            batch["proj_agent_feat"],
            batch["proj_agent_valid"],
            batch["proj_map_feat"],
            batch["proj_map_valid"],
            batch["proj_corridor_feat"],
            batch["proj_corridor_valid"],
        )
        return cls_logits, traj_pred, scores, intent_pred, bev_grid, depth_logits, bev_valid


def configure_trainable_params(model: PartialEndToEndImageBEVPlanner, cfg: E2EConfig):
    # First freeze selected parts.
    if cfg.freeze_bev_backbone:
        for p in model.bev_encoder.backbone.parameters():
            p.requires_grad = False

    # Safer default: freeze all BEV encoder, then unfreeze selected BEV neck layers.
    for p in model.bev_encoder.parameters():
        p.requires_grad = False

    if cfg.train_bev_feat_head:
        for p in model.bev_encoder.feat_head.parameters():
            p.requires_grad = True
    if cfg.train_bev_post:
        for p in model.bev_encoder.post.parameters():
            p.requires_grad = True
    if cfg.train_bev_depth_head:
        for p in model.bev_encoder.depth_head.parameters():
            p.requires_grad = True

    if cfg.freeze_planner:
        for p in model.planner.parameters():
            p.requires_grad = False
    else:
        for p in model.planner.parameters():
            p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    return trainable, total_params, trainable_params


def load_planner_checkpoint(planner: nn.Module, ckpt_path: str, device: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[WARN] Planner checkpoint not found; planner will train from scratch: {ckpt_path}")
        return None, "scratch"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "base_model_state_dict" in ckpt:
        state = ckpt["base_model_state_dict"]
        src = "base_model_state_dict"
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        src = "model_state_dict"
    else:
        state = ckpt
        src = "raw_state_dict"
    cur = planner.state_dict()
    compatible = {k: v for k, v in state.items() if k in cur and tuple(v.shape) == tuple(cur[k].shape)}
    planner.load_state_dict(compatible, strict=False)
    print(f"Loaded planner checkpoint: {ckpt_path}")
    print(f"  source={src}, compatible tensors={len(compatible)} / current tensors={len(cur)}")
    return ckpt, ckpt_path


def load_bev_encoder_checkpoint(bev_encoder: nn.Module, ckpt_path: str, device: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"BEV_ENCODER_CKPT not found: {ckpt_path}\n"
            "You need a pretrained RealImageLSSBEV checkpoint before partial end-to-end training."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    cur = bev_encoder.state_dict()
    compatible = {k: v for k, v in state.items() if k in cur and tuple(v.shape) == tuple(cur[k].shape)}
    missing = [k for k in cur.keys() if k not in compatible]
    bev_encoder.load_state_dict(compatible, strict=False)
    print(f"Loaded BEV encoder checkpoint: {ckpt_path}")
    print(f"  compatible tensors={len(compatible)} / current tensors={len(cur)}, missing/new={len(missing)}")
    if len(compatible) == 0:
        raise RuntimeError("No compatible BEV encoder weights loaded. Check IMAGE_H/W, hidden_dim, depth_bins, backbone config.")
    return ckpt


# ============================================================
# Loss and metrics
# ============================================================

def compute_e2e_total_loss(
    model: PartialEndToEndImageBEVPlanner,
    batch: Dict[str, torch.Tensor],
    cls_criterion: nn.Module,
    e2e_cfg: E2EConfig,
    bev_cfg,
):
    cls_logits, traj_pred, scores, intent_pred, bev_grid, depth_logits, bev_valid = model(batch, return_depth=True)
    y_cls = batch["y_cls"]
    y_traj = batch["y_traj"]

    loss_cls = cls_criterion(cls_logits, y_cls)
    loss_stop = pmod.stop_logit_penalty(cls_logits, y_cls)
    loss_traj_mode, loss_traj_gt, loss_score, best_idx_gt, _, pred_idx, _ = pmod.base_multimodal_losses(traj_pred, y_traj, y_cls, scores)
    traj_best_gt = pmod.gather_modes(traj_pred, best_idx_gt)
    loss_map = pmod.compute_real_map_loss(traj_best_gt, batch["map_y_ref"], batch["map_ref_valid"])
    loss_div = pmod.compute_diversity_loss(traj_pred)
    loss_comfort = pmod.compute_comfort_loss(traj_best_gt)
    loss_collision = pmod.compute_collision_loss(traj_best_gt, batch["agents"], batch["agent_valid"])
    loss_intent = pmod.compute_intent_loss(intent_pred, batch["intent_target"])

    if e2e_cfg.lambda_bev > 0:
        loss_bev, bev_parts = bmod.compute_loss(bev_grid, batch["teacher_bev"], batch["teacher_valid"], bev_cfg)
    else:
        loss_bev = bev_grid.new_tensor(0.0)
        bev_parts = {}

    if e2e_cfg.lambda_depth > 0 and depth_logits is not None:
        loss_depth, depth_pts = bmod.pseudo_depth_loss(
            depth_logits,
            batch["depth_target"],
            batch["depth_mask"],
            batch["depth_weight"],
            bev_cfg,
        )
    else:
        loss_depth = bev_grid.new_tensor(0.0)
        depth_pts = 0.0

    loss = (
        e2e_cfg.lambda_cls * loss_cls
        + e2e_cfg.lambda_stop * loss_stop
        + e2e_cfg.lambda_traj_mode * loss_traj_mode
        + e2e_cfg.lambda_traj_gt * loss_traj_gt
        + e2e_cfg.lambda_score * loss_score
        + e2e_cfg.lambda_map * loss_map
        + e2e_cfg.lambda_diversity * loss_div
        + e2e_cfg.lambda_comfort * loss_comfort
        + e2e_cfg.lambda_collision * loss_collision
        + e2e_cfg.lambda_intent * loss_intent
        + e2e_cfg.lambda_bev * loss_bev
        + e2e_cfg.lambda_depth * loss_depth
    )

    return {
        "loss": loss,
        "loss_cls": loss_cls,
        "loss_stop": loss_stop,
        "loss_traj_mode": loss_traj_mode,
        "loss_traj_gt": loss_traj_gt,
        "loss_score": loss_score,
        "loss_map": loss_map,
        "loss_div": loss_div,
        "loss_comfort": loss_comfort,
        "loss_collision": loss_collision,
        "loss_intent": loss_intent,
        "loss_bev": loss_bev,
        "loss_depth": loss_depth,
        "depth_pts": float(depth_pts),
        "cls_logits": cls_logits,
        "traj_pred": traj_pred,
        "scores": scores,
        "intent_pred": intent_pred,
        "y_cls": y_cls,
        "y_traj": y_traj,
        "best_idx_gt": best_idx_gt,
        "pred_idx": pred_idx,
        "bev_valid": bev_valid,
        "bev_parts": bev_parts,
    }


def train_one_epoch(model, loader, optimizer, scaler, cls_criterion, e2e_cfg: E2EConfig, bev_cfg, epoch: int):
    model.train()
    meters = {k: 0.0 for k in [
        "loss", "loss_cls", "loss_stop", "loss_traj_mode", "loss_traj_gt", "loss_score",
        "loss_map", "loss_div", "loss_comfort", "loss_collision", "loss_intent", "loss_bev", "loss_depth", "depth_pts"
    ]}
    total = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, 1):
        batch = move_batch(batch, e2e_cfg.device)
        use_amp = e2e_cfg.use_amp and e2e_cfg.device.startswith("cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            out = compute_e2e_total_loss(model, batch, cls_criterion, e2e_cfg, bev_cfg)
            loss_for_backward = out["loss"] / max(1, e2e_cfg.grad_accum_steps)

        scaler.scale(loss_for_backward).backward()

        if step % e2e_cfg.grad_accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], e2e_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = batch["y_cls"].size(0)
        total += bs
        for k in meters:
            v = out[k]
            meters[k] += (float(v.item()) if torch.is_tensor(v) else float(v)) * bs

        if step % 20 == 0:
            print(
                f"[train e{epoch:03d} i{step:04d}] "
                f"loss={out['loss'].item():.4f} cls={out['loss_cls'].item():.4f} "
                f"traj_gt={out['loss_traj_gt'].item():.4f} score={out['loss_score'].item():.4f} "
                f"bev={out['loss_bev'].item():.4f} depth={out['loss_depth'].item():.4f}"
            )

    return {k: v / max(total, 1) for k, v in meters.items()}


@torch.no_grad()
def evaluate(model, loader, cls_criterion, e2e_cfg: E2EConfig, bev_cfg):
    model.eval()
    meters = {k: 0.0 for k in [
        "loss", "loss_cls", "loss_stop", "loss_traj_mode", "loss_traj_gt", "loss_score",
        "loss_map", "loss_div", "loss_comfort", "loss_collision", "loss_intent", "loss_bev", "loss_depth", "depth_pts"
    ]}
    total = 0
    preds_all, y_all = [], []
    ade_sel_all, fde_sel_all, ade_orc_all, fde_orc_all = [], [], [], []
    score_hit_all = []

    for batch in loader:
        batch = move_batch(batch, e2e_cfg.device)
        out = compute_e2e_total_loss(model, batch, cls_criterion, e2e_cfg, bev_cfg)
        bs = batch["y_cls"].size(0)
        total += bs
        for k in meters:
            v = out[k]
            meters[k] += (float(v.item()) if torch.is_tensor(v) else float(v)) * bs

        preds = out["cls_logits"].argmax(dim=1)
        ade_sel, fde_sel = pmod.calc_traj_metrics(out["traj_pred"], out["y_traj"], out["pred_idx"])
        ade_orc, fde_orc = pmod.calc_traj_metrics(out["traj_pred"], out["y_traj"], out["best_idx_gt"])
        score_hit = (out["pred_idx"] == out["best_idx_gt"]).float()

        preds_all.append(preds.cpu().numpy())
        y_all.append(out["y_cls"].cpu().numpy())
        ade_sel_all.append(ade_sel.cpu().numpy())
        fde_sel_all.append(fde_sel.cpu().numpy())
        ade_orc_all.append(ade_orc.cpu().numpy())
        fde_orc_all.append(fde_orc.cpu().numpy())
        score_hit_all.append(score_hit.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)
    ade_sel_all = np.concatenate(ade_sel_all)
    fde_sel_all = np.concatenate(fde_sel_all)
    ade_orc_all = np.concatenate(ade_orc_all)
    fde_orc_all = np.concatenate(fde_orc_all)
    score_hit_all = np.concatenate(score_hit_all)

    labels = list(range(len(pmod.CFG.target_classes)))
    acc = accuracy_score(y_all, preds_all)
    macro_f1 = f1_score(y_all, preds_all, labels=labels, average="macro", zero_division=0)
    precisions = precision_score(y_all, preds_all, labels=labels, average=None, zero_division=0)
    recalls = recall_score(y_all, preds_all, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_all, preds_all, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_all, preds_all, labels=labels)

    ret = {k: v / max(total, 1) for k, v in meters.items()}
    ret.update({
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "ADE_selected": float(ade_sel_all.mean()),
        "FDE_selected": float(fde_sel_all.mean()),
        "ADE_oracle": float(ade_orc_all.mean()),
        "FDE_oracle": float(fde_orc_all.mean()),
        "ADE_gap": float(ade_sel_all.mean() - ade_orc_all.mean()),
        "FDE_gap": float(fde_sel_all.mean() - fde_orc_all.mean()),
        "score_hit_rate": float(score_hit_all.mean()),
        "precisions": precisions.tolist(),
        "recalls": recalls.tolist(),
        "f1s": f1s.tolist(),
        "cm": cm.tolist(),
        "y_true": y_all.tolist(),
        "y_pred": preds_all.tolist(),
    })
    return ret


def save_checkpoint(path: str, model, optimizer, scaler_obj, epoch, metrics, e2e_cfg, bev_cfg, hist_scaler):
    torch.save({
        "model_state_dict": model.state_dict(),
        "bev_encoder_state_dict": model.bev_encoder.state_dict(),
        "planner_state_dict": model.planner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "amp_scaler_state_dict": scaler_obj.state_dict() if scaler_obj is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "e2e_config": {k: getattr(e2e_cfg, k) for k in dir(e2e_cfg) if not k.startswith("_") and not callable(getattr(e2e_cfg, k))},
        "bev_config": asdict(bev_cfg),
        "planner_config": asdict(pmod.CFG),
        "class_to_id": pmod.CLASS_TO_ID,
        "id_to_class": pmod.ID_TO_CLASS,
        "history_scaler": hist_scaler.state_dict(),
    }, path)


# ============================================================
# Main
# ============================================================

def main():
    e2e_cfg = E2EConfig()
    set_seed(e2e_cfg.seed)
    ensure_dir(e2e_cfg.save_dir)

    # Build BEV config from env. Must match planner BEV grid.
    bev_cfg = bmod.cfg_from_env()
    bev_cfg.device = e2e_cfg.device
    bev_cfg.batch_size = e2e_cfg.batch_size
    bev_cfg.num_workers = e2e_cfg.num_workers
    # Keep planner and BEV config aligned. If user passes BEV_H/W env, both modules read it.
    pmod.CFG.bev_channels = bev_cfg.out_channels
    pmod.CFG.bev_h = bev_cfg.bev_h
    pmod.CFG.bev_w = bev_cfg.bev_w
    pmod.CFG.device = e2e_cfg.device

    print("=" * 88)
    print("Training: v16.4.1 Partial End-to-End Image-BEV Planner")
    print(f"BEV script     : {BEV_SCRIPT_PATH}")
    print(f"Planner script : {PLANNER_SCRIPT_PATH}")
    print(f"Manifest       : {pmod.CFG.manifest_path}")
    print(f"Shard dir      : {pmod.CFG.shard_dir}")
    print(f"Save dir       : {e2e_cfg.save_dir}")
    print(f"Device         : {e2e_cfg.device}")
    print(f"Image size     : {bev_cfg.image_h}x{bev_cfg.image_w}")
    print(f"BEV grid       : C={bev_cfg.out_channels}, H={bev_cfg.bev_h}, W={bev_cfg.bev_w}")
    print(f"Batch/Accum    : {e2e_cfg.batch_size}/{e2e_cfg.grad_accum_steps}")
    print(f"AMP            : {e2e_cfg.use_amp}")
    print("=" * 88)

    if not os.path.exists(pmod.CFG.manifest_path):
        raise FileNotFoundError(f"manifest not found: {pmod.CFG.manifest_path}")

    nusc = NuScenes(version=bev_cfg.version, dataroot=bev_cfg.dataroot, verbose=True)
    teacher_cache = bmod.load_annotation_bev_cache(bev_cfg.teacher_bev_manifest)
    print(f"Teacher BEV samples: {len(teacher_cache)}")

    # Load optional caches used by planner tokens. Projection visual defaults are off in v16.4.1 config.
    use_any_visual = (
        pmod.CFG.use_global_bev_token
        or pmod.CFG.use_agent_aligned_visual
        or pmod.CFG.use_map_aligned_visual
        or pmod.CFG.use_corridor_visual_token
    )
    if use_any_visual:
        print("Loading projection-aligned visual cache...")
        projection_cache = pmod.load_projection_visual_cache(pmod.CFG.projection_cache_manifest)
    else:
        print("Projection visual branch disabled: zero projection placeholders.")
        projection_cache = None

    if pmod.CFG.use_agent3d_tokens:
        print("Loading v10.0 3D agent cache...")
        agent3d_cache = pmod.load_agent3d_cache(pmod.CFG.agent3d_cache_manifest)
    else:
        print("3D agent tokens disabled: zero placeholders.")
        agent3d_cache = None

    print("Loading train/val planning records...")
    train_records = load_planning_records(
        pmod.CFG.manifest_path,
        pmod.CFG.shard_dir,
        "train",
        teacher_cache,
        agent3d_cache=agent3d_cache,
        projection_cache=projection_cache,
        max_samples=e2e_cfg.max_train,
    )
    val_records = load_planning_records(
        pmod.CFG.manifest_path,
        pmod.CFG.shard_dir,
        "val",
        teacher_cache,
        agent3d_cache=agent3d_cache,
        projection_cache=projection_cache,
        max_samples=e2e_cfg.max_val,
    )
    if not train_records or not val_records:
        raise RuntimeError(f"No valid records. train={len(train_records)}, val={len(val_records)}")
    print(f"Train records: {len(train_records)}")
    print(f"Val records  : {len(val_records)}")

    # Build/load planner first so we can reuse checkpoint scaler if present.
    planner = pmod.build_base_model().to(e2e_cfg.device)
    planner_ckpt, planner_src = load_planner_checkpoint(planner, e2e_cfg.base_ckpt_path, e2e_cfg.device)

    # History scaler: prefer planner checkpoint scaler for consistency.
    hist_scaler = pmod.StandardScalerNP(skip_indices=[14, 15])
    ckpt_scaler = planner_ckpt.get("scaler", None) if isinstance(planner_ckpt, dict) else None
    if isinstance(ckpt_scaler, dict) and ckpt_scaler.get("mean", None) is not None and ckpt_scaler.get("std", None) is not None:
        hist_scaler.mean = np.asarray(ckpt_scaler["mean"], dtype=np.float32)
        hist_scaler.std = np.asarray(ckpt_scaler["std"], dtype=np.float32)
        hist_scaler.skip_indices = set(ckpt_scaler.get("skip_indices", [14, 15]))
        print("History scaler: loaded from planner checkpoint")
    else:
        X_train = np.stack([r["hist"] for r in train_records], axis=0)
        hist_scaler.fit(X_train.reshape(-1, pmod.CFG.history_dim))
        print("History scaler: fitted from current train records")

    train_ds = JointRawImagePlanningDataset(bev_cfg, nusc, train_records, teacher_cache, scaler=hist_scaler)
    val_ds = JointRawImagePlanningDataset(bev_cfg, nusc, val_records, teacher_cache, scaler=hist_scaler)
    train_loader = DataLoader(
        train_ds,
        batch_size=e2e_cfg.batch_size,
        shuffle=True,
        num_workers=e2e_cfg.num_workers,
        pin_memory=e2e_cfg.device.startswith("cuda"),
        collate_fn=collate_joint,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=e2e_cfg.batch_size,
        shuffle=False,
        num_workers=e2e_cfg.num_workers,
        pin_memory=e2e_cfg.device.startswith("cuda"),
        collate_fn=collate_joint,
        drop_last=False,
    )

    bev_encoder = bmod.RealImageLSSBEV(bev_cfg).to(e2e_cfg.device)
    load_bev_encoder_checkpoint(bev_encoder, e2e_cfg.bev_encoder_ckpt, e2e_cfg.device)

    model = PartialEndToEndImageBEVPlanner(bev_encoder, planner).to(e2e_cfg.device)
    trainable, total_params, trainable_params = configure_trainable_params(model, e2e_cfg)
    if not trainable:
        raise RuntimeError("No trainable parameters. Check FREEZE_* and TRAIN_* switches.")

    print("=" * 88)
    print("Trainability")
    print(f"Freeze BEV backbone : {e2e_cfg.freeze_bev_backbone}")
    print(f"Train BEV feat_head : {e2e_cfg.train_bev_feat_head}")
    print(f"Train BEV post      : {e2e_cfg.train_bev_post}")
    print(f"Train BEV depth_head: {e2e_cfg.train_bev_depth_head}")
    print(f"Freeze planner      : {e2e_cfg.freeze_planner}")
    print(f"Trainable params    : {trainable_params:,} / {total_params:,}")
    print("Loss weights")
    print(
        f"cls={e2e_cfg.lambda_cls}, traj_gt={e2e_cfg.lambda_traj_gt}, traj_mode={e2e_cfg.lambda_traj_mode}, "
        f"score={e2e_cfg.lambda_score}, map={e2e_cfg.lambda_map}, collision={e2e_cfg.lambda_collision}, "
        f"bev_aux={e2e_cfg.lambda_bev}, depth_aux={e2e_cfg.lambda_depth}"
    )
    print("=" * 88)

    # Class weights from current subset.
    y_train_cls = np.asarray([r["y_cls"] for r in train_records], dtype=np.int64)
    class_ids = np.arange(len(pmod.CFG.target_classes))
    cls_weights_raw = compute_class_weight(class_weight="balanced", classes=class_ids, y=y_train_cls)
    cls_weights_np = np.clip(cls_weights_raw, pmod.CFG.class_weight_min, pmod.CFG.class_weight_max)
    stop_id = pmod.CLASS_TO_ID["STOP"]
    cls_weights_np[stop_id] = min(cls_weights_np[stop_id], pmod.CFG.stop_class_weight_max)
    cls_weights = torch.tensor(cls_weights_np, dtype=torch.float32, device=e2e_cfg.device)
    print("Class weights raw -> calibrated:")
    for i, w in enumerate(cls_weights.tolist()):
        print(f"  {pmod.ID_TO_CLASS[i]:>7}: {float(cls_weights_raw[i]):.6f} -> {w:.6f}")
    cls_criterion = pmod.FocalLoss(alpha=cls_weights, gamma=pmod.CFG.focal_gamma) if pmod.CFG.use_focal_loss else nn.CrossEntropyLoss(weight=cls_weights)

    optimizer = torch.optim.AdamW(trainable, lr=e2e_cfg.lr, weight_decay=e2e_cfg.weight_decay)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(e2e_cfg.use_amp and e2e_cfg.device.startswith("cuda")))

    best_model_path = os.path.join(e2e_cfg.save_dir, e2e_cfg.best_model_name)
    metrics_path = os.path.join(e2e_cfg.save_dir, e2e_cfg.metrics_name)

    print("\n========== Initial Online E2E Check ==========")
    base_metrics = evaluate(model, val_loader, cls_criterion, e2e_cfg, bev_cfg)
    print(
        f"Init Acc={base_metrics['acc']:.4f} MacroF1={base_metrics['macro_f1']:.4f} "
        f"ADE_sel={base_metrics['ADE_selected']:.4f} FDE_sel={base_metrics['FDE_selected']:.4f} "
        f"ADE_orc={base_metrics['ADE_oracle']:.4f} Gap={base_metrics['ADE_gap']:.4f} "
        f"ScoreHit={base_metrics['score_hit_rate']:.4f}"
    )

    best_epoch = 0
    best_ADE = base_metrics["ADE_selected"]
    best_gap = base_metrics["ADE_gap"]
    best_score_hit = base_metrics["score_hit_rate"]
    best_metrics = base_metrics
    patience = 0
    history = [{"epoch": 0, **{f"init_{k}": v for k, v in base_metrics.items() if isinstance(v, (int, float))}}]
    save_checkpoint(best_model_path, model, optimizer, amp_scaler, 0, base_metrics, e2e_cfg, bev_cfg, hist_scaler)

    for epoch in range(1, e2e_cfg.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, amp_scaler, cls_criterion, e2e_cfg, bev_cfg, epoch)
        val_metrics = evaluate(model, val_loader, cls_criterion, e2e_cfg, bev_cfg)
        history.append({
            "epoch": epoch,
            **{f"train_{k}": float(v) for k, v in train_stats.items()},
            **{f"val_{k}": float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
        })

        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss {train_stats['loss']:.4f} | "
            f"ValLoss {val_metrics['loss']:.4f} | "
            f"Acc {val_metrics['acc']:.4f} | MacroF1 {val_metrics['macro_f1']:.4f} | "
            f"ADE_sel {val_metrics['ADE_selected']:.4f} | FDE_sel {val_metrics['FDE_selected']:.4f} | "
            f"ADE_orc {val_metrics['ADE_oracle']:.4f} | Gap {val_metrics['ADE_gap']:.4f} | "
            f"ScoreHit {val_metrics['score_hit_rate']:.4f} | "
            f"BEV {val_metrics['loss_bev']:.4f} | Depth {val_metrics['loss_depth']:.4f}"
        )

        improved = False
        ade = val_metrics["ADE_selected"]
        gap = val_metrics["ADE_gap"]
        hit = val_metrics["score_hit_rate"]
        if ade < best_ADE - 1e-4:
            improved = True
        elif abs(ade - best_ADE) <= 1e-4 and gap < best_gap - 1e-4:
            improved = True
        elif abs(ade - best_ADE) <= 1e-4 and abs(gap - best_gap) <= 1e-4 and hit > best_score_hit + 1e-4:
            improved = True

        if improved:
            best_epoch = epoch
            best_ADE = ade
            best_gap = gap
            best_score_hit = hit
            best_metrics = val_metrics
            patience = 0
            save_checkpoint(best_model_path, model, optimizer, amp_scaler, epoch, val_metrics, e2e_cfg, bev_cfg, hist_scaler)
            print(f"🔥 Saved best partial-E2E model: epoch={epoch}, ADE={ade:.4f}, gap={gap:.4f}, hit={hit:.4f}")
        else:
            patience += 1
            if patience >= e2e_cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Reload best and write final report.
    ckpt_best = torch.load(best_model_path, map_location=e2e_cfg.device, weights_only=False)
    model.load_state_dict(ckpt_best["model_state_dict"], strict=True)
    final_metrics = evaluate(model, val_loader, cls_criterion, e2e_cfg, bev_cfg)
    report = classification_report(
        final_metrics["y_true"],
        final_metrics["y_pred"],
        labels=list(range(len(pmod.CFG.target_classes))),
        target_names=list(pmod.CFG.target_classes),
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "mode": "partial_end_to_end_online_image_bev_planner",
        "bev_script_path": BEV_SCRIPT_PATH,
        "planner_script_path": PLANNER_SCRIPT_PATH,
        "bev_encoder_ckpt": e2e_cfg.bev_encoder_ckpt,
        "base_ckpt_path": e2e_cfg.base_ckpt_path,
        "best_epoch": int(best_epoch),
        "base_metrics": {k: v for k, v in base_metrics.items() if isinstance(v, (int, float))},
        "best_metrics": {k: v for k, v in best_metrics.items() if isinstance(v, (int, float))},
        "final_metrics": {k: v for k, v in final_metrics.items() if isinstance(v, (int, float))},
        "final_precisions": {pmod.ID_TO_CLASS[i]: float(p) for i, p in enumerate(final_metrics["precisions"])},
        "final_recalls": {pmod.ID_TO_CLASS[i]: float(r) for i, r in enumerate(final_metrics["recalls"])},
        "final_f1s": {pmod.ID_TO_CLASS[i]: float(f) for i, f in enumerate(final_metrics["f1s"])},
        "confusion_matrix": final_metrics["cm"],
        "classification_report": report,
        "history": history,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "bev_config": asdict(bev_cfg),
        "planner_config": asdict(pmod.CFG),
        "e2e_config": {k: getattr(e2e_cfg, k) for k in dir(e2e_cfg) if not k.startswith("_") and not callable(getattr(e2e_cfg, k))},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n========== Final Summary ==========")
    print(f"Best epoch       : {best_epoch}")
    print(f"Final Acc        : {final_metrics['acc']:.4f}")
    print(f"Final MacroF1    : {final_metrics['macro_f1']:.4f}")
    print(f"Final ADE_sel    : {final_metrics['ADE_selected']:.4f}")
    print(f"Final FDE_sel    : {final_metrics['FDE_selected']:.4f}")
    print(f"Final ADE_orc    : {final_metrics['ADE_oracle']:.4f}")
    print(f"Final FDE_orc    : {final_metrics['FDE_oracle']:.4f}")
    print(f"Final ADE_gap    : {final_metrics['ADE_gap']:.4f}")
    print(f"Final ScoreHit   : {final_metrics['score_hit_rate']:.4f}")
    print("Confusion Matrix [rows=true, cols=pred]:")
    for i, row in enumerate(final_metrics["cm"]):
        print(f"{pmod.ID_TO_CLASS[i]:>7}: {row}")
    print(f"Metrics saved to    : {metrics_path}")
    print(f"Best model saved to : {best_model_path}")


if __name__ == "__main__":
    main()
