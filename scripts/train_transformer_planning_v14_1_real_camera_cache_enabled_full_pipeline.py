#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full v14.1 real-camera-cache learned camera-BEV joint planner on nuScenes FULL dataset.

This is a self-contained v14.1 pipeline built on v12.2, upgraded from annotation-BEV consumption to learned camera-BEV generation:

Stage 1 / v10.0 base model:
    Map-Agent-Ego token fusion planner
    + cached 6-camera geometry-aligned visual token fusion
    + class calibration
    + intent auxiliary head
    + behavior-conditioned mode queries

Stage 2 / v9.5 score-only calibration:
    Load or train the v10.0 base model, freeze it, and train only a tiny
    score refiner to reduce selected-vs-oracle trajectory gap without
    corrupting classification or trajectory generation.

Default behavior:
    If BASE_CKPT_PATH points to a v10.0 base checkpoint, skip Stage 1 and run score calibration directly.
    If RUN_BASE_TRAIN=1 or BASE_CKPT_PATH is missing and TRAIN_BASE_IF_MISSING=1,
    train v9.5 first, save it, then run score calibration.

No v8 dataset rebuild is required. Use the existing v8 manifest + shards plus the v9.2 camera feature cache.

Input from v8 build shards:
    history_features      : [B, T=6, 20]
    agent_features        : [B, A=24, 10]
    agent_valid           : [B, A]
    map_polylines         : [B, M=32, P=20, 5]
    map_polyline_valid    : [B, M, P]
    camera_feat           : [B, 6, 256]
    camera_valid          : [B, 6]

Outputs:
    learned_bev_grid      : [B, C=8, H=80, W=60] generated from cached 6-camera features
    cls_logits            : [B, 4]
    traj_pred             : [B, K=3, future_steps=6, 2]
    scores                : [B, K]

Upgrade over v7:
1) Map is input token, not only loss.
2) Nearby agents are explicit tokens.
3) Planning uses K learnable mode queries + Transformer decoder.
4) Keeps Best-of-K, soft score supervision, map loss, diversity loss.
5) Adds comfort loss and simple collision loss.

Recommended:
    MANIFEST_PATH=/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/build_manifest_full_v8_map_agent_t6.json \
    SHARD_DIR=/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/shards_full_v8_map_agent_t6 \
    CAMERA_CACHE_MANIFEST=/home/ubuntu22/decision_on_nuscenes/outputs_v9_5_projection_visual_cache/projection_visual_cache_manifest_v9_5.json \
    SAVE_DIR=/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v9_5_projection_aligned \
    RUN_BASE_TRAIN=1 \
    INIT_CKPT_PATH=/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v8_123_intent_behavior/best_model.pt \
    python train_transformer_planning_v9_5_projection_aligned_full_pipeline.py

To train v9.5 inside this script first:
    RUN_BASE_TRAIN=1 BASE_EPOCHS=80 \
    MANIFEST_PATH=/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/build_manifest_full_v8_map_agent_t6.json \
    SHARD_DIR=/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/shards_full_v8_map_agent_t6 \
    CAMERA_CACHE_MANIFEST=/home/ubuntu22/decision_on_nuscenes/outputs_v9_5_projection_visual_cache/projection_visual_cache_manifest_v9_5.json \
    SAVE_DIR=/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v9_5_projection_aligned \
    INIT_CKPT_PATH=/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v8_123_intent_behavior/best_model.pt \
    python train_transformer_planning_v9_5_projection_aligned_full_pipeline.py
"""

import os
import json
import math
import pickle
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    seed: int = 42
    target_classes: Tuple[str, ...] = ("CRUISE", "FOLLOW", "STOP", "TURN")

    batch_size: int = 96
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    num_workers: int = 0
    early_stop_patience: int = 8

    history_len: int = 6
    history_dim: int = 20
    agent_dim: int = 10
    # v10.0: extra raw nuScenes 3D agent tokens built from sample_annotation.
    # These are separate from v8/v9 basic agent_features, so old experiments remain comparable.
    agent3d_dim: int = 26
    max_agent3d: int = int(os.getenv("MAX_3D_AGENTS", "32"))
    use_agent3d_tokens: bool = os.getenv("USE_AGENT3D_TOKENS", "1").lower() in {"1", "true", "yes", "y", "on"}
    map_dim: int = 5
    # v11.0/v12.0: annotation/map/agent BEV occupancy-risk grid.
    # This controls whether the BEV cache is LOADED. Keep it separate from USE_DENSE_BEV_TOKENS:
    #   USE_ANNOTATION_BEV=1   -> load real annotation-BEV cache so object-aligned fusion has non-zero BEV input
    #   USE_DENSE_BEV_TOKENS=1 -> additionally append dense BEV tokens to the Transformer sequence
    # In v14.1 real-camera-cache learned camera-BEV joint, USE_ANNOTATION_BEV should normally be 1 and USE_DENSE_BEV_TOKENS should normally be 0.
    use_annotation_bev: bool = os.getenv("USE_ANNOTATION_BEV", "1").lower() in {"1", "true", "yes", "y", "on"}
    use_dense_bev_tokens: bool = os.getenv("USE_DENSE_BEV_TOKENS", "0").lower() in {"1", "true", "yes", "y", "on"}
    bev_channels: int = int(os.getenv("BEV_CHANNELS", "8"))
    bev_h: int = int(os.getenv("BEV_H", "80"))
    bev_w: int = int(os.getenv("BEV_W", "60"))
    bev_token_grid: int = int(os.getenv("BEV_TOKEN_GRID", "4"))

    # v12.0: object-aligned BEV fusion. Instead of appending dense BEV tokens,
    # sample local BEV channels around agent/basic-agent/map/corridor objects and
    # inject them as gated residual adapters into the corresponding tokens.
    use_object_aligned_bev: bool = os.getenv("USE_OBJECT_ALIGNED_BEV", "1").lower() in {"1", "true", "yes", "y", "on"}
    use_agent_bev_fusion: bool = os.getenv("USE_AGENT_BEV_FUSION", "1").lower() in {"1", "true", "yes", "y", "on"}
    use_agent3d_bev_fusion: bool = os.getenv("USE_AGENT3D_BEV_FUSION", "1").lower() in {"1", "true", "yes", "y", "on"}
    use_map_bev_fusion: bool = os.getenv("USE_MAP_BEV_FUSION", "1").lower() in {"1", "true", "yes", "y", "on"}
    use_corridor_bev_token: bool = os.getenv("USE_CORRIDOR_BEV_TOKEN", "1").lower() in {"1", "true", "yes", "y", "on"}
    bev_x_min: float = float(os.getenv("BEV_X_MIN", "-40.0"))
    bev_x_max: float = float(os.getenv("BEV_X_MAX", "40.0"))
    bev_y_min: float = float(os.getenv("BEV_Y_MIN", "-30.0"))
    bev_y_max: float = float(os.getenv("BEV_Y_MAX", "30.0"))
    agent3d_pos_scale: float = float(os.getenv("AGENT3D_POS_SCALE", "80.0"))

    max_agents: int = 24
    max_lanes: int = 32
    lane_points: int = 20

    # v9.2 cached 6-camera visual features and BEV-style query fusion.
    num_cameras: int = 6
    camera_feat_dim: int = 256
    num_bev_queries: int = 8

    # v10.0: v9.5 projection-aligned visual token fusion + raw nuScenes 3D agent tokens.
    # Camera features are still cached [6,256], but are no longer used only as one global token.
    # They are aligned to agent positions, lane polyline centers, and a forward ego-corridor token
    # using camera-view bearing weights in ego-local coordinates.
    camera_align_temperature: float = float(os.getenv("CAMERA_ALIGN_TEMPERATURE", "0.55"))
    use_global_bev_token: bool = os.getenv("USE_GLOBAL_BEV_TOKEN", "0").lower() in {"1", "true", "yes", "y", "on"}
    use_agent_aligned_visual: bool = os.getenv("USE_AGENT_ALIGNED_VISUAL", "0").lower() in {"1", "true", "yes", "y", "on"}
    use_map_aligned_visual: bool = os.getenv("USE_MAP_ALIGNED_VISUAL", "0").lower() in {"1", "true", "yes", "y", "on"}
    use_corridor_visual_token: bool = os.getenv("USE_CORRIDOR_VISUAL_TOKEN", "0").lower() in {"1", "true", "yes", "y", "on"}
    ego_corridor_x_m: float = float(os.getenv("EGO_CORRIDOR_X_M", "30.0"))

    future_steps: int = 6
    future_dim: int = 2
    num_modes: int = 3

    hidden_dim: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 2
    nhead: int = 4
    ff_mult: int = 4
    dropout: float = 0.25
    branch_dim: int = 128

    use_focal_loss: bool = True
    focal_gamma: float = 1.5

    lambda_cls: float = 1.0
    lambda_stop_logit_penalty: float = 0.15
    lambda_traj_mode: float = 0.8
    lambda_traj_gt: float = 3.0
    # Frozen-base losses are not optimized in v9.2; these are kept only for compatibility.
    lambda_score: float = 0.35
    lambda_map: float = 0.20
    lambda_diversity: float = 0.06
    lambda_comfort: float = 0.02
    lambda_intent: float = 0.15
    lambda_collision: float = 0.08

    # v14.1: generate BEV from cached 6-camera features and train it jointly with planner.
    # This is the practical bridge toward sensor-level end-to-end without loading raw images.
    use_learned_camera_bev: bool = os.getenv("USE_LEARNED_CAMERA_BEV", "1").lower() in {"1", "true", "yes", "y", "on"}
    # v14.1 hard guard: learned camera-BEV is meaningless if the camera cache is not loaded.
    # Set REQUIRE_REAL_CAMERA_CACHE=1 by default so a zero camera_valid cache fails fast.
    require_real_camera_cache: bool = os.getenv("REQUIRE_REAL_CAMERA_CACHE", "1").lower() in {"1", "true", "yes", "y", "on"}
    min_camera_valid_mean: float = float(os.getenv("MIN_CAMERA_VALID_MEAN", "0.05"))
    lambda_bev_distill: float = float(os.getenv("LAMBDA_BEV_DISTILL", "0.35"))
    camera_bev_hidden_dim: int = int(os.getenv("CAMERA_BEV_HIDDEN_DIM", "128"))
    camera_bev_dropout: float = float(os.getenv("CAMERA_BEV_DROPOUT", "0.10"))

    # v9.2 score-only calibration losses.
    lambda_score_soft: float = 1.0
    lambda_score_rank: float = float(os.getenv("LAMBDA_SCORE_RANK", "0.15"))
    lambda_score_reg: float = 0.03
    score_rank_margin: float = 0.10
    min_ade_improve: float = 1e-4

    traj_scale: float = 20.0
    agent_pos_scale: float = 60.0
    agent_vel_scale: float = 20.0
    map_pos_scale: float = 60.0
    dt: float = 0.5

    score_target_temp: float = 0.35
    score_speed_penalty_weight: float = 0.12
    score_speed_penalty_scale: float = 2.0
    score_label_smoothing: float = 0.02

    div_end_margin_m: float = 0.45
    div_full_margin_m: float = 0.25
    collision_radius_m: float = 2.5

    # v9.2: explicit but conservative score calibration features.
    score_quality_dim: int = 12
    score_refiner_hidden_dim: int = int(os.getenv("SCORE_REFINER_HIDDEN_DIM", "128"))
    score_refiner_dropout: float = 0.10
    score_quality_clip: float = 5.0
    score_delta_scale: float = float(os.getenv("SCORE_DELTA_SCALE", "0.30"))
    score_map_dist_scale: float = 10.0
    score_acc_scale: float = 10.0
    score_jerk_scale: float = 20.0
    score_agent_dist_scale: float = 20.0

    # v8.1: class-weight calibration. Prevent STOP from becoming a false-positive magnet.
    class_weight_min: float = 0.30
    class_weight_max: float = 4.00
    stop_class_weight_max: float = 3.50

    # v8.2: intent auxiliary target normalization.
    intent_speed_scale: float = 20.0
    intent_disp_scale: float = 20.0
    intent_yaw_scale: float = math.pi

    mode_y_offsets_m: Tuple[float, ...] = (-0.8, 0.0, 0.8)
    stop_mode_y_offsets_m: Tuple[float, ...] = (-0.2, 0.0, 0.2)
    mode_target_ramp_start: float = 0.15
    mode_target_ramp_end: float = 1.0

    # Full-pipeline control. Default trains v10.0 base if no v9.5 checkpoint exists.
    run_base_train: bool = os.getenv("RUN_BASE_TRAIN", "0").lower() in {"1", "true", "yes", "y", "on"}
    train_base_if_missing: bool = os.getenv("TRAIN_BASE_IF_MISSING", "1").lower() in {"1", "true", "yes", "y", "on"}
    base_epochs: int = int(os.getenv("BASE_EPOCHS", "80"))
    base_lr: float = float(os.getenv("BASE_LR", "4e-4"))
    base_weight_decay: float = float(os.getenv("BASE_WEIGHT_DECAY", "1e-4"))
    base_early_stop_patience: int = int(os.getenv("BASE_EARLY_STOP_PATIENCE", "20"))
    base_select_min_macro_f1: float = float(os.getenv("BASE_SELECT_MIN_MACRO_F1", "0.755"))
    planning_score_fde_weight: float = float(os.getenv("PLANNING_SCORE_FDE_WEIGHT", "0.50"))
    planning_score_gap_weight: float = float(os.getenv("PLANNING_SCORE_GAP_WEIGHT", "0.50"))
    base_save_subdir: str = os.getenv("BASE_SAVE_SUBDIR", "v14_1_base")
    base_best_model_name: str = "best_v14_1_base_model.pt"

    manifest_path: str = "/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/build_manifest_full_v8_map_agent_t6.json"
    shard_dir: str = "/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/shards_full_v8_map_agent_t6"
    save_dir: str = "/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v14_1_real_camera_cache_enabled"
    base_ckpt_path: str = os.getenv("BASE_CKPT_PATH", "/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v14_1_real_camera_cache_enabled/v14_1_base/best_v14_1_base_model.pt")
    init_ckpt_path: str = os.getenv("INIT_CKPT_PATH", "/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v8_123_intent_behavior/best_model.pt")
    projection_cache_manifest: str = os.getenv("PROJECTION_CACHE_MANIFEST", os.getenv("CAMERA_CACHE_MANIFEST", "/home/ubuntu22/decision_on_nuscenes/outputs_v9_5_projection_visual_cache/projection_visual_cache_manifest_v9_5.json"))
    agent3d_cache_manifest: str = os.getenv("AGENT3D_CACHE_MANIFEST", "/home/ubuntu22/decision_on_nuscenes/outputs_v10_0_3d_agent_cache/agent3d_cache_manifest_v10_0.json")
    bev_cache_manifest: str = os.getenv("BEV_CACHE_MANIFEST", "/home/ubuntu22/decision_on_nuscenes/outputs_v11_0_annotation_bev_cache/annotation_bev_cache_manifest_v11_0.json")
    best_model_name: str = "best_score_model.pt"
    metrics_name: str = "metrics.json"
    model_type: str = "transformer_v14_1_real_camera_cache_enabled_full_pipeline"

    device: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


CFG = Config(
    manifest_path=os.getenv("MANIFEST_PATH", Config.manifest_path),
    shard_dir=os.getenv("SHARD_DIR", Config.shard_dir),
    save_dir=os.getenv("SAVE_DIR", Config.save_dir),
)

CLASS_TO_ID = {c: i for i, c in enumerate(CFG.target_classes)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}


# ============================================================
# Utils
# ============================================================

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


# ============================================================
# Dataset loading
# ============================================================

def _fixed_array(sample: Dict[str, Any], key: str, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    if key not in sample or sample[key] is None:
        return np.zeros(shape, dtype=dtype)
    arr = np.asarray(sample[key], dtype=dtype)
    if arr.shape == shape:
        return arr
    out = np.zeros(shape, dtype=dtype)
    slices = tuple(slice(0, min(arr.shape[i], shape[i])) for i in range(min(arr.ndim, len(shape))))
    out[slices] = arr[slices]
    return out


def load_projection_visual_cache(cache_manifest_path: str) -> Dict[str, Tuple[np.ndarray, ...]]:
    """Load projection-aligned visual cache.

    sample_token -> (camera_feat, camera_valid, proj_agent_feat, proj_agent_valid,
                     proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid)
    """
    if not os.path.exists(cache_manifest_path):
        raise FileNotFoundError(
            f"projection visual cache manifest not found: {cache_manifest_path}\n"
            "Run build_v9_5_projection_visual_cache.py first or set PROJECTION_CACHE_MANIFEST correctly."
        )
    cache_manifest = load_json(cache_manifest_path)
    all_paths = cache_manifest.get("train_cache_shards", []) + cache_manifest.get("val_cache_shards", [])
    cache: Dict[str, Tuple[np.ndarray, ...]] = {}
    for p in all_paths:
        data = np.load(p, allow_pickle=True)
        tokens = data["sample_tokens"]
        camera_feat = data["camera_feat"].astype(np.float32)
        camera_valid = data["camera_valid"].astype(np.float32)
        proj_agent_feat = data["proj_agent_feat"].astype(np.float32)
        proj_agent_valid = data["proj_agent_valid"].astype(np.float32)
        proj_map_feat = data["proj_map_feat"].astype(np.float32)
        proj_map_valid = data["proj_map_valid"].astype(np.float32)
        proj_corridor_feat = data["proj_corridor_feat"].astype(np.float32)
        proj_corridor_valid = data["proj_corridor_valid"].astype(np.float32)
        for vals in zip(tokens, camera_feat, camera_valid, proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid):
            tok = str(vals[0])
            cache[tok] = vals[1:]
    return cache



def load_agent3d_cache(cache_manifest_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load v10.0 3D agent token cache.

    sample_token -> (agent3d_features [A3,D], agent3d_valid [A3])
    """
    if not cache_manifest_path or not os.path.exists(cache_manifest_path):
        raise FileNotFoundError(
            f"3D agent cache manifest not found: {cache_manifest_path}\n"
            "Run build_v10_0_3d_agent_cache.py first or set AGENT3D_CACHE_MANIFEST correctly."
        )
    manifest = load_json(cache_manifest_path)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cache_file in manifest.get("cache_files", []):
        if not os.path.exists(cache_file):
            print(f"[WARN] missing 3D-agent cache file: {cache_file}")
            continue
        data = np.load(cache_file, allow_pickle=True)
        tokens = [str(x) for x in data["sample_tokens"]]
        feat = data["agent3d_features"].astype(np.float32)
        valid = data["agent3d_valid"].astype(np.float32)
        for vals in zip(tokens, feat, valid):
            out[vals[0]] = (vals[1], vals[2])
    return out



def load_annotation_bev_cache(cache_manifest_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load v11.0 annotation-BEV cache.

    sample_token -> (bev_grid [C,H,W], bev_valid [1])
    """
    if not cache_manifest_path or not os.path.exists(cache_manifest_path):
        raise FileNotFoundError(
            f"annotation-BEV cache manifest not found: {cache_manifest_path}\n"
            "Run build_v11_0_annotation_bev_cache.py first or set BEV_CACHE_MANIFEST correctly."
        )
    manifest = load_json(cache_manifest_path)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cache_file in manifest.get("cache_files", []):
        if not os.path.exists(cache_file):
            print(f"[WARN] missing BEV cache file: {cache_file}")
            continue
        data = np.load(cache_file, allow_pickle=True)
        tokens = [str(x) for x in data["sample_tokens"]]
        grid = data["bev_grid"].astype(np.float32)
        valid = data["bev_valid"].astype(np.float32)
        for tok, g, v in zip(tokens, grid, valid):
            out[tok] = (g, v)
    return out

def load_split_from_shards(manifest_path: str, shard_dir: str, split: str = "train", projection_cache: Optional[Dict[str, Tuple[np.ndarray, ...]]] = None, agent3d_cache: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None, bev_cache: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None):
    manifest = load_json(manifest_path)
    shard_names = manifest[f"{split}_shards"]

    hist_list, agent_list, agent_valid_list = [], [], []
    agent3d_list, agent3d_valid_list = [], []
    bev_grid_list, bev_valid_list = [], []
    map_list, map_valid_list = [], []
    y_cls_list, y_traj_list = [], []
    map_y_ref_list, map_ref_valid_list = [], []
    intent_target_list = []
    camera_feat_list, camera_valid_list = [], []
    proj_agent_feat_list, proj_agent_valid_list = [], []
    proj_map_feat_list, proj_map_valid_list = [], []
    proj_corridor_feat_list, proj_corridor_valid_list = [], []

    for shard_name in shard_names:
        shard_path = shard_name if os.path.isabs(shard_name) else os.path.join(shard_dir, shard_name)
        data = load_pickle(shard_path)
        for s in data:
            if not bool(s.get("history_valid", False)):
                continue
            label = normalize_label(s.get("label_name", ""))
            if label not in CLASS_TO_ID:
                continue

            hist = np.asarray(s.get("history_features"), dtype=np.float32)
            if hist.ndim != 2 or hist.shape[0] < CFG.history_len or hist.shape[1] != CFG.history_dim:
                continue
            hist = hist[:CFG.history_len, :CFG.history_dim]

            future_xy_local = s.get("future_xy_local", None)
            if future_xy_local is None:
                continue
            future_xy_local = np.asarray(future_xy_local, dtype=np.float32)
            if future_xy_local.ndim != 2 or future_xy_local.shape[0] < CFG.future_steps or future_xy_local.shape[1] < 2:
                continue
            future_xy_local = future_xy_local[:CFG.future_steps, :2]
            y_traj = future_xy_local / CFG.traj_scale

            # v8.2 intent targets, normalized for stable SmoothL1 supervision:
            # [terminal_speed, total_disp, abs_lateral_disp, abs_yaw_delta]
            future_t = np.asarray(s.get("future_t", np.arange(1, CFG.future_steps + 1, dtype=np.float32) * CFG.dt), dtype=np.float32)
            if len(future_xy_local) >= 2 and len(future_t) >= 2:
                dt_last = float(future_t[min(len(future_t), CFG.future_steps) - 1] - future_t[min(len(future_t), CFG.future_steps) - 2])
                if dt_last <= 1e-6:
                    dt_last = CFG.dt
                terminal_speed = float(np.linalg.norm(future_xy_local[-1] - future_xy_local[-2]) / dt_last)
            else:
                terminal_speed = float(s.get("future_terminal_speed", 0.0))

            total_disp = float(s.get("future_total_disp", np.linalg.norm(future_xy_local[-1])))
            lateral_disp = float(abs(future_xy_local[-1, 1]))
            future_yaw_local = s.get("future_yaw_local", None)
            if future_yaw_local is not None:
                fyaw = np.asarray(future_yaw_local, dtype=np.float32)
                yaw_delta = float(abs(fyaw[min(len(fyaw), CFG.future_steps) - 1])) if len(fyaw) > 0 else 0.0
            else:
                yaw_delta = 0.0

            intent_target = np.array([
                terminal_speed / CFG.intent_speed_scale,
                total_disp / CFG.intent_disp_scale,
                lateral_disp / CFG.intent_disp_scale,
                yaw_delta / CFG.intent_yaw_scale,
            ], dtype=np.float32)

            sample_token = str(s.get("sample_token", ""))
            if agent3d_cache is not None and sample_token in agent3d_cache:
                agent3d_feat, agent3d_valid = agent3d_cache[sample_token]
                agent3d_feat = np.asarray(agent3d_feat, dtype=np.float32)
                agent3d_valid = np.asarray(agent3d_valid, dtype=np.float32)
            else:
                agent3d_feat = np.zeros((CFG.max_agent3d, CFG.agent3d_dim), dtype=np.float32)
                agent3d_valid = np.zeros((CFG.max_agent3d,), dtype=np.float32)

            if bev_cache is not None and sample_token in bev_cache:
                bev_grid, bev_valid = bev_cache[sample_token]
                bev_grid = np.asarray(bev_grid, dtype=np.float32)
                bev_valid = np.asarray(bev_valid, dtype=np.float32)
            else:
                bev_grid = np.zeros((CFG.bev_channels, CFG.bev_h, CFG.bev_w), dtype=np.float32)
                bev_valid = np.zeros((1,), dtype=np.float32)

            if projection_cache is not None and sample_token in projection_cache:
                (camera_feat, camera_valid, proj_agent_feat, proj_agent_valid,
                 proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid) = projection_cache[sample_token]
                camera_feat = np.asarray(camera_feat, dtype=np.float32)
                camera_valid = np.asarray(camera_valid, dtype=np.float32)
                proj_agent_feat = np.asarray(proj_agent_feat, dtype=np.float32)
                proj_agent_valid = np.asarray(proj_agent_valid, dtype=np.float32)
                proj_map_feat = np.asarray(proj_map_feat, dtype=np.float32)
                proj_map_valid = np.asarray(proj_map_valid, dtype=np.float32)
                proj_corridor_feat = np.asarray(proj_corridor_feat, dtype=np.float32)
                proj_corridor_valid = np.asarray(proj_corridor_valid, dtype=np.float32)
            else:
                camera_feat = np.zeros((CFG.num_cameras, CFG.camera_feat_dim), dtype=np.float32)
                camera_valid = np.zeros((CFG.num_cameras,), dtype=np.float32)
                proj_agent_feat = np.zeros((CFG.max_agents, CFG.camera_feat_dim), dtype=np.float32)
                proj_agent_valid = np.zeros((CFG.max_agents,), dtype=np.float32)
                proj_map_feat = np.zeros((CFG.max_lanes, CFG.camera_feat_dim), dtype=np.float32)
                proj_map_valid = np.zeros((CFG.max_lanes,), dtype=np.float32)
                proj_corridor_feat = np.zeros((1, CFG.camera_feat_dim), dtype=np.float32)
                proj_corridor_valid = np.zeros((1,), dtype=np.float32)

            agents = _fixed_array(s, "agent_features", (CFG.max_agents, CFG.agent_dim))
            agent_valid = _fixed_array(s, "agent_valid", (CFG.max_agents,))
            maps = _fixed_array(s, "map_polylines", (CFG.max_lanes, CFG.lane_points, CFG.map_dim))
            map_valid = _fixed_array(s, "map_polyline_valid", (CFG.max_lanes, CFG.lane_points))
            map_y_ref = _fixed_array(s, "map_y_ref", (CFG.future_steps,))
            map_ref_valid = _fixed_array(s, "map_ref_valid", (CFG.future_steps,))

            hist_list.append(hist)
            agent_list.append(agents)
            agent_valid_list.append(agent_valid)
            agent3d_list.append(agent3d_feat)
            agent3d_valid_list.append(agent3d_valid)
            bev_grid_list.append(bev_grid)
            bev_valid_list.append(bev_valid)
            map_list.append(maps)
            map_valid_list.append(map_valid)
            y_cls_list.append(CLASS_TO_ID[label])
            y_traj_list.append(y_traj)
            map_y_ref_list.append(map_y_ref)
            map_ref_valid_list.append(map_ref_valid)
            intent_target_list.append(intent_target)
            camera_feat_list.append(camera_feat)
            camera_valid_list.append(camera_valid)
            proj_agent_feat_list.append(proj_agent_feat)
            proj_agent_valid_list.append(proj_agent_valid)
            proj_map_feat_list.append(proj_map_feat)
            proj_map_valid_list.append(proj_map_valid)
            proj_corridor_feat_list.append(proj_corridor_feat)
            proj_corridor_valid_list.append(proj_corridor_valid)

    if len(hist_list) == 0:
        raise RuntimeError(f"No valid samples found for split={split}")

    return (
        np.stack(hist_list, axis=0),
        np.stack(agent_list, axis=0),
        np.stack(agent_valid_list, axis=0),
        np.stack(agent3d_list, axis=0).astype(np.float32),
        np.stack(agent3d_valid_list, axis=0).astype(np.float32),
        np.stack(bev_grid_list, axis=0).astype(np.float32),
        np.stack(bev_valid_list, axis=0).astype(np.float32),
        np.stack(map_list, axis=0),
        np.stack(map_valid_list, axis=0),
        np.asarray(y_cls_list, dtype=np.int64),
        np.stack(y_traj_list, axis=0).astype(np.float32),
        np.stack(map_y_ref_list, axis=0).astype(np.float32),
        np.stack(map_ref_valid_list, axis=0).astype(np.float32),
        np.stack(intent_target_list, axis=0).astype(np.float32),
        np.stack(camera_feat_list, axis=0).astype(np.float32),
        np.stack(camera_valid_list, axis=0).astype(np.float32),
        np.stack(proj_agent_feat_list, axis=0).astype(np.float32),
        np.stack(proj_agent_valid_list, axis=0).astype(np.float32),
        np.stack(proj_map_feat_list, axis=0).astype(np.float32),
        np.stack(proj_map_valid_list, axis=0).astype(np.float32),
        np.stack(proj_corridor_feat_list, axis=0).astype(np.float32),
        np.stack(proj_corridor_valid_list, axis=0).astype(np.float32),
    )


class JointTokenDataset(Dataset):
    def __init__(self, hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, y_cls, y_traj, map_y_ref, map_ref_valid, intent_target, camera_feat, camera_valid, proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid):
        self.hist = torch.tensor(hist, dtype=torch.float32)
        self.agents = torch.tensor(agents, dtype=torch.float32)
        self.agent_valid = torch.tensor(agent_valid, dtype=torch.float32)
        self.agent3d = torch.tensor(agent3d, dtype=torch.float32)
        self.agent3d_valid = torch.tensor(agent3d_valid, dtype=torch.float32)
        self.bev_grid = torch.tensor(bev_grid, dtype=torch.float32)
        self.bev_valid = torch.tensor(bev_valid, dtype=torch.float32)
        self.maps = torch.tensor(maps, dtype=torch.float32)
        self.map_valid = torch.tensor(map_valid, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_traj = torch.tensor(y_traj, dtype=torch.float32)
        self.map_y_ref = torch.tensor(map_y_ref, dtype=torch.float32)
        self.map_ref_valid = torch.tensor(map_ref_valid, dtype=torch.float32)
        self.intent_target = torch.tensor(intent_target, dtype=torch.float32)
        self.camera_feat = torch.tensor(camera_feat, dtype=torch.float32)
        self.camera_valid = torch.tensor(camera_valid, dtype=torch.float32)
        self.proj_agent_feat = torch.tensor(proj_agent_feat, dtype=torch.float32)
        self.proj_agent_valid = torch.tensor(proj_agent_valid, dtype=torch.float32)
        self.proj_map_feat = torch.tensor(proj_map_feat, dtype=torch.float32)
        self.proj_map_valid = torch.tensor(proj_map_valid, dtype=torch.float32)
        self.proj_corridor_feat = torch.tensor(proj_corridor_feat, dtype=torch.float32)
        self.proj_corridor_valid = torch.tensor(proj_corridor_valid, dtype=torch.float32)

    def __len__(self):
        return len(self.hist)

    def __getitem__(self, idx):
        return (
            self.hist[idx],
            self.agents[idx],
            self.agent_valid[idx],
            self.agent3d[idx],
            self.agent3d_valid[idx],
            self.bev_grid[idx],
            self.bev_valid[idx],
            self.maps[idx],
            self.map_valid[idx],
            self.y_cls[idx],
            self.y_traj[idx],
            self.map_y_ref[idx],
            self.map_ref_valid[idx],
            self.intent_target[idx],
            self.camera_feat[idx],
            self.camera_valid[idx],
            self.proj_agent_feat[idx],
            self.proj_agent_valid[idx],
            self.proj_map_feat[idx],
            self.proj_map_valid[idx],
            self.proj_corridor_feat[idx],
            self.proj_corridor_valid[idx],
        )


# ============================================================
# Model
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1), :]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CameraToBEVGenerator(nn.Module):
    """
    v14.1 lightweight learned camera-BEV generator.

    Input is the existing cached six-camera descriptor [B,6,256]. This is not raw-image
    BEVFormer, but it is a real learnable perception-to-BEV bridge inside the planner
    training loop. Annotation-BEV is used only as a teacher/distillation target.
    """
    def __init__(self, camera_dim: int, num_cameras: int, out_channels: int, hidden_dim: int, bev_h: int, bev_w: int, dropout: float = 0.10):
        super().__init__()
        self.num_cameras = num_cameras
        self.out_channels = out_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.seed_h = max(4, math.ceil(bev_h / 8))
        self.seed_w = max(4, math.ceil(bev_w / 8))
        self.camera_proj = nn.Sequential(
            nn.Linear(camera_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.camera_embed = nn.Parameter(torch.zeros(1, num_cameras, hidden_dim))
        self.to_seed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * self.seed_h * self.seed_w),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, out_channels, kernel_size=3, padding=1),
        )
        nn.init.normal_(self.camera_embed, mean=0.0, std=0.02)

    def forward(self, camera_feat: torch.Tensor, camera_valid: torch.Tensor) -> torch.Tensor:
        B = camera_feat.size(0)
        cam = self.camera_proj(camera_feat) + self.camera_embed[:, :camera_feat.size(1), :]
        valid = (camera_valid > 0.5).unsqueeze(-1).to(dtype=cam.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        pooled = (cam * valid).sum(dim=1) / denom
        seed = self.to_seed(pooled).view(B, -1, self.seed_h, self.seed_w)
        bev_logits = self.decoder(seed)
        if bev_logits.shape[-2:] != (self.bev_h, self.bev_w):
            bev_logits = torch.nn.functional.interpolate(bev_logits, size=(self.bev_h, self.bev_w), mode="bilinear", align_corners=False)
        return torch.sigmoid(bev_logits)


class MapAgentEgoPlanner(nn.Module):
    def __init__(
        self,
        history_dim: int,
        agent_dim: int,
        agent3d_dim: int,
        bev_channels: int,
        map_dim: int,
        camera_dim: int,
        num_cameras: int,
        num_bev_queries: int,
        hidden_dim: int,
        num_classes: int,
        future_steps: int,
        future_dim: int,
        num_modes: int,
        nhead: int,
        ff_mult: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
        branch_dim: int,
    ):
        super().__init__()
        self.K = num_modes
        self.future_steps = future_steps
        self.future_dim = future_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_cameras = num_cameras
        self.num_bev_queries = num_bev_queries

        self.ego_proj = MLP(history_dim, hidden_dim, hidden_dim, dropout)
        self.agent_proj = MLP(agent_dim, hidden_dim, hidden_dim, dropout)
        self.agent3d_proj = MLP(agent3d_dim, hidden_dim, hidden_dim, dropout)
        self.annotation_bev_encoder = nn.Sequential(
            nn.Conv2d(bev_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((CFG.bev_token_grid, CFG.bev_token_grid)),
        )
        self.annotation_bev_norm = nn.LayerNorm(hidden_dim)

        # v14.1 real-camera-cache learned camera-BEV joint adapters. Input is sampled raw BEV channel vector [C]
        # at each object/lane/corridor location. Gates start small so warm-started
        # structured planner is not destroyed early.
        self.bev_local_proj = MLP(bev_channels, hidden_dim, hidden_dim, dropout)
        self.agent_bev_proj = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.agent3d_bev_proj = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.map_bev_proj = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.corridor_bev_proj = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.agent_bev_gate = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.agent3d_bev_gate = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.map_bev_gate = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.corridor_bev_gate = nn.Parameter(torch.tensor(0.50, dtype=torch.float32))

        self.map_point_proj = MLP(map_dim, hidden_dim, hidden_dim, dropout)
        self.map_poly_proj = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.camera_proj = MLP(camera_dim, hidden_dim, hidden_dim, dropout)
        self.camera_bev_generator = CameraToBEVGenerator(
            camera_dim=camera_dim,
            num_cameras=num_cameras,
            out_channels=bev_channels,
            hidden_dim=CFG.camera_bev_hidden_dim,
            bev_h=CFG.bev_h,
            bev_w=CFG.bev_w,
            dropout=CFG.camera_bev_dropout,
        )
        # v9.5: projected image-patch descriptors for agent / lane / corridor tokens.
        self.projected_visual_proj = MLP(camera_dim, hidden_dim, hidden_dim, dropout)

        # scene / ego / basic_agent / map / global_bev / corridor_visual / raw_3d_agent
        self.type_embed = nn.Embedding(8, hidden_dim)
        self.ego_pos_enc = PositionalEncoding(hidden_dim, max_len=64)
        self.scene_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # nuScenes camera order in build_v9_2_camera_feature_cache.py:
        # [FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT]
        # Ego-local convention: x forward, y left. Angles: front=0, left=+60°, right=-60°.
        camera_angles = torch.tensor(
            [0.0, math.pi / 3.0, -math.pi / 3.0, math.pi, 2.0 * math.pi / 3.0, -2.0 * math.pi / 3.0],
            dtype=torch.float32,
        )
        self.register_buffer("camera_view_angles", camera_angles)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # v9.2/v9.5: global BEV-style visual query fusion over cached six-camera features.
        self.bev_queries = nn.Parameter(torch.zeros(num_bev_queries, hidden_dim))
        bev_dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.bev_decoder = nn.TransformerDecoder(bev_dec_layer, num_layers=1)
        self.bev_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # v9.5: geometry-aligned visual adapters. These are residual adapters so
        # warm-starting from v8/v9.2 does not destroy the already-good structured planner.
        self.agent_visual_proj = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.map_visual_proj = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.corridor_visual_proj = MLP(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.agent_visual_gate = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.map_visual_gate = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.corridor_visual_gate = nn.Parameter(torch.tensor(1.00, dtype=torch.float32))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # v9.2/v10.0 base: shared mode queries + behavior-conditioned mode priors.
        self.mode_queries = nn.Parameter(torch.zeros(num_modes, hidden_dim))
        self.behavior_mode_queries = nn.Parameter(torch.zeros(num_classes, num_modes, hidden_dim))
        self.query_norm = nn.LayerNorm(hidden_dim)

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, num_classes),
        )
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_dim, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, 4),
        )

        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, future_steps * future_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim + future_steps * future_dim, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.scene_token, mean=0.0, std=0.02)
        nn.init.normal_(self.mode_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.behavior_mode_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.bev_queries, mean=0.0, std=0.02)

    def encode_map(self, maps: torch.Tensor, map_valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # maps: [B,M,P,C], map_valid: [B,M,P]
        point_emb = self.map_point_proj(maps)  # [B,M,P,H]
        w = map_valid.unsqueeze(-1).float()
        denom = w.sum(dim=2).clamp_min(1.0)
        poly = (point_emb * w).sum(dim=2) / denom  # [B,M,H]
        poly = self.map_poly_proj(poly)
        lane_valid = map_valid.sum(dim=2) > 0.5

        # v9.5: lane center in ego-local normalized coordinates. Only x/y matter for bearing.
        map_xy = maps[..., 0:2]
        center = (map_xy * w).sum(dim=2) / denom
        return poly, lane_valid, center

    def encode_camera_tokens(self, camera_feat: torch.Tensor, camera_valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cam_tok = self.camera_proj(camera_feat)
        cam_valid_bool = camera_valid > 0.5
        cam_key_padding = ~cam_valid_bool
        all_invalid = cam_key_padding.all(dim=1)
        if all_invalid.any():
            cam_key_padding = cam_key_padding.clone()
            cam_key_padding[all_invalid] = False
            cam_tok = cam_tok.clone()
            cam_tok[all_invalid] = 0.0
        return cam_tok, cam_valid_bool, cam_key_padding

    def encode_camera_bev_from_tokens(self, cam_tok: torch.Tensor, cam_valid_bool: torch.Tensor, cam_key_padding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = cam_tok.size(0)
        device = cam_tok.device
        bev_q = self.bev_queries.unsqueeze(0).expand(B, -1, -1)
        bev_feat = self.bev_decoder(tgt=bev_q, memory=cam_tok, memory_key_padding_mask=cam_key_padding)
        bev_token = self.bev_pool(bev_feat.mean(dim=1)).unsqueeze(1)
        bev_token = bev_token + self.type_embed(torch.tensor(4, device=device)).view(1, 1, -1)
        bev_valid = cam_valid_bool.any(dim=1, keepdim=True)
        return bev_token, bev_valid

    def aggregate_camera_by_xy(self, cam_tok: torch.Tensor, cam_valid_bool: torch.Tensor, xy_local_norm: torch.Tensor) -> torch.Tensor:
        """View-angle weighted camera aggregation for arbitrary ego-local points.

        xy_local_norm: [B,N,2], x forward, y left. Scale does not matter for atan2.
        returns: [B,N,H]
        """
        angle = torch.atan2(xy_local_norm[..., 1], xy_local_norm[..., 0])  # [B,N]
        cam_angles = self.camera_view_angles.to(device=xy_local_norm.device, dtype=xy_local_norm.dtype).view(1, 1, -1)
        diff = angle.unsqueeze(-1) - cam_angles
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        logits = torch.cos(diff) / max(float(CFG.camera_align_temperature), 1e-3)
        valid = cam_valid_bool.unsqueeze(1)
        logits = logits.masked_fill(~valid, -1e4)
        all_invalid = ~cam_valid_bool.any(dim=1)
        if all_invalid.any():
            logits = logits.clone()
            logits[all_invalid] = 0.0
        weights = torch.softmax(logits, dim=-1) * valid.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bnc,bch->bnh", weights, cam_tok)

    def sample_bev_at_xy(self, bev_grid: torch.Tensor, xy_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample raw BEV grid channels at ego-local metric points.

        bev_grid: [B,C,H,W], xy_m: [B,N,2] where x is forward, y is left.
        Returns sampled [B,N,C] and inside mask [B,N].
        """
        x = xy_m[..., 0]
        y = xy_m[..., 1]
        gx = 2.0 * (y - CFG.bev_y_min) / max(CFG.bev_y_max - CFG.bev_y_min, 1e-6) - 1.0
        gy = 2.0 * (x - CFG.bev_x_min) / max(CFG.bev_x_max - CFG.bev_x_min, 1e-6) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # [B,N,1,2]
        sampled = torch.nn.functional.grid_sample(
            bev_grid, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # [B,C,N,1]
        sampled = sampled.squeeze(-1).transpose(1, 2).contiguous()  # [B,N,C]
        inside = (gx >= -1.0) & (gx <= 1.0) & (gy >= -1.0) & (gy <= 1.0)
        return sampled, inside

    def object_aligned_bev_embedding(self, bev_grid: torch.Tensor, xy_m: torch.Tensor, obj_valid: torch.Tensor) -> torch.Tensor:
        local_bev, inside = self.sample_bev_at_xy(bev_grid, xy_m)
        emb = self.bev_local_proj(local_bev)
        mask = (obj_valid > 0.5) & inside
        return emb * mask.unsqueeze(-1).float()

    def build_corridor_bev_token(self, bev_grid: torch.Tensor, bev_valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = bev_grid.size(0)
        device = bev_grid.device
        xs = torch.tensor([8.0, 16.0, 24.0, 32.0], device=device, dtype=bev_grid.dtype)
        ys = torch.zeros_like(xs)
        pts = torch.stack([xs, ys], dim=-1).view(1, -1, 2).expand(B, -1, -1)
        valid = torch.ones((B, pts.size(1)), device=device, dtype=bev_grid.dtype)
        emb = self.object_aligned_bev_embedding(bev_grid, pts, valid)
        token = emb.mean(dim=1, keepdim=True)
        token = self.corridor_bev_gate * self.corridor_bev_proj(token)
        token = token + self.type_embed(torch.tensor(7, device=device)).view(1, 1, -1)
        token_valid = (bev_valid > 0.5).view(B, 1)
        return token, token_valid

    def encode_annotation_bev(self, bev_grid: torch.Tensor, bev_valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = bev_grid.size(0)
        device = bev_grid.device
        feat = self.annotation_bev_encoder(bev_grid)  # [B,H,g,g]
        feat = feat.flatten(2).transpose(1, 2).contiguous()  # [B,g*g,H]
        feat = self.annotation_bev_norm(feat)
        feat = feat + self.type_embed(torch.tensor(7, device=device)).view(1, 1, -1)
        valid = (bev_valid > 0.5).view(B, 1).expand(B, feat.size(1))
        return feat, valid

    def build_tokens(self, hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, camera_feat, camera_valid,
                     proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid,
                     proj_corridor_feat, proj_corridor_valid):
        B = hist.size(0)
        device = hist.device

        scene = self.scene_token.expand(B, 1, -1) + self.type_embed(torch.tensor(0, device=device)).view(1, 1, -1)

        ego = self.ego_proj(hist)
        ego = self.ego_pos_enc(ego)
        ego = ego + self.type_embed(torch.tensor(1, device=device)).view(1, 1, -1)
        ego_valid = torch.ones((B, ego.size(1)), dtype=torch.bool, device=device)

        # Global six-camera descriptor is kept only for optional global BEV token.
        # Agent/map/corridor visual information comes from true camera projection cache.
        cam_tok, cam_valid_bool, cam_key_padding = self.encode_camera_tokens(camera_feat, camera_valid)

        agent_base = self.agent_proj(agents)
        if CFG.use_object_aligned_bev and CFG.use_agent_bev_fusion:
            agent_xy_m = agents[..., 0:2] * CFG.agent_pos_scale
            agent_bev = self.object_aligned_bev_embedding(bev_grid, agent_xy_m, agent_valid)
            agent_base = agent_base + self.agent_bev_gate * self.agent_bev_proj(agent_bev)
        if CFG.use_agent_aligned_visual:
            agent_vis = self.projected_visual_proj(proj_agent_feat)
            agent_vis = agent_vis * (proj_agent_valid > 0.5).unsqueeze(-1).float()
            agent_base = agent_base + self.agent_visual_gate * self.agent_visual_proj(agent_vis)
        agent = agent_base + self.type_embed(torch.tensor(2, device=device)).view(1, 1, -1)
        agent_valid_bool = agent_valid > 0.5

        if CFG.use_agent3d_tokens:
            agent3d_base = self.agent3d_proj(agent3d)
            if CFG.use_object_aligned_bev and CFG.use_agent3d_bev_fusion:
                agent3d_xy_m = agent3d[..., 0:2] * CFG.agent3d_pos_scale
                agent3d_bev = self.object_aligned_bev_embedding(bev_grid, agent3d_xy_m, agent3d_valid)
                agent3d_base = agent3d_base + self.agent3d_bev_gate * self.agent3d_bev_proj(agent3d_bev)
            agent3d_tok = agent3d_base + self.type_embed(torch.tensor(6, device=device)).view(1, 1, -1)
            agent3d_valid_bool = agent3d_valid > 0.5
        else:
            agent3d_tok = agent.new_zeros((B, 0, self.hidden_dim))
            agent3d_valid_bool = torch.zeros((B, 0), dtype=torch.bool, device=device)

        map_base, lane_valid, lane_center_xy = self.encode_map(maps, map_valid)
        if CFG.use_object_aligned_bev and CFG.use_map_bev_fusion:
            lane_xy_m = lane_center_xy * CFG.map_pos_scale
            lane_bev = self.object_aligned_bev_embedding(bev_grid, lane_xy_m, lane_valid.float())
            map_base = map_base + self.map_bev_gate * self.map_bev_proj(lane_bev)
        if CFG.use_map_aligned_visual:
            map_vis = self.projected_visual_proj(proj_map_feat)
            map_vis = map_vis * (proj_map_valid > 0.5).unsqueeze(-1).float()
            map_base = map_base + self.map_visual_gate * self.map_visual_proj(map_vis)
        map_tok = map_base + self.type_embed(torch.tensor(3, device=device)).view(1, 1, -1)

        extra_tokens = []
        extra_valids = []

        if CFG.use_global_bev_token:
            bev_token, bev_valid = self.encode_camera_bev_from_tokens(cam_tok, cam_valid_bool, cam_key_padding)
            extra_tokens.append(bev_token)
            extra_valids.append(bev_valid)

        if CFG.use_corridor_visual_token:
            corridor_vis = self.projected_visual_proj(proj_corridor_feat)
            corridor_vis = corridor_vis * (proj_corridor_valid > 0.5).unsqueeze(-1).float()
            corridor_token = self.corridor_visual_gate * self.corridor_visual_proj(corridor_vis)
            corridor_token = corridor_token + self.type_embed(torch.tensor(5, device=device)).view(1, 1, -1)
            corridor_valid = proj_corridor_valid > 0.5
            extra_tokens.append(corridor_token)
            extra_valids.append(corridor_valid)

        if CFG.use_object_aligned_bev and CFG.use_corridor_bev_token:
            corridor_bev_token, corridor_bev_valid = self.build_corridor_bev_token(bev_grid, bev_valid)
            extra_tokens.append(corridor_bev_token)
            extra_valids.append(corridor_bev_valid)

        # Optional dense BEV token path. Default is OFF in v12.0 because v11.0 showed
        # directly appending dense BEV tokens improves oracle but can hurt selected ADE.
        # Object-aligned BEV fusion above only needs USE_ANNOTATION_BEV=1 to load/supply bev_grid.
        if CFG.use_dense_bev_tokens:
            bev_tokens, bev_token_valid = self.encode_annotation_bev(bev_grid, bev_valid)
            extra_tokens.append(bev_tokens)
            extra_valids.append(bev_token_valid)

        if extra_tokens:
            visual_tokens = torch.cat(extra_tokens, dim=1)
            visual_valid = torch.cat(extra_valids, dim=1)
            tokens = torch.cat([scene, ego, agent, agent3d_tok, map_tok, visual_tokens], dim=1)
            valid = torch.cat([
                torch.ones((B, 1), dtype=torch.bool, device=device),
                ego_valid,
                agent_valid_bool,
                agent3d_valid_bool,
                lane_valid,
                visual_valid,
            ], dim=1)
        else:
            tokens = torch.cat([scene, ego, agent, agent3d_tok, map_tok], dim=1)
            valid = torch.cat([
                torch.ones((B, 1), dtype=torch.bool, device=device),
                ego_valid,
                agent_valid_bool,
                agent3d_valid_bool,
                lane_valid,
            ], dim=1)

        key_padding_mask = ~valid
        return tokens, key_padding_mask

    def forward(self, hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, camera_feat, camera_valid,
                proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid,
                proj_corridor_feat, proj_corridor_valid, return_bev_aux: bool = False):
        B = hist.size(0)
        learned_bev_grid = None
        effective_bev_grid = bev_grid
        if CFG.use_learned_camera_bev:
            learned_bev_grid = self.camera_bev_generator(camera_feat, camera_valid)
            # Important: planner consumes camera-generated BEV; annotation-BEV remains teacher only.
            effective_bev_grid = learned_bev_grid
        memory_in, key_padding_mask = self.build_tokens(
            hist, agents, agent_valid, agent3d, agent3d_valid, effective_bev_grid, bev_valid, maps, map_valid, camera_feat, camera_valid,
            proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid,
            proj_corridor_feat, proj_corridor_valid,
        )
        memory = self.encoder(memory_in, src_key_padding_mask=key_padding_mask)

        scene_feat = memory[:, 0, :]
        cls_logits = self.cls_head(scene_feat)
        cls_prob = torch.softmax(cls_logits, dim=-1)
        intent_pred = self.intent_head(scene_feat)

        shared_queries = self.mode_queries.unsqueeze(0).expand(B, -1, -1)
        behavior_queries = torch.einsum("bc,ckh->bkh", cls_prob, self.behavior_mode_queries)
        queries = self.query_norm(shared_queries + behavior_queries)
        mode_feat = self.decoder(tgt=queries, memory=memory, memory_key_padding_mask=key_padding_mask)

        cls_expand = cls_prob.unsqueeze(1).expand(B, self.K, -1)
        traj_input = torch.cat([mode_feat, cls_expand], dim=-1)
        traj = self.traj_head(traj_input).view(B, self.K, self.future_steps, self.future_dim)

        traj_feat = traj.detach().reshape(B, self.K, -1)
        score_input = torch.cat([mode_feat, traj_feat], dim=-1)
        scores = self.score_head(score_input).squeeze(-1)
        if return_bev_aux:
            return cls_logits, traj, scores, intent_pred, learned_bev_grid
        return cls_logits, traj, scores, intent_pred

# ============================================================
# v10.0 base training losses / metrics / train loop
# ============================================================

def stop_logit_penalty(cls_logits: torch.Tensor, y_cls: torch.Tensor) -> torch.Tensor:
    stop_id = CLASS_TO_ID["STOP"]
    non_stop = (y_cls != stop_id).float()
    if non_stop.sum().item() < 1:
        return cls_logits.new_tensor(0.0)
    penalty = torch.nn.functional.softplus(cls_logits[:, stop_id])
    return (penalty * non_stop).sum() / non_stop.sum()


def build_mode_targets(y_traj: torch.Tensor, y_cls: torch.Tensor) -> torch.Tensor:
    B, T, _ = y_traj.shape
    K = CFG.num_modes
    mode_targets = y_traj.unsqueeze(1).repeat(1, K, 1, 1).clone()
    ramp = torch.linspace(
        CFG.mode_target_ramp_start,
        CFG.mode_target_ramp_end,
        steps=T,
        device=y_traj.device,
        dtype=y_traj.dtype,
    ).view(1, 1, T)
    offsets_default = torch.tensor(CFG.mode_y_offsets_m, device=y_traj.device, dtype=y_traj.dtype) / CFG.traj_scale
    offsets_stop = torch.tensor(CFG.stop_mode_y_offsets_m, device=y_traj.device, dtype=y_traj.dtype) / CFG.traj_scale
    stop_id = CLASS_TO_ID["STOP"]
    is_stop = (y_cls == stop_id).view(B, 1)
    offsets = offsets_default.view(1, K).repeat(B, 1)
    offsets = torch.where(is_stop, offsets_stop.view(1, K).repeat(B, 1), offsets)
    mode_targets[:, :, :, 1] = mode_targets[:, :, :, 1] + offsets.unsqueeze(-1) * ramp
    return mode_targets


def base_multimodal_losses(traj_pred: torch.Tensor, y_traj: torch.Tensor, y_cls: torch.Tensor, scores: torch.Tensor):
    mode_targets = build_mode_targets(y_traj, y_cls)
    traj_err_mode = ((traj_pred - mode_targets) ** 2).sum(dim=-1).mean(dim=-1)
    loss_traj_mode = traj_err_mode.mean()

    y_expand = y_traj.unsqueeze(1).expand_as(traj_pred)
    traj_err_gt = ((traj_pred - y_expand) ** 2).sum(dim=-1).mean(dim=-1)
    best_idx_gt = traj_err_gt.argmin(dim=1)
    loss_traj_gt = traj_err_gt.min(dim=1)[0].mean()

    target_prob, ade_like, best_idx_score = compute_soft_score_target(traj_pred, y_traj)
    log_scores = torch.log_softmax(scores, dim=1)
    loss_score = -(target_prob * log_scores).sum(dim=1).mean()
    pred_idx = scores.argmax(dim=1)
    return loss_traj_mode, loss_traj_gt, loss_score, best_idx_gt, best_idx_score, pred_idx, ade_like


def compute_real_map_loss(traj_best: torch.Tensor, map_y_ref: torch.Tensor, map_ref_valid: torch.Tensor) -> torch.Tensor:
    pred_y = traj_best[..., 1]
    sq_err = (pred_y - map_y_ref) ** 2
    w = map_ref_valid.float()
    denom = w.sum()
    if denom.item() < 1e-6:
        return pred_y.new_tensor(0.0)
    return (sq_err * w).sum() / denom


def compute_diversity_loss(traj_pred: torch.Tensor) -> torch.Tensor:
    B, K, T, _ = traj_pred.shape
    if K <= 1:
        return traj_pred.new_tensor(0.0)
    pair_losses = []
    end_margin = CFG.div_end_margin_m / CFG.traj_scale
    full_margin = CFG.div_full_margin_m / CFG.traj_scale
    for i in range(K):
        for j in range(i + 1, K):
            end_dist = torch.norm(traj_pred[:, i, -1, :] - traj_pred[:, j, -1, :], dim=-1)
            full_dist = torch.norm(traj_pred[:, i] - traj_pred[:, j], dim=-1).mean(dim=-1)
            pair_losses.append(0.7 * torch.relu(end_margin - end_dist) + 0.3 * torch.relu(full_margin - full_dist))
    return torch.stack(pair_losses, dim=0).mean()


def compute_comfort_loss(traj_best: torch.Tensor) -> torch.Tensor:
    # v8.1: SmoothL1 comfort loss. The old squared version can explode in early epochs.
    xy = traj_best * CFG.traj_scale
    if xy.size(1) < 3:
        return xy.new_tensor(0.0)
    vel = (xy[:, 1:] - xy[:, :-1]) / CFG.dt
    acc = (vel[:, 1:] - vel[:, :-1]) / CFG.dt
    loss_acc = torch.nn.functional.smooth_l1_loss(acc, torch.zeros_like(acc))
    if acc.size(1) >= 2:
        jerk = (acc[:, 1:] - acc[:, :-1]) / CFG.dt
        loss_jerk = torch.nn.functional.smooth_l1_loss(jerk, torch.zeros_like(jerk))
    else:
        loss_jerk = xy.new_tensor(0.0)
    return loss_acc + 0.2 * loss_jerk


def compute_collision_loss(traj_best: torch.Tensor, agents: torch.Tensor, agent_valid: torch.Tensor) -> torch.Tensor:
    pred = traj_best * CFG.traj_scale
    agent_pos = agents[..., 0:2] * CFG.agent_pos_scale
    agent_vel = agents[..., 2:4] * CFG.agent_vel_scale
    valid = agent_valid.float()
    if valid.sum().item() < 1:
        return pred.new_tensor(0.0)
    times = torch.arange(1, CFG.future_steps + 1, device=pred.device, dtype=pred.dtype).view(1, 1, CFG.future_steps, 1) * CFG.dt
    agent_future = agent_pos.unsqueeze(2) + agent_vel.unsqueeze(2) * times
    pred_expand = pred.unsqueeze(1)
    dist = torch.norm(pred_expand - agent_future, dim=-1)
    penalty = torch.relu(CFG.collision_radius_m - dist) ** 2
    penalty = penalty * valid.unsqueeze(-1)
    return penalty.sum() / (valid.sum() * CFG.future_steps).clamp_min(1.0)


def compute_intent_loss(intent_pred: torch.Tensor, intent_target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(intent_pred, intent_target)


def compute_bev_distill_loss(learned_bev: Optional[torch.Tensor], teacher_bev: torch.Tensor, bev_valid: torch.Tensor) -> torch.Tensor:
    """Distill learned camera-BEV toward annotation-BEV teacher."""
    if learned_bev is None or (not CFG.use_learned_camera_bev) or CFG.lambda_bev_distill <= 0.0:
        return teacher_bev.new_tensor(0.0)
    if learned_bev.shape != teacher_bev.shape:
        teacher_bev = torch.nn.functional.interpolate(teacher_bev, size=learned_bev.shape[-2:], mode="bilinear", align_corners=False)
    mask = (bev_valid > 0.5).view(-1, 1, 1, 1).to(dtype=learned_bev.dtype)
    denom = mask.sum() * learned_bev.size(1) * learned_bev.size(2) * learned_bev.size(3)
    if denom.item() < 1.0:
        return learned_bev.new_tensor(0.0)
    diff = torch.nn.functional.smooth_l1_loss(learned_bev * mask, teacher_bev.detach() * mask, reduction="sum")
    return diff / denom.clamp_min(1.0)


def base_compute_total_loss(model, batch, cls_criterion, device: str):
    hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, y_cls, y_traj, map_y_ref, map_ref_valid, intent_target, camera_feat, camera_valid, proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid = unpack_batch(batch, device)
    cls_logits, traj_pred, scores, intent_pred, learned_bev_grid = model(hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, camera_feat, camera_valid, proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid, return_bev_aux=True)

    loss_cls = cls_criterion(cls_logits, y_cls)
    loss_stop = stop_logit_penalty(cls_logits, y_cls)
    loss_traj_mode, loss_traj_gt, loss_score, best_idx_gt, _, pred_idx, _ = base_multimodal_losses(traj_pred, y_traj, y_cls, scores)
    traj_best_gt = gather_modes(traj_pred, best_idx_gt)
    loss_map = compute_real_map_loss(traj_best_gt, map_y_ref, map_ref_valid)
    loss_div = compute_diversity_loss(traj_pred)
    loss_comfort = compute_comfort_loss(traj_best_gt)
    loss_collision = compute_collision_loss(traj_best_gt, agents, agent_valid)
    loss_intent = compute_intent_loss(intent_pred, intent_target)
    loss_bev_distill = compute_bev_distill_loss(learned_bev_grid, bev_grid, bev_valid)

    loss = (
        CFG.lambda_cls * loss_cls
        + CFG.lambda_stop_logit_penalty * loss_stop
        + CFG.lambda_traj_mode * loss_traj_mode
        + CFG.lambda_traj_gt * loss_traj_gt
        + CFG.lambda_score * loss_score
        + CFG.lambda_map * loss_map
        + CFG.lambda_diversity * loss_div
        + CFG.lambda_comfort * loss_comfort
        + CFG.lambda_collision * loss_collision
        + CFG.lambda_intent * loss_intent
        + CFG.lambda_bev_distill * loss_bev_distill
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
        "loss_bev_distill": loss_bev_distill,
        "cls_logits": cls_logits,
        "traj_pred": traj_pred,
        "scores": scores,
        "intent_pred": intent_pred,
        "intent_target": intent_target,
        "y_cls": y_cls,
        "y_traj": y_traj,
        "best_idx_gt": best_idx_gt,
        "pred_idx": pred_idx,
        "learned_bev_grid": learned_bev_grid,
    }


def base_train_one_epoch(model, loader, cls_criterion, optimizer, device: str):
    model.train()
    meters = {k: 0.0 for k in ["loss", "loss_cls", "loss_stop", "loss_traj_mode", "loss_traj_gt", "loss_score", "loss_map", "loss_div", "loss_comfort", "loss_collision", "loss_intent", "loss_bev_distill"]}
    total = 0
    for batch in loader:
        optimizer.zero_grad()
        out = base_compute_total_loss(model, batch, cls_criterion, device)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        bs = out["y_cls"].size(0)
        total += bs
        for k in meters:
            meters[k] += float(out[k].item()) * bs
    return {k: v / max(total, 1) for k, v in meters.items()}


@torch.no_grad()
def base_evaluate(model, loader, cls_criterion, device: str):
    model.eval()
    meters = {k: 0.0 for k in ["loss", "loss_cls", "loss_stop", "loss_traj_mode", "loss_traj_gt", "loss_score", "loss_map", "loss_div", "loss_comfort", "loss_collision", "loss_intent", "loss_bev_distill"]}
    total = 0
    preds_all, y_all = [], []
    ade_sel_all, fde_sel_all, ade_orc_all, fde_orc_all = [], [], [], []
    intent_mae_all = []

    for batch in loader:
        out = base_compute_total_loss(model, batch, cls_criterion, device)
        bs = out["y_cls"].size(0)
        total += bs
        for k in meters:
            meters[k] += float(out[k].item()) * bs
        preds = out["cls_logits"].argmax(dim=1)
        ade_sel, fde_sel = calc_traj_metrics(out["traj_pred"], out["y_traj"], out["pred_idx"])
        ade_orc, fde_orc = calc_traj_metrics(out["traj_pred"], out["y_traj"], out["best_idx_gt"])
        preds_all.append(preds.cpu().numpy())
        y_all.append(out["y_cls"].cpu().numpy())
        ade_sel_all.append(ade_sel.cpu().numpy())
        fde_sel_all.append(fde_sel.cpu().numpy())
        ade_orc_all.append(ade_orc.cpu().numpy())
        fde_orc_all.append(fde_orc.cpu().numpy())
        intent_mae = torch.abs(decode_intent(out["intent_pred"]) - decode_intent(out["intent_target"])).mean(dim=0)
        intent_mae_all.append(intent_mae.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)
    ade_sel_all = np.concatenate(ade_sel_all)
    fde_sel_all = np.concatenate(fde_sel_all)
    ade_orc_all = np.concatenate(ade_orc_all)
    fde_orc_all = np.concatenate(fde_orc_all)
    intent_mae_all = np.stack(intent_mae_all, axis=0).mean(axis=0) if intent_mae_all else np.zeros((4,), dtype=np.float32)

    labels = list(range(len(CFG.target_classes)))
    acc = accuracy_score(y_all, preds_all)
    precisions = precision_score(y_all, preds_all, labels=labels, average=None, zero_division=0)
    recalls = recall_score(y_all, preds_all, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_all, preds_all, labels=labels, average=None, zero_division=0)
    macro_f1 = f1_score(y_all, preds_all, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_all, preds_all, labels=labels)

    ret = {k: v / max(total, 1) for k, v in meters.items()}
    ret.update({
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "ADE_selected": float(ade_sel_all.mean()),
        "FDE_selected": float(fde_sel_all.mean()),
        "ADE_oracle": float(ade_orc_all.mean()),
        "FDE_oracle": float(fde_orc_all.mean()),
        "intent_mae_terminal_speed_mps": float(intent_mae_all[0]),
        "intent_mae_total_disp_m": float(intent_mae_all[1]),
        "intent_mae_lateral_disp_m": float(intent_mae_all[2]),
        "intent_mae_yaw_delta_rad": float(intent_mae_all[3]),
        "precisions": precisions.tolist(),
        "recalls": recalls.tolist(),
        "f1s": f1s.tolist(),
        "cm": cm.tolist(),
        "y_true": y_all.tolist(),
        "y_pred": preds_all.tolist(),
    })
    return ret


def train_v8_3_base_if_needed(
    train_loader,
    val_loader,
    cls_criterion,
    scaler: StandardScalerNP,
    train_shapes: Dict[str, List[int]],
    val_shapes: Dict[str, List[int]],
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Return a v10.0 base checkpoint dict. Train v10.0 base first if requested or missing."""
    base_dir = os.path.join(CFG.save_dir, CFG.base_save_subdir)
    ensure_dir(base_dir)
    generated_ckpt_path = os.path.join(base_dir, CFG.base_best_model_name)

    should_train = CFG.run_base_train or (not os.path.exists(CFG.base_ckpt_path) and CFG.train_base_if_missing)
    load_path = generated_ckpt_path if CFG.run_base_train else CFG.base_ckpt_path

    if not should_train and os.path.exists(load_path):
        print(f"Loading existing v10.0 base checkpoint: {load_path}")
        ckpt = torch.load(load_path, map_location=CFG.device, weights_only=False)
        return ckpt, load_path, {"trained": False, "path": load_path}

    if not should_train:
        raise FileNotFoundError(
            f"base v10.0 base checkpoint not found: {CFG.base_ckpt_path}\n"
            "Either set BASE_CKPT_PATH correctly, or set RUN_BASE_TRAIN=1."
        )

    print("\n" + "=" * 88)
    print("Stage 1: training v14.1 real-camera-cache learned camera-BEV joint base model inside this script")
    print(f"Base save dir     : {base_dir}")
    print(f"Base epochs       : {CFG.base_epochs}")
    print(f"Base lr           : {CFG.base_lr}")
    print(f"Base patience     : {CFG.base_early_stop_patience}")
    print("=" * 88)

    base_model = build_base_model().to(CFG.device)
    if CFG.init_ckpt_path and os.path.exists(CFG.init_ckpt_path):
        try:
            init_ckpt = torch.load(CFG.init_ckpt_path, map_location=CFG.device, weights_only=False)
            init_state = init_ckpt.get("model_state_dict", init_ckpt)
            cur_state = base_model.state_dict()
            compatible = {k: v for k, v in init_state.items() if k in cur_state and tuple(v.shape) == tuple(cur_state[k].shape)}
            missing = [k for k in cur_state.keys() if k not in compatible]
            base_model.load_state_dict(compatible, strict=False)
            print(f"Warm-started v10.0 base from INIT_CKPT_PATH: {CFG.init_ckpt_path}")
            print(f"Compatible tensors loaded: {len(compatible)} | newly initialized tensors: {len(missing)}")
        except Exception as e:
            print(f"[WARN] Failed to warm-start from INIT_CKPT_PATH={CFG.init_ckpt_path}: {e}")
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=CFG.base_lr, weight_decay=CFG.base_weight_decay)
    best_epoch = -1
    best_macro_f1 = -1.0
    best_ADE = math.inf
    best_FDE = math.inf
    best_gap = math.inf
    best_planning_score = math.inf
    best_metrics: Dict[str, Any] = {}
    patience_counter = 0
    history = []

    for epoch in range(1, CFG.base_epochs + 1):
        train_stats = base_train_one_epoch(base_model, train_loader, cls_criterion, optimizer, CFG.device)
        val_metrics = base_evaluate(base_model, val_loader, cls_criterion, CFG.device)
        macro_f1 = val_metrics["macro_f1"]
        ade = val_metrics["ADE_selected"]
        fde = val_metrics["FDE_selected"]
        ade_orc = val_metrics["ADE_oracle"]
        fde_orc = val_metrics["FDE_oracle"]
        gap = ade - ade_orc
        planning_score = ade + CFG.planning_score_fde_weight * fde + CFG.planning_score_gap_weight * gap
        history.append({
            "epoch": epoch,
            **{f"train_{k}": float(v) for k, v in train_stats.items()},
            "val_acc": float(val_metrics["acc"]),
            "val_macro_f1": float(macro_f1),
            "val_ADE_selected": float(ade),
            "val_FDE_selected": float(fde),
            "val_ADE_oracle": float(ade_orc),
            "val_FDE_oracle": float(fde_orc),
            "val_ADE_gap": float(gap),
            "val_planning_score": float(planning_score),
        })
        print(
            f"[v14.1 base] Epoch {epoch:03d} | TrainLoss {train_stats['loss']:.4f} | "
            f"Cls {train_stats['loss_cls']:.4f} | TrajGT {train_stats['loss_traj_gt']:.4f} | "
            f"Score {train_stats['loss_score']:.4f} | BEVdist {train_stats['loss_bev_distill']:.4f} | ValLoss {val_metrics['loss']:.4f} | "
            f"Acc {val_metrics['acc']:.4f} | MacroF1 {macro_f1:.4f} | "
            f"ADE_sel {ade:.4f} | FDE_sel {fde:.4f} | Gap {gap:.4f} | "
            f"PlanScore {planning_score:.4f} | ADE_orc {ade_orc:.4f} | FDE_orc {fde_orc:.4f}"
        )

        improved = False
        # v14.0 planning-aware checkpoint selection:
        # Once Macro-F1 is acceptable, select by a trajectory score rather than ADE alone.
        # planning_score = ADE_selected + w_fde * FDE_selected + w_gap * ADE_gap.
        candidate_ok = macro_f1 >= CFG.base_select_min_macro_f1
        best_ok = best_macro_f1 >= CFG.base_select_min_macro_f1
        if candidate_ok:
            if (not best_ok) or (planning_score < best_planning_score - 1e-8):
                improved = True
            elif abs(planning_score - best_planning_score) < 1e-8 and macro_f1 > best_macro_f1:
                improved = True
        else:
            if (not best_ok) and macro_f1 > best_macro_f1:
                improved = True

        if improved:
            best_epoch = epoch
            best_macro_f1 = macro_f1
            best_ADE = ade
            best_FDE = fde
            best_gap = gap
            best_planning_score = planning_score
            best_metrics = val_metrics
            patience_counter = 0
            ckpt_obj = {
                "model_state_dict": base_model.state_dict(),
                "model_type": "transformer_v14_1_real_camera_cache_enabled_base",
                "config": asdict(CFG),
                "class_to_id": CLASS_TO_ID,
                "id_to_class": ID_TO_CLASS,
                "scaler": scaler.state_dict(),
                "best_val_acc": float(val_metrics["acc"]),
                "best_val_macro_f1": float(best_macro_f1),
                "best_val_ADE_selected": float(best_ADE),
                "best_val_FDE_selected": float(best_FDE),
                "best_val_ADE_gap": float(best_gap),
                "best_val_planning_score": float(best_planning_score),
                "epoch": best_epoch,
                "train_shapes": train_shapes,
                "val_shapes": val_shapes,
                "history": history,
            }
            torch.save(ckpt_obj, generated_ckpt_path)
            print(f"🔥 Saved planning-aware v14.1 real-camera-cache BEV base at epoch {epoch} (Macro-F1={macro_f1:.4f}, ADE={ade:.4f}, FDE={fde:.4f}, Gap={gap:.4f}, PlanScore={planning_score:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= CFG.base_early_stop_patience:
            print(f"\n[v14.1 base] Early stopping triggered at epoch {epoch}.")
            break

    if not os.path.exists(generated_ckpt_path):
        raise RuntimeError("v10.0 base training finished without saving a checkpoint.")

    ckpt = torch.load(generated_ckpt_path, map_location=CFG.device, weights_only=False)
    stage_metrics = {
        "trained": True,
        "path": generated_ckpt_path,
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_macro_f1),
        "best_val_ADE_selected": float(best_ADE),
        "best_val_FDE_selected": float(best_FDE),
        "best_val_ADE_gap": float(best_gap),
        "best_val_planning_score": float(best_planning_score),
        "best_val_acc": float(best_metrics.get("acc", -1.0)) if best_metrics else -1.0,
    }
    return ckpt, generated_ckpt_path, stage_metrics

# ============================================================
# v9.5 Score-only calibration losses and metrics
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()


class ScoreOnlyCalibrator(nn.Module):
    """
    v9.5: freeze the v10.0 base planner, then train only a tiny score calibrator.

    The frozen base model still generates:
        cls_logits, traj_pred, base_scores, intent_pred
    The calibrator predicts a small delta over base_scores from:
        base_score + explicit quality features + cls_prob + intent_pred

    This avoids the v8.4/v8.6 failure mode where score optimization drags the
    trajectory generator into a ditch.
    """
    def __init__(self, base_model: MapAgentEgoPlanner, quality_dim: int = 12, hidden_dim: int = 64, dropout: float = 0.10):
        super().__init__()
        self.base = base_model
        self.quality_dim = quality_dim
        in_dim = 1 + quality_dim + len(CFG.target_classes) + 4
        self.score_refiner = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Start exactly from the v10.0 base score behavior. Training begins as a safe no-op.
        nn.init.zeros_(self.score_refiner[-1].weight)
        nn.init.zeros_(self.score_refiner[-1].bias)

        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

    @torch.no_grad()
    def compute_score_quality_features(self, traj: torch.Tensor, agents: torch.Tensor, agent_valid: torch.Tensor, maps: torch.Tensor, map_valid: torch.Tensor) -> torch.Tensor:
        """
        Explicit per-mode quality features, normalized and clamped.
        Returns [B,K,12]:
            final_x, abs_final_y, path_length, mean_speed, terminal_speed,
            mean_acc, mean_jerk, straightness, collision_penalty,
            min_agent_dist, map_mean_dist, map_final_dist
        """
        eps = 1e-6
        B, K, T, _ = traj.shape
        device = traj.device
        traj_m = traj.detach() * CFG.traj_scale

        final_xy = traj_m[:, :, -1, :]
        final_x_norm = final_xy[..., 0] / CFG.traj_scale
        abs_final_y_norm = final_xy[..., 1].abs() / CFG.traj_scale

        if T >= 2:
            delta = traj_m[:, :, 1:, :] - traj_m[:, :, :-1, :]
            step_dist = torch.norm(delta, dim=-1)
            path_len = step_dist.sum(dim=-1)
            mean_speed = step_dist.mean(dim=-1) / CFG.dt
            terminal_speed = step_dist[:, :, -1] / CFG.dt
        else:
            path_len = torch.zeros(B, K, device=device)
            mean_speed = torch.zeros(B, K, device=device)
            terminal_speed = torch.zeros(B, K, device=device)

        path_len_norm = path_len / CFG.traj_scale
        mean_speed_norm = mean_speed / CFG.intent_speed_scale
        terminal_speed_norm = terminal_speed / CFG.intent_speed_scale

        if T >= 3:
            vel = (traj_m[:, :, 1:, :] - traj_m[:, :, :-1, :]) / CFG.dt
            acc = (vel[:, :, 1:, :] - vel[:, :, :-1, :]) / CFG.dt
            mean_acc = torch.norm(acc, dim=-1).mean(dim=-1)
        else:
            acc = None
            mean_acc = torch.zeros(B, K, device=device)
        mean_acc_norm = mean_acc / CFG.score_acc_scale

        if acc is not None and acc.size(2) >= 2:
            jerk = (acc[:, :, 1:, :] - acc[:, :, :-1, :]) / CFG.dt
            mean_jerk = torch.norm(jerk, dim=-1).mean(dim=-1)
        else:
            mean_jerk = torch.zeros(B, K, device=device)
        mean_jerk_norm = mean_jerk / CFG.score_jerk_scale

        final_disp = torch.norm(final_xy, dim=-1)
        straightness = (final_disp / (path_len + eps)).clamp(0.0, 1.0)

        valid_agents = agent_valid.float()
        if valid_agents.sum().item() > 0:
            agent_pos = agents[..., 0:2] * CFG.agent_pos_scale
            agent_vel = agents[..., 2:4] * CFG.agent_vel_scale
            times = torch.arange(1, T + 1, device=device, dtype=traj_m.dtype).view(1, 1, 1, T, 1) * CFG.dt
            agent_future = agent_pos[:, None, :, None, :] + agent_vel[:, None, :, None, :] * times
            pred_expand = traj_m[:, :, None, :, :]
            dist = torch.norm(pred_expand - agent_future, dim=-1)
            mask = valid_agents[:, None, :, None] > 0.5
            dist_masked = dist.masked_fill(~mask, 1e6)
            min_agent_dist = dist_masked.amin(dim=(2, 3)).clamp(max=CFG.score_agent_dist_scale)
            collision_penalty = (torch.relu(CFG.collision_radius_m - dist) ** 2 * mask.float()).sum(dim=(2, 3))
            denom = mask.float().sum(dim=(2, 3)).clamp_min(1.0)
            collision_penalty = collision_penalty / denom
        else:
            min_agent_dist = torch.full((B, K), CFG.score_agent_dist_scale, device=device, dtype=traj_m.dtype)
            collision_penalty = torch.zeros(B, K, device=device, dtype=traj_m.dtype)

        min_agent_dist_norm = min_agent_dist / CFG.score_agent_dist_scale
        collision_penalty_norm = collision_penalty / max(CFG.collision_radius_m ** 2, eps)

        map_xy = maps[..., 0:2] * CFG.map_pos_scale
        map_valid_flat = (map_valid.reshape(B, -1) > 0.5)
        map_points = map_xy.reshape(B, -1, 2)
        if map_points.size(1) > 0:
            pred_flat = traj_m.reshape(B, K * T, 2)
            dmap = torch.cdist(pred_flat, map_points)
            dmap = dmap.masked_fill(~map_valid_flat[:, None, :], 1e6)
            min_map = dmap.amin(dim=-1).reshape(B, K, T)
            has_map = map_valid_flat.any(dim=1).view(B, 1, 1)
            min_map = torch.where(has_map, min_map, torch.zeros_like(min_map))
            map_mean_dist = min_map.mean(dim=-1)
            map_final_dist = min_map[:, :, -1]
        else:
            map_mean_dist = torch.zeros(B, K, device=device, dtype=traj_m.dtype)
            map_final_dist = torch.zeros(B, K, device=device, dtype=traj_m.dtype)

        map_mean_dist_norm = map_mean_dist / CFG.score_map_dist_scale
        map_final_dist_norm = map_final_dist / CFG.score_map_dist_scale

        quality = torch.stack([
            final_x_norm, abs_final_y_norm, path_len_norm, mean_speed_norm, terminal_speed_norm,
            mean_acc_norm, mean_jerk_norm, straightness, collision_penalty_norm,
            min_agent_dist_norm, map_mean_dist_norm, map_final_dist_norm,
        ], dim=-1)
        quality = torch.nan_to_num(quality, nan=0.0, posinf=CFG.score_quality_clip, neginf=-CFG.score_quality_clip)
        return torch.clamp(quality, -CFG.score_quality_clip, CFG.score_quality_clip)

    def forward(self, hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, camera_feat, camera_valid,
                proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid,
                proj_corridor_feat, proj_corridor_valid):
        self.base.eval()
        with torch.no_grad():
            cls_logits, traj, base_scores, intent_pred = self.base(
                hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, camera_feat, camera_valid,
                proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid,
                proj_corridor_feat, proj_corridor_valid,
            )
            quality_feat = self.compute_score_quality_features(traj, agents, agent_valid, maps, map_valid)
            cls_prob = torch.softmax(cls_logits, dim=-1)
            cls_expand = cls_prob.unsqueeze(1).expand(-1, traj.size(1), -1)
            intent_expand = intent_pred.unsqueeze(1).expand(-1, traj.size(1), -1)
            base_score_feat = base_scores.unsqueeze(-1)
            refiner_input = torch.cat([base_score_feat, quality_feat, cls_expand, intent_expand], dim=-1)

        delta = self.score_refiner(refiner_input).squeeze(-1)
        scores = base_scores + CFG.score_delta_scale * delta
        return cls_logits, traj, scores, intent_pred, base_scores, quality_feat


def gather_modes(traj_pred: torch.Tensor, mode_idx: torch.Tensor) -> torch.Tensor:
    B = traj_pred.size(0)
    return traj_pred[torch.arange(B, device=traj_pred.device), mode_idx]


def compute_soft_score_target(traj_pred: torch.Tensor, y_traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_expand = y_traj.unsqueeze(1).expand_as(traj_pred)
    diff = (traj_pred - y_expand) * CFG.traj_scale
    ade_like = torch.norm(diff, dim=-1).mean(dim=-1)
    with torch.no_grad():
        traj_delta = (traj_pred[:, :, 1:, :] - traj_pred[:, :, :-1, :]) * CFG.traj_scale
        traj_speed = torch.norm(traj_delta, dim=-1).mean(dim=-1)
        speed_penalty = torch.exp(-CFG.score_speed_penalty_scale * traj_speed)
        combined = ade_like + CFG.score_speed_penalty_weight * speed_penalty
        target_prob = torch.softmax(-combined / CFG.score_target_temp, dim=1)
        if CFG.score_label_smoothing > 0.0:
            uniform = torch.full_like(target_prob, 1.0 / traj_pred.size(1))
            target_prob = (1.0 - CFG.score_label_smoothing) * target_prob + CFG.score_label_smoothing * uniform
        best_idx = combined.argmin(dim=1)
    return target_prob, ade_like, best_idx


def score_calibration_losses(traj_pred: torch.Tensor, y_traj: torch.Tensor, scores: torch.Tensor, base_scores: torch.Tensor):
    target_prob, ade_like, best_idx = compute_soft_score_target(traj_pred, y_traj)
    log_scores = torch.log_softmax(scores, dim=1)
    loss_soft = -(target_prob * log_scores).sum(dim=1).mean()

    best_score = scores.gather(1, best_idx.view(-1, 1)).squeeze(1)
    rank_penalty = torch.relu(CFG.score_rank_margin + scores - best_score.unsqueeze(1))

    # Do NOT use scatter_ here. rank_penalty is the output of ReLU and is part of
    # the autograd graph. In-place modification will trigger:
    # RuntimeError: one of the variables needed for gradient computation has been modified.
    best_mask = torch.nn.functional.one_hot(best_idx, num_classes=scores.size(1)).to(dtype=scores.dtype)
    rank_penalty = rank_penalty * (1.0 - best_mask)
    loss_rank = rank_penalty.mean()

    # Keep calibration conservative; do not let delta dominate the already-trained v10.0 base scores.
    loss_reg = torch.nn.functional.mse_loss(scores, base_scores.detach())
    loss = CFG.lambda_score_soft * loss_soft + CFG.lambda_score_rank * loss_rank + CFG.lambda_score_reg * loss_reg
    pred_idx = scores.argmax(dim=1)
    base_pred_idx = base_scores.argmax(dim=1)
    return loss, loss_soft, loss_rank, loss_reg, best_idx, pred_idx, base_pred_idx, ade_like


def calc_traj_metrics(pred: torch.Tensor, gt: torch.Tensor, mode_idx: torch.Tensor):
    pred_sel = gather_modes(pred, mode_idx)
    pred_meter = pred_sel * CFG.traj_scale
    gt_meter = gt * CFG.traj_scale
    diff = torch.norm(pred_meter - gt_meter, dim=-1)
    return diff.mean(dim=1), diff[:, -1]


def decode_intent(intent_norm: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor(
        [CFG.intent_speed_scale, CFG.intent_disp_scale, CFG.intent_disp_scale, CFG.intent_yaw_scale],
        dtype=intent_norm.dtype,
        device=intent_norm.device,
    )
    return intent_norm * scale


def unpack_batch(batch, device: str):
    (hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, y_cls, y_traj, map_y_ref, map_ref_valid,
     intent_target, camera_feat, camera_valid, proj_agent_feat, proj_agent_valid,
     proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid) = batch
    return (
        hist.to(device),
        agents.to(device),
        agent_valid.to(device),
        agent3d.to(device),
        agent3d_valid.to(device),
        bev_grid.to(device),
        bev_valid.to(device),
        maps.to(device),
        map_valid.to(device),
        y_cls.to(device),
        y_traj.to(device),
        map_y_ref.to(device),
        map_ref_valid.to(device),
        intent_target.to(device),
        camera_feat.to(device),
        camera_valid.to(device),
        proj_agent_feat.to(device),
        proj_agent_valid.to(device),
        proj_map_feat.to(device),
        proj_map_valid.to(device),
        proj_corridor_feat.to(device),
        proj_corridor_valid.to(device),
    )


def compute_total_loss(model: ScoreOnlyCalibrator, batch, cls_criterion, device: str):
    hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, y_cls, y_traj, map_y_ref, map_ref_valid, intent_target, camera_feat, camera_valid, proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid = unpack_batch(batch, device)
    cls_logits, traj_pred, scores, intent_pred, base_scores, quality_feat = model(hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, camera_feat, camera_valid, proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid)

    loss_score, loss_soft, loss_rank, loss_reg, best_idx_gt, pred_idx, base_pred_idx, _ = score_calibration_losses(traj_pred, y_traj, scores, base_scores)
    loss_cls = cls_criterion(cls_logits, y_cls)

    outputs = {
        "loss": loss_score,
        "loss_score": loss_score,
        "loss_score_soft": loss_soft,
        "loss_score_rank": loss_rank,
        "loss_score_reg": loss_reg,
        "loss_cls": loss_cls,
        "cls_logits": cls_logits,
        "traj_pred": traj_pred,
        "scores": scores,
        "base_scores": base_scores,
        "intent_pred": intent_pred,
        "intent_target": intent_target,
        "quality_feat": quality_feat,
        "y_cls": y_cls,
        "y_traj": y_traj,
        "best_idx_gt": best_idx_gt,
        "pred_idx": pred_idx,
        "base_pred_idx": base_pred_idx,
    }
    return outputs


def train_one_epoch(model, loader, cls_criterion, optimizer, device: str):
    model.train()
    model.base.eval()
    meters = {k: 0.0 for k in ["loss", "loss_score", "loss_score_soft", "loss_score_rank", "loss_score_reg", "loss_cls"]}
    total = 0
    for batch in loader:
        optimizer.zero_grad()
        out = compute_total_loss(model, batch, cls_criterion, device)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.score_refiner.parameters(), max_norm=2.0)
        optimizer.step()
        bs = out["y_cls"].size(0)
        total += bs
        for k in meters:
            meters[k] += float(out[k].item()) * bs
    return {k: v / max(total, 1) for k, v in meters.items()}


@torch.no_grad()
def evaluate(model, loader, cls_criterion, device: str):
    model.eval()
    meters = {k: 0.0 for k in ["loss", "loss_score", "loss_score_soft", "loss_score_rank", "loss_score_reg", "loss_cls"]}
    total = 0
    preds_all, y_all = [], []
    ade_sel_all, fde_sel_all, ade_orc_all, fde_orc_all = [], [], [], []
    ade_base_all, fde_base_all = [], []
    score_hit_all, base_score_hit_all = [], []
    intent_mae_all = []

    for batch in loader:
        out = compute_total_loss(model, batch, cls_criterion, device)
        bs = out["y_cls"].size(0)
        total += bs
        for k in meters:
            meters[k] += float(out[k].item()) * bs

        preds = out["cls_logits"].argmax(dim=1)
        ade_sel, fde_sel = calc_traj_metrics(out["traj_pred"], out["y_traj"], out["pred_idx"])
        ade_orc, fde_orc = calc_traj_metrics(out["traj_pred"], out["y_traj"], out["best_idx_gt"])
        ade_base, fde_base = calc_traj_metrics(out["traj_pred"], out["y_traj"], out["base_pred_idx"])
        score_hit = (out["pred_idx"] == out["best_idx_gt"]).float()
        base_score_hit = (out["base_pred_idx"] == out["best_idx_gt"]).float()

        preds_all.append(preds.cpu().numpy())
        y_all.append(out["y_cls"].cpu().numpy())
        ade_sel_all.append(ade_sel.cpu().numpy())
        fde_sel_all.append(fde_sel.cpu().numpy())
        ade_orc_all.append(ade_orc.cpu().numpy())
        fde_orc_all.append(fde_orc.cpu().numpy())
        ade_base_all.append(ade_base.cpu().numpy())
        fde_base_all.append(fde_base.cpu().numpy())
        score_hit_all.append(score_hit.cpu().numpy())
        base_score_hit_all.append(base_score_hit.cpu().numpy())
        intent_mae = torch.abs(decode_intent(out["intent_pred"]) - decode_intent(out["intent_target"])).mean(dim=0)
        intent_mae_all.append(intent_mae.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)
    ade_sel_all = np.concatenate(ade_sel_all)
    fde_sel_all = np.concatenate(fde_sel_all)
    ade_orc_all = np.concatenate(ade_orc_all)
    fde_orc_all = np.concatenate(fde_orc_all)
    ade_base_all = np.concatenate(ade_base_all)
    fde_base_all = np.concatenate(fde_base_all)
    score_hit_all = np.concatenate(score_hit_all)
    base_score_hit_all = np.concatenate(base_score_hit_all)
    intent_mae_all = np.stack(intent_mae_all, axis=0).mean(axis=0) if intent_mae_all else np.zeros((4,), dtype=np.float32)

    labels = list(range(len(CFG.target_classes)))
    acc = accuracy_score(y_all, preds_all)
    precisions = precision_score(y_all, preds_all, labels=labels, average=None, zero_division=0)
    recalls = recall_score(y_all, preds_all, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_all, preds_all, labels=labels, average=None, zero_division=0)
    macro_f1 = f1_score(y_all, preds_all, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_all, preds_all, labels=labels)

    ret = {k: v / max(total, 1) for k, v in meters.items()}
    ret.update({
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "ADE_selected": float(ade_sel_all.mean()),
        "FDE_selected": float(fde_sel_all.mean()),
        "ADE_oracle": float(ade_orc_all.mean()),
        "FDE_oracle": float(fde_orc_all.mean()),
        "ADE_base_selected": float(ade_base_all.mean()),
        "FDE_base_selected": float(fde_base_all.mean()),
        "ADE_gap": float(ade_sel_all.mean() - ade_orc_all.mean()),
        "FDE_gap": float(fde_sel_all.mean() - fde_orc_all.mean()),
        "base_ADE_gap": float(ade_base_all.mean() - ade_orc_all.mean()),
        "score_hit_rate": float(score_hit_all.mean()),
        "base_score_hit_rate": float(base_score_hit_all.mean()),
        "intent_mae_terminal_speed_mps": float(intent_mae_all[0]),
        "intent_mae_total_disp_m": float(intent_mae_all[1]),
        "intent_mae_lateral_disp_m": float(intent_mae_all[2]),
        "intent_mae_yaw_delta_rad": float(intent_mae_all[3]),
        "precisions": precisions.tolist(),
        "recalls": recalls.tolist(),
        "f1s": f1s.tolist(),
        "cm": cm.tolist(),
        "y_true": y_all.tolist(),
        "y_pred": preds_all.tolist(),
    })
    return ret


@torch.no_grad()
def save_traj_visualizations(model, loader, device, save_dir, num_samples=12):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    saved = 0
    for batch in loader:
        hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, y_cls, y_traj, _, _, _, camera_feat, camera_valid, proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid = unpack_batch(batch, device)
        cls_logits, traj_pred, scores, _, _, _ = model(hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, camera_feat, camera_valid, proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid)
        pred_idx = scores.argmax(dim=1)
        traj_best = gather_modes(traj_pred, pred_idx)
        pred_plot = (traj_best * CFG.traj_scale).cpu().numpy()
        gt_plot = (y_traj * CFG.traj_scale).cpu().numpy()
        for i in range(pred_plot.shape[0]):
            plt.figure(figsize=(5, 5))
            plt.plot(gt_plot[i, :, 0], gt_plot[i, :, 1], marker="o", label="GT")
            plt.plot(pred_plot[i, :, 0], pred_plot[i, :, 1], marker="x", label="v9.5 Selected Pred")
            plt.scatter([0.0], [0.0], marker="s", s=80, label="Ego")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.xlabel("x_local (m)")
            plt.ylabel("y_local (m)")
            plt.title(f"sample_{saved:03d}")
            plt.savefig(os.path.join(save_dir, f"traj_{saved:03d}.png"), bbox_inches="tight")
            plt.close()
            saved += 1
            if saved >= num_samples:
                return


def build_base_model() -> MapAgentEgoPlanner:
    return MapAgentEgoPlanner(
        history_dim=CFG.history_dim,
        agent_dim=CFG.agent_dim,
        agent3d_dim=CFG.agent3d_dim,
        bev_channels=CFG.bev_channels,
        map_dim=CFG.map_dim,
        camera_dim=CFG.camera_feat_dim,
        num_cameras=CFG.num_cameras,
        num_bev_queries=CFG.num_bev_queries,
        hidden_dim=CFG.hidden_dim,
        num_classes=len(CFG.target_classes),
        future_steps=CFG.future_steps,
        future_dim=CFG.future_dim,
        num_modes=CFG.num_modes,
        nhead=CFG.nhead,
        ff_mult=CFG.ff_mult,
        num_encoder_layers=CFG.num_encoder_layers,
        num_decoder_layers=CFG.num_decoder_layers,
        dropout=CFG.dropout,
        branch_dim=CFG.branch_dim,
    )


def apply_checkpoint_scaler_or_fit(ckpt: Dict[str, Any], X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScalerNP, str]:
    scaler = StandardScalerNP(skip_indices=[14, 15])
    ckpt_scaler = ckpt.get("scaler", None)
    if isinstance(ckpt_scaler, dict) and ckpt_scaler.get("mean", None) is not None and ckpt_scaler.get("std", None) is not None:
        scaler.mean = np.asarray(ckpt_scaler["mean"], dtype=np.float32)
        scaler.std = np.asarray(ckpt_scaler["std"], dtype=np.float32)
        scaler.skip_indices = set(ckpt_scaler.get("skip_indices", [14, 15]))
        source = "checkpoint"
        train_D = X_train.shape[-1]
        X_train = scaler.transform(X_train.reshape(-1, train_D)).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, train_D)).reshape(X_val.shape)
    else:
        source = "fit_from_train_fallback"
        train_D = X_train.shape[-1]
        X_train = scaler.fit_transform(X_train.reshape(-1, train_D)).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, train_D)).reshape(X_val.shape)
    return X_train, X_val, scaler, source


def main():
    set_seed(CFG.seed)
    ensure_dir(CFG.save_dir)
    best_model_path = os.path.join(CFG.save_dir, CFG.best_model_name)
    metrics_path = os.path.join(CFG.save_dir, CFG.metrics_name)

    print("=" * 88)
    print("Training: train_transformer_planning_v14_1_real_camera_cache_enabled_full_pipeline")
    print(f"Manifest path : {CFG.manifest_path}")
    print(f"Shard dir     : {CFG.shard_dir}")
    print(f"Save dir      : {CFG.save_dir}")
    print(f"Base ckpt     : {CFG.base_ckpt_path}")
    print(f"Device        : {CFG.device}")
    print(f"Load ann-BEV  : {CFG.use_annotation_bev}  (USE_ANNOTATION_BEV)")
    print(f"Dense BEV tok : {CFG.use_dense_bev_tokens}  (USE_DENSE_BEV_TOKENS)")
    print(f"Obj-align BEV : {CFG.use_object_aligned_bev}")
    print(f"Learned camBEV: {CFG.use_learned_camera_bev}  (planner consumes generated BEV)")
    print(f"Real cam cache: require={CFG.require_real_camera_cache}, min_valid={CFG.min_camera_valid_mean}")
    print(f"Camera cache  : {CFG.projection_cache_manifest}")
    print(f"BEV distill   : lambda={CFG.lambda_bev_distill}")
    print("=" * 88)

    if not os.path.exists(CFG.manifest_path):
        raise FileNotFoundError(f"manifest not found: {CFG.manifest_path}")
    if CFG.use_object_aligned_bev and not CFG.use_annotation_bev:
        raise RuntimeError(
            "USE_OBJECT_ALIGNED_BEV=1 but USE_ANNOTATION_BEV=0. "
            "Object-aligned BEV fusion / BEV distillation would lack a teacher BEV. "
            "Set USE_ANNOTATION_BEV=1 or disable USE_OBJECT_ALIGNED_BEV and LAMBDA_BEV_DISTILL."
        )

    use_any_visual = (CFG.use_global_bev_token or CFG.use_agent_aligned_visual or CFG.use_map_aligned_visual or CFG.use_corridor_visual_token)
    need_camera_cache = CFG.use_learned_camera_bev or use_any_visual or CFG.require_real_camera_cache
    if need_camera_cache:
        reason = []
        if CFG.use_learned_camera_bev:
            reason.append("learned camera-BEV")
        if use_any_visual:
            reason.append("projection visual tokens")
        if CFG.require_real_camera_cache:
            reason.append("real-camera guard")
        print("Loading projection/camera cache because: " + ", ".join(reason))
        projection_cache = load_projection_visual_cache(CFG.projection_cache_manifest)
        print(f"Projection/camera cache samples: {len(projection_cache)}")
    else:
        print("Projection visual branch disabled: using zero visual placeholders and NOT loading projection cache.")
        projection_cache = None
    print("Loading v10.0 3D agent cache...")
    agent3d_cache = load_agent3d_cache(CFG.agent3d_cache_manifest)
    print(f"3D agent cache samples: {len(agent3d_cache)}")
    if CFG.use_annotation_bev:
        print("Loading annotation-BEV teacher cache...")
        bev_cache = load_annotation_bev_cache(CFG.bev_cache_manifest)
        print(f"Teacher BEV cache samples: {len(bev_cache)}")
    else:
        print("Annotation BEV disabled: using zero BEV placeholders.")
        bev_cache = None

    print("Loading train/val data from shards...")
    train_data = load_split_from_shards(CFG.manifest_path, CFG.shard_dir, "train", projection_cache=projection_cache, agent3d_cache=agent3d_cache, bev_cache=bev_cache)
    val_data = load_split_from_shards(CFG.manifest_path, CFG.shard_dir, "val", projection_cache=projection_cache, agent3d_cache=agent3d_cache, bev_cache=bev_cache)
    X_train, A_train, Av_train, A3_train, A3v_train, BEV_train, BEVv_train, M_train, Mv_train, y_train_cls, y_train_traj, map_y_train, map_ref_train, intent_train, C_train, Cv_train, PA_train, PAv_train, PM_train, PMv_train, PC_train, PCv_train = train_data
    X_val, A_val, Av_val, A3_val, A3v_val, BEV_val, BEVv_val, M_val, Mv_val, y_val_cls, y_val_traj, map_y_val, map_ref_val, intent_val, C_val, Cv_val, PA_val, PAv_val, PM_val, PMv_val, PC_val, PCv_val = val_data

    print(f"Train size       : {len(X_train)}")
    print(f"Val size         : {len(X_val)}")
    print(f"History shape    : {X_train.shape}")
    print(f"Agent shape      : {A_train.shape}")
    print(f"3D Agent shape   : {A3_train.shape}, valid mean={A3v_train.mean():.4f}")
    print(f"Teacher ann-BEV  : {BEV_train.shape}, valid mean={BEVv_train.mean():.4f}")
    print(f"Map shape        : {M_train.shape}")
    print(f"Future traj shape: {y_train_traj.shape}")
    print(f"Intent shape     : {intent_train.shape}  # [terminal_speed, total_disp, lateral_disp, yaw_delta], normalized")
    print(f"Camera shape     : {C_train.shape}")
    print(f"Camera valid mean: {Cv_train.mean():.4f}")
    if CFG.require_real_camera_cache and CFG.use_learned_camera_bev:
        cam_valid_mean = float(Cv_train.mean())
        if cam_valid_mean < CFG.min_camera_valid_mean:
            raise RuntimeError(
                f"v14.1 requires real camera cache, but Camera valid mean={cam_valid_mean:.6f} < {CFG.min_camera_valid_mean}.\n"
                f"Your learned camera-BEV would be fake/zero. Check PROJECTION_CACHE_MANIFEST/CAMERA_CACHE_MANIFEST: {CFG.projection_cache_manifest}\n"
                "Run or fix the v9.5 projection/camera cache builder, then retry."
            )
    print(f"Proj agent shape : {PA_train.shape}, valid mean={PAv_train.mean():.4f}")
    print(f"Proj map shape   : {PM_train.shape}, valid mean={PMv_train.mean():.4f}")
    print(f"Proj corridor    : {PC_train.shape}, valid mean={PCv_train.mean():.4f}")

    if X_train.shape[1:] != (CFG.history_len, CFG.history_dim):
        raise ValueError(f"Expected history [T={CFG.history_len},D={CFG.history_dim}], got {X_train.shape[1:]}")

    # Fit a provisional scaler for optional v10.0 base training.
    # If an existing base checkpoint is loaded later, its scaler is re-applied exactly.
    temp_scaler = StandardScalerNP(skip_indices=[14, 15])
    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_train = temp_scaler.fit_transform(X_train.reshape(-1, CFG.history_dim)).reshape(X_train.shape)
    X_val = temp_scaler.transform(X_val.reshape(-1, CFG.history_dim)).reshape(X_val.shape)
    scaler = temp_scaler
    scaler_source = "fit_from_train"
    print(f"History scaler   : {scaler_source}")

    class_ids = np.arange(len(CFG.target_classes))
    cls_weights_raw = compute_class_weight(class_weight="balanced", classes=class_ids, y=y_train_cls)
    cls_weights_np = np.clip(cls_weights_raw, CFG.class_weight_min, CFG.class_weight_max)
    stop_id = CLASS_TO_ID["STOP"]
    cls_weights_np[stop_id] = min(cls_weights_np[stop_id], CFG.stop_class_weight_max)
    cls_weights = torch.tensor(cls_weights_np, dtype=torch.float32, device=CFG.device)
    print("Class weights raw -> calibrated:")
    for i, w in enumerate(cls_weights.tolist()):
        print(f"  {ID_TO_CLASS[i]:>7}: {float(cls_weights_raw[i]):.6f} -> {w:.6f}")

    train_ds = JointTokenDataset(X_train, A_train, Av_train, A3_train, A3v_train, BEV_train, BEVv_train, M_train, Mv_train, y_train_cls, y_train_traj, map_y_train, map_ref_train, intent_train, C_train, Cv_train, PA_train, PAv_train, PM_train, PMv_train, PC_train, PCv_train)
    val_ds = JointTokenDataset(X_val, A_val, Av_val, A3_val, A3v_val, BEV_val, BEVv_val, M_val, Mv_val, y_val_cls, y_val_traj, map_y_val, map_ref_val, intent_val, C_val, Cv_val, PA_val, PAv_val, PM_val, PMv_val, PC_val, PCv_val)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    train_shapes = {
        "history": list(X_train.shape), "agents": list(A_train.shape), "agent3d": list(A3_train.shape), "bev": list(BEV_train.shape), "maps": list(M_train.shape),
        "traj": list(y_train_traj.shape), "intent": list(intent_train.shape), "camera": list(C_train.shape),
        "proj_agent": list(PA_train.shape), "proj_map": list(PM_train.shape), "proj_corridor": list(PC_train.shape),
    }
    val_shapes = {
        "history": list(X_val.shape), "agents": list(A_val.shape), "agent3d": list(A3_val.shape), "bev": list(BEV_val.shape), "maps": list(M_val.shape),
        "traj": list(y_val_traj.shape), "intent": list(intent_val.shape), "camera": list(C_val.shape),
    }

    cls_criterion = FocalLoss(alpha=cls_weights, gamma=CFG.focal_gamma) if CFG.use_focal_loss else nn.CrossEntropyLoss(weight=cls_weights)

    ckpt, resolved_base_ckpt_path, stage1_info = train_v8_3_base_if_needed(
        train_loader=train_loader,
        val_loader=val_loader,
        cls_criterion=cls_criterion,
        scaler=scaler,
        train_shapes=train_shapes,
        val_shapes=val_shapes,
    )

    # Re-apply checkpoint scaler and rebuild loaders to exactly match the resolved base checkpoint.
    X_train, X_val, scaler, scaler_source = apply_checkpoint_scaler_or_fit(ckpt, X_train_raw, X_val_raw)
    print(f"Resolved base ckpt: {resolved_base_ckpt_path}")
    print(f"History scaler    : {scaler_source}")
    train_ds = JointTokenDataset(X_train, A_train, Av_train, A3_train, A3v_train, BEV_train, BEVv_train, M_train, Mv_train, y_train_cls, y_train_traj, map_y_train, map_ref_train, intent_train, C_train, Cv_train, PA_train, PAv_train, PM_train, PMv_train, PC_train, PCv_train)
    val_ds = JointTokenDataset(X_val, A_val, Av_val, A3_val, A3v_val, BEV_val, BEVv_val, M_val, Mv_val, y_val_cls, y_val_traj, map_y_val, map_ref_val, intent_val, C_val, Cv_val, PA_val, PAv_val, PM_val, PMv_val, PC_val, PCv_val)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    base_model = build_base_model().to(CFG.device)
    base_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = ScoreOnlyCalibrator(base_model, quality_dim=CFG.score_quality_dim, hidden_dim=CFG.score_refiner_hidden_dim, dropout=CFG.score_refiner_dropout).to(CFG.device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)

    optimizer = torch.optim.AdamW(trainable_params, lr=CFG.lr, weight_decay=CFG.weight_decay)

    print(f"Model type       : {CFG.model_type}")
    print(f"Stage 1 info     : {stage1_info}")
    print(f"Frozen base      : v14.1 Real-Camera-Cache Learned Camera-BEV Joint planner")
    print(f"Trainable params : {trainable_count:,} / {total_params:,}")
    print(f"Score refiner    : hidden={CFG.score_refiner_hidden_dim}, delta_scale={CFG.score_delta_scale}, quality_clip={CFG.score_quality_clip}")
    print(f"Loss lambdas     : soft={CFG.lambda_score_soft}, rank={CFG.lambda_score_rank}, reg={CFG.lambda_score_reg}, margin={CFG.score_rank_margin}")
    print(f"Planning score   : ADE + {CFG.planning_score_fde_weight}*FDE + {CFG.planning_score_gap_weight}*ADE_gap")
    print("Goal             : lower ADE_selected / ADE_gap without touching classification or trajectory generator")

    # Evaluate before training: this should match your frozen v10.0 base selected/oracle behavior.
    base_metrics = evaluate(model, val_loader, cls_criterion, CFG.device)
    print("\n========== Frozen Base Check ==========")
    print(f"Base Acc       : {base_metrics['acc']:.4f}")
    print(f"Base MacroF1   : {base_metrics['macro_f1']:.4f}")
    print(f"Base ADE_sel   : {base_metrics['ADE_selected']:.4f}")
    print(f"Base FDE_sel   : {base_metrics['FDE_selected']:.4f}")
    print(f"Base ADE_orc   : {base_metrics['ADE_oracle']:.4f}")
    print(f"Base FDE_orc   : {base_metrics['FDE_oracle']:.4f}")
    print(f"Base ADE gap   : {base_metrics['ADE_gap']:.4f}")
    print(f"Base ScoreHit  : {base_metrics['score_hit_rate']:.4f}")

    best_epoch = 0
    best_ADE = base_metrics["ADE_selected"]
    best_FDE = base_metrics["FDE_selected"]
    best_gap = base_metrics["ADE_gap"]
    best_score_hit = base_metrics["score_hit_rate"]
    best_planning_score = best_ADE + CFG.planning_score_fde_weight * best_FDE + CFG.planning_score_gap_weight * best_gap
    best_metrics = base_metrics
    patience_counter = 0
    history = [{"epoch": 0, "base_planning_score": float(best_planning_score), **{f"base_{k}": v for k, v in base_metrics.items() if isinstance(v, (int, float))}}]

    # Save initial no-op calibrator as a safe fallback.
    torch.save({
        "model_state_dict": model.state_dict(),
        "base_model_state_dict": base_model.state_dict(),
        "score_refiner_state_dict": model.score_refiner.state_dict(),
        "model_type": CFG.model_type,
        "base_ckpt_path": resolved_base_ckpt_path,
        "projection_cache_manifest": CFG.projection_cache_manifest,
        "config": asdict(CFG),
        "class_to_id": CLASS_TO_ID,
        "id_to_class": ID_TO_CLASS,
        "scaler": scaler.state_dict(),
        "best_val_ADE_selected": float(best_ADE),
        "best_val_FDE_selected": float(best_FDE),
        "best_val_ADE_gap": float(best_gap),
        "best_val_planning_score": float(best_planning_score),
        "epoch": 0,
    }, best_model_path)

    for epoch in range(1, CFG.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, cls_criterion, optimizer, CFG.device)
        val_metrics = evaluate(model, val_loader, cls_criterion, CFG.device)

        ade = val_metrics["ADE_selected"]
        fde = val_metrics["FDE_selected"]
        ade_orc = val_metrics["ADE_oracle"]
        fde_orc = val_metrics["FDE_oracle"]
        gap = val_metrics["ADE_gap"]
        hit = val_metrics["score_hit_rate"]
        planning_score = ade + CFG.planning_score_fde_weight * fde + CFG.planning_score_gap_weight * gap
        base_ade = val_metrics["ADE_base_selected"]
        base_gap = val_metrics["base_ADE_gap"]
        macro_f1 = val_metrics["macro_f1"]
        acc = val_metrics["acc"]

        row = {"epoch": epoch}
        row.update({f"train_{k}": float(v) for k, v in train_stats.items()})
        row.update({
            "val_acc": float(acc),
            "val_macro_f1": float(macro_f1),
            "val_ADE_selected": float(ade),
            "val_FDE_selected": float(fde),
            "val_ADE_oracle": float(ade_orc),
            "val_FDE_oracle": float(fde_orc),
            "val_ADE_gap": float(gap),
            "val_planning_score": float(planning_score),
            "val_score_hit_rate": float(hit),
            "val_ADE_base_selected": float(base_ade),
            "val_base_ADE_gap": float(base_gap),
        })
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss {train_stats['loss']:.4f} | Soft {train_stats['loss_score_soft']:.4f} | "
            f"Rank {train_stats['loss_score_rank']:.4f} | Reg {train_stats['loss_score_reg']:.4f} | "
            f"Val Acc {acc:.4f} | MacroF1 {macro_f1:.4f} | "
            f"ADE_sel {ade:.4f} | FDE_sel {fde:.4f} | "
            f"ADE_orc {ade_orc:.4f} | FDE_orc {fde_orc:.4f} | "
            f"Gap {gap:.4f} | PlanScore {planning_score:.4f} | ScoreHit {hit:.3f} | BaseADE {base_ade:.4f}"
        )

        improved = False
        # v14.1: select the score-calibrated model by planning score, not ADE alone.
        if planning_score < best_planning_score - CFG.min_ade_improve:
            improved = True
        elif abs(planning_score - best_planning_score) <= CFG.min_ade_improve and ade < best_ADE - 1e-4:
            improved = True
        elif abs(planning_score - best_planning_score) <= CFG.min_ade_improve and abs(ade - best_ADE) <= 1e-4 and hit > best_score_hit + 1e-4:
            improved = True

        if improved:
            best_epoch = epoch
            best_ADE = ade
            best_FDE = fde
            best_gap = gap
            best_score_hit = hit
            best_planning_score = planning_score
            best_metrics = val_metrics
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "base_model_state_dict": base_model.state_dict(),
                "score_refiner_state_dict": model.score_refiner.state_dict(),
                "model_type": CFG.model_type,
                "base_ckpt_path": resolved_base_ckpt_path,
        "projection_cache_manifest": CFG.projection_cache_manifest,
                "config": asdict(CFG),
                "class_to_id": CLASS_TO_ID,
                "id_to_class": ID_TO_CLASS,
                "scaler": scaler.state_dict(),
                "best_val_acc": float(acc),
                "best_val_macro_f1": float(macro_f1),
                "best_val_ADE_selected": float(best_ADE),
                "best_val_FDE_selected": float(best_FDE),
                "best_val_ADE_gap": float(best_gap),
                "best_val_planning_score": float(best_planning_score),
                "best_val_score_hit_rate": float(best_score_hit),
                "epoch": best_epoch,
            }, best_model_path)
            print(f"🧭 Saved best score-calibrated model at epoch {epoch} (ADE={ade:.4f}, FDE={fde:.4f}, Gap={gap:.4f}, PlanScore={planning_score:.4f}, ScoreHit={hit:.3f})")
        else:
            patience_counter += 1

        if patience_counter >= CFG.early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    ckpt_best = torch.load(best_model_path, map_location=CFG.device, weights_only=False)
    final_base = build_base_model().to(CFG.device)
    final_base.load_state_dict(ckpt_best["base_model_state_dict"], strict=True)
    best_model = ScoreOnlyCalibrator(final_base, quality_dim=CFG.score_quality_dim, hidden_dim=CFG.score_refiner_hidden_dim, dropout=CFG.score_refiner_dropout).to(CFG.device)
    best_model.load_state_dict(ckpt_best["model_state_dict"], strict=True)
    final_metrics = evaluate(best_model, val_loader, cls_criterion, CFG.device)

    report = classification_report(
        final_metrics["y_true"],
        final_metrics["y_pred"],
        labels=list(range(len(CFG.target_classes))),
        target_names=list(CFG.target_classes),
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    traj_vis_dir = os.path.join(CFG.save_dir, "traj_vis")
    save_traj_visualizations(best_model, val_loader, CFG.device, traj_vis_dir, num_samples=12)

    metrics = {
        "config": asdict(CFG),
        "base_ckpt_path": resolved_base_ckpt_path,
        "projection_cache_manifest": CFG.projection_cache_manifest,
        "stage1_info": stage1_info,
        "train_shapes": {
            "history": list(X_train.shape),
            "agents": list(A_train.shape),
            "maps": list(M_train.shape),
            "traj": list(y_train_traj.shape),
            "intent": list(intent_train.shape), "camera": list(C_train.shape),
        },
        "val_shapes": {
            "history": list(X_val.shape),
            "agents": list(A_val.shape),
            "maps": list(M_val.shape),
            "traj": list(y_val_traj.shape),
            "intent": list(intent_val.shape), "camera": list(C_val.shape),
        },
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "base_metrics": {k: v for k, v in base_metrics.items() if isinstance(v, (int, float))},
        "best_epoch": int(best_epoch),
        "best_val_ADE_selected": float(best_ADE),
        "best_val_FDE_selected": float(best_FDE),
        "best_val_ADE_gap": float(best_gap),
        "best_val_planning_score": float(best_planning_score),
        "best_val_score_hit_rate": float(best_score_hit),
        "final_val_loss": float(final_metrics["loss"]),
        "final_val_acc": float(final_metrics["acc"]),
        "final_val_macro_f1": float(final_metrics["macro_f1"]),
        "final_val_ADE_selected": float(final_metrics["ADE_selected"]),
        "final_val_FDE_selected": float(final_metrics["FDE_selected"]),
        "final_val_ADE_oracle": float(final_metrics["ADE_oracle"]),
        "final_val_FDE_oracle": float(final_metrics["FDE_oracle"]),
        "final_val_ADE_base_selected": float(final_metrics["ADE_base_selected"]),
        "final_val_FDE_base_selected": float(final_metrics["FDE_base_selected"]),
        "final_val_ADE_gap": float(final_metrics["ADE_gap"]),
        "final_val_base_ADE_gap": float(final_metrics["base_ADE_gap"]),
        "final_score_hit_rate": float(final_metrics["score_hit_rate"]),
        "final_base_score_hit_rate": float(final_metrics["base_score_hit_rate"]),
        "final_intent_mae_terminal_speed_mps": float(final_metrics["intent_mae_terminal_speed_mps"]),
        "final_intent_mae_total_disp_m": float(final_metrics["intent_mae_total_disp_m"]),
        "final_intent_mae_lateral_disp_m": float(final_metrics["intent_mae_lateral_disp_m"]),
        "final_intent_mae_yaw_delta_rad": float(final_metrics["intent_mae_yaw_delta_rad"]),
        "final_precisions": {ID_TO_CLASS[i]: float(p) for i, p in enumerate(final_metrics["precisions"])},
        "final_recalls": {ID_TO_CLASS[i]: float(r) for i, r in enumerate(final_metrics["recalls"])},
        "final_f1s": {ID_TO_CLASS[i]: float(f) for i, f in enumerate(final_metrics["f1s"])},
        "confusion_matrix": final_metrics["cm"],
        "classification_report": report,
        "history": history,
        "traj_vis_dir": traj_vis_dir,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n========== Final Summary ==========")
    print(f"Best epoch       : {best_epoch}")
    print(f"Base ADE_sel     : {base_metrics['ADE_selected']:.4f}")
    print(f"Base FDE_sel     : {base_metrics['FDE_selected']:.4f}")
    print(f"Base ADE_gap     : {base_metrics['ADE_gap']:.4f}")
    print(f"Base ScoreHit    : {base_metrics['score_hit_rate']:.4f}")
    print(f"Final Acc        : {final_metrics['acc']:.4f}")
    print(f"Final MacroF1    : {final_metrics['macro_f1']:.4f}")
    print(f"Final ADE_sel    : {final_metrics['ADE_selected']:.4f}")
    print(f"Final FDE_sel    : {final_metrics['FDE_selected']:.4f}")
    print(f"Final ADE_orc    : {final_metrics['ADE_oracle']:.4f}")
    print(f"Final FDE_orc    : {final_metrics['FDE_oracle']:.4f}")
    print(f"Final ADE_gap    : {final_metrics['ADE_gap']:.4f}")
    final_plan_score = final_metrics['ADE_selected'] + CFG.planning_score_fde_weight * final_metrics['FDE_selected'] + CFG.planning_score_gap_weight * final_metrics['ADE_gap']
    print(f"Final ScoreHit   : {final_metrics['score_hit_rate']:.4f}")
    print(f"Final PlanScore  : {final_plan_score:.4f}")
    print(f"Intent MAE       : speed={final_metrics['intent_mae_terminal_speed_mps']:.3f} m/s | total_disp={final_metrics['intent_mae_total_disp_m']:.3f} m | lateral={final_metrics['intent_mae_lateral_disp_m']:.3f} m | yaw={final_metrics['intent_mae_yaw_delta_rad']:.3f} rad")
    print("Confusion Matrix [rows=true, cols=pred]:")
    for i, row in enumerate(final_metrics["cm"]):
        print(f"{ID_TO_CLASS[i]:>7}: {row}")
    print(f"\nTrajectory visualizations saved to: {traj_vis_dir}")
    print(f"Metrics saved to              : {metrics_path}")
    print(f"Best model saved to           : {best_model_path}")


if __name__ == "__main__":
    main()
