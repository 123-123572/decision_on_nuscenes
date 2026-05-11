from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Tuple

import torch

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
    # v11.0: annotation/map/agent BEV occupancy-risk grid.
    # This is not image-BEV; it is a teacher/oracle BEV built from nuScenes annotations and vector map tokens.
    use_dense_bev_tokens: bool = os.getenv("USE_DENSE_BEV_TOKENS", "0").lower() in {"1", "true", "yes", "y", "on"}
    bev_channels: int = int(os.getenv("BEV_CHANNELS", "8"))
    # Must match v16.4 BEV cache builder defaults.
    # v16.3 used 80x60; v16.4 defaults to 120x90.
    bev_h: int = int(os.getenv("BEV_H", "120"))
    bev_w: int = int(os.getenv("BEV_W", "90"))
    bev_token_grid: int = int(os.getenv("BEV_TOKEN_GRID", "4"))
    # v16.4.1 cache contains both raw image_bev_feat and cleaned image_bev_feat_clean.
    # Default to cleaned because your preview showed ray-like false positives.
    use_cleaned_image_bev: bool = os.getenv("USE_CLEANED_IMAGE_BEV", "0").lower() in {"1", "true", "yes", "y", "on"}

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

    # v17: conditional diffusion residual trajectory generator.
    # The old MLP head is kept as a stable base trajectory; diffusion learns a residual over it.
    use_diffusion_residual_decoder: bool = os.getenv("USE_DIFFUSION_RESIDUAL_DECODER", "1").lower() in {"1", "true", "yes", "y", "on"}
    diffusion_steps: int = int(os.getenv("DIFFUSION_STEPS", "20"))
    diffusion_sample_steps: int = int(os.getenv("DIFFUSION_SAMPLE_STEPS", "6"))
    diffusion_hidden_dim: int = int(os.getenv("DIFFUSION_HIDDEN_DIM", "128"))
    diffusion_dropout: float = float(os.getenv("DIFFUSION_DROPOUT", "0.10"))
    diffusion_sample_noise_scale: float = float(os.getenv("DIFFUSION_SAMPLE_NOISE_SCALE", "0.65"))
    diffusion_eval_deterministic: bool = os.getenv("DIFFUSION_EVAL_DETERMINISTIC", "1").lower() in {"1", "true", "yes", "y", "on"}
    diffusion_train_deterministic: bool = os.getenv("DIFFUSION_TRAIN_DETERMINISTIC", "1").lower() in {"1", "true", "yes", "y", "on"}
    diffusion_residual_clip_m: float = float(os.getenv("DIFFUSION_RESIDUAL_CLIP_M", "6.0"))
    lambda_diffusion_noise: float = float(os.getenv("LAMBDA_DIFFUSION_NOISE", "0.35"))
    lambda_diffusion_recon: float = float(os.getenv("LAMBDA_DIFFUSION_RECON", "1.20"))

    # v9.2 score-only calibration losses.
    lambda_score_soft: float = 1.0
    lambda_score_rank: float = float(os.getenv("LAMBDA_SCORE_RANK", "0.35"))
    lambda_score_reg: float = float(os.getenv("LAMBDA_SCORE_REG", "0.08"))
    score_rank_margin: float = float(os.getenv("SCORE_RANK_MARGIN", "0.10"))
    min_ade_improve: float = 1e-4

    traj_scale: float = 20.0
    agent_pos_scale: float = 60.0
    agent_vel_scale: float = 20.0
    map_pos_scale: float = 60.0
    dt: float = 0.5

    score_target_temp: float = float(os.getenv("SCORE_TARGET_TEMP", "0.25"))
    score_speed_penalty_weight: float = 0.12
    score_speed_penalty_scale: float = 2.0
    score_label_smoothing: float = float(os.getenv("SCORE_LABEL_SMOOTHING", "0.01"))
    score_fde_weight: float = float(os.getenv("SCORE_FDE_WEIGHT", "0.50"))
    score_pair_min_delta: float = float(os.getenv("SCORE_PAIR_MIN_DELTA", "0.02"))

    div_end_margin_m: float = 0.45
    div_full_margin_m: float = 0.25
    collision_radius_m: float = 2.5

    # v9.2: explicit but conservative score calibration features.
    score_quality_dim: int = int(os.getenv("SCORE_QUALITY_DIM", "16"))
    score_refiner_hidden_dim: int = int(os.getenv("SCORE_REFINER_HIDDEN_DIM", "128"))
    score_refiner_dropout: float = float(os.getenv("SCORE_REFINER_DROPOUT", "0.10"))
    score_quality_clip: float = 5.0
    score_delta_scale: float = float(os.getenv("SCORE_DELTA_SCALE", "0.25"))
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

    # Full-pipeline control. Default trains v17 base if no v17 checkpoint exists.
    run_base_train: bool = os.getenv("RUN_BASE_TRAIN", "0").lower() in {"1", "true", "yes", "y", "on"}
    train_base_if_missing: bool = os.getenv("TRAIN_BASE_IF_MISSING", "1").lower() in {"1", "true", "yes", "y", "on"}
    base_epochs: int = int(os.getenv("BASE_EPOCHS", "80"))
    base_lr: float = float(os.getenv("BASE_LR", "4e-4"))
    base_weight_decay: float = float(os.getenv("BASE_WEIGHT_DECAY", "1e-4"))
    base_early_stop_patience: int = int(os.getenv("BASE_EARLY_STOP_PATIENCE", "20"))
    base_select_min_macro_f1: float = float(os.getenv("BASE_SELECT_MIN_MACRO_F1", "0.755"))
    base_save_subdir: str = os.getenv("BASE_SAVE_SUBDIR", "v18_5_rawbev_diffusion_base")
    base_best_model_name: str = "best_v18_5_rawbev_diffusion_base_model.pt"

    manifest_path: str = "/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/build_manifest_full_v8_map_agent_t6.json"
    shard_dir: str = "/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/shards_full_v8_map_agent_t6"
    save_dir: str = "/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v18_5_rawbev_diffusion_oracleade_selector"
    base_ckpt_path: str = os.getenv("BASE_CKPT_PATH", "/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v18_5_rawbev_diffusion_oracleade_selector/v18_5_rawbev_diffusion_base/best_v18_5_rawbev_diffusion_base_model.pt")
    init_ckpt_path: str = os.getenv("INIT_CKPT_PATH", "/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v8_123_intent_behavior/best_model.pt")
    projection_cache_manifest: str = os.getenv("PROJECTION_CACHE_MANIFEST", "/home/ubuntu22/decision_on_nuscenes/outputs_v9_5_projection_visual_cache/projection_visual_cache_manifest_v9_5.json")
    agent3d_cache_manifest: str = os.getenv("AGENT3D_CACHE_MANIFEST", "/home/ubuntu22/decision_on_nuscenes/outputs_v10_0_3d_agent_cache/agent3d_cache_manifest_v10_0.json")
    bev_cache_manifest: str = os.getenv("IMAGE_BEV_CACHE_MANIFEST", os.getenv("BEV_CACHE_MANIFEST", "/home/ubuntu22/decision_on_nuscenes/outputs_v16_4_1_highres_depthsup_bgclean_real_image_lss_bev_cache/image_bev_cache_manifest_v16_4_1.json"))
    best_model_name: str = "best_score_model.pt"
    metrics_name: str = "metrics.json"
    model_type: str = "transformer_v18_5_rawbev_diffusion_oracleade_selector_full_pipeline"

    device: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

CFG = Config(
    manifest_path=os.getenv("MANIFEST_PATH", Config.manifest_path),
    shard_dir=os.getenv("SHARD_DIR", Config.shard_dir),
    save_dir=os.getenv("SAVE_DIR", Config.save_dir),
)

CLASS_TO_ID = {c: i for i, c in enumerate(CFG.target_classes)}

ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}

