#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from .config import CFG, CLASS_TO_ID, ID_TO_CLASS
from .utils import set_seed, ensure_dir, StandardScalerNP
from .data import (
    JointTokenDataset, load_agent3d_cache, load_image_bev_cache,
    load_projection_visual_cache, load_split_from_shards,
)
from .losses import FocalLoss
from .model import MapAgentEgoPlanner
from .calibrator import ScoreOnlyCalibrator
from .train_eval import (
    apply_checkpoint_scaler_or_fit, build_base_model, evaluate, save_traj_visualizations,
    train_one_epoch, train_v8_3_base_if_needed,
)

def main():
    set_seed(CFG.seed)
    ensure_dir(CFG.save_dir)
    best_model_path = os.path.join(CFG.save_dir, CFG.best_model_name)
    metrics_path = os.path.join(CFG.save_dir, CFG.metrics_name)

    print("=" * 88)
    print("Training: train_transformer_planning_v18_5_rawbev_diffusion_oracleade_selector_full_pipeline")
    print(f"Manifest path : {CFG.manifest_path}")
    print(f"Shard dir     : {CFG.shard_dir}")
    print(f"Save dir      : {CFG.save_dir}")
    print(f"Base ckpt     : {CFG.base_ckpt_path}")
    print(f"Device        : {CFG.device}")
    print(f"Raw BEV       : {not CFG.use_cleaned_image_bev}  # USE_CLEANED_IMAGE_BEV={int(CFG.use_cleaned_image_bev)}")
    print(f"ADE selector  : quality={CFG.score_quality_dim}D, hidden={CFG.score_refiner_hidden_dim}, delta_scale={CFG.score_delta_scale}, temp={CFG.score_target_temp}, fde_w={CFG.score_fde_weight}")
    print("=" * 88)

    if not os.path.exists(CFG.manifest_path):
        raise FileNotFoundError(f"manifest not found: {CFG.manifest_path}")

    use_any_visual = (CFG.use_global_bev_token or CFG.use_agent_aligned_visual or CFG.use_map_aligned_visual or CFG.use_corridor_visual_token)
    if use_any_visual:
        print("Loading projection-aligned visual cache because at least one visual switch is enabled...")
        projection_cache = load_projection_visual_cache(CFG.projection_cache_manifest)
        print(f"Projection cache samples: {len(projection_cache)}")
    else:
        print("Projection visual branch disabled: using zero visual placeholders and NOT loading projection cache.")
        projection_cache = None
    print("Loading v10.0 3D agent cache...")
    agent3d_cache = load_agent3d_cache(CFG.agent3d_cache_manifest)
    print(f"3D agent cache samples: {len(agent3d_cache)}")
    need_bev_cache = CFG.use_object_aligned_bev or CFG.use_dense_bev_tokens
    if need_bev_cache:
        print("Loading v16.4.1 image-BEV cache for object-aligned / dense BEV fusion...")
        print(f"BEV cache manifest: {CFG.bev_cache_manifest}")
        bev_cache = load_image_bev_cache(CFG.bev_cache_manifest)
        print(f"Image BEV cache samples: {len(bev_cache)}")
    else:
        print("BEV fusion disabled: using zero BEV placeholders.")
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
    print(f"Image BEV         : {BEV_train.shape}, valid mean={BEVv_train.mean():.4f}")
    print(f"Map shape        : {M_train.shape}")
    print(f"Future traj shape: {y_train_traj.shape}")
    print(f"Intent shape     : {intent_train.shape}  # [terminal_speed, total_disp, lateral_disp, yaw_delta], normalized")
    print(f"Camera shape     : {C_train.shape}")
    print(f"Camera valid mean: {Cv_train.mean():.4f}")
    print(f"Proj agent shape : {PA_train.shape}, valid mean={PAv_train.mean():.4f}")
    print(f"Proj map shape   : {PM_train.shape}, valid mean={PMv_train.mean():.4f}")
    print(f"Proj corridor    : {PC_train.shape}, valid mean={PCv_train.mean():.4f}")

    if X_train.shape[1:] != (CFG.history_len, CFG.history_dim):
        raise ValueError(f"Expected history [T={CFG.history_len},D={CFG.history_dim}], got {X_train.shape[1:]}")

    # Fit a provisional scaler for optional v16.4 base training.
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
    print(f"Frozen base      : v18.5 Raw-BEV Diffusion-Residual generator + oracle-ADE-supervised selector")
    print(f"Trainable params : {trainable_count:,} / {total_params:,}")
    print(f"Score refiner    : hidden={CFG.score_refiner_hidden_dim}, delta_scale={CFG.score_delta_scale}, quality_clip={CFG.score_quality_clip}")
    print(f"Loss lambdas     : softADE={CFG.lambda_score_soft}, pairRank={CFG.lambda_score_rank}, reg={CFG.lambda_score_reg}, margin={CFG.score_rank_margin}")
    print(f"Diffusion v18.5    : enabled={CFG.use_diffusion_residual_decoder}, steps={CFG.diffusion_steps}, sample_steps={CFG.diffusion_sample_steps}, noise_lambda={CFG.lambda_diffusion_noise}, recon_lambda={CFG.lambda_diffusion_recon}")
    print("Goal             : train selector directly on per-mode ADE+FDE oracle labels over rawBEV + diffusion trajectories")

    # Evaluate before training: this should match your frozen v16.4 base selected/oracle behavior.
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
    best_metrics = base_metrics
    patience_counter = 0
    history = [{"epoch": 0, **{f"base_{k}": v for k, v in base_metrics.items() if isinstance(v, (int, float))}}]

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
            f"Gap {gap:.4f} | ScoreHit {hit:.3f} | BaseADE {base_ade:.4f}"
        )

        improved = False
        # Be conservative: selected ADE must improve; if tied, gap/hit decides.
        if ade < best_ADE - CFG.min_ade_improve:
            improved = True
        elif abs(ade - best_ADE) <= CFG.min_ade_improve and gap < best_gap - 1e-4:
            improved = True
        elif abs(ade - best_ADE) <= CFG.min_ade_improve and abs(gap - best_gap) <= 1e-4 and hit > best_score_hit + 1e-4:
            improved = True

        if improved:
            best_epoch = epoch
            best_ADE = ade
            best_FDE = fde
            best_gap = gap
            best_score_hit = hit
            best_metrics = val_metrics
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "base_model_state_dict": base_model.state_dict(),
                "score_refiner_state_dict": model.score_refiner.state_dict(),
                "model_type": CFG.model_type,
                "base_ckpt_path": resolved_base_ckpt_path,
        "projection_cache_manifest": CFG.projection_cache_manifest,
                "image_bev_cache_manifest": CFG.bev_cache_manifest,
                "config": asdict(CFG),
                "class_to_id": CLASS_TO_ID,
                "id_to_class": ID_TO_CLASS,
                "scaler": scaler.state_dict(),
                "best_val_acc": float(acc),
                "best_val_macro_f1": float(macro_f1),
                "best_val_ADE_selected": float(best_ADE),
                "best_val_FDE_selected": float(best_FDE),
                "best_val_ADE_gap": float(best_gap),
                "best_val_score_hit_rate": float(best_score_hit),
                "epoch": best_epoch,
            }, best_model_path)
            print(f"🧭 Saved best score-calibrated model at epoch {epoch} (ADE={ade:.4f}, Gap={gap:.4f}, ScoreHit={hit:.3f})")
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
        "image_bev_cache_manifest": CFG.bev_cache_manifest,
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
    print(f"Final ScoreHit   : {final_metrics['score_hit_rate']:.4f}")
    print(f"Intent MAE       : speed={final_metrics['intent_mae_terminal_speed_mps']:.3f} m/s | total_disp={final_metrics['intent_mae_total_disp_m']:.3f} m | lateral={final_metrics['intent_mae_lateral_disp_m']:.3f} m | yaw={final_metrics['intent_mae_yaw_delta_rad']:.3f} rad")
    print("Confusion Matrix [rows=true, cols=pred]:")
    for i, row in enumerate(final_metrics["cm"]):
        print(f"{ID_TO_CLASS[i]:>7}: {row}")
    print(f"\nTrajectory visualizations saved to: {traj_vis_dir}")
    print(f"Metrics saved to              : {metrics_path}")
    print(f"Best model saved to           : {best_model_path}")



if __name__ == "__main__":
    main()
