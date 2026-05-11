from __future__ import annotations

import os
import math
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .config import CFG, CLASS_TO_ID, ID_TO_CLASS
from .utils import ensure_dir, StandardScalerNP
from .model import MapAgentEgoPlanner
from .calibrator import ScoreOnlyCalibrator
from .losses import (
    stop_logit_penalty, base_multimodal_losses, compute_real_map_loss,
    compute_diversity_loss, compute_comfort_loss, compute_collision_loss,
    compute_intent_loss, gather_modes, calc_traj_metrics, decode_intent,
    score_calibration_losses,
)

def base_compute_total_loss(model, batch, cls_criterion, device: str):
    hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, y_cls, y_traj, map_y_ref, map_ref_valid, intent_target, camera_feat, camera_valid, proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid = unpack_batch(batch, device)
    cls_logits, traj_pred, scores, intent_pred = model(hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, camera_feat, camera_valid, proj_agent_feat, proj_agent_valid, proj_map_feat, proj_map_valid, proj_corridor_feat, proj_corridor_valid)

    loss_cls = cls_criterion(cls_logits, y_cls)
    loss_stop = stop_logit_penalty(cls_logits, y_cls)
    loss_traj_mode, loss_traj_gt, loss_score, best_idx_gt, _, pred_idx, _ = base_multimodal_losses(traj_pred, y_traj, y_cls, scores)
    traj_best_gt = gather_modes(traj_pred, best_idx_gt)
    loss_map = compute_real_map_loss(traj_best_gt, map_y_ref, map_ref_valid)
    loss_div = compute_diversity_loss(traj_pred)
    loss_comfort = compute_comfort_loss(traj_best_gt)
    loss_collision = compute_collision_loss(traj_best_gt, agents, agent_valid)
    loss_intent = compute_intent_loss(intent_pred, intent_target)
    if hasattr(model, "diffusion_training_losses"):
        loss_diff_noise, loss_diff_recon = model.diffusion_training_losses(y_traj, y_cls)
    else:
        loss_diff_noise = y_traj.new_tensor(0.0)
        loss_diff_recon = y_traj.new_tensor(0.0)

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
        + CFG.lambda_diffusion_noise * loss_diff_noise
        + CFG.lambda_diffusion_recon * loss_diff_recon
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
        "loss_diff_noise": loss_diff_noise,
        "loss_diff_recon": loss_diff_recon,
        "cls_logits": cls_logits,
        "traj_pred": traj_pred,
        "scores": scores,
        "intent_pred": intent_pred,
        "intent_target": intent_target,
        "y_cls": y_cls,
        "y_traj": y_traj,
        "best_idx_gt": best_idx_gt,
        "pred_idx": pred_idx,
    }


def base_train_one_epoch(model, loader, cls_criterion, optimizer, device: str):
    model.train()
    meters = {k: 0.0 for k in ["loss", "loss_cls", "loss_stop", "loss_traj_mode", "loss_traj_gt", "loss_score", "loss_map", "loss_div", "loss_comfort", "loss_collision", "loss_intent", "loss_diff_noise", "loss_diff_recon"]}
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
    meters = {k: 0.0 for k in ["loss", "loss_cls", "loss_stop", "loss_traj_mode", "loss_traj_gt", "loss_score", "loss_map", "loss_div", "loss_comfort", "loss_collision", "loss_intent", "loss_diff_noise", "loss_diff_recon"]}
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
    """Return a v16.4.1 base checkpoint dict. Train v16.4 base first if requested or missing."""
    base_dir = os.path.join(CFG.save_dir, CFG.base_save_subdir)
    ensure_dir(base_dir)
    generated_ckpt_path = os.path.join(base_dir, CFG.base_best_model_name)

    should_train = CFG.run_base_train or (not os.path.exists(CFG.base_ckpt_path) and CFG.train_base_if_missing)
    load_path = generated_ckpt_path if CFG.run_base_train else CFG.base_ckpt_path

    if not should_train and os.path.exists(load_path):
        print(f"Loading existing v16.4.1 base checkpoint: {load_path}")
        ckpt = torch.load(load_path, map_location=CFG.device, weights_only=False)
        return ckpt, load_path, {"trained": False, "path": load_path}

    if not should_train:
        raise FileNotFoundError(
            f"base v16.4.1 base checkpoint not found: {CFG.base_ckpt_path}\n"
            "Either set BASE_CKPT_PATH correctly, or set RUN_BASE_TRAIN=1."
        )

    print("\n" + "=" * 88)
    print("Stage 1: training v18.5 raw-BEV diffusion-residual base generator inside this script")
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
            print(f"Warm-started v16.4 base from INIT_CKPT_PATH: {CFG.init_ckpt_path}")
            print(f"Compatible tensors loaded: {len(compatible)} | newly initialized tensors: {len(missing)}")
        except Exception as e:
            print(f"[WARN] Failed to warm-start from INIT_CKPT_PATH={CFG.init_ckpt_path}: {e}")
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=CFG.base_lr, weight_decay=CFG.base_weight_decay)
    best_epoch = -1
    best_macro_f1 = -1.0
    best_ADE = math.inf
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
        history.append({
            "epoch": epoch,
            **{f"train_{k}": float(v) for k, v in train_stats.items()},
            "val_acc": float(val_metrics["acc"]),
            "val_macro_f1": float(macro_f1),
            "val_ADE_selected": float(ade),
            "val_FDE_selected": float(fde),
            "val_ADE_oracle": float(ade_orc),
            "val_FDE_oracle": float(fde_orc),
        })
        print(
            f"[v18.5 rawBEV diffusion base] Epoch {epoch:03d} | TrainLoss {train_stats['loss']:.4f} | "
            f"Cls {train_stats['loss_cls']:.4f} | TrajGT {train_stats['loss_traj_gt']:.4f} | "
            f"Score {train_stats['loss_score']:.4f} | DiffN {train_stats['loss_diff_noise']:.4f} | DiffR {train_stats['loss_diff_recon']:.4f} | ValLoss {val_metrics['loss']:.4f} | "
            f"Acc {val_metrics['acc']:.4f} | MacroF1 {macro_f1:.4f} | "
            f"ADE_sel {ade:.4f} | FDE_sel {fde:.4f} | ADE_orc {ade_orc:.4f} | FDE_orc {fde_orc:.4f}"
        )

        improved = False
        # ADE-aware checkpoint selection for planning:
        # If Macro-F1 is acceptable, select the checkpoint with the lowest ADE_selected.
        # Before any checkpoint reaches the threshold, fall back to best Macro-F1.
        candidate_ok = macro_f1 >= CFG.base_select_min_macro_f1
        best_ok = best_macro_f1 >= CFG.base_select_min_macro_f1
        if candidate_ok:
            if (not best_ok) or (ade < best_ADE - 1e-8):
                improved = True
            elif abs(ade - best_ADE) < 1e-8 and macro_f1 > best_macro_f1:
                improved = True
        else:
            if (not best_ok) and macro_f1 > best_macro_f1:
                improved = True

        if improved:
            best_epoch = epoch
            best_macro_f1 = macro_f1
            best_ADE = ade
            best_metrics = val_metrics
            patience_counter = 0
            ckpt_obj = {
                "model_state_dict": base_model.state_dict(),
                "model_type": "transformer_v17_diffusion_residual_bgclean_highres_depthsup_image_bev_base",
                "config": asdict(CFG),
                "class_to_id": CLASS_TO_ID,
                "id_to_class": ID_TO_CLASS,
                "scaler": scaler.state_dict(),
                "best_val_acc": float(val_metrics["acc"]),
                "best_val_macro_f1": float(best_macro_f1),
                "best_val_ADE_selected": float(best_ADE),
                "best_val_FDE_selected": float(fde),
                "epoch": best_epoch,
                "train_shapes": train_shapes,
                "val_shapes": val_shapes,
                "history": history,
            }
            torch.save(ckpt_obj, generated_ckpt_path)
            print(f"🔥 Saved ADE-aware v17 diffusion-residual bg-clean high-res depth-supervised image-BEV base at epoch {epoch} (Macro-F1={macro_f1:.4f}, ADE={ade:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= CFG.base_early_stop_patience:
            print(f"\n[v18.5 rawBEV diffusion base] Early stopping triggered at epoch {epoch}.")
            break

    if not os.path.exists(generated_ckpt_path):
        raise RuntimeError("v16.4 base training finished without saving a checkpoint.")

    ckpt = torch.load(generated_ckpt_path, map_location=CFG.device, weights_only=False)
    stage_metrics = {
        "trained": True,
        "path": generated_ckpt_path,
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_macro_f1),
        "best_val_ADE_selected": float(best_ADE),
        "best_val_acc": float(best_metrics.get("acc", -1.0)) if best_metrics else -1.0,
    }
    return ckpt, generated_ckpt_path, stage_metrics


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

