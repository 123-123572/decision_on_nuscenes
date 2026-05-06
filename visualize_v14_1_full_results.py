#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone visualization script for v14.1 real-camera-cache learned camera-BEV joint planner.

Features
--------
1) Load the trained best_score_model.pt checkpoint.
2) Rebuild the validation loader from manifest + shards + caches.
3) Save:
   - prediction_debug.npz
   - K-mode trajectory visualizations (GT / selected / oracle / base-selected)
   - BEV visualizations (teacher vs learned vs error)
   - confusion matrix figure
   - training history curves from metrics.json
   - optional multi-version comparison chart from multiple metrics.json files

Example
-------
python visualize_v14_1_full_results.py \
    --train_script /home/ubuntu22/decision_on_nuscenes/scripts/train_transformer_planning_v14_1_real_camera_cache_enabled_full_pipeline.py \
    --checkpoint /home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v14_1_real_camera_cache_enabled/best_score_model.pt \
    --metrics /home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v14_1_real_camera_cache_enabled/metrics.json \
    --output_dir /home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v14_1_real_camera_cache_enabled/vis_full \
    --traj_samples 48 \
    --bev_samples 24 \
    --save_debug_npz
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full visualization for v14.1 planner results")
    parser.add_argument("--train_script", type=str,
                        default="/home/ubuntu22/decision_on_nuscenes/scripts/train_transformer_planning_v14_1_real_camera_cache_enabled_full_pipeline.py")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v14_1_real_camera_cache_enabled/best_score_model.pt")
    parser.add_argument("--metrics", type=str,
                        default="/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v14_1_real_camera_cache_enabled/metrics.json")
    parser.add_argument("--output_dir", type=str,
                        default="/home/ubuntu22/decision_on_nuscenes/outputs_transformer_planning_v14_1_real_camera_cache_enabled/vis_full")

    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--shard_dir", type=str, default=None)
    parser.add_argument("--projection_cache_manifest", type=str, default=None)
    parser.add_argument("--agent3d_cache_manifest", type=str, default=None)
    parser.add_argument("--bev_cache_manifest", type=str, default=None)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--traj_samples", type=int, default=48)
    parser.add_argument("--bev_samples", type=int, default=24)
    parser.add_argument("--max_debug_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_debug_npz", action="store_true")

    parser.add_argument("--compare_metrics", nargs="*", default=None,
                        help="Optional list of metrics.json files for version comparison.")
    parser.add_argument("--compare_labels", nargs="*", default=None,
                        help="Optional labels matching --compare_metrics.")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dynamic_import(module_path: str):
    module_name = Path(module_path).stem + "_visimport"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def maybe_set_cfg_paths(mod, ckpt: Dict[str, Any], args: argparse.Namespace) -> None:
    ckcfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    mod.CFG.manifest_path = args.manifest_path or ckcfg.get("manifest_path", mod.CFG.manifest_path)
    mod.CFG.shard_dir = args.shard_dir or ckcfg.get("shard_dir", mod.CFG.shard_dir)
    mod.CFG.projection_cache_manifest = args.projection_cache_manifest or ckcfg.get("projection_cache_manifest", mod.CFG.projection_cache_manifest)
    mod.CFG.agent3d_cache_manifest = args.agent3d_cache_manifest or ckcfg.get("agent3d_cache_manifest", mod.CFG.agent3d_cache_manifest)
    mod.CFG.bev_cache_manifest = args.bev_cache_manifest or ckcfg.get("bev_cache_manifest", mod.CFG.bev_cache_manifest)
    mod.CFG.device = args.device
    mod.CFG.batch_size = args.batch_size
    mod.CFG.num_workers = args.num_workers



def build_val_loader(mod, ckpt: Dict[str, Any], args: argparse.Namespace):
    maybe_set_cfg_paths(mod, ckpt, args)

    projection_cache = None
    agent3d_cache = None
    bev_cache = None

    use_any_visual = (
        bool(mod.CFG.use_learned_camera_bev)
        or bool(mod.CFG.use_global_bev_token)
        or bool(mod.CFG.use_agent_aligned_visual)
        or bool(mod.CFG.use_map_aligned_visual)
        or bool(mod.CFG.use_corridor_visual_token)
        or bool(mod.CFG.require_real_camera_cache)
    )
    if use_any_visual and mod.CFG.projection_cache_manifest and os.path.exists(mod.CFG.projection_cache_manifest):
        projection_cache = mod.load_projection_visual_cache(mod.CFG.projection_cache_manifest)

    if bool(mod.CFG.use_agent3d_tokens) and mod.CFG.agent3d_cache_manifest and os.path.exists(mod.CFG.agent3d_cache_manifest):
        agent3d_cache = mod.load_agent3d_cache(mod.CFG.agent3d_cache_manifest)

    if bool(mod.CFG.use_annotation_bev) and mod.CFG.bev_cache_manifest and os.path.exists(mod.CFG.bev_cache_manifest):
        bev_cache = mod.load_annotation_bev_cache(mod.CFG.bev_cache_manifest)

    data = mod.load_split_from_shards(
        mod.CFG.manifest_path,
        mod.CFG.shard_dir,
        split="val",
        projection_cache=projection_cache,
        agent3d_cache=agent3d_cache,
        bev_cache=bev_cache,
    )

    (
        X_val, A_val, A_valid_val, A3_val, A3_valid_val, BEV_val, BEV_valid_val,
        M_val, M_valid_val, y_val_cls, y_val_traj, map_y_ref_val, map_ref_valid_val,
        intent_val, C_val, C_valid_val,
        PA_val, PA_valid_val, PM_val, PM_valid_val, PC_val, PC_valid_val,
    ) = data

    scaler = mod.StandardScalerNP(skip_indices=[14, 15])
    scaler_state = ckpt.get("scaler", None)
    if isinstance(scaler_state, dict) and scaler_state.get("mean") is not None and scaler_state.get("std") is not None:
        scaler.mean = np.asarray(scaler_state["mean"], dtype=np.float32)
        scaler.std = np.asarray(scaler_state["std"], dtype=np.float32)
    else:
        raise RuntimeError("Checkpoint scaler missing. Please use a checkpoint that stores the scaler.")

    X_val = scaler.transform(X_val)

    ds = mod.JointTokenDataset(
        X_val, A_val, A_valid_val, A3_val, A3_valid_val, BEV_val, BEV_valid_val,
        M_val, M_valid_val, y_val_cls, y_val_traj, map_y_ref_val, map_ref_valid_val,
        intent_val, C_val, C_valid_val, PA_val, PA_valid_val, PM_val, PM_valid_val,
        PC_val, PC_valid_val,
    )
    loader = mod.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(args.device == "cuda"))
    return loader



def load_model(mod, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base = mod.build_base_model().to(device)
    base_state = ckpt.get("base_model_state_dict", None)
    full_state = ckpt.get("model_state_dict", None)
    if base_state is None or full_state is None:
        raise RuntimeError("Checkpoint must contain both base_model_state_dict and model_state_dict.")
    base.load_state_dict(base_state, strict=True)
    model = mod.ScoreOnlyCalibrator(
        base,
        quality_dim=mod.CFG.score_quality_dim,
        hidden_dim=mod.CFG.score_refiner_hidden_dim,
        dropout=mod.CFG.score_refiner_dropout,
    ).to(device)
    model.load_state_dict(full_state, strict=True)
    model.eval()
    return ckpt, model


@torch.no_grad()
def save_prediction_debug_npz(mod, model, loader, device: str, save_path: str, max_samples: int = 512):
    model.eval()
    all_traj_pred = []
    all_traj_gt = []
    all_scores = []
    all_base_scores = []
    all_cls_logits = []
    all_y_cls = []

    count = 0
    for batch in loader:
        (hist, agents, agent_valid, agent3d, agent3d_valid,
         bev_grid, bev_valid, maps, map_valid,
         y_cls, y_traj, map_y_ref, map_ref_valid,
         intent_target, camera_feat, camera_valid,
         proj_agent_feat, proj_agent_valid,
         proj_map_feat, proj_map_valid,
         proj_corridor_feat, proj_corridor_valid) = mod.unpack_batch(batch, device)

        cls_logits, traj_pred, scores, intent_pred, base_scores, quality_feat = model(
            hist, agents, agent_valid, agent3d, agent3d_valid,
            bev_grid, bev_valid, maps, map_valid,
            camera_feat, camera_valid,
            proj_agent_feat, proj_agent_valid,
            proj_map_feat, proj_map_valid,
            proj_corridor_feat, proj_corridor_valid,
        )

        all_traj_pred.append((traj_pred * mod.CFG.traj_scale).cpu().numpy())
        all_traj_gt.append((y_traj * mod.CFG.traj_scale).cpu().numpy())
        all_scores.append(scores.cpu().numpy())
        all_base_scores.append(base_scores.cpu().numpy())
        all_cls_logits.append(cls_logits.cpu().numpy())
        all_y_cls.append(y_cls.cpu().numpy())

        count += traj_pred.shape[0]
        if count >= max_samples:
            break

    traj_pred = np.concatenate(all_traj_pred, axis=0)[:max_samples]
    traj_gt = np.concatenate(all_traj_gt, axis=0)[:max_samples]
    scores = np.concatenate(all_scores, axis=0)[:max_samples]
    base_scores = np.concatenate(all_base_scores, axis=0)[:max_samples]
    cls_logits = np.concatenate(all_cls_logits, axis=0)[:max_samples]
    y_cls = np.concatenate(all_y_cls, axis=0)[:max_samples]

    np.savez_compressed(
        save_path,
        traj_pred=traj_pred,
        traj_gt=traj_gt,
        scores=scores,
        base_scores=base_scores,
        cls_logits=cls_logits,
        y_cls=y_cls,
    )
    print(f"[vis] prediction debug saved to: {save_path}")


@torch.no_grad()
def save_kmode_trajectory_visualizations(mod, model, loader, device: str, save_dir: str, max_samples: int = 48):
    ensure_dir(save_dir)
    model.eval()
    saved = 0

    for batch in loader:
        (hist, agents, agent_valid, agent3d, agent3d_valid,
         bev_grid, bev_valid, maps, map_valid,
         y_cls, y_traj, map_y_ref, map_ref_valid,
         intent_target, camera_feat, camera_valid,
         proj_agent_feat, proj_agent_valid,
         proj_map_feat, proj_map_valid,
         proj_corridor_feat, proj_corridor_valid) = mod.unpack_batch(batch, device)

        cls_logits, traj_pred, scores, intent_pred, base_scores, quality_feat = model(
            hist, agents, agent_valid, agent3d, agent3d_valid,
            bev_grid, bev_valid, maps, map_valid,
            camera_feat, camera_valid,
            proj_agent_feat, proj_agent_valid,
            proj_map_feat, proj_map_valid,
            proj_corridor_feat, proj_corridor_valid,
        )

        traj_m = (traj_pred * mod.CFG.traj_scale).cpu().numpy()
        gt_m = (y_traj * mod.CFG.traj_scale).cpu().numpy()
        score_np = scores.cpu().numpy()
        base_score_np = base_scores.cpu().numpy()
        cls_prob = torch.softmax(cls_logits, dim=-1).cpu().numpy()
        y_cls_np = y_cls.cpu().numpy()

        ade_per_mode = np.linalg.norm(traj_m - gt_m[:, None, :, :], axis=-1).mean(axis=-1)
        selected_idx = score_np.argmax(axis=1)
        base_selected_idx = base_score_np.argmax(axis=1)
        oracle_idx = ade_per_mode.argmin(axis=1)

        for i in range(traj_m.shape[0]):
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111)

            for k in range(traj_m.shape[1]):
                ax.plot(
                    traj_m[i, k, :, 0],
                    traj_m[i, k, :, 1],
                    marker="x",
                    linewidth=1.0,
                    alpha=0.35,
                    label=f"mode{k} ade={ade_per_mode[i, k]:.2f} score={score_np[i, k]:.2f}",
                )

            ax.plot(gt_m[i, :, 0], gt_m[i, :, 1], marker="o", linewidth=2.8, label="GT")
            ax.plot(traj_m[i, selected_idx[i], :, 0], traj_m[i, selected_idx[i], :, 1],
                    marker="s", linewidth=2.4, label=f"Selected {selected_idx[i]}")
            ax.plot(traj_m[i, oracle_idx[i], :, 0], traj_m[i, oracle_idx[i], :, 1],
                    marker="^", linewidth=2.2, label=f"Oracle {oracle_idx[i]}")
            ax.plot(traj_m[i, base_selected_idx[i], :, 0], traj_m[i, base_selected_idx[i], :, 1],
                    marker="d", linewidth=1.8, linestyle="--", label=f"BaseSel {base_selected_idx[i]}")
            ax.scatter([0.0], [0.0], marker="*", s=120, label="Ego")

            pred_label = mod.ID_TO_CLASS[int(np.argmax(cls_prob[i]))] if hasattr(mod, "ID_TO_CLASS") else str(int(np.argmax(cls_prob[i])))
            true_label = mod.ID_TO_CLASS[int(y_cls_np[i])] if hasattr(mod, "ID_TO_CLASS") else str(int(y_cls_np[i]))
            ax.set_title(
                f"sample_{saved:04d} | true={true_label} pred={pred_label} | "
                f"selADE={ade_per_mode[i, selected_idx[i]]:.3f} orcADE={ade_per_mode[i, oracle_idx[i]]:.3f}"
            )
            ax.set_xlabel("x_local (m)")
            ax.set_ylabel("y_local (m)")
            ax.grid(True)
            ax.axis("equal")
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"sample_{saved:04d}.png"), dpi=160)
            plt.close(fig)

            saved += 1
            if saved >= max_samples:
                print(f"[vis] saved K-mode trajectory visualizations to: {save_dir}")
                return

    print(f"[vis] saved K-mode trajectory visualizations to: {save_dir}")


@torch.no_grad()
def save_bev_visualizations(mod, model, loader, device: str, save_dir: str, max_samples: int = 24):
    ensure_dir(save_dir)
    model.eval()
    saved = 0

    for batch in loader:
        (hist, agents, agent_valid, agent3d, agent3d_valid,
         bev_grid, bev_valid, maps, map_valid,
         y_cls, y_traj, map_y_ref, map_ref_valid,
         intent_target, camera_feat, camera_valid,
         proj_agent_feat, proj_agent_valid,
         proj_map_feat, proj_map_valid,
         proj_corridor_feat, proj_corridor_valid) = mod.unpack_batch(batch, device)

        cls_logits, traj_pred, scores, intent_pred = model.base(
            hist, agents, agent_valid, agent3d, agent3d_valid,
            bev_grid, bev_valid, maps, map_valid,
            camera_feat, camera_valid,
            proj_agent_feat, proj_agent_valid,
            proj_map_feat, proj_map_valid,
            proj_corridor_feat, proj_corridor_valid,
        )
        # learned BEV from the base planner perception branch
        _, _, _, _, learned_bev = model.base(
            hist, agents, agent_valid, agent3d, agent3d_valid,
            bev_grid, bev_valid, maps, map_valid,
            camera_feat, camera_valid,
            proj_agent_feat, proj_agent_valid,
            proj_map_feat, proj_map_valid,
            proj_corridor_feat, proj_corridor_valid,
            return_bev_aux=True,
        )

        teacher = bev_grid.detach().cpu().numpy()
        learned = learned_bev.detach().cpu().numpy()
        valid = bev_valid.detach().cpu().numpy().reshape(-1)

        for i in range(teacher.shape[0]):
            if saved >= max_samples:
                print(f"[vis] saved BEV visualizations to: {save_dir}")
                return
            if valid[i] <= 0.5:
                continue

            t = teacher[i]
            l = learned[i]
            err = np.abs(t - l)

            t_max = t.max(axis=0)
            l_max = l.max(axis=0)
            e_mean = err.mean(axis=0)

            num_show_channels = min(3, t.shape[0])
            fig = plt.figure(figsize=(14, 8))

            ax = fig.add_subplot(2, 3, 1)
            im = ax.imshow(t_max, origin="lower")
            ax.set_title("Teacher BEV max-over-ch")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax = fig.add_subplot(2, 3, 2)
            im = ax.imshow(l_max, origin="lower")
            ax.set_title("Learned BEV max-over-ch")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax = fig.add_subplot(2, 3, 3)
            im = ax.imshow(e_mean, origin="lower")
            ax.set_title("Mean Abs Error")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for c in range(num_show_channels):
                ax = fig.add_subplot(2, 3, 4 + c)
                im = ax.imshow(np.concatenate([t[c], l[c], err[c]], axis=1), origin="lower")
                ax.set_title(f"ch{c}: teacher | learned | abs err")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.suptitle(f"BEV sample_{saved:04d}")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"bev_{saved:04d}.png"), dpi=160)
            plt.close(fig)
            saved += 1

    print(f"[vis] saved BEV visualizations to: {save_dir}")



def save_confusion_matrix(metrics: Dict[str, Any], out_path: str, class_names: Optional[Sequence[str]] = None):
    cm = np.asarray(metrics.get("confusion_matrix", []), dtype=np.float32)
    if cm.size == 0:
        return
    if class_names is None:
        class_names = list(metrics.get("config", {}).get("target_classes", [str(i) for i in range(cm.shape[0])]))

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[vis] confusion matrix saved to: {out_path}")



def save_history_curves(metrics: Dict[str, Any], out_dir: str):
    history = metrics.get("history", None)
    if not isinstance(history, dict) or len(history) == 0:
        return
    ensure_dir(out_dir)

    curve_specs = [
        ("train_loss", "val_loss", "loss_curve.png", "Loss"),
        ("val_acc", "val_macro_f1", "cls_curve.png", "Classification"),
        ("val_ADE_selected", "val_FDE_selected", "traj_curve.png", "Trajectory Selected"),
        ("val_ADE_oracle", "val_FDE_oracle", "traj_oracle_curve.png", "Trajectory Oracle"),
        ("val_ADE_gap", None, "gap_curve.png", "ADE Gap"),
        ("val_score_hit_rate", None, "score_hit_curve.png", "Score Hit Rate"),
    ]

    for key1, key2, fname, title in curve_specs:
        vals1 = history.get(key1, None)
        vals2 = history.get(key2, None) if key2 else None
        if vals1 is None:
            continue
        epochs = np.arange(1, len(vals1) + 1)
        fig = plt.figure(figsize=(7, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(epochs, vals1, marker="o", label=key1)
        if vals2 is not None:
            ax.plot(epochs, vals2, marker="x", label=key2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close(fig)
    print(f"[vis] history curves saved to: {out_dir}")



def _extract_metric_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    if key in metrics:
        try:
            return float(metrics[key])
        except Exception:
            return None
    aliases = {
        "acc": ["final_val_acc"],
        "macro_f1": ["final_val_macro_f1"],
        "ade_sel": ["final_val_ADE_selected"],
        "fde_sel": ["final_val_FDE_selected"],
        "ade_orc": ["final_val_ADE_oracle"],
        "fde_orc": ["final_val_FDE_oracle"],
        "ade_gap": ["final_val_ADE_gap"],
        "score_hit": ["final_score_hit_rate"],
    }
    for alt in aliases.get(key, []):
        if alt in metrics:
            try:
                return float(metrics[alt])
            except Exception:
                return None
    return None



def save_comparison_chart(metric_paths: Sequence[str], labels: Optional[Sequence[str]], out_path: str):
    if not metric_paths:
        return
    if labels is None or len(labels) != len(metric_paths):
        labels = [Path(p).parent.name or Path(p).stem for p in metric_paths]

    keys = ["acc", "macro_f1", "ade_sel", "fde_sel", "ade_orc", "fde_orc", "ade_gap", "score_hit"]
    display_names = ["Acc", "MacroF1", "ADE_sel", "FDE_sel", "ADE_orc", "FDE_orc", "ADE_gap", "ScoreHit"]

    loaded = [load_json(p) for p in metric_paths]
    x = np.arange(len(keys))
    width = 0.8 / max(1, len(loaded))

    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(111)
    for i, (lab, metrics) in enumerate(zip(labels, loaded)):
        vals = [(_extract_metric_value(metrics, k) if _extract_metric_value(metrics, k) is not None else 0.0) for k in keys]
        ax.bar(x + (i - (len(loaded) - 1) / 2) * width, vals, width=width, label=lab)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=20, ha="right")
    ax.set_title("Version Comparison")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[vis] comparison chart saved to: {out_path}")



def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    metrics = load_json(args.metrics) if args.metrics and os.path.exists(args.metrics) else {}
    mod = dynamic_import(args.train_script)

    ckpt, model = load_model(mod, args.checkpoint, args.device)
    loader = build_val_loader(mod, ckpt, args)

    if args.save_debug_npz:
        save_prediction_debug_npz(
            mod, model, loader, args.device,
            os.path.join(args.output_dir, "predictions_debug.npz"),
            max_samples=args.max_debug_samples,
        )

    traj_dir = os.path.join(args.output_dir, "traj_k_modes")
    save_kmode_trajectory_visualizations(mod, model, loader, args.device, traj_dir, max_samples=args.traj_samples)

    bev_dir = os.path.join(args.output_dir, "bev_vis")
    save_bev_visualizations(mod, model, loader, args.device, bev_dir, max_samples=args.bev_samples)

    if metrics:
        save_confusion_matrix(metrics, os.path.join(args.output_dir, "confusion_matrix.png"))
        save_history_curves(metrics, os.path.join(args.output_dir, "history_curves"))

    if args.compare_metrics:
        save_comparison_chart(args.compare_metrics, args.compare_labels, os.path.join(args.output_dir, "version_comparison.png"))

    summary = {
        "checkpoint": args.checkpoint,
        "metrics": args.metrics,
        "output_dir": args.output_dir,
        "traj_dir": traj_dir,
        "bev_dir": bev_dir,
        "compare_metrics": args.compare_metrics,
    }
    with open(os.path.join(args.output_dir, "visualization_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[vis] done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
