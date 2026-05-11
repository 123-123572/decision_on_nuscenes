from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG, CLASS_TO_ID

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


def gather_modes(traj_pred: torch.Tensor, mode_idx: torch.Tensor) -> torch.Tensor:
    B = traj_pred.size(0)
    return traj_pred[torch.arange(B, device=traj_pred.device), mode_idx]


def compute_soft_score_target(traj_pred: torch.Tensor, y_traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build direct oracle-ADE/FDE soft labels for selector training.

    target_metric_k = ADE_k + SCORE_FDE_WEIGHT * FDE_k

    This is intentionally more direct than v18.4's cost-aware proxy target.
    """
    y_expand = y_traj.unsqueeze(1).expand_as(traj_pred)
    diff = (traj_pred - y_expand) * CFG.traj_scale
    point_dist = torch.norm(diff, dim=-1)       # [B,K,T]
    ade = point_dist.mean(dim=-1)               # [B,K]
    fde = point_dist[:, :, -1]                  # [B,K]
    with torch.no_grad():
        target_metric = ade + float(CFG.score_fde_weight) * fde
        target_prob = torch.softmax(-target_metric / max(float(CFG.score_target_temp), 1e-4), dim=1)
        if CFG.score_label_smoothing > 0.0:
            uniform = torch.full_like(target_prob, 1.0 / traj_pred.size(1))
            target_prob = (1.0 - CFG.score_label_smoothing) * target_prob + CFG.score_label_smoothing * uniform
        best_idx = target_metric.argmin(dim=1)
    return target_prob, target_metric.detach(), best_idx


def score_calibration_losses(traj_pred: torch.Tensor, y_traj: torch.Tensor, scores: torch.Tensor, base_scores: torch.Tensor):
    target_prob, target_metric, best_idx = compute_soft_score_target(traj_pred, y_traj)
    log_scores = torch.log_softmax(scores, dim=1)
    loss_soft = -(target_prob * log_scores).sum(dim=1).mean()

    # Pairwise oracle-ADE/FDE ranking:
    # if metric_i is meaningfully better than metric_j, score_i should exceed score_j by margin.
    metric_i = target_metric.unsqueeze(2)  # [B,K,1]
    metric_j = target_metric.unsqueeze(1)  # [B,1,K]
    score_i = scores.unsqueeze(2)
    score_j = scores.unsqueeze(1)
    better_mask = (metric_i + float(CFG.score_pair_min_delta)) < metric_j
    pair_loss = torch.relu(float(CFG.score_rank_margin) - (score_i - score_j))
    if better_mask.any():
        loss_rank = (pair_loss * better_mask.float()).sum() / better_mask.float().sum().clamp_min(1.0)
    else:
        loss_rank = scores.new_tensor(0.0)

    # Conservative delta regularization. Keeps the selector from destroying a strong base score.
    loss_reg = torch.nn.functional.mse_loss(scores, base_scores.detach())
    loss = CFG.lambda_score_soft * loss_soft + CFG.lambda_score_rank * loss_rank + CFG.lambda_score_reg * loss_reg
    pred_idx = scores.argmax(dim=1)
    base_pred_idx = base_scores.argmax(dim=1)
    return loss, loss_soft, loss_rank, loss_reg, best_idx, pred_idx, base_pred_idx, target_metric


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

