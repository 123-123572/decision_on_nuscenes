from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import CFG, CLASS_TO_ID
from .model import MapAgentEgoPlanner

class ScoreOnlyCalibrator(nn.Module):
    """
    v18.5: freeze the raw-BEV + diffusion-residual base generator, then train only
    an oracle-ADE-supervised score calibrator.

    Difference from v18.4:
      v18.4 used many cost-aware proxy features. ScoreHit could improve while ADE got worse.
      v18.5 makes the selector objective direct:

          target_metric_k = ADE_k + SCORE_FDE_WEIGHT * FDE_k
          target_prob     = softmax(-target_metric_k / SCORE_TARGET_TEMP)

    The selector still gets compact trajectory / map / interaction features, but these are
    only input features. The training target is not a hand-crafted cost proxy anymore.

    The calibrator predicts a conservative delta over frozen base scores:
        refined_score = base_score + score_delta_scale * delta
    """
    def __init__(self, base_model: MapAgentEgoPlanner, quality_dim: int = 16, hidden_dim: int = 128, dropout: float = 0.10):
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
        # Safe start: epoch 0 exactly matches frozen base scores.
        nn.init.zeros_(self.score_refiner[-1].weight)
        nn.init.zeros_(self.score_refiner[-1].bias)

        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

    @torch.no_grad()
    def compute_score_quality_features(
        self,
        traj: torch.Tensor,
        agents: torch.Tensor,
        agent_valid: torch.Tensor,
        maps: torch.Tensor,
        map_valid: torch.Tensor,
        bev_grid: torch.Tensor,
        bev_valid: torch.Tensor,
        cls_prob: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compact per-mode selector features, normalized and clamped.

        Returns [B,K,16]:
          01 base final_x / traj_scale
          02 abs(final_y) / traj_scale
          03 final displacement / traj_scale
          04 path length / traj_scale
          05 mean speed / speed_scale
          06 terminal speed / speed_scale
          07 mean acceleration / acc_scale
          08 mean jerk / jerk_scale
          09 straightness ratio
          10 curvature / pi
          11 map mean distance / map_dist_scale
          12 map final distance / map_dist_scale
          13 min agent distance / agent_dist_scale
          14 collision penalty normalized
          15 endpoint diversity / traj_scale
          16 STOP-aware terminal speed penalty

        Important: raw BEV is already fused into the generator. v18.5 deliberately does
        not sample BEV again inside the selector, because v18.4 showed proxy features can
        bias the ranking away from actual ADE/FDE.
        """
        eps = 1e-6
        B, K, T, _ = traj.shape
        device = traj.device
        traj_m = traj.detach() * CFG.traj_scale

        final_xy = traj_m[:, :, -1, :]
        final_x_norm = final_xy[..., 0] / CFG.traj_scale
        abs_final_y_norm = final_xy[..., 1].abs() / CFG.traj_scale
        final_disp = torch.norm(final_xy, dim=-1)
        final_disp_norm = final_disp / CFG.traj_scale

        if T >= 2:
            delta = traj_m[:, :, 1:, :] - traj_m[:, :, :-1, :]
            step_dist = torch.norm(delta, dim=-1)
            path_len = step_dist.sum(dim=-1)
            mean_speed = step_dist.mean(dim=-1) / CFG.dt
            terminal_speed = step_dist[:, :, -1] / CFG.dt
            heading = torch.atan2(delta[..., 1], delta[..., 0].clamp_min(1e-3))
            if heading.size(-1) >= 2:
                d_heading = heading[:, :, 1:] - heading[:, :, :-1]
                d_heading = torch.atan2(torch.sin(d_heading), torch.cos(d_heading))
                curvature = d_heading.abs().mean(dim=-1)
            else:
                curvature = torch.zeros(B, K, device=device, dtype=traj_m.dtype)
        else:
            path_len = torch.zeros(B, K, device=device, dtype=traj_m.dtype)
            mean_speed = torch.zeros(B, K, device=device, dtype=traj_m.dtype)
            terminal_speed = torch.zeros(B, K, device=device, dtype=traj_m.dtype)
            curvature = torch.zeros(B, K, device=device, dtype=traj_m.dtype)

        path_len_norm = path_len / CFG.traj_scale
        mean_speed_norm = mean_speed / CFG.intent_speed_scale
        terminal_speed_norm = terminal_speed / CFG.intent_speed_scale
        curvature_norm = curvature / math.pi

        if T >= 3:
            vel = (traj_m[:, :, 1:, :] - traj_m[:, :, :-1, :]) / CFG.dt
            acc = (vel[:, :, 1:, :] - vel[:, :, :-1, :]) / CFG.dt
            mean_acc = torch.norm(acc, dim=-1).mean(dim=-1)
        else:
            acc = None
            mean_acc = torch.zeros(B, K, device=device, dtype=traj_m.dtype)
        mean_acc_norm = mean_acc / CFG.score_acc_scale

        if acc is not None and acc.size(2) >= 2:
            jerk = (acc[:, :, 1:, :] - acc[:, :, :-1, :]) / CFG.dt
            mean_jerk = torch.norm(jerk, dim=-1).mean(dim=-1)
        else:
            mean_jerk = torch.zeros(B, K, device=device, dtype=traj_m.dtype)
        mean_jerk_norm = mean_jerk / CFG.score_jerk_scale

        straightness = (final_disp / (path_len + eps)).clamp(0.0, 1.0)

        if K > 1:
            end_pair_dist = torch.cdist(final_xy, final_xy)
            eye = torch.eye(K, device=device, dtype=torch.bool).view(1, K, K)
            end_pair_dist = end_pair_dist.masked_fill(eye, 0.0)
            endpoint_diversity = end_pair_dist.sum(dim=-1) / max(K - 1, 1)
        else:
            endpoint_diversity = torch.zeros(B, K, device=device, dtype=traj_m.dtype)
        endpoint_diversity_norm = endpoint_diversity / CFG.traj_scale

        # Agent interaction features.
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

        # Map nearest distance features.
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

        stop_id = CLASS_TO_ID.get("STOP", 2)
        stop_prob = cls_prob[:, stop_id].view(B, 1).expand(B, K)
        stop_speed_penalty = stop_prob * terminal_speed_norm

        quality = torch.stack([
            final_x_norm,
            abs_final_y_norm,
            final_disp_norm,
            path_len_norm,
            mean_speed_norm,
            terminal_speed_norm,
            mean_acc_norm,
            mean_jerk_norm,
            straightness,
            curvature_norm,
            map_mean_dist_norm,
            map_final_dist_norm,
            min_agent_dist_norm,
            collision_penalty_norm,
            endpoint_diversity_norm,
            stop_speed_penalty,
        ], dim=-1)

        if quality.size(-1) != self.quality_dim:
            if quality.size(-1) > self.quality_dim:
                quality = quality[..., :self.quality_dim]
            else:
                pad = quality.new_zeros(*quality.shape[:-1], self.quality_dim - quality.size(-1))
                quality = torch.cat([quality, pad], dim=-1)

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
            cls_prob = torch.softmax(cls_logits, dim=-1)
            quality_feat = self.compute_score_quality_features(traj, agents, agent_valid, maps, map_valid, bev_grid, bev_valid, cls_prob)
            cls_expand = cls_prob.unsqueeze(1).expand(-1, traj.size(1), -1)
            intent_expand = intent_pred.unsqueeze(1).expand(-1, traj.size(1), -1)
            base_score_feat = base_scores.unsqueeze(-1)
            refiner_input = torch.cat([base_score_feat, quality_feat, cls_expand, intent_expand], dim=-1)

        delta = self.score_refiner(refiner_input).squeeze(-1)
        scores = base_scores + CFG.score_delta_scale * delta
        return cls_logits, traj, scores, intent_pred, base_scores, quality_feat

