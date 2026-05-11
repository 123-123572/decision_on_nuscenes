from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG
from .losses import build_mode_targets

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


class DiffusionResidualDecoder(nn.Module):
    """Conditional diffusion residual decoder for v17 trajectory generation.

    It keeps the old K-mode MLP trajectory as base_traj and learns a denoising
    model over residuals:
        final_traj = base_traj + diffusion_residual

    This is deliberately small enough for an 8GB laptop GPU. It is not a huge
    image diffusion model; it is a conditional trajectory denoiser.
    """
    def __init__(self, hidden_dim: int, num_classes: int, future_steps: int, future_dim: int, dropout: float = 0.10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.future_steps = future_steps
        self.future_dim = future_dim
        self.traj_dim = future_steps * future_dim
        self.time_dim = hidden_dim

        steps = max(2, int(CFG.diffusion_steps))
        betas = torch.linspace(1e-4, 2e-2, steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        in_dim = self.traj_dim + hidden_dim + num_classes + self.traj_dim + self.time_dim
        h = int(CFG.diffusion_hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, self.traj_dim),
        )

    def timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        # t: any shape, int or float. Return [..., dim]
        half = dim // 2
        dtype = torch.float32
        device = t.device
        freq = torch.exp(
            torch.arange(half, device=device, dtype=dtype) * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t.to(dtype).unsqueeze(-1) * freq.view(*([1] * t.ndim), half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def predict_noise(self, noisy_residual: torch.Tensor, t: torch.Tensor, mode_feat: torch.Tensor,
                      cls_prob: torch.Tensor, base_traj: torch.Tensor) -> torch.Tensor:
        B, K, T, D = noisy_residual.shape
        if t.ndim == 1:
            t = t.view(B, 1).expand(B, K)
        if t.ndim == 0:
            t = t.view(1, 1).expand(B, K)
        t = t.to(noisy_residual.device)
        cls_expand = cls_prob.unsqueeze(1).expand(B, K, -1)
        time_emb = self.timestep_embedding(t, self.time_dim).to(dtype=noisy_residual.dtype)
        x = torch.cat([
            noisy_residual.reshape(B, K, -1),
            mode_feat,
            cls_expand,
            base_traj.detach().reshape(B, K, -1),
            time_emb,
        ], dim=-1)
        eps = self.net(x).view(B, K, T, D)
        return eps

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].to(device=x0.device, dtype=x0.dtype).view(x0.size(0), x0.size(1), 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar).clamp_min(1e-8) * noise

    def training_losses(self, base_traj: torch.Tensor, mode_feat: torch.Tensor, cls_prob: torch.Tensor,
                        y_traj: torch.Tensor, y_cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mode_targets = build_mode_targets(y_traj, y_cls).detach()
        base_detached = base_traj.detach()
        residual_target = mode_targets - base_detached
        clip = float(CFG.diffusion_residual_clip_m) / max(float(CFG.traj_scale), 1e-6)
        residual_target = residual_target.clamp(-clip, clip)

        B, K, _, _ = residual_target.shape
        t = torch.randint(0, self.alpha_bars.numel(), (B, K), device=residual_target.device)
        noise = torch.randn_like(residual_target)
        noisy = self.q_sample(residual_target, t, noise)
        eps_pred = self.predict_noise(noisy, t, mode_feat, cls_prob, base_detached)
        loss_noise = F.mse_loss(eps_pred, noise)

        alpha_bar = self.alpha_bars[t].to(device=residual_target.device, dtype=residual_target.dtype).view(B, K, 1, 1)
        x0_pred = (noisy - torch.sqrt(1.0 - alpha_bar).clamp_min(1e-8) * eps_pred) / torch.sqrt(alpha_bar).clamp_min(1e-8)
        x0_pred = x0_pred.clamp(-clip, clip)
        refined = base_detached + x0_pred
        loss_recon = F.smooth_l1_loss(refined, mode_targets)
        return loss_noise, loss_recon

    def sample(self, base_traj: torch.Tensor, mode_feat: torch.Tensor, cls_prob: torch.Tensor) -> torch.Tensor:
        B, K, T, D = base_traj.shape
        device, dtype = base_traj.device, base_traj.dtype
        steps_total = self.alpha_bars.numel()
        sample_steps = max(1, min(int(CFG.diffusion_sample_steps), steps_total))
        ts = torch.linspace(steps_total - 1, 0, sample_steps, device=device).long()

        deterministic_start = ((not self.training) and bool(CFG.diffusion_eval_deterministic)) or (self.training and bool(CFG.diffusion_train_deterministic))
        if deterministic_start:
            x = torch.zeros_like(base_traj)
        else:
            x = torch.randn_like(base_traj) * float(CFG.diffusion_sample_noise_scale)

        clip = float(CFG.diffusion_residual_clip_m) / max(float(CFG.traj_scale), 1e-6)
        for i, t_scalar in enumerate(ts):
            t = t_scalar.view(1, 1).expand(B, K)
            eps = self.predict_noise(x, t, mode_feat, cls_prob, base_traj)
            ab_t = self.alpha_bars[t_scalar].to(device=device, dtype=dtype)
            x0 = (x - torch.sqrt(1.0 - ab_t).clamp_min(1e-8) * eps) / torch.sqrt(ab_t).clamp_min(1e-8)
            x0 = x0.clamp(-clip, clip)
            if i == len(ts) - 1:
                x = x0
            else:
                t_next = ts[i + 1]
                ab_next = self.alpha_bars[t_next].to(device=device, dtype=dtype)
                # DDIM-style deterministic update. No extra random noise in the reverse chain.
                x = torch.sqrt(ab_next) * x0 + torch.sqrt(1.0 - ab_next).clamp_min(1e-8) * eps
                x = x.clamp(-clip, clip)
        return base_traj + x


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

        # v16.4 high-res depth-supervised image-BEV adapters. Input is sampled raw BEV channel vector [C]
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

        # v9.2/v16.4 base: shared mode queries + behavior-conditioned mode priors.
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

        # v17 keeps this as a stable base trajectory generator. The diffusion decoder
        # learns residual corrections on top of it.
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, future_steps * future_dim),
        )
        self.diffusion_residual_decoder = DiffusionResidualDecoder(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            future_steps=future_steps,
            future_dim=future_dim,
            dropout=float(CFG.diffusion_dropout),
        )
        self._last_v17_aux: Dict[str, torch.Tensor] = {}
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
                proj_corridor_feat, proj_corridor_valid):
        B = hist.size(0)
        memory_in, key_padding_mask = self.build_tokens(
            hist, agents, agent_valid, agent3d, agent3d_valid, bev_grid, bev_valid, maps, map_valid, camera_feat, camera_valid,
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
        base_traj = self.traj_head(traj_input).view(B, self.K, self.future_steps, self.future_dim)

        if bool(CFG.use_diffusion_residual_decoder):
            traj = self.diffusion_residual_decoder.sample(base_traj, mode_feat, cls_prob)
        else:
            traj = base_traj

        self._last_v17_aux = {
            "base_traj": base_traj,
            "mode_feat": mode_feat,
            "cls_prob": cls_prob,
        }

        traj_feat = traj.detach().reshape(B, self.K, -1)
        score_input = torch.cat([mode_feat, traj_feat], dim=-1)
        scores = self.score_head(score_input).squeeze(-1)
        return cls_logits, traj, scores, intent_pred

    def diffusion_training_losses(self, y_traj: torch.Tensor, y_cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if (not bool(CFG.use_diffusion_residual_decoder)) or (not self._last_v17_aux):
            z = y_traj.new_tensor(0.0)
            return z, z
        return self.diffusion_residual_decoder.training_losses(
            base_traj=self._last_v17_aux["base_traj"],
            mode_feat=self._last_v17_aux["mode_feat"],
            cls_prob=self._last_v17_aux["cls_prob"],
            y_traj=y_traj,
            y_cls=y_cls,
        )

