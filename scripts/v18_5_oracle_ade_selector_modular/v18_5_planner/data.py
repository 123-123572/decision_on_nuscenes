from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CFG, CLASS_TO_ID
from .utils import load_json, load_pickle, normalize_label, resize_bev_grid_np

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


def load_image_bev_cache(cache_manifest_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load v16.x image-BEV cache or legacy annotation-BEV cache.

    Supports both:
      v16.x: image_bev_feat / image_bev_valid / sample_tokens
      v11.x: bev_grid / bev_valid / sample_tokens

    sample_token -> (bev_grid [C,H,W], bev_valid [1])
    """
    if not cache_manifest_path or not os.path.exists(cache_manifest_path):
        raise FileNotFoundError(
            f"BEV cache manifest not found: {cache_manifest_path}\n"
            "For v16.4.1, set IMAGE_BEV_CACHE_MANIFEST to image_bev_cache_manifest_v16_4_1.json."
        )
    manifest = load_json(cache_manifest_path)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    cache_files = manifest.get("cache_files", [])
    if not cache_files:
        cache_files = manifest.get("train_cache_shards", []) + manifest.get("val_cache_shards", [])
    bev_key_default = manifest.get("bev_key", None)
    clean_bev_key_default = manifest.get("clean_bev_key", "image_bev_feat_clean")
    valid_key_default = manifest.get("valid_key", None)
    sample_token_key = manifest.get("sample_token_key", "sample_tokens")
    prefer_cleaned = bool(CFG.use_cleaned_image_bev)
    print(f"BEV cache key preference: {'cleaned' if prefer_cleaned else 'raw'}")

    for cache_file in cache_files:
        if not os.path.exists(cache_file):
            print(f"[WARN] missing BEV cache file: {cache_file}")
            continue
        data = np.load(cache_file, allow_pickle=True)
        keys = set(data.files)
        token_key = sample_token_key if sample_token_key in keys else "sample_tokens"
        if prefer_cleaned and clean_bev_key_default in keys:
            bev_key = clean_bev_key_default
        elif bev_key_default is None:
            if "image_bev_feat" in keys:
                bev_key = "image_bev_feat"
            elif "bev_grid" in keys:
                bev_key = "bev_grid"
            else:
                raise KeyError(f"No BEV grid key found in {cache_file}. keys={sorted(keys)}")
        else:
            bev_key = bev_key_default
        if valid_key_default is None:
            if "image_bev_valid" in keys:
                valid_key = "image_bev_valid"
            elif "bev_valid" in keys:
                valid_key = "bev_valid"
            else:
                valid_key = None
        else:
            valid_key = valid_key_default

        tokens = [str(x) for x in data[token_key]]
        grid = data[bev_key].astype(np.float32)
        if valid_key is not None and valid_key in keys:
            valid = data[valid_key].astype(np.float32)
        else:
            valid = np.ones((len(tokens), 1), dtype=np.float32)
        for tok, g, v in zip(tokens, grid, valid):
            g = np.asarray(g, dtype=np.float32)
            if g.shape != (CFG.bev_channels, CFG.bev_h, CFG.bev_w):
                g = resize_bev_grid_np(g, CFG.bev_channels, CFG.bev_h, CFG.bev_w)
            out[tok] = (g, np.asarray(v, dtype=np.float32).reshape(-1)[:1])
    return out


def load_annotation_bev_cache(cache_manifest_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    # Backward-compatible alias.
    return load_image_bev_cache(cache_manifest_path)


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

