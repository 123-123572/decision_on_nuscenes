# v18.5 Oracle-ADE Selector 模块化规划模型

本目录是 `v18.5 Oracle-ADE Selector` 自动驾驶决策规划模型的模块化版本。

原始 v18.5 版本是一个单文件完整训练脚本，包含配置、数据加载、模型结构、损失函数、训练流程、评估逻辑和 score calibration。为了便于阅读、调试、实验复现和面试讲解，这里将其拆分为多个相互独立的模块。

---

## 1. 项目定位

v18.5 面向 nuScenes v1.0-trainval 数据集，目标是构建一个决策规划联合建模框架。

模型输入多源场景信息，包括：

- Ego 历史运动状态
- 周围 Agent 结构化特征
- nuScenes 3D annotation 构建的 Agent token
- 局部地图 polyline
- 图像 BEV cache
- 可选 projection visual cache

模型统一输出：

- 驾驶行为类别
- K 条未来候选轨迹
- 每条候选轨迹的 score
- intent 辅助预测
- diffusion residual 修正后的轨迹

需要注意：该项目不是完整的 camera-to-control 端到端自动驾驶系统。感知 BEV 是离线构建好的 cache，不参与 planner 训练时的反向传播。

更准确的表述是：

> 基于离线感知缓存特征的决策规划层端到端联合建模。

---

## 2. v18.5 要解决的核心问题

v18.5 的重点不是单纯再堆一个模型，而是解决轨迹生成和轨迹选择之间的不一致。

在多模态轨迹预测中，模型会输出 K 条候选轨迹：

```text
traj_pred: [B, K, T, 2]
scores:    [B, K]
```

评估时有两个关键指标：

```text
Oracle ADE/FDE   = K 条候选轨迹中，与 GT 最接近的那条轨迹误差
Selected ADE/FDE = score head 实际选中的那条轨迹误差
```

如果：

```text
Oracle ADE/FDE 很好
Selected ADE/FDE 不好
```

说明模型其实已经生成了较好的候选轨迹，但 score selector 没有稳定选中它。

v18.5 的核心改进就是：

> 冻结 base generator，只训练轻量级 score refiner，用 oracle ADE/FDE 构造直接监督，让 selector 更稳定地选中候选轨迹中的最优轨迹。

---

## 3. 整体数据流

整体 pipeline 可以概括为：

```text
nuScenes 原始数据
    ↓
v8 Map-Agent-Ego 主数据集
    ↓ sample_token 对齐
v10 3D Agent cache
v16.4.1 Image-BEV cache
v9.5 Projection Visual cache（可选）
    ↓
JointTokenDataset
    ↓
Map-Agent-Ego-BEV Transformer Planner
    ↓
行为分类 + K 模态轨迹生成 + 轨迹评分
    ↓
Diffusion Residual Decoder
    ↓
Score-only Calibration
    ↓
Selected trajectory
```

---

## 4. 使用的预构建数据

本模型训练时不直接在线解析完整 nuScenes 原始数据，而是读取已经构建好的 shard/cache。

### 4.1 v8 Map-Agent-Ego 主数据集

默认路径示例：

```bash
/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/build_manifest_full_v8_map_agent_t6.json
/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/shards_full_v8_map_agent_t6
```

主要提供：

```text
history_features      [B, 6, 20]
agent_features        [B, 24, 10]
agent_valid           [B, 24]
map_polylines         [B, 32, 20, 5]
map_polyline_valid    [B, 32, 20]
future_xy_local       [B, 6, 2]
label_name            CRUISE / FOLLOW / STOP / TURN
map_y_ref             [B, 6]
map_ref_valid         [B, 6]
sample_token
```

v8 是主数据集，也是后续所有 cache 对齐的基准。

### 4.2 v10 3D Agent Cache

默认路径示例：

```bash
/home/ubuntu22/decision_on_nuscenes/outputs_v10_0_3d_agent_cache/agent3d_cache_manifest_v10_0.json
```

主要提供：

```text
agent3d_features      [B, 32, 26]
agent3d_valid         [B, 32]
```

它基于 nuScenes `sample_annotation` 构建，包含目标相对位置、速度、尺寸、类别、TTC 和点云观测质量等信息。

### 4.3 v16.4.1 Image-BEV Cache

默认路径示例：

```bash
/home/ubuntu22/decision_on_nuscenes/outputs_v16_4_1_highres_depthsup_bgclean_real_image_lss_bev_cache/image_bev_cache_manifest_v16_4_1.json
```

主要提供：

```text
image_bev_feat        [B, 8, 120, 90]
image_bev_feat_clean  [B, 8, 120, 90]
image_bev_valid       [B, 1]
```

v18.5 默认可以选择 raw BEV 或 cleaned BEV。该 modular 版本默认 `USE_CLEANED_IMAGE_BEV=0`，也就是使用 raw BEV。

### 4.4 v9.5 Projection Visual Cache

默认路径示例：

```bash
/home/ubuntu22/decision_on_nuscenes/outputs_v9_5_projection_visual_cache/projection_visual_cache_manifest_v9_5.json
```

主要提供：

```text
camera_feat             [B, 6, 256]
camera_valid            [B, 6]
proj_agent_feat         [B, 24, 256]
proj_agent_valid        [B, 24]
proj_map_feat           [B, 32, 256]
proj_map_valid          [B, 32]
proj_corridor_feat      [B, 1, 256]
proj_corridor_valid     [B, 1]
```

注意：projection visual 分支默认关闭。只有启用相关 visual switch 时才加载该 cache。

---

## 5. 输入输出张量

### 输入

```text
history_features      [B, 6, 20]
agent_features        [B, 24, 10]
agent_valid           [B, 24]
agent3d_features      [B, 32, 26]
agent3d_valid         [B, 32]
image_bev             [B, 8, 120, 90]
image_bev_valid       [B, 1]
map_polylines         [B, 32, 20, 5]
map_polyline_valid    [B, 32, 20]
camera_feat           [B, 6, 256]
camera_valid          [B, 6]
proj_agent_feat       [B, 24, 256]
proj_agent_valid      [B, 24]
proj_map_feat         [B, 32, 256]
proj_map_valid        [B, 32]
proj_corridor_feat    [B, 1, 256]
proj_corridor_valid   [B, 1]
```

### 监督标签

```text
y_cls                 [B]
y_traj                [B, 6, 2]
map_y_ref             [B, 6]
map_ref_valid         [B, 6]
intent_target         [B, 4]
```

其中：

```text
intent_target = [
    terminal_speed,
    total_displacement,
    lateral_displacement,
    yaw_delta
]
```

### 输出

```text
cls_logits            [B, 4]
traj_pred             [B, K=3, 6, 2]
scores                [B, K=3]
intent_pred           [B, 4]
```

---

## 6. 模型结构

v18.5 主模型是 Map-Agent-Ego-BEV Transformer Planner。

核心结构：

```text
Ego history token
Agent tokens
3D Agent tokens
Map polyline tokens
BEV-aligned tokens
Projection visual tokens（可选）
        ↓
Transformer Encoder
        ↓
Scene context
        ↓
Mode Query Transformer Decoder
        ↓
K 条候选轨迹
        ↓
Diffusion residual refinement
        ↓
Score selector
        ↓
Selected trajectory
```

---

## 7. 关键设计点

### 7.1 多源 token 融合

模型没有简单把所有输入 concat 成一个向量，而是将不同场景元素 token 化：

- ego history token
- agent token
- 3D agent token
- map polyline token
- BEV token / BEV-aligned feature
- corridor token

这样 Transformer 可以建模 ego、agent、map、BEV 之间的交互关系。

### 7.2 Object-aligned BEV Fusion

直接把整张 BEV 展平为 dense token 会带来几个问题：

- token 数量大
- 计算成本高
- 噪声多
- 很多 BEV 区域和当前规划无关

因此 v18.5 采用 object-aligned BEV fusion：

```text
在 agent / agent3d / map / corridor 的空间位置上采样局部 BEV 特征
再通过 gated residual adapter 注入对应 token
```

这样可以让 BEV 信息服务于规划对象，而不是让噪声视觉特征主导模型。

### 7.3 Diffusion Residual Decoder

这里的 diffusion 不是图像生成模型，而是条件轨迹残差扩散模型。

基础轨迹头先输出：

```text
base_traj
```

diffusion decoder 学习残差：

```text
diffusion_residual
```

最终轨迹为：

```text
final_traj = base_traj + diffusion_residual
```

这样做比从纯噪声直接生成轨迹更稳定，也更适合 8GB 显存环境。

### 7.4 Score-only Calibration

v18.5 第二阶段冻结 base generator，只训练 score refiner。

目标是缩小：

```text
ADE_gap = ADE_selected - ADE_oracle
```

score target 基于：

```text
target_metric_k = ADE_k + SCORE_FDE_WEIGHT * FDE_k
target_prob = softmax(-target_metric_k / SCORE_TARGET_TEMP)
```

这比手工 cost-aware proxy 更直接，因为监督目标就是轨迹与 GT 的真实误差。

---

## 8. 目录结构

```text
v18_5_oracle_ade_selector_modular/
├── README.md
├── run_v18_5_modular.py
└── v18_5_planner/
    ├── __init__.py
    ├── config.py
    ├── utils.py
    ├── data.py
    ├── model.py
    ├── losses.py
    ├── calibrator.py
    ├── train_eval.py
    └── main.py
```

---

## 9. 模块说明

### 9.1 `config.py`

配置中心。

负责管理：

- 随机种子
- 数据路径
- 输入维度
- 模型维度
- BEV 参数
- diffusion 参数
- loss 权重
- checkpoint 路径
- 训练开关

典型开关：

```bash
USE_AGENT3D_TOKENS=1
USE_OBJECT_ALIGNED_BEV=1
USE_DIFFUSION_RESIDUAL_DECODER=1
USE_CLEANED_IMAGE_BEV=0
RUN_BASE_TRAIN=0
```

### 9.2 `utils.py`

通用工具函数，包括：

- `set_seed`
- `ensure_dir`
- `load_json`
- `load_pickle`
- `normalize_label`
- `resize_bev_grid_np`
- `StandardScalerNP`

### 9.3 `data.py`

数据加载与对齐模块。

主要功能：

- 加载 projection visual cache
- 加载 3D agent cache
- 加载 image-BEV cache
- 从 v8 shards 加载 train/val split
- 通过 `sample_token` 对齐多源数据
- 构建 `JointTokenDataset`

这是整个 pipeline 的数据入口。

### 9.4 `model.py`

核心模型模块。

包含：

- `PositionalEncoding`
- `MLP`
- `DiffusionResidualDecoder`
- `MapAgentEgoPlanner`

其中 `MapAgentEgoPlanner` 是主规划模型。

### 9.5 `losses.py`

损失函数模块。

包括：

- `FocalLoss`
- `build_mode_targets`
- `base_multimodal_losses`
- `compute_soft_score_target`
- `compute_real_map_loss`
- `compute_diversity_loss`
- `compute_comfort_loss`
- `compute_collision_loss`
- `compute_intent_loss`
- `score_calibration_losses`
- `calc_traj_metrics`

### 9.6 `calibrator.py`

轨迹选择器校准模块。

核心类：

```python
ScoreOnlyCalibrator
```

它冻结 base planner，只训练 score refiner。

主要目的：

```text
让 score 更准确地选择 K 条候选轨迹中最接近 GT 的那条。
```

### 9.7 `train_eval.py`

训练与评估模块。

负责：

- base model 训练
- score calibration 训练
- loss 统计
- 指标计算
- checkpoint 保存
- 轨迹可视化
- classification report

### 9.8 `main.py`

完整流程编排模块。

执行顺序：

1. 设置随机种子
2. 创建保存目录
3. 加载 projection cache（如果启用）
4. 加载 3D agent cache
5. 加载 image-BEV cache
6. 加载 v8 train/val shards
7. 归一化 history feature
8. 构建 Dataset 和 DataLoader
9. 构建模型
10. 加载或训练 base generator
11. 冻结 base generator
12. 训练 score selector
13. 保存最终指标和模型

### 9.9 `run_v18_5_modular.py`

最外层入口脚本。

运行：

```bash
python run_v18_5_modular.py
```

---

## 10. 运行方式

### 10.1 基本运行

```bash
cd scripts/v18_5_oracle_ade_selector_modular
python run_v18_5_modular.py
```

### 10.2 指定数据路径运行

```bash
MANIFEST_PATH=/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/build_manifest_full_v8_map_agent_t6.json \
SHARD_DIR=/home/ubuntu22/decision_on_nuscenes/outputs_v8_map_agent/shards_full_v8_map_agent_t6 \
IMAGE_BEV_CACHE_MANIFEST=/home/ubuntu22/decision_on_nuscenes/outputs_v16_4_1_highres_depthsup_bgclean_real_image_lss_bev_cache/image_bev_cache_manifest_v16_4_1.json \
AGENT3D_CACHE_MANIFEST=/home/ubuntu22/decision_on_nuscenes/outputs_v10_0_3d_agent_cache/agent3d_cache_manifest_v10_0.json \
RUN_BASE_TRAIN=0 \
python run_v18_5_modular.py
```

### 10.3 只跑 score calibration

如果已有 base checkpoint：

```bash
RUN_BASE_TRAIN=0 \
BASE_CKPT_PATH=/path/to/best_v18_5_rawbev_diffusion_base_model.pt \
python run_v18_5_modular.py
```

### 10.4 重新训练 base generator

```bash
RUN_BASE_TRAIN=1 \
BASE_EPOCHS=80 \
python run_v18_5_modular.py
```

---

## 11. 评估指标

### 11.1 分类指标

```text
Accuracy
Macro-F1
Precision
Recall
Confusion Matrix
```

其中 Macro-F1 比 Accuracy 更重要，因为行为类别存在不均衡。

### 11.2 轨迹指标

```text
ADE_selected
FDE_selected
ADE_oracle
FDE_oracle
ADE_gap
ScoreHit
```

含义：

```text
ADE_selected: score 选中的轨迹平均误差
FDE_selected: score 选中的轨迹终点误差
ADE_oracle:   K 条候选轨迹中最优轨迹的平均误差
FDE_oracle:   K 条候选轨迹中最优轨迹的终点误差
ADE_gap:      ADE_selected - ADE_oracle
ScoreHit:     score 选中的 mode 是否等于 oracle best mode
```

重点看：

```text
ADE_gap
```

如果 gap 大，说明 generator 有能力，但 selector 没选好。

---

## 12. 面试讲法

可以这样介绍 v18.5：

> v18.5 是一个基于 nuScenes 的决策规划联合建模框架。模型输入 ego 历史状态、周围 agent、3D agent、地图 polyline 和 image-BEV cache，将这些信息统一编码成场景 token，并通过 Transformer 进行融合。输出端同时完成驾驶行为分类、K 模态未来轨迹生成和候选轨迹评分。相比前一版，v18.5 的重点是解决 oracle 轨迹和 selected 轨迹之间的 gap，因此第二阶段冻结 base generator，只训练 oracle-ADE-supervised score selector，让模型更稳定地选中候选轨迹中的最优轨迹。

---

## 13. 不能夸大的地方

不要说：

```text
这是完整端到端自动驾驶系统。
```

更准确地说：

```text
这是决策规划层端到端联合建模。
```

原因：

- 图像 BEV 是离线 cache
- 没有在线感知网络联合反传
- 没有接入控制模块
- 没有闭环仿真验证
- 训练方式仍然主要是 open-loop imitation learning

---

## 14. 当前版本不足

v18.5 仍然存在几个问题：

1. 不是完整 camera-to-control 端到端。
2. BEV cache 离线生成，planner 训练时不更新感知模块。
3. 主要是 open-loop 评估，没有闭环仿真。
4. score selector 虽然加强了，但 selected 和 oracle 之间仍可能存在 gap。
5. nuScenes 人类驾驶轨迹是模仿学习标签，不等价于最优规划策略。


