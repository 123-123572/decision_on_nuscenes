# decision_on_nuscenes

基于 nuScenes 全量数据集的自动驾驶行为决策与多模态轨迹规划项目。

本项目融合 ego 历史状态、周围 Agent、地图 polyline、3D Agent 与 BEV 表征，构建行为决策、轨迹生成与轨迹评分一体化模型。

## Features

- nuScenes 数据处理与行为标签构建
- CRUISE / FOLLOW / STOP / TURN 行为分类
- Best-of-K 多模态轨迹预测
- Transformer-based Map-Agent-Ego token fusion
- Map Loss / Comfort Loss / Collision Loss 多任务约束
- Real-image LSS BEV cache 构建
- Object-aligned BEV fusion for planning

## Main Results

| Metric | Result |
|---|---:|
| Accuracy | 0.8621 |
| Macro-F1 | 0.7689 |
| ADE selected | 0.7765 m |
| FDE selected | 1.5973 m |

## Note

本仓库仅包含代码、配置和项目说明，不包含 nuScenes 原始数据、训练输出、模型权重和缓存文件。
