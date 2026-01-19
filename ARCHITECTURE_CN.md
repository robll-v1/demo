# 数据流 + 架构图（中文版）

下面是本 demo 的数据流与架构位置说明。

```text
数据源
  - CSV（车道轨迹）
  - Kaggle CSV（机械臂）
  - 合成轨迹生成器
        |
        v
HDF5 存储（data/*.h5）
  - episodes/ep_xxxxx/...
  - observations/actions/rewards/dones/timestamps
        |
        v
分析流水线（analyze_demo.py）
  - RLDS 转换（steps）或流式统计
  - 统计指标：路径长度、平均步长、动作质量
  - 异常检测 + 轨迹聚类
  - 车道统计（如存在）
        |
        +-------------------------------+
        |                               |
        v                               v
SQLite（outputs/trajectory_stats.db）  InfluxDB（trajectory_metrics）
  - runs 表                             - run_metrics
  - episodes 表                         - episode_metrics
  - 车道统计（JSON 字段）               - lane_metrics（可选）
        |                               |
        v                               v
查询 / 可视化
  - SQLite SQL                          - Flux / UI 图表
  - 报表与审计                          - 监控与趋势分析
```

## 各层角色说明

- **HDF5**：大规模原始轨迹存储（高效顺序读写）。  
- **RLDS**：标准化 step 格式，便于下游训练与评估。  
- **SQLite**：离线分析与报表（episode 级统计）。  
- **InfluxDB**：时序监控与可视化（趋势与仪表盘）。
- **流式模式**：大规模数据的低内存统计（无法导出 RLDS）。
