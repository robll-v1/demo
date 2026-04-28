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
        |
        v
MatrixOne（demo 数据库）
  - runs 表
  - episodes 表
  - trajectory_embeddings 表
  - trajectory_steps 表（逐步时序）
        |
        v
查询
  - SQL 分析
  - 相似检索（CLI）
```

## 各层角色说明

- **HDF5**：大规模原始轨迹存储（高效顺序读写）。  
- **RLDS**：标准化 step 格式，便于下游训练与评估。  
- **MatrixOne**：统一承载分析与相似检索。  
- **流式模式**：大规模数据的低内存统计（无法导出 RLDS）。
