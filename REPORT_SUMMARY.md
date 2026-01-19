## 回报目标与阶段技术栈

- 数据采集与原始存储：HDF5 / 文件存储  
- 标准化与结构转换：RLDS 风格 steps  
- 离线分析与报表：Python + SQLite（runs/episodes）  
- 时序监控与趋势：InfluxDB（run_metrics / episode_metrics / lane_metrics）  
- 查询层：SQLite SQL + InfluxDB Flux  
- 可视化：Matplotlib（本地）+ InfluxDB UI（可选）  
- 大规模处理：流式统计（不构建 RLDS，低内存）  

## 生产级技术栈对照

下面给出“Demo 版”与“生产版”在各个环节的对应关系，说明可平滑替换的路径。

### 1) 原始数据存储
- Demo：HDF5 本地文件
- 生产常用：S3/OSS/GCS（对象存储） + Parquet/TFRecord/ROS bag
- 说明：生产中需要低成本、海量数据与高并发读取；HDF5 适合原型或单机。
  - 典型实践：原始传感器（图像/点云/IMU）与轨迹拆分分层存储；冷数据归档到低成本存储。
  - 迁移要点：统一对象键规范（任务/日期/设备/场景），写入元数据索引表。

### 2) 元数据与索引层
- Demo：SQLite
- 生产常用：PostgreSQL / MySQL（主流 OLTP）
- 说明：生产需要权限管理、多用户并发、事务与审计。
  - 典型表结构：runs / episodes / sensors / labels / assets / users / audits。
  - 关键索引：时间、设备、场景、任务、标签、质量等级。
  - 与文件存储结合：表中保存对象存储路径与校验信息（如 hash）。
  - 查询模式：先用元数据库筛选 episode，再去对象存储/HDF5 读取原始数据。

### 3) 时序监控
- Demo：InfluxDB（单机）
- 生产常用：InfluxDB / TimescaleDB / Prometheus + Grafana
- 说明：生产更强调高可用与告警体系。
  - 常见指标：速度、功耗、温度、帧率、控制延迟、碰撞/异常事件。
  - 关键能力：实时告警、阈值触发、异常趋势监控。
  - 与业务库结合：时序库只存高频指标，业务表存事件与任务；两者通过 run_id/episode_id 关联。

### 4) 分析与报表
- Demo：Python 统计 + SQLite 查询
- 生产常用：ClickHouse / BigQuery / Snowflake / Spark
- 说明：生产场景的汇总分析通常需要列式数据库或分布式计算。
  - 常见报表：成功率、轨迹长度分布、异常类型分布、设备/场景对比。
  - 查询模式：按时间窗口、设备/任务维度聚合；支持交互式分析。
  - 数据链路：对象存储 → ETL → 列式库；元数据来自 OLTP 表进行 join。

### 5) 相似检索与轨迹检索
- Demo：未实现
- 生产常用：Milvus / pgvector / Faiss（向量检索）
- 说明：用于“轨迹相似性检索、故障案例召回、数据去重”。
  - 典型特征：轨迹 embedding、关键事件片段 embedding。
  - 典型应用：异常复盘、故障溯源、相似任务检索。
  - 与元数据结合：向量检索得到候选后回查 OLTP/OLAP 获取完整上下文。

### 6) 训练与评估
- Demo：RLDS 风格 steps + JSONL 导出
- 生产常用：TFDS / WebDataset / 数据版本管理（DVC/MLflow）
- 说明：生产强调可复现与可审计的数据版本。
  - 训练元数据：数据版本、模型版本、超参、评估指标。
  - 评估链路：离线评测 + 在线 A/B + 回放对比。

### 7) 数据治理与权限
- Demo：未覆盖
- 生产常用：数据资产管理、权限控制、审计日志
- 说明：尤其在自动驾驶/医疗等场景必须具备合规体系。
  - 治理要点：数据分级、脱敏、访问审批、使用记录留痕。
  - 合规示例：隐私/安全/行业合规检查。

## 结论
- 本 demo 采用的是**“生产链路的轻量化映射”**：方向一致、规模可控。  
- 可平滑升级路径明确：  
  - HDF5 → 对象存储 + Parquet/TFRecord  
  - SQLite → PostgreSQL/MySQL  
  - 本地统计 → ClickHouse/Spark  
  - InfluxDB 单机 → 高可用时序监控  
- RLDS JSONL → TFDS / MLflow 数据版本  

## 生产落地建议（可选）
- 先落地“元数据索引 + 统计面板”，再扩展到时序监控与大规模分析。  
- 用对象存储承载原始数据，关系型数据库记录索引与权限。  
- 明确数据版本与模型版本的映射关系，保证可追溯与复现。  
- 统一 episode_id/run_id 作为贯穿各库的关联键，避免跨系统无法对齐。  
