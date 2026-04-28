## 汇报目标与阶段技术栈

- 数据采集与原始存储：HDF5 / 文件存储  
- 标准化与结构转换：RLDS 风格 steps  
- 离线分析与报表：Python + MatrixOne（runs/episodes）  
- 查询层：MatrixOne SQL（含逐步时序表）  
- 可视化：Matplotlib（本地）  
- 大规模处理：流式统计（不构建 RLDS，低内存）  

Rlds 的基本结构如图下（通用）：
```text
dataset/
  └─ episodes/
       ├─ episode_000/
       │     ├─ step_000
       │     │     ├─ observation/
       │     │     ├─ action
       │     │     ├─ reward
       │     │     └─ …
       │     ├─ step_001
       │     └─ …
       └─ episode_001/
             └─ …
```
说明：RLDS 的核心思想是“按 episode 组织、按 step 序列化”，便于统一下游训练与评估。
Rlds：
	•	把数据按 episode → step 层次组织
	•	每个 step 包含：observation（可能是多模态）、action、reward、is_first、is_last 等
	•	与 TensorFlow Datasets (TFDS) 紧密集成
	•	提供强大的序列变换 API（如 windowing、N‑step 转换、轨迹重组）
	•	支持 Offline Reinforcement Learning、Imitation Learning 等应用
HDF5 通用结构如图下（行业实践）：
```text
trajectories.h5
  /episodes
    /episode_000001
      observations/...
      actions/...
      rewards
      dones
      timestamps (可选)
      /metadata
        task_id / device_id / scene / success / steps / source_file
    /episode_000002
      ...
```
说明：HDF5 主要用于高效存储大规模原始数据，适合顺序读写与分组管理。
	•	像文件系统一样的层次结构：Group + Dataset
	•	Dataset 是多维数组（可存图像、轨迹、传感器序列等）
	•	支持压缩、切片读写、并行 I/O
	•	支持跨语言调用（Python、C++、MATLAB 等）
	•	被多个具身数据平台采用作为底层存储
             
## 生产流程数据流（前五个步骤）

下面按“生产落地”视角描述从数据导入到检索的完整链路，并拆出前五个步骤的数据流向。

### 数据血缘与读写时机（如何追踪“数据从哪来、何时写、何时读”）

**血缘链路（本 demo）**  
`CSV/外部数据 → HDF5 轨迹文件 → 分析程序 → MatrixOne 表（runs/episodes/trajectory_steps/trajectory_embeddings） → 查询/检索`

**读写时机（时序监控为例）**  
- 写入：运行 `analyze_demo.py --mo --mo-steps` 时写入 `trajectory_steps`  
- 读取：执行时序 SQL（按时间窗口聚合）时读取 `trajectory_steps`

**大规模文件定位（百万文件场景）**  
生产环境不会靠文件名找文件，而是靠元数据索引表进行定位：

```text
files(file_id, path, hash, created_at, size, device_id, scene)
episodes(episode_id, run_id, file_id, start_ts, end_ts, success)
```

查询某条轨迹对应的文件：
```sql
SELECT f.path
FROM episodes e
JOIN files f ON e.file_id = f.file_id
WHERE e.episode_id = 'ep_00123';
```

### ASCII 数据流图（前五步）

```text
数据源（设备/CSV/外部数据集）
          |
          v
Step 1 采集与导入
  - 原始文件落地（对象存储/文件系统）
  - ETL/导入程序清洗 + 标准化
          |
          v
Step 2 元数据与索引（MatrixOne）
  - runs / episodes
          |
          +------------------------------+
          |                              |
          v                              v
Step 3 时序监控（MatrixOne）        Step 4 分析与报表（MatrixOne）
  - trajectory_steps                  - 统计指标落库
          |                              |
          +---------------+--------------+
                          |
                          v
Step 5 相似检索/轨迹检索（MatrixOne）
  - trajectory_embeddings + 向量索引
  - SQL 条件过滤 + 相似召回
```

### Step 1 采集与导入（数据如何进入系统）
数据来源可以是：
- 设备/机器人实时采集（传感器、控制指令、状态等）
- 现场日志/CSV 批量离线文件
- 外部数据集（如 Kaggle）

导入流向：
1) 原始数据落地到对象存储或文件系统（例如 S3/OSS/HDFS/本地文件）
2) ETL/导入程序读取原始数据，做基础校验/清洗（缺失值、非法值、格式不一致）
3) 输出标准化的“轨迹文件”（例如 HDF5、Parquet 或 TFRecord），并产生每条轨迹的唯一 ID（episode_id）

**数据血缘（Step 1）**  
`原始传感器/CSV → ETL/清洗 → 轨迹文件（HDF5/Parquet）`

**常见追问**  
- 问：百万文件如何定位？  
  答：通过元数据索引表（files/episodes）按 device/scene/time 查询路径。  
- 问：原始数据会被重复写入吗？  
  答：原始数据只落一次，后续都是派生指标/特征。  

**示例（真实场景）**：  
工厂机器人每 10ms 记录一次关节角与速度。日志批量落地到对象存储后，ETL 把一段连续时间切成 episode，写入 HDF5/Parquet 并记录 episode_id。

**Demo 对应做法（命令/文件）**  
- CSV → HDF5：`python3 import_csv_to_hdf5.py --csv-dir data/csv --out data/trajectories.h5`  
- 合成轨迹：`python3 generate_demo_data.py --episodes 20 --max-steps 80`  
- 产物：`data/trajectories.h5`（HDF5 文件）

**Demo vs 生产（Step 1）**  
- Demo：HDF5 本地文件  
- 生产：对象存储（S3/OSS/GCS）+ Parquet/TFRecord

### Step 2 元数据与索引（元数据如何落库）
元数据表用于快速筛选与管理（run、episode、场景、设备、时间范围等）。

流向：
1) ETL/分析程序从轨迹文件读取元数据
2) 将 run/episode 级信息写入元数据库（MatrixOne）
3) 建立索引以支持过滤/排序（时间、设备、场景、质量分数等）

元数据层（OLTP）：PostgreSQL / MySQL
表：runs / episodes / files

什么时候写：数据导入/切片完成时（批处理或实时）
数据从哪来：原始文件 + ETL 解析出的元信息
用途：定位文件、筛选轨迹
**数据血缘（Step 2）**  
`轨迹文件 → 解析元数据 → runs/episodes 表`

**常见追问**  
- 问：如何从 episodes 追溯到原始文件？  
  答：生产用 files 表记录路径，episodes 关联 file_id。  
- 问：指标口径是否一致？  
  答：episode 级指标统一落在 episodes 表，用 run_id/episode_id 关联。  

**示例**：  
每次批处理写入 runs 表（数据路径、时间、步数），每条轨迹写入 episodes 表（路径长度、成功率、质量指标）。

**写入内容与 schema（本 demo）**  
写入的是数据库条目（元数据/指标），不是文件本体。文件仍在 HDF5/对象存储中。  
当前表结构示例：
```text
runs(id, data_path, file_size, read_time, episodes, steps, created_at)
episodes(run_id, episode_id, path_length, total_reward, avg_step, success, steps,
         action_mag, action_smoothness, quality_score, anomaly, cluster_id,
         lane_counts, lane_mean_speed)
```

**索引（本 demo）**  
- 结构化索引：`run_id`、`path_length`、`quality_score`、`anomaly`、`cluster_id`  
- 时序索引：`trajectory_steps.timestamp`（若启用 `--mo-steps`）  
- 向量索引：`trajectory_vectors.embedding`（HNSW，演示用表）

**Demo 对应做法（命令/表）**  
- 写入 MatrixOne：`python3 analyze_demo.py --data data/trajectories.h5 --mo --mo-database demo`  
- 写入表：`runs`、`episodes`、`trajectory_embeddings`  
- `episode_id` 在 demo 中就是 HDF5 的分组名（如 `ep_00000`）

**Demo vs 生产（Step 2）**  
- Demo：MatrixOne  
- 生产：PostgreSQL/MySQL（OLTP）

**正常查询示例（已运行）**  
最新一次 run：
```sql
SELECT id, created_at, episodes, steps
FROM runs
ORDER BY id DESC
LIMIT 1;
```
结果：
```text
id  created_at           episodes  steps
3   2026-01-19 18:28:25  4         164
```

### Step 3 时序监控（逐步数据如何进入时序表）
时序监控并非只存汇总指标，核心在“逐步记录可查询”。

流向：
1) 在导入或分析时，将 step 级别的 timestamp/reward/step_mag 写入时序表（trajectory_steps）
2) 按时间窗口聚合查询（分钟/小时/日）
3) 用于趋势监控和异常检测
时序层：TimescaleDB / Prometheus / ClickHouse
表：steps_timeseries（逐步指标）
什么时候写：采集实时流或离线分析时写（可采样）
数据从哪来：原始轨迹的逐步数据（timestamp/速度/奖励等）
**数据血缘（Step 3）**  
`轨迹文件 → step 切分 → trajectory_steps → 时序聚合查询`

**生产表职责对照（为什么时序监控读它）**  
- raw_files：原始文件索引（路径/hash/设备/时间），用于溯源定位  
- episodes：episode 级指标（成功率、步数、质量分数），用于筛选/报表  
- steps_timeseries：逐步时序表（timestamp/速度/奖励/延迟），用于趋势监控  
- metrics_daily / metrics_hourly：聚合表，用于 BI 报表  
- trajectory_embeddings：向量表，用于相似检索  
因此时序监控只读 `steps_timeseries`，不读原始文件或 episodes。

**常见追问**  
- 问：timestamp 口径是什么？  
  答：有真实时间戳就用它；没有就用 step_index。  
- 问：时序表太大怎么办？  
  答：按采样率写入或写聚合表（metrics_daily/hourly）。  

**示例**：  
对每个 episode 的每一条 step 写入 trajectory_steps，后续可查询“每日 step 数量趋势”“平均 reward 随时间变化”等。

**Demo 对应做法（命令/口径）**  
- 写入时序表：`python3 analyze_demo.py --data data/trajectories.h5 --mo --mo-database demo --mo-steps`  
- `timestamp` 口径：  
  - 来自 HDF5 的 `timestamps`（若存在）  
  - 若没有，则使用 step_index（0..N-1）

**Demo vs 生产（Step 3）**  
- Demo：MatrixOne（单机时序表）  
- 生产：TimescaleDB / Prometheus / ClickHouse

**时序查询示例（已运行）**  
按时间桶统计 step 数（run_id=3）：
```sql
SELECT FLOOR(timestamp) AS t, COUNT(*) AS steps
FROM trajectory_steps
WHERE run_id = 3
GROUP BY t
ORDER BY t
LIMIT 10;
```
结果：
```text
t  steps
0  4
1  4
2  4
3  4
4  4
5  4
6  4
7  4
8  4
9  4
```

### Step 4 分析与报表（指标如何产生）
离线分析通常基于 episode 级特征统计。

流向：
1) 统计程序读取轨迹文件（HDF5/Parquet）
2) 计算指标：path_length / avg_step / action_quality / anomaly 等
3) 将结果写入 episodes 表（或独立的分析表）
4) SQL 报表与可视化直接从 MatrixOne 查询
分析报表层（数仓/OLAP）：ClickHouse / BigQuery / Snowflake
表：metrics_daily/hourly 等聚合结果
报表/聚合表（metrics_daily/hourly）
什么时候写：定时任务（小时/天）
数据从哪来：episodes + steps_timeseries 的聚合计算
**数据血缘（Step 4）**  
`轨迹文件 → 指标计算 → episodes（分析字段） → 报表查询`

**常见追问**  
- 问：为什么不直接读 HDF5？  
  答：HDF5 不适合高频聚合，episodes/聚合表是为查询优化的。  
- 问：如何做设备/场景对比？  
  答：在元数据表增加 device_id/scene 字段并按维度聚合。  

**示例**：  
每日批处理跑一次 analyze 任务，输出异常轨迹比例、成功率、质量评分 TopK。

**Demo 对应做法（命令/字段）**  
- 分析入口：`python3 analyze_demo.py --data data/trajectories.h5`  
- 主要指标：`path_length`、`avg_step`、`quality_score`、`anomaly`

**Demo vs 生产（Step 4）**  
- Demo：Python 统计 + MatrixOne 查询  
- 生产：ClickHouse / BigQuery / Snowflake / Spark

**分析报表查询示例（已运行）**  
按 run 汇总质量指标：
```sql
SELECT COUNT(*) AS episodes,
       AVG(quality_score) AS avg_quality,
       SUM(anomaly) AS anomalies
FROM episodes
WHERE run_id = 3;
```
结果：
```text
episodes  avg_quality          anomalies
4         0.8644783996837619   0
```

### Step 5 相似检索与轨迹检索（向量/条件检索如何落库与查询）
相似检索依赖 embedding（向量），轨迹检索依赖结构化过滤条件。

流向：
1) 轨迹 embedding 在分析时计算并写入向量表（trajectory_embeddings 或独立向量表）
2) 构建向量索引（HNSW/IVF）以加速相似检索
3) 检索时先用向量索引召回 TopK，再回表拿元数据

向量表（trajectory_embeddings）

什么时候写：分析/特征计算完成后
数据从哪来：轨迹模型编码出的 embedding
**数据血缘（Step 5）**  
`轨迹文件 → embedding 生成 → 向量表/索引 → 相似检索 → episodes 过滤`

**常见追问**  
- 问：embedding 如何生成？  
  答：demo 用统计特征拼接；生产用模型编码向量。  
- 问：向量检索结果如何过滤？  
  答：先 TopK 召回，再按 episodes 条件过滤。  

**示例**：  
异常轨迹排查时，用 embedding 找到相似历史轨迹，再结合 episodes 表过滤“成功率=0 或 anomaly=1”的候选。

**Demo 对应做法（来源/表）**  
- embedding 来源：`analyze_demo.py` 中的 `build_embedding`（统计特征拼接）  
- 向量检索表：`trajectory_vectors`（SDK 示例中单独创建）

**Demo vs 生产（Step 5）**  
- Demo：MatrixOne（embedding 表 + 向量索引）  
- 生产：Milvus / pgvector / Faiss

**相似检索 SQL 示例（已运行）**  
向量检索（HNSW，cosine 距离）：
```sql
SELECT *, cosine_distance(embedding, '[0.24228373, 0.045815706, 0.014837439, 0.16541404, 0.14555432, 0.02642166, 0.0032144655, 0.06627838, 0.00449408, 0.0013890794, 0, 0.08171055, 0.49573788, 0.096091606, 0.018221453, 0.24513164, 0, 0, 0, 0, 0.7502457, 0.018756142]')
  AS distance
FROM trajectory_vectors
ORDER BY distance
LIMIT 3;
```
说明：`trajectory_vectors` 为向量检索演示表（HNSW 索引）。
结果：
```text
id  episode_id  distance
1   ep_00000    0
2   ep_00001    1.955032166733872e-05
4   ep_00003    0.00013434885477181524
```

**轨迹检索 SQL 示例（已运行）**  
按条件筛选轨迹（run_id=3，路径长度 Top3）：
```sql
SELECT episode_id, path_length, avg_step
FROM episodes
WHERE run_id = 3
ORDER BY path_length DESC
LIMIT 3;
```
结果：
```text
episode_id  path_length          avg_step
ep_00002    9.673707962036133    0.24184270203113556
ep_00003    9.30693531036377     0.23267337679862976
ep_00000    9.181748390197754    0.22954371571540833
```

## 是否需要分开写入（元数据/时序/报表/检索）

### 结论
- **不必须每次都写入全部内容**，可以按需求分层写入。
- 生产系统通常分为：**导入层（原始数据）→ 元数据层 → 分析层 → 检索层**。
- 这几层可以异步、批量、按频率写入。

### 推荐写入策略
1) **元数据表（runs/episodes）**  
   - 每次导入/批处理都写，作为最基础索引层  
2) **时序表（trajectory_steps）**  
   - 只有需要趋势分析/监控时写  
   - 可按采样率写（不一定每个 step 全量写）  
3) **分析报表指标**  
   - 可按日/按批次写（不必实时）  
4) **向量检索表**  
   - 若相似检索是核心需求，建议在分析完成后异步写入  
   - 索引构建可离线批量完成  

### 真实案例（示意）
**自动驾驶数据中心**  
- 每日导入 100TB 传感器数据 → 生成 episode  
- 元数据表实时更新（runs/episodes）  
- 时序表只存关键指标（速度、加速度、碰撞事件），不存全量  
- 分析报表每日离线统计  
- 向量检索每周增量更新 embedding  

**机器人生产线质检**  
- 轨迹日志实时写入对象存储  
- 每班次写一次 runs/episodes  
- 异常检测在班次结束后批处理  
- 相似检索仅在事故复盘时触发  
### q：
是否需要把同一份数据写入四次？：

不需要把同一份原始数据写四次。
实际做法是一次原始落地，然后按用途生成不同“衍生视图/表”：

原始数据：只落一次（对象存储/HDF5/Parquet）
元数据表：抽取必要字段（episode/run）
时序表：只写需要监控的 step/采样数据
分析报表：写聚合结果（不是原始全量）
向量检索：只写 embedding（不是原始轨迹）
所以是同一份原始数据，多种“派生数据”，不是四次完整复制。
只有在系统解耦或性能要求很高时，才会有部分重复写入（但通常是“抽样/聚合后的副本”，不是全量重复）。
