# Query Guide

This guide provides ready-to-run queries for MatrixOne to report on trajectory
analysis results, including action quality, anomalies, clustering, and optional
lane statistics.

## MatrixOne (analytics)

Connect with the MySQL client:

```bash
mysql -h 127.0.0.1 -P 6001 -u root -p111
```

Use the target database (e.g. demo):

```sql
USE demo;
```

### Latest run info

```sql
SELECT * FROM runs ORDER BY id DESC LIMIT 1;
```

### Top 10 by path length

```sql
SELECT episode_id, path_length, avg_step
FROM episodes
ORDER BY path_length DESC
LIMIT 10;
```

### Lowest action quality (Top 10)

```sql
SELECT episode_id, quality_score
FROM episodes
ORDER BY quality_score ASC
LIMIT 10;
```

### Anomalous episodes

```sql
SELECT episode_id, path_length, avg_step, action_smoothness
FROM episodes
WHERE anomaly = 1;
```

### Cluster distribution

```sql
SELECT cluster_id, COUNT(*) AS cnt
FROM episodes
GROUP BY cluster_id
ORDER BY cnt DESC;
```

### Lane stats (only if lane is present)

```sql
SELECT episode_id, lane_counts, lane_mean_speed
FROM episodes
WHERE lane_counts IS NOT NULL
LIMIT 10;
```

## Time-series queries (per-step timestamps)

Requires writing per-step metrics with `--mo-steps`.

### Steps per time bucket (timestamp floored)

```sql
SELECT FLOOR(timestamp) AS t, COUNT(*) AS steps
FROM trajectory_steps
GROUP BY t
ORDER BY t;
```

### Average reward per time bucket

```sql
SELECT FLOOR(timestamp) AS t, AVG(reward) AS avg_reward
FROM trajectory_steps
GROUP BY t
ORDER BY t;
```

### Step magnitude trend

```sql
SELECT FLOOR(timestamp) AS t, AVG(step_mag) AS avg_step_mag
FROM trajectory_steps
GROUP BY t
ORDER BY t;
```

### Terminal events over time

```sql
SELECT FLOOR(timestamp) AS t, SUM(is_terminal) AS terminal_steps
FROM trajectory_steps
GROUP BY t
ORDER BY t;
```

## Trajectory search (CLI)

Similarity search by episode id:

```bash
python3 trajectory_search.py similar \
  --mo-database demo \
  --data data/trajectories.h5 \
  --episode-id ep_00001 \
  --top-k 5
```

Filter and export trajectories:

```bash
python3 trajectory_search.py retrieve \
  --mo-database demo \
  --success 1 \
  --min-path-length 5 \
  --limit 10 \
  --data data/trajectories.h5 \
  --export outputs/retrieved_episodes.jsonl
```
