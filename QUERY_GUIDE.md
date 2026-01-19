# Query Guide

This guide provides ready-to-run queries for SQLite and InfluxDB to report on
trajectory analysis results, including action quality, anomalies, clustering,
and optional lane statistics.

## SQLite (offline analytics)

Open the database:

```bash
sqlite3 outputs/trajectory_stats.db
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

## InfluxDB (Flux)

Make sure InfluxDB is running and data has been written to bucket
`trajectory_metrics`.

### Latest run metrics

```flux
from(bucket: "trajectory_metrics")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "run_metrics")
  |> last()
```

### Path length distribution

```flux
from(bucket: "trajectory_metrics")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "episode_metrics" and r._field == "path_length")
  |> histogram(bins: 10)
```

### Lowest action quality (Top 10)

```flux
from(bucket: "trajectory_metrics")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "episode_metrics" and r._field == "quality_score")
  |> sort(columns: ["_value"], desc: false)
  |> limit(n: 10)
```

### Anomaly count trend

```flux
from(bucket: "trajectory_metrics")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "episode_metrics" and r._field == "anomaly")
  |> aggregateWindow(every: 1d, fn: sum)
```

### Cluster distribution

```flux
from(bucket: "trajectory_metrics")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "episode_metrics" and r._field == "cluster_id")
  |> group(columns: ["_value"])
  |> count()
```

### Lane stats (if lane_metrics exists)

```flux
from(bucket: "trajectory_metrics")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "lane_metrics" and r._field == "steps")
  |> group(columns: ["lane_id"])
  |> sum()
```
