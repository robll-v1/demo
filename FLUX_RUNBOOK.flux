// InfluxDB Flux runbook for trajectory_metrics
// Usage:
// influx query --org datademo --token "$(cat token)" --file FLUX_RUNBOOK.flux

// 1) Latest run metrics
from(bucket: "trajectory_metrics")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "run_metrics")
  |> last()
  |> yield(name: "run_metrics_latest")

// 2) Path length distribution
from(bucket: "trajectory_metrics")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "episode_metrics" and r._field == "path_length")
  |> histogram(bins: [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
  |> yield(name: "path_length_hist")

// 3) Lowest action quality (Top 10)
from(bucket: "trajectory_metrics")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "episode_metrics" and r._field == "quality_score")
  |> sort(columns: ["_value"], desc: false)
  |> limit(n: 10)
  |> yield(name: "quality_score_lowest")

// 4) Anomaly count trend (daily)
from(bucket: "trajectory_metrics")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "episode_metrics" and r._field == "anomaly")
  |> aggregateWindow(every: 1d, fn: sum)
  |> yield(name: "anomaly_trend")

// 5) Cluster distribution
from(bucket: "trajectory_metrics")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "episode_metrics" and r._field == "cluster_id")
  |> group(columns: ["_value"])
  |> count()
  |> yield(name: "cluster_distribution")

// 6) Lane stats (if lane_metrics exists)
from(bucket: "trajectory_metrics")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "lane_metrics" and r._field == "steps")
  |> group(columns: ["lane_id"])
  |> sum()
  |> yield(name: "lane_steps_sum")
