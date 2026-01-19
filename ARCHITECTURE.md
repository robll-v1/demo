# Data Flow + Architecture

This diagram shows how data moves through the demo and where each storage layer fits.

```text
Raw Data Sources
  - CSV (lane trajectories)
  - Kaggle CSV (robot arm)
  - Synthetic generator
        |
        v
HDF5 Storage (data/*.h5)
  - episodes/ep_xxxxx/...
  - observations/actions/rewards/dones/timestamps
        |
        v
Analysis Pipeline (analyze_demo.py)
  - RLDS conversion (steps) or streaming stats
  - Metrics: path_length, avg_step, action quality
  - Anomaly detection + clustering
  - Lane stats (if present)
        |
        +-------------------------------+
        |                               |
        v                               v
SQLite (outputs/trajectory_stats.db)   InfluxDB (trajectory_metrics)
  - runs table                          - run_metrics
  - episodes table                      - episode_metrics
  - lane stats (JSON text)              - lane_metrics (optional)
        |                               |
        v                               v
Queries / Dashboards
  - SQLite SQL                          - Flux / UI charts
  - Reports & audits                    - Monitoring & trends
```

## Key Roles

- **HDF5**: Large raw trajectory storage (fast sequential reads).
- **RLDS**: Standardized step format for downstream ML.
- **SQLite**: Offline analytics and reporting (episode-level).
- **InfluxDB**: Time-series monitoring and dashboards.
- **Streaming mode**: Low-memory stats when data is large (no RLDS export).
