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
        |
        v
MatrixOne (demo database)
  - runs table
  - episodes table
  - trajectory_embeddings table
  - trajectory_steps table (per-step time series)
        |
        v
Queries
  - SQL analytics
  - similarity search (via CLI)
```

## Key Roles

- **HDF5**: Large raw trajectory storage (fast sequential reads).
- **RLDS**: Standardized step format for downstream ML.
- **MatrixOne**: Unified storage for analytics + similarity search.
- **Streaming mode**: Low-memory stats when data is large (no RLDS export).
