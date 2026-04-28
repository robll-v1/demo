# RLDS + HDF5 Embodied Trajectory Demo

This is a small, self-contained demo that generates synthetic embodied-agent
trajectories, stores them in HDF5, and exposes them via an RLDS-style adapter
for simple trajectory analysis and visualization.

## Quick start

1) Create data

```bash
python3 generate_demo_data.py --episodes 20 --max-steps 80
```

2) Analyze (text)

```bash
python3 analyze_demo.py
```

3) Analyze (with plots)

```bash
python3 analyze_demo.py --plot
```

To limit how many episodes are drawn (avoid overplotting):

```bash
python3 analyze_demo.py --plot --plot-samples 20
```

4) Export RLDS-style episodes (JSONL)

```bash
python3 analyze_demo.py --export outputs/rlds_episodes.jsonl
```

5) Write stats to MatrixOne

```bash
mysql -h 127.0.0.1 -P 6001 -u root -p111 -e "CREATE DATABASE IF NOT EXISTS demo"
python3 analyze_demo.py --data data/robotic_arm.h5 --mo --mo-database demo
```

To enable time-series queries over per-step timestamps:

```bash
python3 analyze_demo.py --data data/robotic_arm.h5 --mo --mo-database demo --mo-steps
```

6) Similarity search + trajectory retrieval (MatrixOne + HDF5)

```bash
# Similarity search by episode id (uses embeddings stored in MatrixOne)
python3 trajectory_search.py similar \
  --mo-database demo \
  --data data/robotic_arm.h5 \
  --episode-id ep_00001 \
  --top-k 5

# Filter/retrieve episodes and optionally export to JSONL
python3 trajectory_search.py retrieve \
  --mo-database demo \
  --success 1 \
  --min-path-length 5 \
  --limit 10 \
  --data data/robotic_arm.h5 \
  --export outputs/retrieved_episodes.jsonl
```

Outputs:
- HDF5 file: `data/trajectories.h5`
- Plots: `outputs/trajectories.png`
- RLDS-style JSONL: `outputs/rlds_episodes.jsonl`
- Analysis logs include action quality, anomaly count, cluster summary, and lane stats (if present).
- Streaming mode supports large datasets without building RLDS episodes.

## Streaming stats (large data)

For very large datasets, avoid building RLDS episodes in memory:

```bash
python3 analyze_demo.py --data data/robotic_arm.h5 --stream --log auto --mo --mo-database demo
```

Note: `--stream` disables `--export`.

## Import CSV trajectories

If you have real trajectories in CSV format (one file per episode), put them in
`data/csv/` and run:

```bash
python3 import_csv_to_hdf5.py --csv-dir data/csv --out data/trajectories.h5
```

Required CSV columns:
- `timestamp`
- `x`
- `y`
- `velocity`

Optional columns:
- `lane`

## Generate robot CSVs (synthetic)

You can also generate synthetic robot paths as CSVs:

```bash
python3 generate_robot_csv.py --out-dir data/csv_robot --episodes 5 --pattern zigzag
python3 import_csv_to_hdf5.py --csv-dir data/csv_robot --out data/trajectories.h5
python3 analyze_demo.py --plot
```

## Import Kaggle robotic arm dataset

If you downloaded the Kaggle dataset `meghrajbagde/robotic-arm-dataset-multiple-trajectories`,
use this script to convert the single CSV into HDF5 episodes:

```bash
python3 import_kaggle_robotic_arm.py \
  --csv /path/to/robotic_arm_dataset_multiple_trajectories.csv \
  --out data/robotic_arm.h5 \
  --episode-length 200
python3 analyze_demo.py --data data/robotic_arm.h5 --plot
```

## Export TFDS dataset

To export a formal TFDS dataset (TFRecord + dataset_info):

```bash
python3 export_tfds_dataset.py --data data/trajectories.h5 --out datasets
```

Requirements:

```bash
python3 -m pip install tensorflow tensorflow-datasets
```

## Notes

- The adapter uses RLDS constants if `rlds` is installed; otherwise it falls
  back to string keys with the same names.
- The trajectories can be 2D paths or robot arm joint angles. The plots show joint angles over time plus joint-space statistics.

## Requirements

See `requirements.txt`. The `rlds` dependency is optional; `pymysql` is used for
MatrixOne access from Python.
