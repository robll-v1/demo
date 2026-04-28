# Embodied Trajectory Demo

Self-contained pipeline for embodied agent trajectory analysis: generation, storage, analysis, and similarity search.

Supports synthetic 2D paths, robotic arm joint trajectories, and real-world CSV imports. Stores data in HDF5, exports to RLDS/TFDS, and integrates with MatrixOne for SQL-based trajectory search.

## Architecture

```
CSV / Kaggle / Synthetic
        │
        ▼
   ┌─────────┐
   │  HDF5   │  ← episodes / observations / actions / rewards
   └────┬────┘
        │
        ▼
  ┌───────────┐     ┌──────────┐
  │  Analyze  │────▶│  Export   │  → RLDS JSONL / TFDS TFRecord
  └─────┬─────┘     └──────────┘
        │
        ▼
  ┌───────────┐     ┌──────────────┐
  │ MatrixOne │────▶│  Similarity  │  → embedding-based trajectory search
  └───────────┘     │   Search     │
                    └──────────────┘
```

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic data
python3 generate_demo_data.py --episodes 20 --max-steps 80

# Analyze and plot
python3 analyze_demo.py --plot

# Export RLDS-style JSONL
python3 analyze_demo.py --export outputs/rlds_episodes.jsonl
```

## Data Import

**CSV trajectories** (one file per episode, columns: `timestamp`, `x`, `y`, `velocity`, optional `lane`):

```bash
python3 import_csv_to_hdf5.py --csv-dir data/csv --out data/trajectories.h5
```

**Synthetic robot paths** (zigzag / circle patterns):

```bash
python3 generate_robot_csv.py --out-dir data/csv_robot --episodes 5 --pattern zigzag
python3 import_csv_to_hdf5.py --csv-dir data/csv_robot --out data/trajectories.h5
```

**Kaggle robotic arm dataset** (`meghrajbagde/robotic-arm-dataset-multiple-trajectories`):

```bash
python3 import_kaggle_robotic_arm.py \
  --csv /path/to/robotic_arm_dataset_multiple_trajectories.csv \
  --out data/robotic_arm.h5 \
  --episode-length 200
```

## MatrixOne Integration

Write trajectory analytics and embeddings to MatrixOne for SQL-based querying:

```bash
mysql -h 127.0.0.1 -P 6001 -u root -p111 -e "CREATE DATABASE IF NOT EXISTS demo"
python3 analyze_demo.py --data data/robotic_arm.h5 --mo --mo-database demo
```

With per-step time-series data:

```bash
python3 analyze_demo.py --data data/robotic_arm.h5 --mo --mo-database demo --mo-steps
```

## Similarity Search

Find similar trajectories by embedding distance, or filter/retrieve by metrics:

```bash
# Find 5 most similar trajectories to ep_00001
python3 trajectory_search.py similar \
  --mo-database demo --data data/robotic_arm.h5 \
  --episode-id ep_00001 --top-k 5

# Retrieve successful episodes with path length > 5, export to JSONL
python3 trajectory_search.py retrieve \
  --mo-database demo --success 1 --min-path-length 5 --limit 10 \
  --data data/robotic_arm.h5 --export outputs/retrieved_episodes.jsonl
```

## Streaming Mode

For large datasets, stream analysis without loading all RLDS episodes into memory:

```bash
python3 analyze_demo.py --data data/robotic_arm.h5 --stream --log auto --mo --mo-database demo
```

Note: `--stream` disables `--export`.

## TFDS Export

Export to TensorFlow Datasets format:

```bash
pip install tensorflow tensorflow-datasets
python3 export_tfds_dataset.py --data data/trajectories.h5 --out datasets
```

## Project Structure

```
├── generate_demo_data.py          # Synthetic 2D trajectory generation
├── generate_robot_csv.py          # Synthetic robot path CSV generation
├── import_csv_to_hdf5.py          # CSV → HDF5 conversion
├── import_kaggle_robotic_arm.py   # Kaggle dataset import
├── analyze_demo.py                # Main analysis pipeline
├── trajectory_search.py           # Similarity search CLI
├── export_tfds_dataset.py         # HDF5 → TFDS export
├── mo_client.py                   # MatrixOne database abstraction
├── rlds_adapter.py                # RLDS constant management
├── SQL_RUNBOOK.sql                # MatrixOne schema and queries
├── SQL_TEST_CASE.sql              # Manual test data
├── QUERY_GUIDE.md                 # SQL / CLI query reference
├── ARCHITECTURE.md                # English architecture overview
├── ARCHITECTURE_CN.md             # Chinese architecture details
└── REPORT_SUMMARY.md              # Production reference guide
```

## Analysis Output

- Trajectory metrics: path length, action quality, smoothness
- Anomaly detection (z-score based)
- K-means clustering
- Lane statistics (if lane data present)
- Plots: joint angles over time, distribution histograms

## Requirements

Core dependencies in `requirements.txt`. Optional:
- `rlds` — for RLDS constant names (falls back to string keys)
- `tensorflow`, `tensorflow-datasets` — for TFDS export only
- MatrixOne or any MySQL-compatible database — for SQL queries and search

## License

MIT
