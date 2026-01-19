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

Outputs:
- HDF5 file: `data/trajectories.h5`
- Plots: `outputs/trajectories.png`
- RLDS-style JSONL: `outputs/rlds_episodes.jsonl`

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

See `requirements.txt`. The `rlds` dependency is optional.
