import argparse
import csv
from pathlib import Path

import numpy as np


def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def split_episodes(rows, episode_length, max_episodes):
    total = len(rows)
    episodes = total // episode_length
    if total % episode_length:
        episodes += 1
    if max_episodes is not None:
        episodes = min(episodes, max_episodes)
    for idx in range(episodes):
        start = idx * episode_length
        end = min(total, (idx + 1) * episode_length)
        yield idx, rows[start:end]


def build_episode(rows, episode_index, source_file):
    observations = []
    for row in rows:
        observations.append(
            [
                float(row["Axis_0_Angle"]),
                float(row["Axis_1_Angle"]),
                float(row["Axis_2_Angle"]),
            ]
        )
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.zeros((len(observations), 3), dtype=np.float32)
    rewards = np.zeros(len(observations), dtype=np.float32)
    dones = np.zeros(len(observations), dtype=np.bool_)
    dones[-1] = True
    timestamps = np.arange(len(observations), dtype=np.float32)

    start = observations[0, :2]
    goal = observations[-1, :2]
    metadata = {
        "goal": goal.astype(np.float32),
        "start": start.astype(np.float32),
        "success": np.bool_(True),
        "steps": np.int32(len(observations)),
        "source_file": source_file,
        "episode_index": np.int32(episode_index),
    }
    return observations, actions, rewards, dones, timestamps, metadata


def write_hdf5(path, episodes):
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        root = h5.create_group("episodes")
        for idx, (obs, acts, rewards, dones, timestamps, meta) in enumerate(episodes):
            group = root.create_group(f"ep_{idx:05d}")
            group.create_dataset("observations", data=obs, compression="gzip")
            group.create_dataset("actions", data=acts, compression="gzip")
            group.create_dataset("rewards", data=rewards, compression="gzip")
            group.create_dataset("dones", data=dones, compression="gzip")
            group.create_dataset("timestamps", data=timestamps, compression="gzip")
            meta_group = group.create_group("metadata")
            meta_group.create_dataset("goal", data=meta["goal"])
            meta_group.create_dataset("start", data=meta["start"])
            meta_group.create_dataset("success", data=meta["success"])
            meta_group.create_dataset("steps", data=meta["steps"])
            meta_group.create_dataset("source_file", data=np.bytes_(meta["source_file"]))
            meta_group.create_dataset("episode_index", data=meta["episode_index"])


def main():
    parser = argparse.ArgumentParser(
        description="Import Kaggle robotic arm dataset into HDF5."
    )
    parser.add_argument(
        "--csv",
        default="robotic_arm_dataset_multiple_trajectories.csv",
        help="Path to Kaggle CSV file.",
    )
    parser.add_argument("--out", default="data/robotic_arm.h5")
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows = load_rows(csv_path)
    source_file = csv_path.name

    episodes = []
    for idx, chunk in split_episodes(rows, args.episode_length, args.max_episodes):
        episodes.append(build_episode(chunk, idx, source_file))

    write_hdf5(Path(args.out), episodes)
    print(f"Imported {len(episodes)} episodes to {args.out}")


if __name__ == "__main__":
    main()
