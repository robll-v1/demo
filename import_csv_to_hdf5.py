import argparse
import csv
from pathlib import Path

import numpy as np


def load_csv_episode(path):
    # 读取单个 CSV 并转换为一条 episode。
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows in {path}")

    has_lane = "lane" in rows[0]
    observations = []
    timestamps = []

    for row in rows:
        # 观测向量：x, y, velocity (+ lane 可选)。
        obs = [
            float(row["x"]),
            float(row["y"]),
            float(row["velocity"]),
        ]
        if has_lane:
            obs.append(float(row["lane"]))
        observations.append(obs)
        # 时间戳按 float 保存，方便后续导出。
        timestamps.append(float(row["timestamp"]))

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.zeros((len(observations), 2), dtype=np.float32)
    rewards = np.zeros(len(observations), dtype=np.float32)
    dones = np.zeros(len(observations), dtype=np.bool_)
    dones[-1] = True
    timestamps = np.asarray(timestamps, dtype=np.float32)

    start = observations[0, :2]
    goal = observations[-1, :2]

    # 元信息：起点、终点、步数与来源文件名。
    metadata = {
        "goal": goal.astype(np.float32),
        "start": start.astype(np.float32),
        "success": np.bool_(True),
        "steps": np.int32(len(observations)),
        "source_file": path.name,
    }

    return observations, actions, rewards, dones, timestamps, metadata


def write_hdf5_from_csv(out_path, csv_files):
    # 从多个 CSV 写入统一的 HDF5 结构。
    out_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = len(csv_files)

    # Reuse the HDF5 writer to keep the same storage layout.
    def episode_generator():
        for path in csv_files:
            yield load_csv_episode(path)

    with write_hdf5_from_generator(out_path, episode_generator(), episodes):
        pass


class write_hdf5_from_generator:
    def __init__(self, path, episode_iter, episodes):
        self.path = path
        self.episode_iter = episode_iter
        self.episodes = episodes

    def __enter__(self):
        # 以固定布局写入 HDF5，保持与 demo 一致。
        import h5py

        self.h5 = h5py.File(self.path, "w")
        root = self.h5.create_group("episodes")
        for idx in range(self.episodes):
            obs, acts, rewards, dones, timestamps, metadata = next(self.episode_iter)
            group = root.create_group(f"ep_{idx:05d}")
            group.create_dataset("observations", data=obs, compression="gzip")
            group.create_dataset("actions", data=acts, compression="gzip")
            group.create_dataset("rewards", data=rewards, compression="gzip")
            group.create_dataset("dones", data=dones, compression="gzip")
            group.create_dataset("timestamps", data=timestamps, compression="gzip")
            meta_group = group.create_group("metadata")
            meta_group.create_dataset("goal", data=metadata["goal"])
            meta_group.create_dataset("start", data=metadata["start"])
            meta_group.create_dataset("success", data=metadata["success"])
            meta_group.create_dataset("steps", data=metadata["steps"])
            meta_group.create_dataset("source_file", data=np.bytes_(metadata["source_file"]))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.h5.close()
        return False


def main():
    # 命令行入口：读取 CSV 目录并输出 HDF5。
    parser = argparse.ArgumentParser(description="Import CSV trajectories into HDF5.")
    parser.add_argument("--csv-dir", default="data/csv")
    parser.add_argument("--out", default="data/trajectories.h5")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {csv_dir}")

    write_hdf5_from_csv(Path(args.out), csv_files)
    print(f"Imported {len(csv_files)} episodes to {args.out}")


if __name__ == "__main__":
    main()
