import argparse
from pathlib import Path

import numpy as np

def main():
    # 命令行入口：导出 TFDS 数据集。
    parser = argparse.ArgumentParser(description="Export HDF5 trajectories to TFDS format.")
    parser.add_argument("--data", default="data/trajectories.h5")
    parser.add_argument("--out", default="datasets")
    args = parser.parse_args()

    try:
        import h5py
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:
        # 依赖缺失时给出安装提示。
        raise SystemExit(
            "Missing dependency. Install with: python3 -m pip install tensorflow tensorflow-datasets"
        ) from exc

    data_path = Path(args.data)
    out_dir = Path(args.out)

    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    with h5py.File(data_path, "r") as h5:
        # 推断观测与动作维度。
        first = next(iter(h5["episodes"].values()))
        obs_dim = int(first["observations"].shape[1])
        act_dim = int(first["actions"].shape[1])
        has_timestamps = "timestamps" in first

    class TrajectoryBuilder(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("0.1.0")

        def __init__(self, data_path, **kwargs):
            # 保存数据路径，交给生成器读取。
            self._data_path = data_path
            super().__init__(**kwargs)

        def _info(self):
            # 定义 TFDS 特征结构（steps + episode_metadata）。
            step_features = {
                "observation": tfds.features.Tensor(shape=(obs_dim,), dtype=np.float32),
                "action": tfds.features.Tensor(shape=(act_dim,), dtype=np.float32),
                "reward": np.float32,
                "is_first": np.bool_,
                "is_last": np.bool_,
                "is_terminal": np.bool_,
            }
            if has_timestamps:
                step_features["timestamp"] = np.float32
            return tfds.core.DatasetInfo(
                builder=self,
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Sequence(step_features),
                        "episode_metadata": tfds.features.FeaturesDict(
                            {
                                "start": tfds.features.Tensor(shape=(2,), dtype=np.float32),
                                "goal": tfds.features.Tensor(shape=(2,), dtype=np.float32),
                                "success": np.bool_,
                                "steps": np.int32,
                                "episode_id": tfds.features.Text(),
                                "source_file": tfds.features.Text(),
                            }
                        ),
                    }
                ),
            )

        def _split_generators(self, dl_manager):
            # 单一 split：train。
            return {"train": self._generate_examples()}

        def _generate_examples(self):
            # 逐条读取 HDF5 episode 并产出样本。
            with h5py.File(self._data_path, "r") as h5:
                for name, group in h5["episodes"].items():
                    obs = group["observations"][:]
                    actions = group["actions"][:]
                    rewards = group["rewards"][:]
                    dones = group["dones"][:]
                    timestamps = group["timestamps"][:] if "timestamps" in group else None

                    meta_group = group["metadata"]
                    source_file = ""
                    if "source_file" in meta_group:
                        # 兼容 bytes/string 两种存储。
                        raw = meta_group["source_file"][()]
                        if isinstance(raw, (bytes, bytearray)):
                            source_file = raw.decode("utf-8", errors="ignore")
                        else:
                            source_file = str(raw)

                    meta = {
                        "start": meta_group["start"][:],
                        "goal": meta_group["goal"][:],
                        "success": bool(meta_group["success"][()]),
                        "steps": int(meta_group["steps"][()]),
                        "episode_id": name,
                        "source_file": source_file,
                    }

                    steps = []
                    for i in range(len(obs)):
                        step = {
                            "observation": obs[i],
                            "action": actions[i],
                            "reward": float(rewards[i]),
                            "is_first": i == 0,
                            "is_last": i == len(obs) - 1,
                            "is_terminal": bool(dones[i]),
                        }
                        if timestamps is not None:
                            step["timestamp"] = float(timestamps[i])
                        steps.append(step)

                    yield name, {"steps": steps, "episode_metadata": meta}

    builder = TrajectoryBuilder(data_path=str(data_path), data_dir=str(out_dir))
    builder.download_and_prepare()
    print(f"TFDS dataset written to {out_dir}")


if __name__ == "__main__":
    main()
