import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from rlds_adapter import build_rlds_episode, rlds_keys


def load_hdf5(path):
    # 从 HDF5 读入数据并逐条产出 episode（包含可选时间戳）。
    with h5py.File(path, "r") as h5:
        for name, group in h5["episodes"].items():
            obs = group["observations"][:]
            actions = group["actions"][:]
            rewards = group["rewards"][:]
            dones = group["dones"][:]
            timestamps = group["timestamps"][:] if "timestamps" in group else None
            meta = {
                "goal": group["metadata"]["goal"][:],
                "start": group["metadata"]["start"][:],
                "success": bool(group["metadata"]["success"][()]),
                "steps": int(group["metadata"]["steps"][()]),
                "episode_id": name,
            }
            if "source_file" in group["metadata"]:
                source = group["metadata"]["source_file"][()]
                if isinstance(source, (bytes, bytearray)):
                    source = source.decode("utf-8", errors="ignore")
                meta["source_file"] = str(source)
            yield obs, actions, rewards, dones, timestamps, meta


def to_rlds_steps(obs, actions, rewards, dones, timestamps):
    # 把一条轨迹转换为 RLDS steps 列表（可附带 timestamp）。
    keys = rlds_keys()
    steps = []
    for i in range(len(obs)):
        step = {
            keys["observation"]: obs[i],
            keys["action"]: actions[i],
            keys["reward"]: rewards[i],
            keys["is_first"]: i == 0,
            keys["is_last"]: i == len(obs) - 1,
            keys["is_terminal"]: bool(dones[i]),
        }
        if timestamps is not None:
            step["timestamp"] = float(timestamps[i])
        steps.append(step)
    return steps


def analyze(path):
    # 从 HDF5 读入 -> 转成 RLDS -> 计算轨迹统计。
    stats = []
    episodes = []
    for obs, actions, rewards, dones, timestamps, meta in load_hdf5(path):
        delta = np.diff(obs[:, :2], axis=0)
        path_length = np.sum(np.linalg.norm(delta, axis=1))
        total_reward = np.sum(rewards)
        if obs.shape[1] == 4:
            speed = np.mean(obs[:, 2])
        else:
            speed = np.mean(np.linalg.norm(obs[:, 2:4], axis=1))
        stats.append((path_length, total_reward, speed, meta["success"]))
        steps = to_rlds_steps(obs, actions, rewards, dones, timestamps)
        episodes.append(build_rlds_episode(steps, meta))

    if not stats:
        return [], []

    stats_arr = np.asarray(stats, dtype=np.float32)
    return stats_arr, episodes


def _extract_lane(obs):
    # 仅当 obs 形状匹配 CSV（x,y,velocity,lane）时提取车道。
    if obs.shape[1] == 4:
        return obs[:, 3].astype(np.int32)
    return None


def collect_lane_info(path):
    # 汇总车道级别统计（步数与均速）。
    lane_counts = {}
    lane_speed_sum = {}
    lane_speed_count = {}

    for obs, _, _, _, _, _ in load_hdf5(path):
        lanes = _extract_lane(obs)
        if lanes is None:
            continue
        speeds = obs[:, 2]
        for lane, speed in zip(lanes, speeds):
            lane_counts[lane] = lane_counts.get(lane, 0) + 1
            lane_speed_sum[lane] = lane_speed_sum.get(lane, 0.0) + float(speed)
            lane_speed_count[lane] = lane_speed_count.get(lane, 0) + 1

    if not lane_counts:
        return None

    lane_mean_speed = {
        lane: lane_speed_sum[lane] / lane_speed_count[lane]
        for lane in lane_speed_sum
    }
    return {"counts": lane_counts, "mean_speed": lane_mean_speed}


def summarize(stats, lane_info):
    # 打印轨迹统计摘要。
    path_len = stats[:, 0]
    total_reward = stats[:, 1]
    speed = stats[:, 2]
    success = stats[:, 3]

    print("Trajectory summary")
    print(f"Episodes: {len(stats)}")
    print(f"Success rate: {np.mean(success) * 100:.1f}%")
    print(f"Path length (avg): {np.mean(path_len):.2f}")
    print(f"Total reward (avg): {np.mean(total_reward):.2f}")
    print(f"Mean speed (avg): {np.mean(speed):.2f}")
    if lane_info is not None:
        print("Lane stats:")
        for lane in sorted(lane_info["counts"].keys()):
            count = lane_info["counts"][lane]
            mean_speed = lane_info["mean_speed"][lane]
            print(f"  lane {lane}: steps={count}, mean_speed={mean_speed:.2f}")


def plot_trajectories(path, out_path):
    # 绘制轨迹与分布图。
    import matplotlib.pyplot as plt

    lane_info = collect_lane_info(path)
    if lane_info is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
        ax_traj = ax[0]
        ax_dist = ax[1]
        ax_speed = None
        ax_lane = None
    else:
        fig, ax = plt.subplots(2, 2, figsize=(10, 7), dpi=120)
        ax_traj = ax[0, 0]
        ax_dist = ax[0, 1]
        ax_speed = ax[1, 0]
        ax_lane = ax[1, 1]
    goal_points = []
    success_flags = []
    all_speeds = []

    for obs, _, _, _, _, meta in load_hdf5(path):
        ax_traj.plot(obs[:, 0], obs[:, 1], alpha=0.7, color="#3b6ea5")
        if lane_info is not None:
            lanes = _extract_lane(obs)
            ax_traj.scatter(
                obs[:, 0],
                obs[:, 1],
                c=lanes,
                cmap="tab10",
                s=12,
                alpha=0.9,
            )
        goal_points.append(meta["goal"])
        success_flags.append(meta["success"])
        if obs.shape[1] == 4:
            all_speeds.extend(obs[:, 2].tolist())
        else:
            all_speeds.extend(np.linalg.norm(obs[:, 2:4], axis=1).tolist())

    goal_points = np.asarray(goal_points)
    success_flags = np.asarray(success_flags)

    ax_traj.scatter(goal_points[:, 0], goal_points[:, 1], c=success_flags, cmap="coolwarm")
    ax_traj.set_title("Trajectories and goals")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.axis("equal")

    stats, _ = analyze(path)
    ax_dist.hist(stats[:, 0], bins=10, alpha=0.8)
    ax_dist.set_title("Path length distribution")
    ax_dist.set_xlabel("path length")
    ax_dist.set_ylabel("count")

    if ax_speed is not None:
        ax_speed.hist(all_speeds, bins=10, alpha=0.8, color="#557a2d")
        ax_speed.set_title("Speed distribution")
        ax_speed.set_xlabel("speed")
        ax_speed.set_ylabel("count")

    if ax_lane is not None and lane_info is not None:
        lanes = sorted(lane_info["counts"].keys())
        counts = [lane_info["counts"][lane] for lane in lanes]
        ax_lane.bar([str(l) for l in lanes], counts, color="#7a4f9e", alpha=0.8)
        ax_lane.set_title("Lane step counts")
        ax_lane.set_xlabel("lane")
        ax_lane.set_ylabel("steps")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _to_serializable(value):
    # 将 numpy 类型转换为可 JSON 序列化的对象。
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def export_episodes(episodes, out_path):
    # 将 RLDS episodes 导出为 JSON 或 JSONL。
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_to_serializable(ep) for ep in episodes]

    if out_path.suffix == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for ep in payload:
                f.write(json.dumps(ep, ensure_ascii=True))
                f.write("\n")
        return

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def main():
    # 命令行入口：统计、绘图、导出。
    parser = argparse.ArgumentParser(description="Analyze HDF5 trajectories with RLDS-style adapter.")
    parser.add_argument("--data", default="data/trajectories.h5")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", default="outputs/trajectories.png")
    parser.add_argument("--export", default=None, help="Export RLDS episodes to JSON or JSONL.")
    args = parser.parse_args()

    stats, episodes = analyze(Path(args.data))
    if len(stats) == 0:
        print("No episodes found.")
        return
    lane_info = collect_lane_info(Path(args.data))
    summarize(stats, lane_info)
    print(f"RLDS episodes loaded: {len(episodes)}")

    if args.plot:
        plot_trajectories(Path(args.data), Path(args.out))
        print(f"Saved plot to {args.out}")
    if args.export:
        export_episodes(episodes, Path(args.export))
        print(f"Exported RLDS episodes to {args.export}")


if __name__ == "__main__":
    main()
