import argparse
import json
import time
from datetime import datetime
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
    total_steps = 0
    for obs, actions, rewards, dones, timestamps, meta in load_hdf5(path):
        # 关节空间路径长度：相邻关节角变化的 L2 距离。
        delta = np.diff(obs, axis=0)
        path_length = np.sum(np.linalg.norm(delta, axis=1))
        total_reward = np.sum(rewards)
        avg_step = np.mean(np.linalg.norm(delta, axis=1)) if len(delta) else 0.0
        stats.append((path_length, total_reward, avg_step, meta["success"]))
        steps = to_rlds_steps(obs, actions, rewards, dones, timestamps)
        episodes.append(build_rlds_episode(steps, meta))
        total_steps += len(obs)

    if not stats:
        return [], [], {"episodes": 0, "steps": 0}

    stats_arr = np.asarray(stats, dtype=np.float32)
    return stats_arr, episodes, {"episodes": len(stats_arr), "steps": total_steps}


def summarize(stats, lane_info, info, emit):
    # 打印轨迹统计摘要。
    path_len = stats[:, 0]
    total_reward = stats[:, 1]
    avg_step = stats[:, 2]
    success = stats[:, 3]

    emit("Trajectory summary")
    emit(f"Episodes: {len(stats)}")
    emit(f"Success rate: {np.mean(success) * 100:.1f}%")
    emit(f"Path length (avg): {np.mean(path_len):.2f}")
    emit(f"Total reward (avg): {np.mean(total_reward):.2f}")
    emit(f"Mean step (avg): {np.mean(avg_step):.2f}")
    emit(f"Total steps: {info['steps']}")
    emit(f"Avg steps per episode: {info['steps'] / max(info['episodes'], 1):.2f}")


def plot_trajectories(path, out_path, max_episodes=20):
    # 绘制关节角曲线与分布图（每个关节单独子图）。
    import matplotlib.pyplot as plt

    sampled = []
    obs_dim = None
    for idx, (obs, _, _, _, timestamps, _) in enumerate(load_hdf5(path)):
        if idx >= max_episodes:
            break
        sampled.append((obs, timestamps))
        if obs_dim is None:
            obs_dim = obs.shape[1]

    if not sampled:
        return

    rows = obs_dim + 1
    fig, ax = plt.subplots(rows, 2, figsize=(10, 2.6 * rows), dpi=120)

    step_mags = []
    for obs, _ in sampled:
        if len(obs) > 1:
            step = np.linalg.norm(np.diff(obs, axis=0), axis=1)
            step_mags.extend(step.tolist())

    for j in range(obs_dim):
        ax_time = ax[j, 0]
        ax_hist = ax[j, 1]
        joint_vals = []
        for obs, timestamps in sampled:
            t = timestamps if timestamps is not None else np.arange(len(obs))
            ax_time.plot(t, obs[:, j], alpha=0.7, linewidth=1.0)
            joint_vals.append(obs[:, j])
        joint_vals = np.concatenate(joint_vals, axis=0)
        ax_time.set_title(f"Joint {j} angle over time")
        ax_time.set_xlabel("time step")
        ax_time.set_ylabel("angle")
        ax_hist.hist(joint_vals, bins=20, alpha=0.8, color="#557a2d")
        ax_hist.set_title(f"Joint {j} angle distribution")
        ax_hist.set_xlabel("angle")
        ax_hist.set_ylabel("count")

    stats, _, _ = analyze(path)
    ax_len = ax[rows - 1, 0]
    ax_step = ax[rows - 1, 1]
    ax_len.hist(stats[:, 0], bins=10, alpha=0.8)
    ax_len.set_title("Joint-space path length")
    ax_len.set_xlabel("path length")
    ax_len.set_ylabel("count")
    ax_step.hist(step_mags, bins=20, alpha=0.8, color="#7a4f9e")
    ax_step.set_title("Step magnitude distribution")
    ax_step.set_xlabel("delta angle")
    ax_step.set_ylabel("count")

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
    parser.add_argument("--plot-samples", type=int, default=20)
    parser.add_argument("--export", default=None, help="Export RLDS episodes to JSON or JSONL.")
    parser.add_argument("--log", default=None, help="Write summary output to a log file.")
    args = parser.parse_args()

    data_path = Path(args.data)
    file_size = data_path.stat().st_size if data_path.exists() else 0
    t0 = time.perf_counter()
    stats, episodes, info = analyze(data_path)
    t1 = time.perf_counter()
    if len(stats) == 0:
        print("No episodes found.")
        return
    log_file = None
    if args.log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.log in ("auto", "now"):
            log_value = f"logs/run_{timestamp}.log"
        else:
            log_value = args.log.replace("{timestamp}", timestamp)
        log_path = Path(log_value)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")

    def emit(message):
        print(message)
        if log_file is not None:
            log_file.write(message + "\n")

    summarize(stats, None, info, emit)
    emit(f"File size (bytes): {file_size}")
    emit(f"Read+analysis time (s): {t1 - t0:.3f}")
    emit(f"RLDS episodes loaded: {len(episodes)}")

    if args.plot:
        plot_trajectories(data_path, Path(args.out), max_episodes=args.plot_samples)
        emit(f"Saved plot to {args.out}")
    if args.export:
        export_episodes(episodes, Path(args.export))
        emit(f"Exported RLDS episodes to {args.export}")

    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    main()
