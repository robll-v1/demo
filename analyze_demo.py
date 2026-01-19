import argparse
import json
import sqlite3
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
    per_episode = []
    total_steps = 0
    for obs, actions, rewards, dones, timestamps, meta in load_hdf5(path):
        # 关节空间路径长度：相邻关节角变化的 L2 距离。
        delta = np.diff(obs, axis=0)
        path_length = np.sum(np.linalg.norm(delta, axis=1))
        total_reward = np.sum(rewards)
        avg_step = np.mean(np.linalg.norm(delta, axis=1)) if len(delta) else 0.0
        stats.append((path_length, total_reward, avg_step, meta["success"]))
        action_mag, action_smoothness, quality_score = compute_action_metrics(obs, actions)
        lane_counts, lane_mean_speed = extract_lane_stats(obs)
        per_episode.append(
            {
                "episode_id": meta.get("episode_id", ""),
                "path_length": float(path_length),
                "total_reward": float(total_reward),
                "avg_step": float(avg_step),
                "success": bool(meta["success"]),
                "steps": int(len(obs)),
                "action_mag": float(action_mag),
                "action_smoothness": float(action_smoothness),
                "quality_score": float(quality_score),
                "anomaly": False,
                "cluster_id": -1,
                "lane_counts": lane_counts,
                "lane_mean_speed": lane_mean_speed,
            }
        )
        steps = to_rlds_steps(obs, actions, rewards, dones, timestamps)
        episodes.append(build_rlds_episode(steps, meta))
        total_steps += len(obs)

    if not stats:
        return [], [], {"episodes": 0, "steps": 0}, []

    stats_arr = np.asarray(stats, dtype=np.float32)
    per_episode = apply_anomaly_detection(per_episode)
    per_episode = apply_clustering(per_episode, k=3)
    return stats_arr, episodes, {"episodes": len(stats_arr), "steps": total_steps}, per_episode


def analyze_streaming(path):
    # 流式统计：不构建 RLDS episodes，内存占用更低。
    stats = []
    per_episode = []
    total_steps = 0
    for obs, actions, rewards, dones, _, meta in load_hdf5(path):
        delta = np.diff(obs, axis=0)
        path_length = np.sum(np.linalg.norm(delta, axis=1))
        total_reward = np.sum(rewards)
        avg_step = np.mean(np.linalg.norm(delta, axis=1)) if len(delta) else 0.0
        stats.append((path_length, total_reward, avg_step, meta["success"]))
        action_mag, action_smoothness, quality_score = compute_action_metrics(obs, actions)
        lane_counts, lane_mean_speed = extract_lane_stats(obs)
        per_episode.append(
            {
                "episode_id": meta.get("episode_id", ""),
                "path_length": float(path_length),
                "total_reward": float(total_reward),
                "avg_step": float(avg_step),
                "success": bool(meta["success"]),
                "steps": int(len(obs)),
                "action_mag": float(action_mag),
                "action_smoothness": float(action_smoothness),
                "quality_score": float(quality_score),
                "anomaly": False,
                "cluster_id": -1,
                "lane_counts": lane_counts,
                "lane_mean_speed": lane_mean_speed,
            }
        )
        total_steps += len(obs)

    if not stats:
        return [], {"episodes": 0, "steps": 0}, []

    stats_arr = np.asarray(stats, dtype=np.float32)
    per_episode = apply_anomaly_detection(per_episode)
    per_episode = apply_clustering(per_episode, k=3)
    return stats_arr, {"episodes": len(stats_arr), "steps": total_steps}, per_episode


def summarize(stats, lane_info, info, per_episode, emit):
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
    emit(f"Action magnitude (avg): {np.mean([p['action_mag'] for p in per_episode]):.2f}")
    emit(f"Action smoothness (avg): {np.mean([p['action_smoothness'] for p in per_episode]):.2f}")
    emit(f"Action quality (avg): {np.mean([p['quality_score'] for p in per_episode]):.3f}")
    emit(f"Anomalies (count): {sum(1 for p in per_episode if p['anomaly'])}")
    cluster_counts = {}
    for item in per_episode:
        cluster_counts[item["cluster_id"]] = cluster_counts.get(item["cluster_id"], 0) + 1
    emit(f"Clusters: {cluster_counts}")
    lane_totals = aggregate_lane_stats(per_episode)
    if lane_totals:
        emit(f"Lane counts: {lane_totals['counts']}")
        emit(f"Lane mean speed: {lane_totals['mean_speed']}")


def plot_trajectories(path, out_path, max_episodes=20, joint_index=0, stats_override=None):
    # 绘制关节角曲线与分布图（单关节，避免过密）。
    import matplotlib.pyplot as plt

    max_episodes = min(max_episodes, 1)
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

    joint_index = max(0, min(joint_index, obs_dim - 1))
    rows = 2
    fig, ax = plt.subplots(rows, 2, figsize=(10, 5.2), dpi=120)

    step_mags = []
    for obs, _ in sampled:
        if len(obs) > 1:
            step = np.linalg.norm(np.diff(obs, axis=0), axis=1)
            step_mags.extend(step.tolist())

    ax_time = ax[0, 0]
    ax_hist = ax[0, 1]
    joint_vals = []
    for obs, timestamps in sampled:
        t = timestamps if timestamps is not None else np.arange(len(obs))
        ax_time.plot(t, obs[:, joint_index], alpha=0.8, linewidth=1.2)
        joint_vals.append(obs[:, joint_index])
    joint_vals = np.concatenate(joint_vals, axis=0)
    ax_time.set_title(f"Joint {joint_index} angle over time")
    ax_time.set_xlabel("time step")
    ax_time.set_ylabel("angle")
    ax_hist.hist(joint_vals, bins=20, alpha=0.8, color="#557a2d")
    ax_hist.set_title(f"Joint {joint_index} angle distribution")
    ax_hist.set_xlabel("angle")
    ax_hist.set_ylabel("count")

    stats = stats_override
    if stats is None:
        stats, _, _, _ = analyze(path)
    ax_len = ax[1, 0]
    ax_step = ax[1, 1]
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


def compute_action_metrics(obs, actions):
    # 动作质量代理指标：优先用 actions，否则用关节变化近似。
    if np.any(actions):
        primary = actions
    else:
        primary = np.diff(obs, axis=0)
    if len(primary) == 0:
        return 0.0, 0.0, 0.0
    action_mag = float(np.mean(np.linalg.norm(primary, axis=1)))
    if len(primary) > 1:
        action_smoothness = float(np.mean(np.linalg.norm(np.diff(primary, axis=0), axis=1)))
    else:
        action_smoothness = 0.0
    quality_score = 1.0 / (1.0 + action_smoothness)
    return action_mag, action_smoothness, quality_score


def extract_lane_stats(obs):
    # 尝试从 obs 的第 4 列提取车道信息（x,y,velocity,lane）。
    if obs.shape[1] < 4:
        return None, None
    lane = obs[:, 3]
    if not np.allclose(lane, np.round(lane), atol=1e-3):
        return None, None
    lane_int = lane.astype(np.int32)
    velocity = obs[:, 2]
    counts = {}
    speed_sum = {}
    speed_cnt = {}
    for l, v in zip(lane_int, velocity):
        counts[int(l)] = counts.get(int(l), 0) + 1
        speed_sum[int(l)] = speed_sum.get(int(l), 0.0) + float(v)
        speed_cnt[int(l)] = speed_cnt.get(int(l), 0) + 1
    mean_speed = {k: speed_sum[k] / speed_cnt[k] for k in counts}
    return counts, mean_speed


def aggregate_lane_stats(per_episode):
    # 汇总所有 episode 的车道统计。
    total_counts = {}
    speed_sum = {}
    speed_cnt = {}
    found = False
    for item in per_episode:
        counts = item.get("lane_counts")
        mean_speed = item.get("lane_mean_speed")
        if not counts or not mean_speed:
            continue
        found = True
        for lane_id, cnt in counts.items():
            total_counts[lane_id] = total_counts.get(lane_id, 0) + int(cnt)
        for lane_id, ms in mean_speed.items():
            speed_sum[lane_id] = speed_sum.get(lane_id, 0.0) + float(ms)
            speed_cnt[lane_id] = speed_cnt.get(lane_id, 0) + 1
    if not found:
        return None
    mean_speed = {k: speed_sum[k] / speed_cnt[k] for k in speed_sum}
    return {"counts": total_counts, "mean_speed": mean_speed}


def apply_anomaly_detection(per_episode):
    # 简单异常检测：基于 z-score 的多指标阈值。
    if not per_episode:
        return per_episode
    features = np.array(
        [
            [p["path_length"], p["avg_step"], p["action_smoothness"]]
            for p in per_episode
        ],
        dtype=np.float32,
    )
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    z = np.abs((features - mean) / std)
    for idx, item in enumerate(per_episode):
        item["anomaly"] = bool(np.any(z[idx] > 3.0))
    return per_episode


def apply_clustering(per_episode, k=3, max_iter=20):
    # 简单 KMeans 聚类，基于 path_length/avg_step/action_smoothness。
    if not per_episode:
        return per_episode
    n = len(per_episode)
    k = max(1, min(k, n))
    data = np.array(
        [
            [p["path_length"], p["avg_step"], p["action_smoothness"]]
            for p in per_episode
        ],
        dtype=np.float32,
    )
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-6
    data = (data - mean) / std
    rng = np.random.default_rng(7)
    centroids = data[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            members = data[labels == i]
            if len(members) > 0:
                centroids[i] = members.mean(axis=0)
    for idx, item in enumerate(per_episode):
        item["cluster_id"] = int(labels[idx])
    return per_episode


def write_sqlite(db_path, per_episode, run_info):
    # 记录 episode 级别指标到 SQLite，便于查询分析。
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_path TEXT,
                file_size INTEGER,
                read_time REAL,
                episodes INTEGER,
                steps INTEGER,
                created_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                run_id INTEGER,
                episode_id TEXT,
                path_length REAL,
                total_reward REAL,
                avg_step REAL,
                success INTEGER,
                steps INTEGER,
                action_mag REAL,
                action_smoothness REAL,
                quality_score REAL,
                anomaly INTEGER,
                cluster_id INTEGER,
                lane_counts TEXT,
                lane_mean_speed TEXT
            )
            """
        )
        cur.execute("PRAGMA table_info(episodes)")
        existing_cols = {row[1] for row in cur.fetchall()}
        for col, col_type in [
            ("action_mag", "REAL"),
            ("action_smoothness", "REAL"),
            ("quality_score", "REAL"),
            ("anomaly", "INTEGER"),
            ("cluster_id", "INTEGER"),
            ("lane_counts", "TEXT"),
            ("lane_mean_speed", "TEXT"),
        ]:
            if col not in existing_cols:
                cur.execute(f"ALTER TABLE episodes ADD COLUMN {col} {col_type}")
        cur.execute(
            """
            INSERT INTO runs (data_path, file_size, read_time, episodes, steps, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_info["data_path"],
                run_info["file_size"],
                run_info["read_time"],
                run_info["episodes"],
                run_info["steps"],
                run_info["created_at"],
            ),
        )
        run_id = cur.lastrowid
        cur.executemany(
            """
            INSERT INTO episodes (
                run_id,
                episode_id,
                path_length,
                total_reward,
                avg_step,
                success,
                steps,
                action_mag,
                action_smoothness,
                quality_score,
                anomaly,
                cluster_id,
                lane_counts,
                lane_mean_speed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item["episode_id"],
                    item["path_length"],
                    item["total_reward"],
                    item["avg_step"],
                    1 if item["success"] else 0,
                    item["steps"],
                    item["action_mag"],
                    item["action_smoothness"],
                    item["quality_score"],
                    1 if item["anomaly"] else 0,
                    item["cluster_id"],
                    json.dumps(item["lane_counts"]) if item["lane_counts"] else None,
                    json.dumps(item["lane_mean_speed"]) if item["lane_mean_speed"] else None,
                )
                for item in per_episode
            ],
        )
        conn.commit()
    finally:
        conn.close()


def write_influx(config, per_episode, run_info):
    # 写入 InfluxDB（2.x）用于时序分析与监控。
    try:
        from influxdb_client import InfluxDBClient, Point
        from influxdb_client.client.write_api import SYNCHRONOUS
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Missing dependency. Install with: python3 -m pip install influxdb-client"
        ) from exc

    url = config["url"]
    token = config["token"]
    org = config["org"]
    bucket = config["bucket"]
    measurement = config["measurement"]
    timestamp = datetime.utcnow()

    with InfluxDBClient(url=url, token=token, org=org) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)

        run_point = (
            Point("run_metrics")
            .tag("data_path", run_info["data_path"])
            .field("file_size", int(run_info["file_size"]))
            .field("read_time", float(run_info["read_time"]))
            .field("episodes", int(run_info["episodes"]))
            .field("steps", int(run_info["steps"]))
            .time(timestamp)
        )
        write_api.write(bucket=bucket, org=org, record=run_point)

        points = []
        lane_points = []
        for item in per_episode:
            points.append(
                Point(measurement)
                .tag("episode_id", item["episode_id"])
                .tag("success", str(int(item["success"])))
                .tag("cluster_id", str(int(item["cluster_id"])))
                .field("path_length", float(item["path_length"]))
                .field("avg_step", float(item["avg_step"]))
                .field("steps", int(item["steps"]))
                .field("action_mag", float(item["action_mag"]))
                .field("action_smoothness", float(item["action_smoothness"]))
                .field("quality_score", float(item["quality_score"]))
                .field("anomaly", int(item["anomaly"]))
                .time(timestamp)
            )
            if item["lane_counts"] and item["lane_mean_speed"]:
                for lane_id, cnt in item["lane_counts"].items():
                    lane_points.append(
                        Point("lane_metrics")
                        .tag("episode_id", item["episode_id"])
                        .tag("lane_id", str(lane_id))
                        .field("steps", int(cnt))
                        .field("mean_speed", float(item["lane_mean_speed"][lane_id]))
                        .time(timestamp)
                    )
        write_api.write(bucket=bucket, org=org, record=points)
        if lane_points:
            write_api.write(bucket=bucket, org=org, record=lane_points)


def main():
    # 命令行入口：统计、绘图、导出。
    parser = argparse.ArgumentParser(description="Analyze HDF5 trajectories with RLDS-style adapter.")
    parser.add_argument("--data", default="data/trajectories.h5")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", default="outputs/trajectories.png")
    parser.add_argument("--plot-samples", type=int, default=1)
    parser.add_argument("--plot-joint", type=int, default=0)
    parser.add_argument("--export", default=None, help="Export RLDS episodes to JSON or JSONL.")
    parser.add_argument("--log", default=None, help="Write summary output to a log file.")
    parser.add_argument("--sqlite", default=None, help="Write episode stats into a SQLite DB.")
    parser.add_argument("--stream", action="store_true", help="Stream stats without building RLDS episodes.")
    parser.add_argument("--influx-url", default=None, help="InfluxDB URL, e.g. http://localhost:8086")
    parser.add_argument("--influx-token", default=None, help="InfluxDB access token")
    parser.add_argument("--influx-org", default=None, help="InfluxDB organization name")
    parser.add_argument("--influx-bucket", default=None, help="InfluxDB bucket name")
    parser.add_argument("--influx-measurement", default="episode_metrics")
    args = parser.parse_args()

    data_path = Path(args.data)
    file_size = data_path.stat().st_size if data_path.exists() else 0
    t0 = time.perf_counter()
    if args.stream and args.export:
        print("Streaming mode cannot export RLDS episodes. Disable --stream to export.")
        return

    if args.stream:
        mode_label = "Streaming mode"
        stats, info, per_episode = analyze_streaming(data_path)
        episodes = []
    else:
        mode_label = "Default mode"
        stats, episodes, info, per_episode = analyze(data_path)
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

    emit(f"Mode: {mode_label}")
    summarize(stats, None, info, per_episode, emit)
    emit(f"File size (bytes): {file_size}")
    emit(f"Read+analysis time (s): {t1 - t0:.3f}")
    emit(f"RLDS episodes loaded: {len(episodes)}")

    if args.plot:
        plot_trajectories(
            data_path,
            Path(args.out),
            max_episodes=args.plot_samples,
            joint_index=args.plot_joint,
            stats_override=stats,
        )
        emit(f"Saved plot to {args.out}")
    if args.export:
        export_episodes(episodes, Path(args.export))
        emit(f"Exported RLDS episodes to {args.export}")
    if args.sqlite:
        run_info = {
            "data_path": str(data_path),
            "file_size": int(file_size),
            "read_time": float(t1 - t0),
            "episodes": int(info["episodes"]),
            "steps": int(info["steps"]),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_sqlite(Path(args.sqlite), per_episode, run_info)
        emit(f"Wrote SQLite stats to {args.sqlite}")
    if args.influx_url and args.influx_token and args.influx_org and args.influx_bucket:
        run_info = {
            "data_path": str(data_path),
            "file_size": int(file_size),
            "read_time": float(t1 - t0),
            "episodes": int(info["episodes"]),
            "steps": int(info["steps"]),
        }
        write_influx(
            {
                "url": args.influx_url,
                "token": args.influx_token,
                "org": args.influx_org,
                "bucket": args.influx_bucket,
                "measurement": args.influx_measurement,
            },
            per_episode,
            run_info,
        )
        emit("Wrote InfluxDB metrics")

    if log_file is not None:
        log_file.close()


if __name__ == "__main__":
    main()
