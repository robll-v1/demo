import argparse
from pathlib import Path

import numpy as np


def _generate_waypoints(rng, pattern):
    if pattern == "zigzag":
        xs = np.linspace(0.0, 6.0, 9)
        ys = np.array([0.0, 0.4, 0.1, 0.8, 0.2, 1.0, 0.4, 1.2, 0.8])
    elif pattern == "circle":
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        xs = 2.0 * np.cos(angles)
        ys = 2.0 * np.sin(angles)
    else:
        xs = np.linspace(0.0, 4.0, 8)
        ys = np.linspace(0.0, 2.0, 8)
    points = np.stack([xs, ys], axis=1)
    points += rng.normal(scale=0.05, size=points.shape)
    return points


def _interpolate_path(points, steps_per_segment):
    path = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        for t in range(steps_per_segment):
            alpha = t / float(steps_per_segment)
            path.append((1 - alpha) * start + alpha * end)
    path.append(points[-1])
    return np.asarray(path, dtype=np.float32)


def generate_episode(rng, pattern, steps_per_segment):
    # 生成一条机器人路径（折线插值 + 噪声）。
    waypoints = _generate_waypoints(rng, pattern)
    path = _interpolate_path(waypoints, steps_per_segment)

    velocity = np.zeros(len(path), dtype=np.float32)
    velocity[1:] = np.linalg.norm(np.diff(path, axis=0), axis=1)

    lane = np.zeros(len(path), dtype=np.int32)
    lane[: len(path) // 3] = 1
    lane[len(path) // 3 : 2 * len(path) // 3] = 2
    lane[2 * len(path) // 3 :] = 3

    timestamps = np.arange(len(path), dtype=np.float32)
    return timestamps, path[:, 0], path[:, 1], velocity, lane


def write_csv(path, timestamps, xs, ys, velocity, lane):
    # 输出 CSV：timestamp,x,y,velocity,lane。
    lines = ["timestamp,x,y,velocity,lane"]
    for t, x, y, v, l in zip(timestamps, xs, ys, velocity, lane):
        lines.append(f"{t:.2f},{x:.3f},{y:.3f},{v:.3f},{int(l)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate robot trajectory CSV files.")
    parser.add_argument("--out-dir", default="data/csv_robot")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--pattern", choices=["line", "zigzag", "circle"], default="zigzag")
    parser.add_argument("--steps-per-segment", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.episodes):
        timestamps, xs, ys, velocity, lane = generate_episode(
            rng, args.pattern, args.steps_per_segment
        )
        path = out_dir / f"robot_{idx:03d}.csv"
        write_csv(path, timestamps, xs, ys, velocity, lane)
    print(f"Wrote {args.episodes} robot CSVs to {out_dir}")


if __name__ == "__main__":
    main()
