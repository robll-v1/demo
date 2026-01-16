import argparse
from pathlib import Path

import h5py
import numpy as np


def generate_episode(rng, max_steps):
    # 生成一条带目标点的 2D 轨迹。
    goal = rng.uniform(-4.0, 4.0, size=(2,))
    position = rng.uniform(-1.0, 1.0, size=(2,))
    start = position.copy()
    velocity = np.zeros(2, dtype=np.float32)

    observations = []
    actions = []
    rewards = []
    dones = []

    for step in range(max_steps):
        # 简单控制：朝向目标前进并加入噪声。
        direction = goal - position
        dist = np.linalg.norm(direction) + 1e-6
        desired_velocity = 0.2 * direction / dist
        noise = rng.normal(scale=0.03, size=(2,))
        action = desired_velocity + noise
        velocity = 0.7 * velocity + 0.3 * action
        position = position + velocity

        obs = np.array(
            [
                position[0],
                position[1],
                velocity[0],
                velocity[1],
                goal[0],
                goal[1],
                dist,
            ],
            dtype=np.float32,
        )
        reward = -dist
        done = dist < 0.2 or step == max_steps - 1

        observations.append(obs)
        actions.append(action.astype(np.float32))
        rewards.append(np.float32(reward))
        dones.append(done)

        if done:
            break

    # 汇总并生成元信息。
    observations = np.stack(observations, axis=0)
    actions = np.stack(actions, axis=0)
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.bool_)
    success = np.bool_(dones[-1] and (np.linalg.norm(goal - position) < 0.2))

    metadata = {
        "goal": goal.astype(np.float32),
        "start": start.astype(np.float32),
        "success": success,
        "steps": np.int32(len(observations)),
    }
    return observations, actions, rewards, dones, metadata


def write_hdf5(path, episodes, max_steps, seed):
    # 写入 HDF5：把生成的轨迹存到 episodes/ep_xxxxx/...
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as h5:
        root = h5.create_group("episodes")
        for idx in range(episodes):
            obs, acts, rewards, dones, metadata = generate_episode(rng, max_steps)
            group = root.create_group(f"ep_{idx:05d}")
            group.create_dataset("observations", data=obs, compression="gzip")
            group.create_dataset("actions", data=acts, compression="gzip")
            group.create_dataset("rewards", data=rewards, compression="gzip")
            group.create_dataset("dones", data=dones, compression="gzip")
            meta_group = group.create_group("metadata")
            meta_group.create_dataset("goal", data=metadata["goal"])
            meta_group.create_dataset("start", data=metadata["start"])
            meta_group.create_dataset("success", data=metadata["success"])
            meta_group.create_dataset("steps", data=metadata["steps"])


def main():
    # 命令行入口：生成指定数量的轨迹。
    parser = argparse.ArgumentParser(description="Generate synthetic HDF5 trajectories.")
    parser.add_argument("--out", default="data/trajectories.h5")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    write_hdf5(Path(args.out), args.episodes, args.max_steps, args.seed)
    print(f"Wrote {args.episodes} episodes to {args.out}")


if __name__ == "__main__":
    main()
