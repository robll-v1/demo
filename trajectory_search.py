import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from analyze_demo import build_embedding
from mo_client import connect_mo


def _to_serializable(value):
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


def mo_client_from_args(args):
    return connect_mo(
        {
            "host": args.mo_host,
            "port": args.mo_port,
            "user": args.mo_user,
            "password": args.mo_password,
            "database": args.mo_database,
        }
    )


def load_embeddings(client):
    rows = client.execute(
        "SELECT run_id, episode_id, embedding FROM trajectory_embeddings"
    ).fetchall()
    embeddings = []
    for run_id, episode_id, embedding_json in rows:
        embedding = np.asarray(json.loads(embedding_json), dtype=np.float32)
        embeddings.append((run_id, episode_id, embedding))
    return embeddings


def load_episode_metrics(client):
    rows = client.execute(
        """
        SELECT run_id, episode_id, path_length, avg_step, success, steps,
               quality_score, cluster_id
        FROM episodes
        """
    ).fetchall()
    metrics = {}
    for row in rows:
        metrics[(row[0], row[1])] = {
            "path_length": row[2],
            "avg_step": row[3],
            "success": row[4],
            "steps": row[5],
            "quality_score": row[6],
            "cluster_id": row[7],
        }
    return metrics


def cosine_similarity(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def l2_distance(a, b):
    return float(np.linalg.norm(a - b))


def _sql_literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def load_episode_from_hdf5(data_path, episode_id):
    with h5py.File(data_path, "r") as h5:
        group = h5["episodes"][episode_id]
        obs = group["observations"][:]
        actions = group["actions"][:]
        rewards = group["rewards"][:]
        dones = group["dones"][:]
        timestamps = group["timestamps"][:] if "timestamps" in group else None
        meta_group = group["metadata"]
        source_file = ""
        if "source_file" in meta_group:
            raw = meta_group["source_file"][()]
            if isinstance(raw, (bytes, bytearray)):
                source_file = raw.decode("utf-8", errors="ignore")
            else:
                source_file = str(raw)
        metadata = {
            "start": meta_group["start"][:],
            "goal": meta_group["goal"][:],
            "success": bool(meta_group["success"][()]),
            "steps": int(meta_group["steps"][()]),
            "episode_id": episode_id,
            "source_file": source_file,
        }
    return obs, actions, rewards, dones, timestamps, metadata


def write_jsonl(records, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_to_serializable(record), ensure_ascii=True))
            f.write("\n")


def run_similarity(args):
    client = mo_client_from_args(args)
    try:
        embeddings = load_embeddings(client)
        if not embeddings:
            raise SystemExit("No embeddings found. Run analyze_demo.py --mo first.")

        query_embedding = None
        if args.data:
            obs, actions, _, _, _, _ = load_episode_from_hdf5(args.data, args.episode_id)
            query_embedding = build_embedding(obs, actions)
        else:
            for run_id, episode_id, embedding in embeddings:
                if episode_id == args.episode_id and (
                    args.run_id is None or run_id == args.run_id
                ):
                    query_embedding = embedding
                    break
        if query_embedding is None:
            raise SystemExit("Query episode not found in HDF5 or embeddings table.")

        metrics = load_episode_metrics(client)
        scored = []
        for run_id, episode_id, embedding in embeddings:
            if not args.include_self and episode_id == args.episode_id:
                continue
            if args.run_id is not None and run_id != args.run_id:
                continue
            if args.metric == "cosine":
                score = cosine_similarity(query_embedding, embedding)
            else:
                score = -l2_distance(query_embedding, embedding)
            scored.append((score, run_id, episode_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: args.top_k]
        results = []
        for score, run_id, episode_id in top:
            meta = metrics.get((run_id, episode_id), {})
            results.append(
                {
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "score": score,
                    **meta,
                }
            )
        for item in results:
            print(
                f"{item['episode_id']}\t{item['score']:.4f}\t"
                f"path_length={item.get('path_length')}\t"
                f"quality_score={item.get('quality_score')}"
            )
        if args.out:
            write_jsonl(results, Path(args.out))
    finally:
        client.disconnect()


def run_retrieve(args):
    conditions = []
    if args.run_id is not None:
        conditions.append(f"run_id = {_sql_literal(args.run_id)}")
    if args.success is not None:
        conditions.append(f"success = {_sql_literal(args.success)}")
    if args.cluster_id is not None:
        conditions.append(f"cluster_id = {_sql_literal(args.cluster_id)}")
    if args.min_path_length is not None:
        conditions.append(f"path_length >= {_sql_literal(args.min_path_length)}")
    if args.max_path_length is not None:
        conditions.append(f"path_length <= {_sql_literal(args.max_path_length)}")
    if args.min_quality is not None:
        conditions.append(f"quality_score >= {_sql_literal(args.min_quality)}")
    if args.max_quality is not None:
        conditions.append(f"quality_score <= {_sql_literal(args.max_quality)}")

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT run_id, episode_id, path_length, avg_step, success, steps,
               quality_score, cluster_id
        FROM episodes
        {where}
        ORDER BY path_length DESC
        LIMIT {int(args.limit)}
    """

    client = mo_client_from_args(args)
    try:
        rows = client.execute(query).fetchall()
    finally:
        client.disconnect()

    results = []
    for row in rows:
        results.append(
            {
                "run_id": row[0],
                "episode_id": row[1],
                "path_length": row[2],
                "avg_step": row[3],
                "success": row[4],
                "steps": row[5],
                "quality_score": row[6],
                "cluster_id": row[7],
            }
        )

    for item in results:
        print(
            f"{item['episode_id']}\tpath_length={item['path_length']}\t"
            f"quality_score={item['quality_score']}\tcluster_id={item['cluster_id']}"
        )

    if args.export:
        if not args.data:
            raise SystemExit("--data is required to export trajectories.")
        records = []
        for item in results:
            obs, actions, rewards, dones, timestamps, metadata = load_episode_from_hdf5(
                args.data, item["episode_id"]
            )
            record = {
                "episode_id": item["episode_id"],
                "observations": obs,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "timestamps": timestamps,
                "metadata": metadata,
            }
            records.append(record)
        write_jsonl(records, Path(args.export))


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory similarity search and retrieval (MatrixOne + HDF5)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    similar = subparsers.add_parser("similar", help="Find similar trajectories.")
    similar.add_argument("--mo-host", default="127.0.0.1")
    similar.add_argument("--mo-port", type=int, default=6001)
    similar.add_argument("--mo-user", default="root")
    similar.add_argument("--mo-password", default="111")
    similar.add_argument("--mo-database", default="demo")
    similar.add_argument("--data", default=None, help="HDF5 file for query episode.")
    similar.add_argument("--episode-id", required=True)
    similar.add_argument("--run-id", type=int, default=None)
    similar.add_argument("--top-k", type=int, default=5)
    similar.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    similar.add_argument("--include-self", action="store_true")
    similar.add_argument("--out", default=None, help="Write results to JSONL.")

    retrieve = subparsers.add_parser("retrieve", help="Filter and retrieve trajectories.")
    retrieve.add_argument("--mo-host", default="127.0.0.1")
    retrieve.add_argument("--mo-port", type=int, default=6001)
    retrieve.add_argument("--mo-user", default="root")
    retrieve.add_argument("--mo-password", default="111")
    retrieve.add_argument("--mo-database", default="demo")
    retrieve.add_argument("--run-id", type=int, default=None)
    retrieve.add_argument("--success", type=int, choices=[0, 1], default=None)
    retrieve.add_argument("--cluster-id", type=int, default=None)
    retrieve.add_argument("--min-path-length", type=float, default=None)
    retrieve.add_argument("--max-path-length", type=float, default=None)
    retrieve.add_argument("--min-quality", type=float, default=None)
    retrieve.add_argument("--max-quality", type=float, default=None)
    retrieve.add_argument("--limit", type=int, default=20)
    retrieve.add_argument("--data", default=None, help="HDF5 file for exporting trajectories.")
    retrieve.add_argument("--export", default=None, help="Export matched episodes to JSONL.")

    args = parser.parse_args()
    if args.command == "similar":
        run_similarity(args)
    else:
        run_retrieve(args)


if __name__ == "__main__":
    main()
