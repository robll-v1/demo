-- SQLite runbook: create schema and run common queries.

-- 1) Create tables (matches analyze_demo.py)
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data_path TEXT,
    file_size INTEGER,
    read_time REAL,
    episodes INTEGER,
    steps INTEGER,
    created_at TEXT
);

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
);

-- 2) Useful indexes
CREATE INDEX IF NOT EXISTS idx_episodes_run_id ON episodes(run_id);
CREATE INDEX IF NOT EXISTS idx_episodes_path_length ON episodes(path_length);
CREATE INDEX IF NOT EXISTS idx_episodes_quality ON episodes(quality_score);
CREATE INDEX IF NOT EXISTS idx_episodes_anomaly ON episodes(anomaly);
CREATE INDEX IF NOT EXISTS idx_episodes_cluster ON episodes(cluster_id);

-- 3) Queries
-- Latest run
SELECT * FROM runs ORDER BY id DESC LIMIT 1;

-- Top 10 by path length
SELECT episode_id, path_length, avg_step
FROM episodes
ORDER BY path_length DESC
LIMIT 10;

-- Lowest action quality
SELECT episode_id, quality_score
FROM episodes
ORDER BY quality_score ASC
LIMIT 10;

-- Anomalies
SELECT episode_id, path_length, avg_step, action_smoothness
FROM episodes
WHERE anomaly = 1;

-- Cluster distribution
SELECT cluster_id, COUNT(*) AS cnt
FROM episodes
GROUP BY cluster_id
ORDER BY cnt DESC;

-- Lane stats (if present)
SELECT episode_id, lane_counts, lane_mean_speed
FROM episodes
WHERE lane_counts IS NOT NULL
LIMIT 10;
