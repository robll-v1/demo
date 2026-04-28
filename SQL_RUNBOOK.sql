-- MatrixOne runbook: create schema and run common queries.

-- 1) Create database and tables (matches analyze_demo.py)
CREATE DATABASE IF NOT EXISTS demo;
USE demo;
CREATE TABLE IF NOT EXISTS runs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    data_path VARCHAR(512),
    file_size BIGINT,
    read_time DOUBLE,
    episodes INT,
    steps INT,
    created_at VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS episodes (
    run_id BIGINT,
    episode_id VARCHAR(64),
    path_length DOUBLE,
    total_reward DOUBLE,
    avg_step DOUBLE,
    success TINYINT,
    steps INT,
    action_mag DOUBLE,
    action_smoothness DOUBLE,
    quality_score DOUBLE,
    anomaly TINYINT,
    cluster_id INT,
    lane_counts TEXT,
    lane_mean_speed TEXT
);

CREATE TABLE IF NOT EXISTS trajectory_embeddings (
    run_id BIGINT,
    episode_id VARCHAR(64),
    embedding TEXT,
    PRIMARY KEY (run_id, episode_id)
);

CREATE TABLE IF NOT EXISTS trajectory_steps (
    run_id BIGINT,
    episode_id VARCHAR(64),
    step_index INT,
    timestamp DOUBLE,
    reward DOUBLE,
    is_terminal TINYINT,
    step_mag DOUBLE
);

-- 2) Useful indexes (run once; remove if already created)
-- CREATE INDEX idx_episodes_run_id ON episodes(run_id);
-- CREATE INDEX idx_episodes_path_length ON episodes(path_length);
-- CREATE INDEX idx_episodes_quality ON episodes(quality_score);
-- CREATE INDEX idx_episodes_anomaly ON episodes(anomaly);
-- CREATE INDEX idx_episodes_cluster ON episodes(cluster_id);
-- CREATE INDEX idx_embeddings_episode_id ON trajectory_embeddings(episode_id);

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

-- Embeddings (sample)
SELECT episode_id, embedding
FROM trajectory_embeddings
LIMIT 5;

-- Time-series: steps per time bucket
SELECT FLOOR(timestamp) AS t, COUNT(*) AS steps
FROM trajectory_steps
GROUP BY t
ORDER BY t;

-- Time-series: average reward per time bucket
SELECT FLOOR(timestamp) AS t, AVG(reward) AS avg_reward
FROM trajectory_steps
GROUP BY t
ORDER BY t;

-- Time-series: average step magnitude per time bucket
SELECT FLOOR(timestamp) AS t, AVG(step_mag) AS avg_step_mag
FROM trajectory_steps
GROUP BY t
ORDER BY t;

-- Time-series: terminal events per time bucket
SELECT FLOOR(timestamp) AS t, SUM(is_terminal) AS terminal_steps
FROM trajectory_steps
GROUP BY t
ORDER BY t;

-- Time-series: per-episode step counts (top 10 by length)
SELECT episode_id, COUNT(*) AS steps
FROM trajectory_steps
GROUP BY episode_id
ORDER BY steps DESC
LIMIT 10;

-- Time-series: timestamp range per episode
SELECT episode_id, MIN(timestamp) AS t_min, MAX(timestamp) AS t_max
FROM trajectory_steps
GROUP BY episode_id
ORDER BY episode_id;

-- Time-series: time range filter (example: t between 5 and 15)
SELECT FLOOR(timestamp) AS t, COUNT(*) AS steps
FROM trajectory_steps
WHERE timestamp BETWEEN 5 AND 15
GROUP BY t
ORDER BY t;

-- Time-series: cumulative steps over time
SELECT t, SUM(steps) OVER (ORDER BY t) AS cumulative_steps
FROM (
  SELECT FLOOR(timestamp) AS t, COUNT(*) AS steps
  FROM trajectory_steps
  GROUP BY t
) AS buckets
ORDER BY t;

-- Time-series: step magnitude delta (change vs previous bucket)
SELECT t, avg_step_mag,
       avg_step_mag - LAG(avg_step_mag) OVER (ORDER BY t) AS delta_step_mag
FROM (
  SELECT FLOOR(timestamp) AS t, AVG(step_mag) AS avg_step_mag
  FROM trajectory_steps
  GROUP BY t
) AS mags
ORDER BY t;

-- Time-series: top 5 time buckets by step magnitude
SELECT t, avg_step_mag
FROM (
  SELECT FLOOR(timestamp) AS t, AVG(step_mag) AS avg_step_mag
  FROM trajectory_steps
  GROUP BY t
) AS mags
ORDER BY avg_step_mag DESC
LIMIT 5;
