-- Test case data for SQLite
DELETE FROM episodes;
DELETE FROM runs;

INSERT INTO runs (data_path, file_size, read_time, episodes, steps, created_at)
VALUES ('data/test.h5', 123456, 0.42, 2, 400, '2026-01-19 16:00:00');

INSERT INTO episodes (
  run_id, episode_id, path_length, total_reward, avg_step, success, steps,
  action_mag, action_smoothness, quality_score, anomaly, cluster_id,
  lane_counts, lane_mean_speed
) VALUES
  (1, 'ep_00000', 10.5, -5.0, 0.05, 1, 200, 0.10, 0.20, 0.833, 0, 1,
   '{"1":120,"2":80}', '{"1":0.20,"2":0.30}'),
  (1, 'ep_00001', 15.2, -8.0, 0.08, 1, 200, 0.12, 0.25, 0.800, 1, 2,
   NULL, NULL);

-- Queries
SELECT * FROM runs ORDER BY id DESC LIMIT 1;
SELECT episode_id, path_length, avg_step FROM episodes ORDER BY path_length DESC LIMIT 10;
SELECT episode_id, quality_score FROM episodes ORDER BY quality_score ASC LIMIT 10;
SELECT episode_id, path_length, avg_step, action_smoothness FROM episodes WHERE anomaly = 1;
SELECT cluster_id, COUNT(*) AS cnt FROM episodes GROUP BY cluster_id ORDER BY cnt DESC;
SELECT episode_id, lane_counts, lane_mean_speed FROM episodes WHERE lane_counts IS NOT NULL LIMIT 10;
