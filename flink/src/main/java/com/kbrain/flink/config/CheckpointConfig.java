package com.kbrain.flink.config;

import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class CheckpointConfig {

    static final long CHECKPOINT_INTERVAL_MS = 60_000L;
    static final long MIN_PAUSE_BETWEEN_CHECKPOINTS_MS = 30_000L;
    static final long CHECKPOINT_TIMEOUT_MS = 300_000L;

    public static void configure(StreamExecutionEnvironment env) {
        env.enableCheckpointing(CHECKPOINT_INTERVAL_MS,
            org.apache.flink.streaming.api.environment.CheckpointConfig.CheckpointingMode.EXACTLY_ONCE);
        env.setMinPauseBetweenCheckpoints(MIN_PAUSE_BETWEEN_CHECKPOINTS_MS);
        env.setCheckpointTimeout(CHECKPOINT_TIMEOUT_MS);
        env.setMaxConcurrentCheckpoints(1);
        env.setFailOnCheckpointingErrors(true);
        env.getCheckpointConfig().enableExternalizedCheckpoints(
            org.apache.flink.streaming.api.environment.CheckpointConfig.ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);
    }
}
