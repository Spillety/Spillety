package com.kbrain.flink.config;

import org.apache.flink.api.common.state.StateTtlConfig;
import org.apache.flink.api.common.time.Time;
import org.apache.flink.contrib.streaming.state.RocksDBStateBackend;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class RocksDBStateConfig {

    static final int HAWKES_TTL_DAYS = 14;
    static final int PAGERANK_TTL_DAYS = 30;

    static StreamExecutionEnvironment createEnvironment() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setStateBackend(new RocksDBStateBackend("s3://flink-checkpoints/", true));
        return env;
    }

    static StateTtlConfig createHawkesTtlConfig() {
        return StateTtlConfig.newBuilder(Time.days(HAWKES_TTL_DAYS))
            .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite)
            .setStateVisibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
            .setCleanupFullSnapshotMode(StateTtlConfig.CleanupFullSnapshotMode.RETAIN)
            .build();
    }

    static StateTtlConfig createPageRankTtlConfig() {
        return StateTtlConfig.newBuilder(Time.days(PAGERANK_TTL_DAYS))
            .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite)
            .setStateVisibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
            .setCleanupFullSnapshotMode(StateTtlConfig.CleanupFullSnapshotMode.RETAIN)
            .build();
    }
}
