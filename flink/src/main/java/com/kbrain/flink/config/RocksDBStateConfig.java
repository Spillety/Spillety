package com.kbrain.flink.config;

import org.apache.flink.contrib.streaming.state.RocksDBStateBackend;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class RocksDBStateConfig {

    public static StreamExecutionEnvironment createEnvironment() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setStateBackend(new RocksDBStateBackend("s3://flink-checkpoints/", true));
        return env;
    }
}
