package com.kbrain.flink.config;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import com.kbrain.flink.model.OnChainEvent;
import java.time.Duration;

public class EventTimeConfig {

    static final Duration MAX_OUT_OF_ORDERNESS = Duration.ofSeconds(30);

    public static WatermarkStrategy<OnChainEvent> watermarkStrategy() {
        return WatermarkStrategy.<OnChainEvent>forBoundedOutOfOrderness(MAX_OUT_OF_ORDERNESS)
            .withTimestampAssigner((event, recordTimestamp) -> event.getTimestamp());
    }
}
