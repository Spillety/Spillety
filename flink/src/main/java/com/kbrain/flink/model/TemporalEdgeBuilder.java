package com.kbrain.flink.model;

import org.apache.flink.api.common.functions.MapFunction;

// # ponytail: close superseded edges by setting validTo instead of emitting open-only edges.
public class TemporalEdgeBuilder implements MapFunction<OnChainEvent, TemporalEdge> {

    @Override
    public TemporalEdge map(OnChainEvent event) {
        return new TemporalEdge(
            "TRANSACTS",
            event.getFromAddress(),
            event.getToAddress(),
            event.getTxHash(),
            event.getTimestamp(),
            null);
    }
}
