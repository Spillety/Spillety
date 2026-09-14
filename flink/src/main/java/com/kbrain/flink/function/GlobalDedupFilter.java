package com.kbrain.flink.function;

import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;
import com.kbrain.flink.model.OnChainEvent;

// # ponytail: add StateTtlConfig on seenState once volume requires compaction.
public class GlobalDedupFilter extends KeyedProcessFunction<String, OnChainEvent, OnChainEvent> {

    private transient ValueState<Boolean> seenState;

    @Override
    public void open(Configuration parameters) {
        ValueStateDescriptor<Boolean> seenDescriptor =
            new ValueStateDescriptor<>("dedup-seen", Types.BOOLEAN);
        seenState = getRuntimeContext().getValueState(seenDescriptor);
    }

    @Override
    public void processElement(
            OnChainEvent event,
            Context ctx,
            Collector<OnChainEvent> out) throws Exception {
        if (seenState.value() != null) {
            return;
        }
        seenState.update(true);
        out.collect(event);
    }
}
