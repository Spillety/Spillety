package com.kbrain.flink.function;

import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;
import com.kbrain.flink.function.PowerLawKernel;
import com.kbrain.flink.model.OnChainEvent;
import com.kbrain.flink.model.HawkesUpdate;

import java.util.ArrayList;
import java.util.List;

public class PowerLawHawkes extends KeyedProcessFunction<String, OnChainEvent, HawkesUpdate> {

    private transient MapState<String, OnChainEvent> eventLogState;
    private transient ValueState<Double> intensityState;

    @Override
    public void open(Configuration parameters) {
        MapStateDescriptor<String, OnChainEvent> eventLogDescriptor =
            new MapStateDescriptor<>("event-log", Types.STRING, OnChainEvent.class);
        eventLogState = getRuntimeContext().getMapState(eventLogDescriptor);

        ValueStateDescriptor<Double> intensityDescriptor =
            new ValueStateDescriptor<>("hawkes-intensity", Types.DOUBLE);
        intensityState = getRuntimeContext().getValueState(intensityDescriptor);
    }

    @Override
    public void processElement(
            OnChainEvent event,
            Context ctx,
            Collector<HawkesUpdate> out) throws Exception {

        if (eventLogState.get(event.getEventId()) != null) {
            return;
        }

        eventLogState.put(event.getEventId(), event);

        List<Double> timestamps = new ArrayList<>();
        for (String key : eventLogState.keys()) {
            timestamps.add(eventLogState.get(key).getTimestamp());
        }

        double newIntensity = PowerLawKernel.computeIntensity(
            System.currentTimeMillis(), timestamps);
        intensityState.update(newIntensity);
        out.collect(new HawkesUpdate(event.getAddress(), newIntensity));
    }
}
