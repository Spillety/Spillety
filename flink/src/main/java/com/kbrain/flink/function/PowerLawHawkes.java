package com.kbrain.flink.function;

import org.apache.flink.api.common.state.StateTtlConfig;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.time.Time;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;
import com.kbrain.flink.model.OnChainEvent;
import com.kbrain.flink.model.HawkesUpdate;

public class PowerLawHawkes extends KeyedProcessFunction<String, OnChainEvent, HawkesUpdate> {

    private transient ValueState<Double> intensityState;
    private transient ValueState<Long> lastTimeState;
    private transient ValueState<Long> countState;
    private transient ValueState<PowerLawKernel.HawkesParams> paramsState;

    private static StateTtlConfig hawkesTtl() {
        return StateTtlConfig.newBuilder(Time.days(PowerLawKernel.TTL_DAYS))
            .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite)
            .setStateVisibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
            .build();
    }

    @Override
    public void open(Configuration parameters) {
        ValueStateDescriptor<Double> intensityDescriptor =
            new ValueStateDescriptor<>("hawkes-intensity", Types.DOUBLE);
        intensityDescriptor.enableTimeToLive(hawkesTtl());
        intensityState = getRuntimeContext().getValueState(intensityDescriptor);

        ValueStateDescriptor<Long> lastTimeDescriptor =
            new ValueStateDescriptor<>("hawkes-last-time", Types.LONG);
        lastTimeDescriptor.enableTimeToLive(hawkesTtl());
        lastTimeState = getRuntimeContext().getValueState(lastTimeDescriptor);

        ValueStateDescriptor<Long> countDescriptor =
            new ValueStateDescriptor<>("hawkes-count", Types.LONG);
        countDescriptor.enableTimeToLive(hawkesTtl());
        countState = getRuntimeContext().getValueState(countDescriptor);

        ValueStateDescriptor<PowerLawKernel.HawkesParams> paramsDescriptor =
            new ValueStateDescriptor<>(
                "hawkes-params", Types.POJO(PowerLawKernel.HawkesParams.class));
        paramsDescriptor.enableTimeToLive(hawkesTtl());
        paramsState = getRuntimeContext().getValueState(paramsDescriptor);
    }

    @Override
    public void processElement(
            OnChainEvent event,
            Context ctx,
            Collector<HawkesUpdate> out) throws Exception {

        PowerLawKernel.HawkesParams params = paramsState.value();
        if (params == null) {
            params = new PowerLawKernel.HawkesParams();
            paramsState.update(params);
        }

        Double prevLambda = intensityState.value();
        Long prevTime = lastTimeState.value();
        Long count = countState.value();
        if (count == null) {
            count = 0L;
        }

        // Event-time from the payload; wall-clock is never substituted (wave A2).
        long eventTime = event.getTimestamp();
        double base = (prevLambda == null) ? params.getMu() : prevLambda;
        long baseTime = (prevTime == null) ? -1L : prevTime;

        // Out-of-order events skip decay so intensity stays monotone within the key.
        double newIntensity =
            PowerLawKernel.updateIncremental(base, baseTime, eventTime, params);

        intensityState.update(newIntensity);
        if (prevTime == null || eventTime > prevTime) {
            lastTimeState.update(eventTime);
        }
        countState.update(count + 1);
        out.collect(new HawkesUpdate(event.getToAddress(), newIntensity));
    }
}
