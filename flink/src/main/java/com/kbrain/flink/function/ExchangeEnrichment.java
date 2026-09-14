package com.kbrain.flink.function;

import org.apache.flink.api.common.functions.RichMapFunction;
import com.kbrain.flink.model.OnChainEvent;
import java.util.Set;

// # ponytail: replace ctor-injected snapshot with BroadcastStream join.
public class ExchangeEnrichment extends RichMapFunction<OnChainEvent, OnChainEvent> {

    private final Set<String> exchangeAddresses;

    public ExchangeEnrichment(Set<String> exchangeAddresses) {
        this.exchangeAddresses = exchangeAddresses;
    }

    @Override
    public OnChainEvent map(OnChainEvent event) {
        boolean internal = exchangeAddresses.contains(event.getFromAddress())
            || exchangeAddresses.contains(event.getToAddress());
        event.setExchangeInternal(internal);
        return event;
    }
}
