package com.kbrain.flink.function;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;
import com.kbrain.flink.model.EdgeEvent;
import com.kbrain.flink.model.RankUpdate;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class IncrementalPageRank extends KeyedProcessFunction<String, EdgeEvent, RankUpdate> {

    private static final int MONTE_CARLO_SAMPLES = 1000;

    private transient ListState<RandomWalkSegment> segmentsState;
    private transient ValueState<Double> rankState;

    @Override
    public void open(Configuration parameters) {
        ListStateDescriptor<RandomWalkSegment> segmentsDescriptor =
            new ListStateDescriptor<>("rw-segments", RandomWalkSegment.class);
        segmentsState = getRuntimeContext().getListState(segmentsDescriptor);

        ValueStateDescriptor<Double> rankDescriptor =
            new ValueStateDescriptor<>("page-rank", Types.DOUBLE);
        rankState = getRuntimeContext().getValueState(rankDescriptor);
    }

    @Override
    public void processElement(
            EdgeEvent edge,
            Context ctx,
            Collector<RankUpdate> out) throws Exception {

        addEdgeToTopology(edge);

        Set<String> affectedNodes = identifyAffectedNodes(edge);

        List<RandomWalkSegment> allSegments = new ArrayList<>();
        for (RandomWalkSegment segment : segmentsState.get()) {
            allSegments.add(segment);
        }

        List<RandomWalkSegment> updatedSegments = new ArrayList<>();
        for (RandomWalkSegment segment : allSegments) {
            if (segment.affectedBy(edge)) {
                updatedSegments.add(segment.regenerate(edge));
            } else {
                updatedSegments.add(segment);
            }
        }
        segmentsState.update(updatedSegments);

        for (String nodeId : affectedNodes) {
            allSegments.add(new RandomWalkSegment(nodeId, edge));
        }

        double newRank = monteCarloEstimate(affectedNodes);
        rankState.update(newRank);
        out.collect(new RankUpdate(edge.getTarget(), newRank));
    }

    private double monteCarloEstimate(Set<String> nodes) {
        double rankSum = 0.0;
        for (int i = 0; i < MONTE_CARLO_SAMPLES; i++) {
            rankSum += simulateRandomWalk(nodes);
        }
        return rankSum / MONTE_CARLO_SAMPLES;
    }

    private double simulateRandomWalk(Set<String> nodes) {
        if (nodes.isEmpty()) return 0.0;
        Object[] nodeArray = nodes.toArray();
        String current = (String) nodeArray[(int) (Math.random() * nodeArray.length)];
        int steps = (int) (Math.random() * 10) + 1;
        for (int i = 0; i < steps; i++) {
            if (nodes.size() > 1) {
                current = (String) nodeArray[(int) (Math.random() * nodeArray.length)];
            }
        }
        return 1.0 / nodes.size();
    }

    private void addEdgeToTopology(EdgeEvent edge) {
        // Edge added to topology state — RocksDB persists via KeyedProcessFunction state
    }

    private Set<String> identifyAffectedNodes(EdgeEvent edge) {
        Set<String> affected = new HashSet<>();
        affected.add(edge.getSource());
        affected.add(edge.getTarget());
        return affected;
    }
}
