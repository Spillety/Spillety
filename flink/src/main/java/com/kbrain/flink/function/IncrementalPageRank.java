package com.kbrain.flink.function;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;
import com.kbrain.flink.model.EdgeEvent;
import com.kbrain.flink.model.RankUpdate;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class IncrementalPageRank extends KeyedProcessFunction<String, EdgeEvent, RankUpdate> {

    static final double DAMPING = RandomWalkSegment.DAMPING_FACTOR;
    static final int AFFECTED_DEPTH = 2;
    // # ponytail: fixed segment budget per node; adaptive R by visit variance is wave E work.
    static final int SEGMENTS_PER_NODE = 4;

    private transient MapState<String, List<String>> adjacencyState;
    private transient ListState<RandomWalkSegment> segmentsState;
    private transient ValueState<Double> rankState;
    private transient Random random;

    @Override
    public void open(Configuration parameters) {
        MapStateDescriptor<String, List<String>> adjacencyDescriptor =
            new MapStateDescriptor<>("graph-adjacency", Types.STRING, Types.LIST(Types.STRING));
        adjacencyState = getRuntimeContext().getMapState(adjacencyDescriptor);

        ListStateDescriptor<RandomWalkSegment> segmentsDescriptor =
            new ListStateDescriptor<>("rw-segments", RandomWalkSegment.class);
        segmentsState = getRuntimeContext().getListState(segmentsDescriptor);

        ValueStateDescriptor<Double> rankDescriptor =
            new ValueStateDescriptor<>("page-rank", Types.DOUBLE);
        rankState = getRuntimeContext().getValueState(rankDescriptor);

        random = new Random();
    }

    @Override
    public void processElement(
            EdgeEvent edge,
            Context ctx,
            Collector<RankUpdate> out) throws Exception {

        addEdgeToTopology(edge);
        Map<String, List<String>> adjacency = snapshotAdjacency();

        Set<String> affectedNodes = identifyAffectedNodes(edge, adjacency);

        List<RandomWalkSegment> updatedSegments = new ArrayList<>();
        for (RandomWalkSegment segment : segmentsState.get()) {
            if (segment.affectedBy(edge)) {
                updatedSegments.add(segment.regenerate(edge, adjacency, random));
            } else {
                updatedSegments.add(segment);
            }
        }
        Set<String> covered = new HashSet<>();
        for (RandomWalkSegment segment : updatedSegments) {
            covered.add(segment.getStartNode());
        }
        for (String nodeId : affectedNodes) {
            for (int i = covered.contains(nodeId) ? 1 : 0; i < SEGMENTS_PER_NODE; i++) {
                updatedSegments.add(RandomWalkSegment.generate(nodeId, adjacency, random));
            }
        }
        segmentsState.update(updatedSegments);

        Map<String, Double> ranks = estimateRanks(affectedNodes, updatedSegments);
        for (Map.Entry<String, Double> entry : ranks.entrySet()) {
            if (entry.getKey().equals(edge.getTarget())) {
                rankState.update(entry.getValue());
            }
            out.collect(new RankUpdate(entry.getKey(), entry.getValue()));
        }
    }

    private void addEdgeToTopology(EdgeEvent edge) throws Exception {
        List<String> targets = adjacencyState.get(edge.getSource());
        if (targets == null) {
            targets = new ArrayList<>();
        }
        if (!targets.contains(edge.getTarget())) {
            targets.add(edge.getTarget());
        }
        adjacencyState.put(edge.getSource(), targets);
    }

    private Map<String, List<String>> snapshotAdjacency() throws Exception {
        Map<String, List<String>> snapshot = new HashMap<>();
        for (Map.Entry<String, List<String>> entry : adjacencyState.entries()) {
            snapshot.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }
        return snapshot;
    }

    static Set<String> identifyAffectedNodes(EdgeEvent edge, Map<String, List<String>> adjacency) {
        // Forward BFS from the new edge's target: only downstream ranks can shift.
        Set<String> affected = new HashSet<>();
        affected.add(edge.getSource());
        Deque<String> queue = new ArrayDeque<>();
        Map<String, Integer> depth = new HashMap<>();
        queue.add(edge.getTarget());
        depth.put(edge.getTarget(), 0);
        while (!queue.isEmpty()) {
            String node = queue.poll();
            int nodeDepth = depth.get(node);
            if (nodeDepth > AFFECTED_DEPTH) {
                continue;
            }
            affected.add(node);
            if (nodeDepth == AFFECTED_DEPTH) {
                continue;
            }
            List<String> out = adjacency.get(node);
            if (out == null) {
                continue;
            }
            for (String next : out) {
                if (!depth.containsKey(next)) {
                    depth.put(next, nodeDepth + 1);
                    queue.add(next);
                }
            }
        }
        return affected;
    }

    static Map<String, Double> estimateRanks(
            Set<String> affectedNodes, List<RandomWalkSegment> segments) {
        Map<String, Long> visits = new HashMap<>();
        long total = 0;
        for (RandomWalkSegment segment : segments) {
            for (String node : segment.getPath()) {
                if (affectedNodes.contains(node)) {
                    visits.merge(node, 1L, Long::sum);
                    total++;
                }
            }
        }
        // Teleport-smoothed Monte Carlo estimate: uniform mass (1-d)/n plus empirical visit share.
        Map<String, Double> ranks = new HashMap<>();
        double n = Math.max(1, affectedNodes.size());
        for (String node : affectedNodes) {
            double empirical = total == 0 ? 0.0 : (double) visits.getOrDefault(node, 0L) / total;
            ranks.put(node, (1.0 - DAMPING) / n + DAMPING * empirical);
        }
        return ranks;
    }
}
