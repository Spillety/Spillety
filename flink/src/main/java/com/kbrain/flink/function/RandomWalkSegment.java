package com.kbrain.flink.function;

import com.kbrain.flink.model.EdgeEvent;

public class RandomWalkSegment {

    private final String nodeId;
    private final EdgeEvent seedEdge;

    public RandomWalkSegment(String nodeId, EdgeEvent seedEdge) {
        this.nodeId = nodeId;
        this.seedEdge = seedEdge;
    }

    public boolean affectedBy(EdgeEvent edge) {
        return edge.getSource().equals(nodeId) || edge.getTarget().equals(nodeId);
    }

    public RandomWalkSegment regenerate(EdgeEvent edge) {
        return new RandomWalkSegment(nodeId, edge);
    }

    public String getNodeId() { return nodeId; }
}
