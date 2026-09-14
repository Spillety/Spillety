package com.kbrain.flink.function;

import com.kbrain.flink.model.EdgeEvent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class RandomWalkSegment {

    public static final double DAMPING_FACTOR = 0.85;
    public static final int MAX_WALK_LENGTH = 32;

    private String startNode;
    private List<String> path;

    public RandomWalkSegment() {
        this.startNode = "";
        this.path = new ArrayList<>();
    }

    public RandomWalkSegment(String startNode, List<String> path) {
        this.startNode = startNode;
        this.path = new ArrayList<>(path);
    }

    public static RandomWalkSegment generate(
            String startNode, Map<String, List<String>> adjacency, Random random) {
        List<String> path = new ArrayList<>();
        String current = startNode;
        path.add(current);
        // Geometrically distributed length with mean 1/(1-d): each hop survives w.p. d.
        while (path.size() < MAX_WALK_LENGTH && random.nextDouble() < DAMPING_FACTOR) {
            List<String> out = adjacency.getOrDefault(current, Collections.emptyList());
            if (out.isEmpty()) {
                break;
            }
            current = out.get(random.nextInt(out.size()));
            path.add(current);
        }
        return new RandomWalkSegment(startNode, path);
    }

    public boolean affectedBy(EdgeEvent edge) {
        // Only walks visiting the changed source observe a new out-neighbor distribution.
        return path.contains(edge.getSource());
    }

    public RandomWalkSegment regenerate(
            EdgeEvent edge, Map<String, List<String>> adjacency, Random random) {
        // # ponytail: reruns the suffix walk with the shared task Random; per-key RNG state is wave E work.
        List<String> prefix = new ArrayList<>();
        for (String node : path) {
            prefix.add(node);
            if (node.equals(edge.getSource())) {
                break;
            }
        }
        String current = prefix.get(prefix.size() - 1);
        while (prefix.size() < MAX_WALK_LENGTH && random.nextDouble() < DAMPING_FACTOR) {
            List<String> out = adjacency.getOrDefault(current, Collections.emptyList());
            if (out.isEmpty()) {
                break;
            }
            current = out.get(random.nextInt(out.size()));
            prefix.add(current);
        }
        return new RandomWalkSegment(startNode, prefix);
    }

    public String getStartNode() { return startNode; }
    public void setStartNode(String startNode) { this.startNode = startNode; }
    public List<String> getPath() { return path; }
    public void setPath(List<String> path) { this.path = path; }
}
