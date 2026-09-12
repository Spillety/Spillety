package com.kbrain.flink.model;

public class RankUpdate {
    private String node;
    private double rank;

    public RankUpdate() {}

    public RankUpdate(String node, double rank) {
        this.node = node;
        this.rank = rank;
    }

    public String getNode() { return node; }
    public void setNode(String node) { this.node = node; }
    public double getRank() { return rank; }
    public void setRank(double rank) { this.rank = rank; }
}
