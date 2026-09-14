package com.kbrain.flink.model;

public class EdgeEvent {
    private String source;
    private String target;
    private long timestamp;

    public EdgeEvent() {}

    public EdgeEvent(String source, String target, long timestamp) {
        this.source = source;
        this.target = target;
        this.timestamp = timestamp;
    }

    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }
    public String getTarget() { return target; }
    public void setTarget(String target) { this.target = target; }
    public long getTimestamp() { return timestamp; }
    public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
}
