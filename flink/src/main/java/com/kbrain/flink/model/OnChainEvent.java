package com.kbrain.flink.model;

public class OnChainEvent {
    private String eventId;
    private String fromAddress;
    private String address;
    private long timestamp;

    public OnChainEvent() {}

    public OnChainEvent(String eventId, String fromAddress, String address, long timestamp) {
        this.eventId = eventId;
        this.fromAddress = fromAddress;
        this.address = address;
        this.timestamp = timestamp;
    }

    public String getEventId() { return eventId; }
    public void setEventId(String eventId) { this.eventId = eventId; }
    public String getFromAddress() { return fromAddress; }
    public void setFromAddress(String fromAddress) { this.fromAddress = fromAddress; }
    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }
    public long getTimestamp() { return timestamp; }
    public void setTimestamp(long timestamp) { this.timestamp = timestamp; }

    public static OnChainEvent fromJson(String json) {
        String[] parts = json.replace("{", "").replace("}", "").split(",");
        String eventId = parts[0].split(":")[1].trim().replace("\"", "");
        String fromAddress = parts[1].split(":")[1].trim().replace("\"", "");
        String address = parts[2].split(":")[1].trim().replace("\"", "");
        return new OnChainEvent(eventId, fromAddress, address, System.currentTimeMillis());
    }
}
