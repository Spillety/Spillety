package com.kbrain.flink.model;

public class OnChainEvent {
    private String eventId;
    private String txHash;
    private String fromAddress;
    private String toAddress;
    private double amount;
    private String chainId;
    private long timestamp;
    private boolean exchangeInternal;

    public OnChainEvent() {}

    public OnChainEvent(String eventId, String txHash, String fromAddress, String toAddress,
                        double amount, String chainId, long timestamp) {
        this.eventId = eventId;
        this.txHash = txHash;
        this.fromAddress = fromAddress;
        this.toAddress = toAddress;
        this.amount = amount;
        this.chainId = chainId;
        this.timestamp = timestamp;
    }

    public String getEventId() { return eventId; }
    public void setEventId(String eventId) { this.eventId = eventId; }
    public String getTxHash() { return txHash; }
    public void setTxHash(String txHash) { this.txHash = txHash; }
    public String getFromAddress() { return fromAddress; }
    public void setFromAddress(String fromAddress) { this.fromAddress = fromAddress; }
    public String getToAddress() { return toAddress; }
    public void setToAddress(String toAddress) { this.toAddress = toAddress; }
    public double getAmount() { return amount; }
    public void setAmount(double amount) { this.amount = amount; }
    public String getChainId() { return chainId; }
    public void setChainId(String chainId) { this.chainId = chainId; }
    public long getTimestamp() { return timestamp; }
    public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
    public boolean isExchangeInternal() { return exchangeInternal; }
    public void setExchangeInternal(boolean exchangeInternal) { this.exchangeInternal = exchangeInternal; }

    public static OnChainEvent fromJson(String json) {
        java.util.Map<String, String> fields = new java.util.HashMap<>();
        for (String part : json.replace("{", "").replace("}", "").split(",")) {
            String[] kv = part.split(":", 2);
            if (kv.length == 2) {
                fields.put(kv[0].trim().replace("\"", ""), kv[1].trim().replace("\"", ""));
            }
        }
        String eventId = fields.getOrDefault("eventId", "");
        String toAddress = fields.getOrDefault("toAddress", fields.getOrDefault("address", ""));
        String timestampRaw = fields.get("timestamp");
        if (timestampRaw == null || timestampRaw.isEmpty()) {
            throw new IllegalArgumentException("OnChainEvent JSON missing required field: timestamp");
        }
        long timestamp;
        try {
            timestamp = Long.parseLong(timestampRaw);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("OnChainEvent JSON has invalid timestamp: " + timestampRaw, e);
        }
        OnChainEvent event = new OnChainEvent(
            eventId,
            fields.getOrDefault("txHash", eventId),
            fields.getOrDefault("fromAddress", ""),
            toAddress,
            Double.parseDouble(fields.getOrDefault("amount", "0.0")),
            fields.getOrDefault("chainId", ""),
            timestamp);
        return event;
    }

    // Parses the canonical Protobuf bytes the Kafka producer writes (events.proto:
    // 1/2/3/6 strings, 4 fixed64 double, 5 varint timestamp). Unknown fields skipped.
    // Dependency-free wire reader; no protobuf-java on the classpath.
    // # ponytail: replace with protoc --java_out generated classes once the build has network.
    public static OnChainEvent fromProtoBytes(byte[] bytes) throws Exception {
        String txHash = "";
        String fromAddress = "";
        String toAddress = "";
        double amount = 0.0;
        long timestamp = 0L;
        String chainId = "";
        int pos = 0;
        while (pos < bytes.length) {
            int tag = (int) readVarint(bytes, pos);
            pos += varintSize(bytes, pos);
            int field = tag >>> 3;
            int wire = tag & 0x7;
            if (wire == 0) {
                long value = readVarint(bytes, pos);
                pos += varintSize(bytes, pos);
                if (field == 5) {
                    timestamp = value;
                }
            } else if (wire == 1) {
                long bits = 0L;
                for (int i = 0; i < 8; i++) {
                    bits |= ((long) bytes[pos + i] & 0xFF) << (8 * i);
                }
                pos += 8;
                if (field == 4) {
                    amount = Double.longBitsToDouble(bits);
                }
            } else if (wire == 2) {
                int length = (int) readVarint(bytes, pos);
                pos += varintSize(bytes, pos);
                String value = new String(bytes, pos, length, java.nio.charset.StandardCharsets.UTF_8);
                pos += length;
                if (field == 1) {
                    txHash = value;
                } else if (field == 2) {
                    fromAddress = value;
                } else if (field == 3) {
                    toAddress = value;
                } else if (field == 6) {
                    chainId = value;
                }
            } else {
                throw new IllegalArgumentException("Unsupported wire type: " + wire);
            }
        }
        String eventId = txHash.isEmpty() ? fromAddress + ":" + timestamp : txHash;
        return new OnChainEvent(eventId, txHash, fromAddress, toAddress, amount, chainId, timestamp);
    }

    private static long readVarint(byte[] bytes, int pos) {
        long result = 0L;
        int shift = 0;
        while (true) {
            byte b = bytes[pos++];
            result |= (long) (b & 0x7F) << shift;
            if ((b & 0x80) == 0) {
                return result;
            }
            shift += 7;
        }
    }

    private static int varintSize(byte[] bytes, int pos) {
        int size = 0;
        while ((bytes[pos++] & 0x80) != 0) {
            size++;
        }
        return size + 1;
    }
}
