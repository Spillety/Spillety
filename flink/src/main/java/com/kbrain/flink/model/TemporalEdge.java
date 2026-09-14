package com.kbrain.flink.model;

// validTo == null means the edge is still open.
public class TemporalEdge {
    private String type;
    private String source;
    private String target;
    private String txHash;
    private long validFrom;
    private Long validTo;

    public TemporalEdge() {}

    public TemporalEdge(String type, String source, String target, String txHash, long validFrom, Long validTo) {
        this.type = type;
        this.source = source;
        this.target = target;
        this.txHash = txHash;
        this.validFrom = validFrom;
        this.validTo = validTo;
    }

    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }
    public String getTarget() { return target; }
    public void setTarget(String target) { this.target = target; }
    public String getTxHash() { return txHash; }
    public void setTxHash(String txHash) { this.txHash = txHash; }
    public long getValidFrom() { return validFrom; }
    public void setValidFrom(long validFrom) { this.validFrom = validFrom; }
    public Long getValidTo() { return validTo; }
    public void setValidTo(Long validTo) { this.validTo = validTo; }
}
