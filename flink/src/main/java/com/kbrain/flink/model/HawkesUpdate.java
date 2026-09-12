package com.kbrain.flink.model;

public class HawkesUpdate {
    private String address;
    private double intensity;

    public HawkesUpdate() {}

    public HawkesUpdate(String address, double intensity) {
        this.address = address;
        this.intensity = intensity;
    }

    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }
    public double getIntensity() { return intensity; }
    public void setIntensity(double intensity) { this.intensity = intensity; }
}
