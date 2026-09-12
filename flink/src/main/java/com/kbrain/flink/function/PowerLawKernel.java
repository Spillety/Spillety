package com.kbrain.flink.function;

import java.util.List;

public class PowerLawKernel {

    static final double ALPHA = 0.5;
    static final double BETA = 0.3;
    static final double MU = 0.01;

    /**
     * Compute λ(t) via numerical integration of power-law kernel.
     * Integration uses trapezoidal rule over event timestamps.
     * Complexity: O(n) per event where n = event log size.
     * # ponytail: O(1) approximate updater, add when numerical integration exceeds 5ms latency.
     */
static double computeIntensity(double currentTime, List<Double> eventTimestamps) {
        double intensity = MU;
        for (double t_i : eventTimestamps) {
            double dt = currentTime - t_i;
            if (dt > 0) {
                intensity += ALPHA * Math.pow(dt, -BETA);
            }
        }
        return intensity;
    }

    static double integratePowerLaw(double t, double alpha, double beta) {
        if (t <= 0) return 0.0;
        int steps = 1000;
        double h = t / steps;
        double sum = 0.5 * kernel(0, alpha, beta) + 0.5 * kernel(t, alpha, beta);
        for (int i = 1; i < steps; i++) {
            sum += kernel(i * h, alpha, beta);
        }
        return sum * h;
    }

    static double kernel(double dt, double alpha, double beta) {
        return alpha * Math.pow(dt, -beta);
    }
}
