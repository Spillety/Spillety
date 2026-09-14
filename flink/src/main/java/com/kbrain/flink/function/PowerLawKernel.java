package com.kbrain.flink.function;

import java.io.Serializable;

public class PowerLawKernel {

    public static final double DEFAULT_ALPHA = 0.5;
    public static final double DEFAULT_BETA = 0.3;
    public static final double DEFAULT_MU = 0.01;
    public static final long TTL_DAYS = 14L;

    public static class HawkesParams implements Serializable {
        private double alpha = DEFAULT_ALPHA;
        private double beta = DEFAULT_BETA;
        private double mu = DEFAULT_MU;

        public HawkesParams() {}

        public HawkesParams(double alpha, double beta, double mu) {
            this.alpha = alpha;
            this.beta = beta;
            this.mu = mu;
        }

        public double getAlpha() { return alpha; }
        public void setAlpha(double alpha) { this.alpha = alpha; }
        public double getBeta() { return beta; }
        public void setBeta(double beta) { this.beta = beta; }
        public double getMu() { return mu; }
        public void setMu(double mu) { this.mu = mu; }
    }

    // O(1) recursive update: decay aggregate past excitation, then add fresh event mass.
    // Power-law aggregate has no exact Markov factorization, so the decay uses a
    // heavy-tail factor (1 + dtSec)^-beta that preserves long memory better than exp().
    // # ponytail: exact power-law sum needs full history; this aggregate is an approximation.
    public static double updateIncremental(
            double prevLambda, long prevTimeMs, long currentTimeMs, HawkesParams params) {
        double mu = params.getMu();
        if (prevTimeMs < 0) {
            return mu + params.getAlpha();
        }
        long dtMs = currentTimeMs - prevTimeMs;
        if (dtMs <= 0) {
            return prevLambda + params.getAlpha();
        }
        double dtSec = dtMs / 1000.0;
        double decay = Math.pow(1.0 + dtSec, -params.getBeta());
        double survived = (prevLambda - mu) * decay;
        if (survived < 0) {
            survived = 0;
        }
        return mu + survived + params.getAlpha();
    }
}
