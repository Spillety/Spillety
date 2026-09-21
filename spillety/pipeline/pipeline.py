import numpy as np
import shap

from spillety.cost.operating import alerts_at, assign_tier, cost_at, select_tau
from spillety.evidence.evidence import build_evidence
from spillety.features.build import build_feature_matrix
from spillety.metrics import pr_auc_score
from spillety.models.calibration import brier_score, calibrate, ece_score
from spillety.models.gbdt import select_num_leaves, train_gbdt


class SpilletyPipeline:
    """
    ## GBDT decision path over upstream arrays (§7.4–7.8)

    Parameters
    ----------
    config : dict | None
        `random_state`, `c_fp`, `c_fn`, `budget`, `n_boot`.

    Returns
    ----------
    SpilletyPipeline
        Unfitted pipeline; call `fit` before predicting.
    """

    def __init__(self, config=None):
        cfg = config or {}
        self.random_state = int(cfg.get("random_state", 72))
        self.c_fp = float(cfg.get("c_fp", 1.0))
        self.c_fn = float(cfg.get("c_fn", 10.0))
        self.budget = cfg.get("budget", None)
        if self.budget is not None:
            self.budget = int(self.budget)
        self.n_boot = int(cfg.get("n_boot", 1000))
        self.model_ = None
        self.calibrator_ = None
        self.calibrator_name_ = None
        self.num_leaves_ = None
        self.feature_names_ = None
        self.tau1_ = None
        self.tau2_ = None
        self.tau3_ = None

    def fit(self, frames_train, y_train, frames_valid, y_valid):
        """
        ## Fit features → GBDT → calibrator → operating thresholds

        Parameters
        ----------
        frames_train, frames_valid : dict[str, np.ndarray]
            Upstream blocks for `build_feature_matrix`.
        y_train, y_valid : np.ndarray
            Binary labels aligned with frames rows.

        Returns
        ----------
        SpilletyPipeline
            Fitted pipeline.
        """
        y_train = np.asarray(y_train).astype(int)
        y_valid = np.asarray(y_valid).astype(int)
        x_train, names = build_feature_matrix(frames_train)
        x_valid, _ = build_feature_matrix(frames_valid)
        self.feature_names_ = names

        self.num_leaves_, _ = select_num_leaves(
            x_train,
            y_train,
            x_valid,
            y_valid,
            n_boot=self.n_boot,
            random_state=self.random_state,
        )
        self.model_ = train_gbdt(
            x_train,
            y_train,
            x_valid,
            y_valid,
            random_state=self.random_state,
            n_boot=self.n_boot,
        )

        raw_valid = np.asarray(self.model_.predict(x_valid), dtype=float)
        # ponytail: valid halves stand in for validation/test; dedicated test split
        # arrives with the temporal-validation monitor (§7.9 follow-up).
        cut = max(1, len(y_valid) // 2)
        cal = calibrate(
            raw_valid[:cut],
            y_valid[:cut],
            raw_valid[cut:],
            y_valid[cut:],
            n_boot=self.n_boot,
            random_state=self.random_state,
        )
        self.calibrator_ = cal["best_estimator"]
        self.calibrator_name_ = cal["best"]

        q_valid = np.asarray(
            self.calibrator_.predict_proba(raw_valid)[:, 1], dtype=float
        )
        self.tau1_ = select_tau(
            y_valid, q_valid, c_fp=self.c_fp, c_fn=self.c_fn, budget=self.budget
        )
        # ponytail: τ2/τ3 as valid quantiles clipped below τ1; cost-optimal
        # tier spacing needs analyst-capacity data (§7.7.3 follow-up).
        self.tau2_ = min(self.tau1_, float(np.quantile(q_valid, 0.90)))
        self.tau3_ = min(self.tau2_, float(np.quantile(q_valid, 0.75)))
        return self

    def _check_fitted(self):
        if self.model_ is None or self.calibrator_ is None:
            raise RuntimeError("pipeline not fitted")

    def predict_proba(self, frames):
        """
        ## Calibrated P(illicit | x) for upstream frames

        Parameters
        ----------
        frames : dict[str, np.ndarray]
            Upstream blocks for `build_feature_matrix`.

        Returns
        ----------
        np.ndarray
            Calibrated probabilities, shape (n,).
        """
        self._check_fitted()
        x, _ = build_feature_matrix(frames)
        raw = np.asarray(self.model_.predict(x), dtype=float)
        return np.asarray(self.calibrator_.predict_proba(raw)[:, 1], dtype=float)

    def _shap_top10(self, x):
        explainer = shap.TreeExplainer(self.model_)
        values = explainer.shap_values(x)
        if isinstance(values, list):
            # Binary LightGBM may return [class0, class1]; explain P(illicit).
            values = values[1] if len(values) == 2 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        return values

    def predict(
        self,
        frames,
        *,
        causal_passed=None,
        e_value=None,
        gamma=None,
        retrieval_anchors=None,
        causal_path=None,
        provenance=None,
    ):
        """
        ## Scores, tiers and evidence records for upstream frames

        Parameters
        ----------
        frames : dict[str, np.ndarray]
            Upstream blocks for `build_feature_matrix`.
        causal_passed : np.ndarray | None
            Per-row upstream causal filter outcome.
        e_value, gamma : np.ndarray | None
            Per-row sensitivity metrics.
        retrieval_anchors : list[list[dict]] | None
            Per-row top-K anchors: [[{wallet, source, distance}, ...], ...].
        causal_path : list[list[dict]] | None
            Per-row DAG edges: [[{effect, gamma}, ...], ...].
        provenance : dict | None
            Build metadata: model_version, encoder_version, hnsw_params, calibrator, tau, cost_ratio.

        Returns
        ----------
        tuple[np.ndarray, list[str], list[dict]]
            Calibrated scores, tier labels, evidence records.
        """
        self._check_fitted()
        x, _ = build_feature_matrix(frames)
        n = x.shape[0]
        scores = self.predict_proba(frames)
        shap_values = self._shap_top10(x)
        causal = (
            np.zeros(n, dtype=bool)
            if causal_passed is None
            else np.asarray(causal_passed).astype(bool)
        )
        e_vals = (
            np.full(n, np.nan) if e_value is None else np.asarray(e_value, dtype=float)
        )
        gammas = (
            np.full(n, np.nan) if gamma is None else np.asarray(gamma, dtype=float)
        )

        tiers, evidences = [], []
        for i in range(n):
            ev = float(e_vals[i]) if not np.isnan(e_vals[i]) else 0.0
            ga = float(gammas[i]) if not np.isnan(gammas[i]) else 0.0
            tier = assign_tier(
                float(scores[i]),
                self.tau1_,
                self.tau2_,
                self.tau3_,
                bool(causal[i]),
                ev,
                ga,
            )
            order = np.argsort(np.abs(shap_values[i]))[::-1][:10]
            top = {
                self.feature_names_[j]: float(shap_values[i, j]) for j in order
            }
            tiers.append(tier)
            anchors_i = retrieval_anchors[i] if retrieval_anchors else None
            causal_path_i = causal_path[i] if causal_path else None
            evidences.append(
                build_evidence(
                    float(scores[i]),
                    tier,
                    top,
                    {
                        "e_value": None if np.isnan(e_vals[i]) else float(e_vals[i]),
                        "gamma": None if np.isnan(gammas[i]) else float(gammas[i]),
                        "causal_passed": bool(causal[i]),
                    },
                    anchors=anchors_i,
                    causal_path=causal_path_i,
                    provenance=provenance,
                )
            )
        return scores, tiers, evidences

    def evaluate(self, y_true, scores):
        """
        ## Rank, calibration and operating metrics at fitted thresholds

        Parameters
        ----------
        y_true : np.ndarray
            Binary labels.
        scores : np.ndarray
            Calibrated probabilities.

        Returns
        ----------
        dict
            `pr_auc`, `brier`, `ece`, `cost`, `alerts`, thresholds,
            `calibrator` and `num_leaves`.
        """
        self._check_fitted()
        y_true = np.asarray(y_true).astype(int)
        scores = np.asarray(scores, dtype=float)
        return {
            "pr_auc": pr_auc_score(y_true, scores),
            "brier": brier_score(y_true, scores),
            "ece": ece_score(y_true, scores),
            "cost": cost_at(y_true, scores, self.tau1_, self.c_fp, self.c_fn),
            "alerts": alerts_at(scores, self.tau1_),
            "tau1": self.tau1_,
            "tau2": self.tau2_,
            "tau3": self.tau3_,
            "calibrator": self.calibrator_name_,
            "num_leaves": self.num_leaves_,
        }


__all__ = ["SpilletyPipeline"]
