"""
evaluation/metrics.py
=====================
Evaluation framework for ML-based link adaptation models.

Computes:
  - Classification accuracy / top-k accuracy
  - Throughput gain vs. CQI baseline
  - BLER statistics
  - Spectral efficiency
  - Conservative / aggressive bias analysis
  - Per-SNR-bin analysis

Author: ML Link Adaptation Project
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

from src.channel.simulator import (
    bler_model, effective_throughput, NUM_MCS, MCS_TABLE
)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    name: str
    predictions: np.ndarray
    true_labels: np.ndarray
    snr_trace: np.ndarray
    channel_type: str

    # Computed fields (filled by evaluate())
    accuracy: float = 0.0
    top3_accuracy: float = 0.0
    mean_bler: float = 0.0
    mean_throughput_mbps: float = 0.0
    throughput_gain_pct: float = 0.0
    spectral_efficiency: float = 0.0
    mcs_mae: float = 0.0
    conservative_bias: float = 0.0    # % of predictions below optimal
    aggressive_bias: float = 0.0      # % of predictions above optimal
    per_snr_throughput: Optional[np.ndarray] = None
    report: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class LinkAdaptationEvaluator:
    """
    Comprehensive evaluator for link adaptation models.

    Usage:
        evaluator = LinkAdaptationEvaluator(snr_trace, true_mcs, channel_type)
        result = evaluator.evaluate(predictions, "DQN")
        evaluator.compare([result1, result2, ...])
    """

    BASELINE_TPUT = None   # Set on first compute

    def __init__(self, snr_trace: np.ndarray, true_mcs: np.ndarray,
                 channel_type: str = "rayleigh",
                 bandwidth_hz: float = 20e6):
        self.snr_trace = snr_trace
        self.true_mcs  = true_mcs
        self.channel_type = channel_type
        self.bandwidth_hz = bandwidth_hz
        self.n = len(true_mcs)

        # Pre-compute oracle (optimal) throughput
        self.oracle_tput = np.array([
            effective_throughput(snr, mcs, channel_type)
            for snr, mcs in zip(snr_trace, true_mcs)
        ])

        # SNR bins for per-bin analysis
        self.snr_bins = np.arange(-10, 35, 5)

    # ── Core metrics ──────────────────────────────────────────────────────────
    def compute_throughput(self, predictions: np.ndarray) -> np.ndarray:
        return np.array([
            effective_throughput(snr, mcs, self.channel_type)
            for snr, mcs in zip(self.snr_trace, predictions)
        ])

    def compute_bler(self, predictions: np.ndarray) -> np.ndarray:
        return np.array([
            bler_model(snr, mcs, self.channel_type)
            for snr, mcs in zip(self.snr_trace, predictions)
        ])

    def top_k_accuracy(self, predictions: np.ndarray,
                        true_labels: np.ndarray, k: int = 3) -> float:
        n_correct = sum(
            p in range(max(0, t - k // 2), min(NUM_MCS, t + k // 2 + 1))
            for p, t in zip(predictions, true_labels)
        )
        return n_correct / len(predictions)

    # ── Full evaluation ───────────────────────────────────────────────────────
    def evaluate(self, predictions: np.ndarray, name: str) -> ModelResult:
        result = ModelResult(
            name=name,
            predictions=predictions,
            true_labels=self.true_mcs,
            snr_trace=self.snr_trace,
            channel_type=self.channel_type,
        )

        tput_arr = self.compute_throughput(predictions)
        bler_arr = self.compute_bler(predictions)

        result.accuracy       = accuracy_score(self.true_mcs, predictions)
        result.top3_accuracy  = self.top_k_accuracy(predictions, self.true_mcs, k=3)
        result.mean_bler      = float(bler_arr.mean())
        result.mean_throughput_mbps = float(tput_arr.mean()) / 1e6
        result.throughput_gain_pct  = float(
            (tput_arr.mean() / self.oracle_tput.mean() - 1) * 100)
        result.mcs_mae       = float(np.abs(predictions - self.true_mcs).mean())
        result.conservative_bias = float((predictions < self.true_mcs).mean() * 100)
        result.aggressive_bias   = float((predictions > self.true_mcs).mean() * 100)

        # Spectral efficiency (bits/s/Hz)
        result.spectral_efficiency = float(tput_arr.mean()) / self.bandwidth_hz

        # Per-SNR-bin throughput
        per_snr = []
        for lo in self.snr_bins[:-1]:
            hi = lo + 5
            mask = (self.snr_trace >= lo) & (self.snr_trace < hi)
            if mask.sum() > 0:
                per_snr.append(tput_arr[mask].mean() / 1e6)
            else:
                per_snr.append(np.nan)
        result.per_snr_throughput = np.array(per_snr)

        result.report = classification_report(
            self.true_mcs, predictions,
            labels=list(range(NUM_MCS)),
            zero_division=0,
        )

        return result

    # ── Comparison table ──────────────────────────────────────────────────────
    def compare(self, results: List[ModelResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            rows.append({
                "Model":             r.name,
                "Accuracy (%)":      f"{r.accuracy * 100:.2f}",
                "Top-3 Acc (%)":     f"{r.top3_accuracy * 100:.2f}",
                "Mean BLER":         f"{r.mean_bler:.4f}",
                "Throughput (Mbps)": f"{r.mean_throughput_mbps:.2f}",
                "Gain vs Oracle (%)": f"{r.throughput_gain_pct:+.2f}",
                "SE (bits/s/Hz)":    f"{r.spectral_efficiency:.4f}",
                "MCS MAE":           f"{r.mcs_mae:.3f}",
                "Conservative (%)":  f"{r.conservative_bias:.1f}",
                "Aggressive (%)":    f"{r.aggressive_bias:.1f}",
            })
        df = pd.DataFrame(rows)
        df = df.set_index("Model")
        return df

    # ── Oracle / theoretical upper bound ─────────────────────────────────────
    def oracle_metrics(self) -> dict:
        return {
            "mean_throughput_mbps": self.oracle_tput.mean() / 1e6,
            "spectral_efficiency":  self.oracle_tput.mean() / self.bandwidth_hz,
            "mean_bler": np.array([
                bler_model(snr, mcs, self.channel_type)
                for snr, mcs in zip(self.snr_trace, self.true_mcs)
            ]).mean(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Throughput simulation for a given policy
# ─────────────────────────────────────────────────────────────────────────────

def simulate_policy(predict_fn: Callable[[np.ndarray], int],
                    features: np.ndarray,
                    snr_trace: np.ndarray,
                    channel_type: str = "rayleigh") -> dict:
    """
    Run a policy function frame-by-frame and collect stats.

    predict_fn: callable(feature_vector) → mcs_index
    """
    n = len(features)
    tputs, blers, mcs_choices = [], [], []

    for i in range(n):
        mcs  = predict_fn(features[i])
        snr  = snr_trace[i]
        tputs.append(effective_throughput(snr, mcs, channel_type))
        blers.append(bler_model(snr, mcs, channel_type))
        mcs_choices.append(mcs)

    return {
        "throughput":  np.array(tputs),
        "bler":        np.array(blers),
        "mcs":         np.array(mcs_choices),
        "mean_tput_mbps": np.mean(tputs) / 1e6,
        "mean_bler":   np.mean(blers),
    }
