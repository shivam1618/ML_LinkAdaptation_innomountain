"""
evaluation/visualize.py
=======================
Plotting utilities for ML-based link adaptation results.

Generates publication-quality figures:
  1. SNR trace + MCS overlay
  2. Throughput comparison bar chart
  3. BLER vs SNR curves
  4. Per-SNR-bin throughput comparison
  5. Confusion matrix
  6. MCS distribution
  7. DQN training curves
  8. Feature importance (RF)
  9. Spectral efficiency vs SNR

Author: ML Link Adaptation Project
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict
from sklearn.metrics import confusion_matrix

from src.channel.simulator import NUM_MCS, MCS_TABLE
from src.evaluation.metrics import ModelResult

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "DQN":            "#4C72B0",
    "LSTM":           "#55A868",
    "Random Forest":  "#C44E52",
    "XGBoost":        "#8172B2",
    "CQI Baseline":   "#937860",
    "Oracle":         "#222222",
}
FIGSIZE_WIDE   = (12, 5)
FIGSIZE_SQUARE = (8, 7)
FIGSIZE_TALL   = (10, 8)
DPI = 150


def _save(fig, path: str):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. SNR trace + MCS overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_snr_mcs_trace(snr_trace: np.ndarray, mcs_true: np.ndarray,
                        mcs_pred: np.ndarray, model_name: str,
                        n_frames: int = 500, save_path: str = None):
    t = np.arange(n_frames)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_WIDE,
                                    sharex=True, gridspec_kw={"hspace": 0.08})

    ax1.plot(t, snr_trace[:n_frames], color="#444", linewidth=0.8, alpha=0.9)
    ax1.set_ylabel("SNR (dB)", fontsize=10)
    ax1.set_title("Channel SNR and MCS Selection", fontsize=12, pad=8)
    ax1.grid(True, alpha=0.3)

    ax2.step(t, mcs_true[:n_frames], where="post",
             color="#222", linewidth=1.2, label="Oracle MCS", alpha=0.7)
    ax2.step(t, mcs_pred[:n_frames], where="post",
             color=COLORS.get(model_name, "#4C72B0"),
             linewidth=1.0, label=f"{model_name} MCS",
             linestyle="--", alpha=0.85)
    ax2.set_ylabel("MCS Index", fontsize=10)
    ax2.set_xlabel("Frame Index", fontsize=10)
    ax2.set_ylim(-0.5, NUM_MCS - 0.5)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Throughput comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_throughput_comparison(results: List[ModelResult],
                                oracle_tput: float,
                                save_path: str = None):
    names  = [r.name for r in results] + ["Oracle"]
    tputs  = [r.mean_throughput_mbps for r in results] + [oracle_tput]
    colors = [COLORS.get(n, "#888888") for n in names[:-1]] + [COLORS["Oracle"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, tputs, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.55)

    for bar, val in zip(bars, tputs):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} Mbps", va="center", ha="left", fontsize=9)

    ax.set_xlabel("Mean Effective Throughput (Mbps)", fontsize=10)
    ax.set_title("Throughput Comparison Across Models", fontsize=12, pad=10)
    ax.set_xlim(0, max(tputs) * 1.18)
    ax.axvline(oracle_tput, color="#222", linestyle=":", linewidth=1.2,
               label="Oracle (upper bound)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=9)

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. BLER vs SNR
# ─────────────────────────────────────────────────────────────────────────────

def plot_bler_vs_snr(results: List[ModelResult], snr_bins: np.ndarray,
                     save_path: str = None):
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bin_centers = (snr_bins[:-1] + snr_bins[1:]) / 2

    for r in results:
        per_bin = []
        for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
            mask = (r.snr_trace >= lo) & (r.snr_trace < hi)
            if mask.sum() > 0:
                from src.channel.simulator import bler_model
                blers = [bler_model(snr, mcs, r.channel_type)
                         for snr, mcs in zip(r.snr_trace[mask], r.predictions[mask])]
                per_bin.append(np.mean(blers))
            else:
                per_bin.append(np.nan)

        ax.plot(bin_centers, per_bin, marker="o", markersize=4,
                label=r.name, color=COLORS.get(r.name, None), linewidth=1.5)

    ax.axhline(0.1, color="red", linestyle="--", linewidth=1.0,
               alpha=0.7, label="BLER target (0.1)")
    ax.set_xlabel("SNR (dB)", fontsize=10)
    ax.set_ylabel("Mean BLER", fontsize=10)
    ax.set_title("Block Error Rate vs. SNR", fontsize=12, pad=8)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.5)

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-SNR-bin throughput
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_snr_throughput(results: List[ModelResult],
                             snr_bins: np.ndarray,
                             save_path: str = None):
    bin_labels = [f"{int(lo)}–{int(lo+5)}" for lo in snr_bins[:-1]]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for r in results:
        if r.per_snr_throughput is not None:
            ax.plot(bin_labels, r.per_snr_throughput, marker="s",
                    markersize=5, label=r.name,
                    color=COLORS.get(r.name, None), linewidth=1.5)

    ax.set_xlabel("SNR Bin (dB)", fontsize=10)
    ax.set_ylabel("Mean Throughput (Mbps)", fontsize=10)
    ax.set_title("Throughput vs. SNR Bin", fontsize=12, pad=8)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=30)

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(true: np.ndarray, pred: np.ndarray,
                           model_name: str, save_path: str = None):
    cm = confusion_matrix(true, pred, labels=list(range(NUM_MCS)),
                          normalize="true")
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.heatmap(cm, cmap="Blues", ax=ax, vmax=1.0,
                linewidths=0.0, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted MCS", fontsize=10)
    ax.set_ylabel("True MCS", fontsize=10)
    ax.set_title(f"Normalised Confusion Matrix — {model_name}", fontsize=11, pad=8)

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. MCS distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_mcs_distribution(true: np.ndarray, results: List[ModelResult],
                           save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(NUM_MCS)
    width = 0.15
    n = len(results) + 1

    # Oracle
    counts_true = np.bincount(true, minlength=NUM_MCS) / len(true)
    ax.bar(x - width * (n / 2), counts_true, width, label="Oracle",
           color=COLORS["Oracle"], alpha=0.85)

    for i, r in enumerate(results):
        counts = np.bincount(r.predictions, minlength=NUM_MCS) / len(r.predictions)
        ax.bar(x - width * (n / 2 - i - 1), counts, width,
               label=r.name, color=COLORS.get(r.name, None), alpha=0.85)

    ax.set_xlabel("MCS Index", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("MCS Selection Distribution", fontsize=12, pad=8)
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. DQN training curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_dqn_training(history: dict, save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles = ["Episode Reward", "Mean BLER", "Mean Throughput (Mbps)"]
    keys   = ["episode_rewards", "mean_bler", "mean_throughput"]
    scales = [1.0, 1.0, 1e-6]

    for ax, title, key, scale in zip(axes, titles, keys, scales):
        data = np.array(history.get(key, [])) * scale
        if len(data) > 0:
            ax.plot(data, color="#4C72B0", linewidth=1.5, alpha=0.9)
            # Moving average
            if len(data) >= 5:
                ma = np.convolve(data, np.ones(5) / 5, mode="valid")
                ax.plot(range(4, len(data)), ma, color="#C44E52",
                        linewidth=2, label="5-ep MA")
                ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Episode", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("DQN Training Progress", fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Feature importance (RF / XGB)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(importances: np.ndarray,
                             feature_names: List[str],
                             model_name: str,
                             top_k: int = 15,
                             save_path: str = None):
    idx  = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([feature_names[i] for i in idx[::-1]],
            importances[idx][::-1],
            color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title(f"Top-{top_k} Feature Importances — {model_name}", fontsize=11, pad=8)
    ax.grid(axis="x", alpha=0.3)

    if save_path:
        _save(fig, save_path)
    return fig
