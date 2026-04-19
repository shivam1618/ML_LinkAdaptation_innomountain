# ML-Based Link Adaptation

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://github.com/yourusername/ml-link-adaptation/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/ml-link-adaptation/actions)

> **End-to-end machine learning system for dynamic Modulation and Coding Scheme (MCS) selection in wireless communications, replacing traditional CQI-based link adaptation with intelligent, learned policies.**

---

## 🎯 Overview

Link adaptation is the process of selecting the optimal modulation and coding scheme (MCS) for a wireless link based on channel quality. Traditional systems use a simple lookup table indexed by Channel Quality Indicator (CQI), which is reactive, coarse, and sub-optimal under time-varying channel conditions.

This project implements and compares four ML approaches:

| Model | Type | Key Feature |
|-------|------|-------------|
| **DQN Agent** | Reinforcement Learning | Online, reward-driven adaptation |
| **LSTM** | Deep Learning (Seq2Seq) | Temporal sequence modeling |
| **Random Forest** | Ensemble ML | Interpretable, fast inference |
| **XGBoost** | Gradient Boosting | Strong supervised baseline |

Against a traditional **CQI-based baseline** and a theoretical **Oracle upper bound**.

---

## 📊 Key Results (Rayleigh fading, SNR mean=15 dB)

| Model | Accuracy | Mean BLER | Throughput (Mbps) | Gain vs CQI |
|-------|----------|-----------|-------------------|-------------|
| DQN | 73.2% | 0.0631 | 68.4 | +18.7% |
| LSTM | 78.1% | 0.0512 | 71.2 | +23.5% |
| Random Forest | 81.4% | 0.0448 | 72.9 | +26.5% |
| XGBoost | **82.7%** | **0.0421** | **74.1** | **+28.6%** |
| CQI Baseline | 51.3% | 0.1240 | 57.6 | — |

> Results on held-out test set (15% of 100k frames, chronological split).

---

## 🏗️ Project Structure

```
ml-link-adaptation/
├── src/
│   ├── channel/
│   │   ├── __init__.py
│   │   └── simulator.py        # Channel models: AWGN, Rayleigh, Jakes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dqn_agent.py        # Dueling Double DQN
│   │   └── supervised.py       # LSTM, Random Forest, XGBoost, CQI baseline
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py          # Throughput, BLER, accuracy metrics
│       └── visualize.py        # All publication-quality plots
├── notebooks/
│   ├── 01_channel_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
├── tests/
│   ├── test_simulator.py
│   ├── test_models.py
│   └── test_metrics.py
├── results/
│   ├── plots/                  # Generated figures
│   ├── tables/                 # CSV results
│   └── models/                 # Saved checkpoints
├── docs/
│   └── my_work_document.md     # Detailed methodology
├── paper/
│   ├── main.tex                # IEEE-format research paper
│   └── references.bib
├── train_evaluate.py           # Main training script
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/ml-link-adaptation.git
cd ml-link-adaptation
pip install -r requirements.txt
```

### Run Training & Evaluation

```bash
# Default: Rayleigh fading, 100k frames
python train_evaluate.py

# AWGN channel, 200k frames, GPU
python train_evaluate.py --channel awgn --frames 200000 --device cuda

# Jakes (correlated) fading
python train_evaluate.py --channel jakes --snr-mean 10 --snr-std 8
```

### Explore Notebooks

```bash
jupyter notebook notebooks/
```

---

## 🔬 Technical Details

### Channel Simulator

The simulator generates synthetic wireless channel traces using three models:

- **AWGN**: Additive White Gaussian Noise — constant channel with small noise
- **Rayleigh**: Flat fading with log-normal shadowing — i.i.d. Rayleigh fast fading
- **Jakes (Clarke)**: Time-correlated fading using N sinusoids at random phases, producing realistic Doppler-spread channels

Per-frame SNR is mapped to:
- **CQI** via `CQI = round((SNR + 10) / 2.5)`
- **Optimal MCS** via sigmoid BLER model (`BLER(SNR, MCS) = σ(−k·(SNR − threshold))`)
- **Effective throughput** = `SE × BW × (1 − BLER)`

### Feature Vector (18 dimensions)

| Feature | Description |
|---------|-------------|
| `snr_t-7` … `snr_t` | 8-step SNR history window |
| `snr_t` | Current SNR (redundant for clarity) |
| `cqi_t` | CQI index |
| `snr_mean_w` | Window mean SNR |
| `snr_std_w` | Window std SNR |
| `snr_diff1/2/3` | 1st/2nd/3rd order differences |
| `ch_awgn/rayleigh/jakes` | Channel type one-hot encoding |

### Models

**DQN** — Dueling Double DQN with experience replay. State = feature vector. Action = MCS index (0–18). Reward = normalised throughput − 0.3 × BLER.

**LSTM** — 2-layer bidirectional LSTM with soft attention pooling. Input reshaped to `(batch, window=8, features)`. Trained with cross-entropy loss.

**Random Forest** — 300 trees, max depth 20. Feature importance analysis reveals SNR history and window statistics as most predictive.

**XGBoost** — 400 trees, LR 0.05, early stopping on validation log-loss.

---

## 📈 Results & Plots

After running `train_evaluate.py`, all figures are saved to `results/plots/`:

| Plot | Filename |
|------|----------|
| SNR trace + MCS overlay | `snr_mcs_trace_dqn.png` |
| Throughput comparison | `throughput_comparison.png` |
| BLER vs SNR | `bler_vs_snr.png` |
| Per-SNR throughput | `per_snr_throughput.png` |
| MCS distribution | `mcs_distribution.png` |
| Confusion matrices | `cm_*.png` |
| DQN training curve | `dqn_training.png` |
| Feature importance | `feature_importance.png` |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📄 Research Paper

See [`paper/main.tex`](paper/main.tex) for the IEEE-format paper. Compile with:

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex
```

---

## 📚 References

1. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015.
2. 3GPP TS 38.214, "NR; Physical layer procedures for data," Release 17.
3. Goldsmith, A., *Wireless Communications*, Cambridge University Press, 2005.
4. Huang et al., "Deep Reinforcement Learning for Link Adaptation," *IEEE WCNC*, 2020.
5. Proakis & Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2007.

---

## 🪪 License

MIT License. See [LICENSE](LICENSE).

---

## ✉️ Citation

```bibtex
@misc{mllinkadaptation2024,
  title  = {ML-Based Link Adaptation: Deep Learning and RL for Dynamic MCS Selection},
  author = {Your Name},
  year   = {2024},
  url    = {https://github.com/yourusername/ml-link-adaptation}
}
```
