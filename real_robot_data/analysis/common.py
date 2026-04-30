"""
Shared data loading, run registry, and plot style for real-robot analysis.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def apply_style():
    """Apply ICML2024-compatible rcParams (works without tueplots installed)."""
    try:
        from tueplots import bundles
        plt.rcParams.update(bundles.icml2024(usetex=False))
    except ImportError:
        plt.rcParams.update({
            "figure.dpi":        300,
            "font.size":         7,
            "axes.labelsize":    7,
            "axes.titlesize":    7,
            "xtick.labelsize":   6,
            "ytick.labelsize":   6,
            "legend.fontsize":   6,
            "lines.linewidth":   0.9,
            "axes.linewidth":    0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype":      42,
            "ps.fonttype":       42,
        })

# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent        # real_robot_data/
_PPO  = _ROOT / "run_ppo_hybrid_franka"
_BASE = _ROOT / "run_baseline_franka"

PPO_RUNS = {
    ("nocc",    9): _PPO / "20260427_114610" / "data_hybrid.npz",
    ("control", 9): _PPO / "20260427_115418" / "data_hybrid.npz",
    ("contact", 9): _PPO / "20260427_115857" / "data_hybrid.npz",
    ("cc",      9): _PPO / "20260427_120142" / "data_hybrid.npz",
    ("cc",      6): _PPO / "20260427_122215" / "data_hybrid.npz",
    ("control", 6): _PPO / "20260427_122315" / "data_hybrid.npz",
    ("contact", 6): _PPO / "20260427_122521" / "data_hybrid.npz",
    ("nocc",    6): _PPO / "20260427_122631" / "data_hybrid.npz",
    ("nocc",    3): _PPO / "20260427_122815" / "data_hybrid.npz",
    ("control", 3): _PPO / "20260427_122913" / "data_hybrid.npz",
    ("cc",      3): _PPO / "20260427_123054" / "data_hybrid.npz",
}

BASELINE_RUNS = {
    9: _BASE / "20260427_121815" / "data_0.0.npz",
    6: _BASE / "20260427_122035" / "data_0.0.npz",
    3: _BASE / "20260427_123007" / "data_0.0.npz",
}

# Display order and colors per condition
COND_ORDER  = ["nocc", "control", "contact", "cc", "baseline"]
COLORS = {
    "nocc":     "tab:gray",
    "control":  "tab:blue",
    "contact":  "tab:orange",
    "cc":       "tab:green",
    "baseline": "tab:red",
}
LABELS = {
    "nocc":     "No comp.",
    "control":  "Control comp.",
    "contact":  "Contact comp.",
    "cc":       "Both comp.",
    "baseline": "Baseline",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TRANSIENT = 1000    # samples at 1 kHz to skip for statistics (first 1 s)

def load_ppo(path: Path) -> dict:
    """Load PPO hybrid run.  Returns t [s], force_error [N], position_error [m]."""
    d = np.load(path)
    force_err = d["desired_forces"] - d["contact_forces"][:, 2]
    t = np.arange(len(force_err)) / 1000.0
    return {
        "t":             t,
        "force_error":   force_err,
        "position_error": d["position_error"],
    }

def load_baseline(path: Path) -> dict:
    """Load baseline run.  Returns t [s], force_error [N], position_error [m]."""
    d = np.load(path)
    t = np.arange(len(d["force_error"])) / 1000.0
    return {
        "t":             t,
        "force_error":   d["force_error"],
        "position_error": d["position_error"],
    }

def metrics(data: dict) -> tuple:
    """Return (mean_abs_force_error_N, mean_position_error_m) after transient."""
    fe = data["force_error"][TRANSIENT:]
    pe = data["position_error"][TRANSIENT:]
    return float(np.mean(np.abs(fe))), float(np.mean(pe))

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUT = Path(__file__).parent   # real_robot_data/analysis/
