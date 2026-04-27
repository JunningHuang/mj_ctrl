"""
E7 plot — Cross-policy comparison: single grouped bar with three conditions.

Run from repo root:
    python -m ppo_sim_eval_thesis.E7_cross_policy.plot
"""

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tueplots import bundles
    plt.rcParams.update(bundles.icml2024(usetex=False))
except ImportError:
    pass

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

RES_DIR  = os.path.join(os.path.dirname(__file__), "results")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

_CONDS   = ["hfdc", "ppo_exp_a", "ppo_exp_b"]
_LABELS  = ["HFDC", "HFDC+PPO\n(Exp-A)", "HFDC+PPO\n(Exp-B)"]
_COLORS  = ["tab:blue", "tab:orange", "tab:green"]


def main():
    path = os.path.join(RES_DIR, "results.json")
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    means   = [results[c]["mean"] for c in _CONDS]
    stds    = [results[c]["std"]  for c in _CONDS]

    fig, ax = plt.subplots(figsize=(3.25, 2))
    x = np.arange(len(_CONDS))
    bars = ax.bar(x, means, 0.5, yerr=stds,
                  color=_COLORS, capsize=3, error_kw={"lw": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(_LABELS, fontsize=7)
    ax.set_ylabel(r"Mean $|e_F|$ [N]")

    # Annotate bars with numeric values
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.2f}", ha="center", va="bottom", fontsize=6)

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "E7_cross_policy.svg")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[E7] Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
