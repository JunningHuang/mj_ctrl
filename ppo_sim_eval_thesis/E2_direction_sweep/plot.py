"""
E2 plot — Direction generalization: grouped bar chart.

Run from repo root:
    python -m ppo_sim_eval_thesis.E2_direction_sweep.plot
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

from ppo_sim_eval_thesis.common import COLOR_HFDC, COLOR_PPO, LABEL_HFDC, LABEL_PPO

RES_DIR  = os.path.join(os.path.dirname(__file__), "results")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")


def main():
    path = os.path.join(RES_DIR, "results.json")
    with open(path) as f:
        data = json.load(f)

    meta    = data["metadata"]
    results = data["results"]
    dirs    = meta["directions_deg"]
    x_labels = [f"{d}°" for d in dirs]

    m_hfdc = [results["hfdc"][str(d)]["mean"] for d in dirs]
    s_hfdc = [results["hfdc"][str(d)]["std"]  for d in dirs]
    m_ppo  = [results["hfdc_ppo"][str(d)]["mean"] for d in dirs]
    s_ppo  = [results["hfdc_ppo"][str(d)]["std"]  for d in dirs]

    x     = np.arange(len(dirs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.25, 2))
    ax.bar(x - width/2, m_hfdc, width, yerr=s_hfdc, label=LABEL_HFDC,
           color=COLOR_HFDC, capsize=3, error_kw={"lw": 0.8})
    ax.bar(x + width/2, m_ppo,  width, yerr=s_ppo,  label=LABEL_PPO,
           color=COLOR_PPO,  capsize=3, error_kw={"lw": 0.8})

    # Annotate the training direction
    ax.annotate("trained", xy=(0, 0), xytext=(0 - width/2, max(m_hfdc) * 1.15),
                fontsize=5, ha="center", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Direction angle")
    ax.set_ylabel(r"Mean $|e_F|$ [N]")
    ax.legend(fontsize=7)

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "E2_direction_sweep.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[E2] Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
