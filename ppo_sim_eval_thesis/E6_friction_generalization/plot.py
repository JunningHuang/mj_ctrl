"""
E6 plot — Friction generalization (Exp-B policy, OOD at μ<1.0).

Grouped bar chart: μ on x-axis, bars for HFDC and HFDC+PPO.

Run from repo root:
    python -m ppo_sim_eval_thesis.E6_friction_generalization.plot
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

    meta     = data["metadata"]
    results  = data["results"]
    mus      = meta["mu_values"]
    train_mu = meta.get("train_mu", 1.0)
    keys     = [f"{m:.1f}" for m in mus]
    x_labels = [f"μ={m:.1f}" + (" (train)" if m == train_mu else " (OOD)") for m in mus]

    m_hfdc = [results["hfdc"][k]["mean"] for k in keys]
    s_hfdc = [results["hfdc"][k]["std"]  for k in keys]
    m_ppo  = [results["hfdc_ppo"][k]["mean"] for k in keys]
    s_ppo  = [results["hfdc_ppo"][k]["std"]  for k in keys]

    x     = np.arange(len(mus))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.25, 2))
    ax.bar(x - width/2, m_hfdc, width, yerr=s_hfdc, label=LABEL_HFDC,
           color=COLOR_HFDC, capsize=3, error_kw={"lw": 0.8})
    ax.bar(x + width/2, m_ppo,  width, yerr=s_ppo,  label=LABEL_PPO,
           color=COLOR_PPO,  capsize=3, error_kw={"lw": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=15, ha="right")
    ax.set_ylabel(r"Mean $|e_F|$ [N]")
    ax.legend(fontsize=7)

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "E6_friction_generalization.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[E6] Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
