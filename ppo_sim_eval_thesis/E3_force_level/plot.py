"""
E3 plot — Desired force level: grouped bar chart (|Fd| on x-axis).

Run from repo root:
    python -m ppo_sim_eval_thesis.E3_force_level.plot
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
    fd_vals = meta["fd_values"]
    keys    = [str(int(abs(fd))) for fd in fd_vals]
    x_labels = [f"{abs(fd):.0f} N" for fd in fd_vals]

    m_hfdc = [results["hfdc"][k]["mean"] for k in keys]
    s_hfdc = [results["hfdc"][k]["std"]  for k in keys]
    m_ppo  = [results["hfdc_ppo"][k]["mean"] for k in keys]
    s_ppo  = [results["hfdc_ppo"][k]["std"]  for k in keys]

    x     = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.25, 2))
    ax.bar(x - width/2, m_hfdc, width, yerr=s_hfdc, label=LABEL_HFDC,
           color=COLOR_HFDC, capsize=3, error_kw={"lw": 0.8})
    ax.bar(x + width/2, m_ppo,  width, yerr=s_ppo,  label=LABEL_PPO,
           color=COLOR_PPO,  capsize=3, error_kw={"lw": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"Desired force $|F_d|$")
    ax.set_ylabel(r"Mean $|e_F|$ [N]")
    ax.legend(fontsize=7)

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "E3_force_level.svg")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[E3] Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
