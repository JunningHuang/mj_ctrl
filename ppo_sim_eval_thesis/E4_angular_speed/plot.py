"""
E4 plot — Angular speed sweep: mean |e_F| vs ω for HFDC and HFDC+PPO.

Run from repo root:
    python -m ppo_sim_eval_thesis.E4_angular_speed.plot
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
    mults   = meta["omega_multiples"]
    keys    = [f"{m:.1f}pi" for m in mults]
    x_ticks = [f"{m:.1f}π" for m in mults]

    m_hfdc = [results["hfdc"][k]["mean"] for k in keys]
    s_hfdc = [results["hfdc"][k]["std"]  for k in keys]
    m_ppo  = [results["hfdc_ppo"][k]["mean"] for k in keys]
    s_ppo  = [results["hfdc_ppo"][k]["std"]  for k in keys]

    # Shade the training ω range (2π–3π rad/s)
    train_lo, train_hi = meta.get("train_omega_range", [2.0, 3.0])

    fig, ax = plt.subplots(figsize=(3.25, 2))

    ax.axvspan(train_lo, train_hi, alpha=0.08, color="gray", label="Train range")

    ax.plot(mults, m_hfdc, color=COLOR_HFDC, marker="o", ms=3, lw=1.2, label=LABEL_HFDC)
    ax.fill_between(mults, np.array(m_hfdc) - np.array(s_hfdc),
                           np.array(m_hfdc) + np.array(s_hfdc),
                    color=COLOR_HFDC, alpha=0.2)

    ax.plot(mults, m_ppo, color=COLOR_PPO, marker="s", ms=3, lw=1.2, label=LABEL_PPO)
    ax.fill_between(mults, np.array(m_ppo) - np.array(s_ppo),
                           np.array(m_ppo) + np.array(s_ppo),
                    color=COLOR_PPO, alpha=0.2)

    ax.set_xticks(mults)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel(r"Angular speed $\omega$ [rad/s]")
    ax.set_ylabel(r"Mean $|e_F|$ [N]")
    ax.legend(fontsize=7)

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "E4_angular_speed.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[E4] Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
