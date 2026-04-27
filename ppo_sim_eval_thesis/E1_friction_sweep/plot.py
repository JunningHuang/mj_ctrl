"""
E1 plot — Friction sweep: mean |e_F| vs μ for HFDC and HFDC+PPO.

Run from repo root:
    python -m ppo_sim_eval_thesis.E1_friction_sweep.plot
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
    mus     = sorted(float(k) for k in results["hfdc"])
    ood_bnd = meta.get("ood_boundary", 0.3)

    def _extract(cond):
        means = [results[cond][f"{m:.1f}"]["mean"] for m in mus]
        stds  = [results[cond][f"{m:.1f}"]["std"]  for m in mus]
        return np.array(means), np.array(stds)

    m_hfdc,    s_hfdc    = _extract("hfdc")
    m_ppo,     s_ppo     = _extract("hfdc_ppo")

    fig, ax = plt.subplots(figsize=(3.25, 2))

    ax.plot(mus, m_hfdc, color=COLOR_HFDC, marker="o", ms=3, lw=1.2, label=LABEL_HFDC)
    ax.fill_between(mus, m_hfdc - s_hfdc, m_hfdc + s_hfdc,
                    color=COLOR_HFDC, alpha=0.2)

    ax.plot(mus, m_ppo, color=COLOR_PPO, marker="s", ms=3, lw=1.2, label=LABEL_PPO)
    ax.fill_between(mus, m_ppo - s_ppo, m_ppo + s_ppo,
                    color=COLOR_PPO, alpha=0.2)

    ax.axvline(ood_bnd, color="gray", ls="--", lw=0.8)
    ax.text(ood_bnd - 0.01, ax.get_ylim()[1] * 0.95, "OOD",
            ha="right", va="top", fontsize=6, color="gray")

    ax.set_xlabel(r"Surface friction $\mu$")
    ax.set_ylabel(r"Mean $|e_F|$ [N]")
    ax.set_xticks(mus)
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=7, loc="upper left")

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "E1_friction_sweep.svg")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[E1] Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
