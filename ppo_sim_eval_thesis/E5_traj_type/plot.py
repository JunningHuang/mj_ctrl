"""
E5 plot — Per-trajectory-type: grouped bar chart with HFDC + models B/C/D.

Run from repo root:
    python -m ppo_sim_eval_thesis.E5_traj_type.plot
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

_TRAJ_LABELS = {
    "circle":     "Circle",
    "sinusoidal": "Sinusoidal",
    "lissajous":  "Lissajous",
    "ramp_hold":  "Ramp-Hold",
}

# Condition display order and colours
_COND_ORDER  = ["hfdc", "ppo_B", "ppo_C", "ppo_D"]
_COND_COLORS = {
    "hfdc":  "tab:blue",
    "ppo_B": "tab:orange",
    "ppo_C": "tab:green",
    "ppo_D": "tab:red",
}


def main():
    path = os.path.join(RES_DIR, "results.json")
    with open(path) as f:
        data = json.load(f)

    meta    = data["metadata"]
    results = data["results"]
    types   = meta["traj_types"]
    conds   = meta["conditions"]          # {cond: {checkpoint, label}}

    x_labels  = [_TRAJ_LABELS.get(t, t) for t in types]
    n_conds   = len(_COND_ORDER)
    n_types   = len(types)
    width     = 0.18
    offsets   = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * width
    x         = np.arange(n_types)

    # Scale height for 4-bar groups
    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    for offset, cond in zip(offsets, _COND_ORDER):
        label  = conds[cond]["label"]
        color  = _COND_COLORS[cond]
        means  = [results[cond][t]["mean"] for t in types]
        stds   = [results[cond][t]["std"]  for t in types]
        ax.bar(x + offset, means, width, yerr=stds, label=label,
               color=color, capsize=2, error_kw={"lw": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=15, ha="right")
    ax.set_xlabel("Trajectory type")
    ax.set_ylabel(r"Mean $|e_F|$ [N]")
    ax.legend(fontsize=6, loc="upper right")

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "E5_traj_type.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[E5] Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
