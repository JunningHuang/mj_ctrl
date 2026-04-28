"""
Trajectory visualisation for thesis Chapter 5.

2×2 panel:
  (a) Sinusoidal  — three direction angles θ ∈ {0°, 45°, 90°}, amplitude A annotated
  (b) Circle      — full revolution, radius r annotated
  (c) Lissajous   — ratio 1:2 figure-8, half-amplitudes Ax/Ay annotated
  (d) Ramp-Hold   — x(t) time profile coloured by phase (move / hold)

Run from repo root:
    python -m ppo_sim_eval_thesis.plot_trajectories
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

try:
    from tueplots import bundles
    plt.rcParams.update(bundles.icml2024(usetex=False))
except ImportError:
    pass

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

from src import (
    CircleTrajectory,
    LissajousTrajectory,
    RampHoldTrajectory,
    SinusoidalTrajectory,
)

_ORIGIN = np.zeros(3)
_R_EYE  = np.eye(3)
_N      = 3000          # sampling resolution
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample(traj, t_end, n=_N):
    """Return (x_cm, y_cm) arrays sampled over [0, t_end]."""
    ts  = np.linspace(0, t_end, n)
    pts = np.array([traj(t)[0] for t in ts])
    return pts[:, 0] * 1e2, pts[:, 1] * 1e2   # m → cm


def _direction_arrow(ax, x, y, frac, color, mut=7):
    """Place a hollow arrow indicating travel direction at position frac ∈ (0,1)."""
    n    = len(x)
    i    = int(frac * n)
    step = max(3, n // 80)
    j    = min(i + step, n - 1)
    ax.annotate(
        "", xy=(x[j], y[j]), xytext=(x[i], y[i]),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=0.7, mutation_scale=mut),
    )


def _double_arrow(ax, xy0, xy1, color="gray", lw=0.7, mut=6):
    """Double-headed annotation arrow between two points (in data coordinates)."""
    ax.annotate("", xy=xy1, xytext=xy0,
                arrowprops=dict(arrowstyle="<->", color=color,
                                lw=lw, mutation_scale=mut))


# ---------------------------------------------------------------------------
# Panel (a): Sinusoidal — three direction angles
# ---------------------------------------------------------------------------

def panel_sinusoidal(ax):
    A   = 0.07    # half-amplitude [m]
    f   = 1.85    # frequency [Hz]
    dur = 1.0 / f # one full oscillation

    cfg = [
        (0.0,       r"$\theta = 0°$",  "tab:blue",   0.25),
        (np.pi / 4, r"$\theta = 45°$", "tab:orange", 0.25),
        (np.pi / 2, r"$\theta = 90°$", "tab:green",  0.26),
    ]

    for angle, lbl, col, frac in cfg:
        traj = SinusoidalTrajectory(
            start_pos=_ORIGIN, amplitude=A, frequency=f,
            R_slope=_R_EYE, size_z=0.0, direction_angle=angle,
        )
        x, y = _sample(traj, dur)
        ax.plot(x, y, color=col, lw=1.3, label=lbl, zorder=3)
        _direction_arrow(ax, x, y, frac=frac, color=col)

    A_cm = A * 1e2

    # Double-headed arrow for amplitude on the θ=0° axis
    _double_arrow(ax, (0, 0), (A_cm, 0), color="tab:blue")
    ax.text(A_cm / 2, 0.55, r"$A$", ha="center", va="bottom",
            fontsize=7, color="tab:blue")

    # Arc showing θ for the 45° direction
    arc_r = 0.38 * A_cm
    arc = Arc((0, 0), 2 * arc_r, 2 * arc_r,
              angle=0, theta1=0, theta2=45,
              color="tab:orange", lw=0.9, linestyle="--", zorder=4)
    ax.add_patch(arc)
    mid = np.pi / 8
    ax.text(arc_r * 1.3 * np.cos(mid), arc_r * 1.3 * np.sin(mid),
            r"$\theta$", ha="center", va="center",
            fontsize=7, color="tab:orange")

    # Origin marker
    ax.plot(0, 0, "k.", ms=3, zorder=5)

    lim = A_cm * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [cm]")
    ax.set_ylabel("$y$ [cm]")
    ax.set_title("(a) Sinusoidal", fontsize=8, pad=4)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.85,
              handlelength=1.0, borderpad=0.4)


# ---------------------------------------------------------------------------
# Panel (b): Circle
# ---------------------------------------------------------------------------

def panel_circle(ax):
    r    = 0.10     # radius [m]
    omega = 2 * np.pi
    dur  = 2 * np.pi / omega   # one full revolution

    traj = CircleTrajectory(
        center=_ORIGIN, radius=r, angular_speed=omega,
        R_slope=_R_EYE, size_z=0.0,
    )
    x, y = _sample(traj, dur)
    ax.plot(x, y, color="tab:blue", lw=1.3, zorder=3)
    _direction_arrow(ax, x, y, frac=0.10, color="tab:blue")

    r_cm = r * 1e2

    # Radius line + label
    ang = np.pi / 4
    ax.plot([0, r_cm * np.cos(ang)], [0, r_cm * np.sin(ang)],
            color="gray", lw=0.8, ls="--", zorder=2)
    ax.text(r_cm / 2 * np.cos(ang) - 0.4,
            r_cm / 2 * np.sin(ang) + 0.5,
            r"$r$", ha="right", va="bottom", fontsize=7, color="gray")

    # Centre marker
    ax.plot(0, 0, "k+", ms=5, mew=0.9, zorder=5)

    lim = r_cm * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [cm]")
    ax.set_ylabel("$y$ [cm]")
    ax.set_title("(b) Circle", fontsize=8, pad=4)


# ---------------------------------------------------------------------------
# Panel (c): Lissajous 1:2 (figure-8)
# ---------------------------------------------------------------------------

def panel_lissajous(ax):
    A      = 0.07
    base_f = 1.0
    # Period of the 1:2 pattern = 1 / base_freq (x completes 1 cycle, y completes 2)
    dur    = 1.0 / base_f

    traj = LissajousTrajectory(
        center=_ORIGIN, x_amplitude=A, y_amplitude=A,
        base_freq=base_f, freq_ratio_x=1, freq_ratio_y=2,
        phase=float(np.pi / 2), R_slope=_R_EYE, size_z=0.0,
    )
    x, y = _sample(traj, dur)
    ax.plot(x, y, color="tab:blue", lw=1.3, zorder=3)
    _direction_arrow(ax, x, y, frac=0.10, color="tab:blue")
    _direction_arrow(ax, x, y, frac=0.60, color="tab:blue")

    A_cm = A * 1e2

    # Ax: half-width annotation (horizontal, below figure)
    y_ann = -A_cm * 1.28
    _double_arrow(ax, (0, y_ann), (A_cm, y_ann), color="gray")
    ax.text(A_cm / 2, y_ann - 0.6, r"$A_x$", ha="center", va="top",
            fontsize=7, color="gray")

    # Ay: half-height of top lobe (vertical, left of figure)
    x_ann = -A_cm * 1.28
    _double_arrow(ax, (x_ann, 0), (x_ann, A_cm), color="gray")
    ax.text(x_ann - 0.4, A_cm / 2, r"$A_y$", ha="right", va="center",
            fontsize=7, color="gray")

    lim = A_cm * 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [cm]")
    ax.set_ylabel("$y$ [cm]")
    ax.set_title(r"(c) Lissajous ($f_x{:}f_y = 1{:}2$)", fontsize=8, pad=4)


# ---------------------------------------------------------------------------
# Panel (d): Ramp-Hold — x(t) time profile
# ---------------------------------------------------------------------------

def panel_ramphold(ax):
    move_dur = 3.0
    hold_dur = 2.0
    d_cm     = 3.5          # offset in cm

    start = np.array([0.0, 0.0, 0.0])
    end   = np.array([d_cm / 1e2, 0.0, 0.0])
    traj  = RampHoldTrajectory(
        start_pos=start, end_pos=end,
        move_duration=move_dur, hold_duration=hold_dur,
    )

    half  = move_dur + hold_dur
    T     = 2 * half
    ts    = np.linspace(0, T, 4000)
    xs    = np.array([traj(t)[0][0] * 1e2 for t in ts])

    # Colour segments: move=blue, hold=orange
    phases = [
        (0,          move_dur, "tab:blue",   "move"),
        (move_dur,   half,     "tab:orange", "hold"),
        (half,       half + move_dur, "tab:blue",   None),
        (half + move_dur, T,   "tab:orange", None),
    ]
    for t0, t1, col, lbl in phases:
        mask = (ts >= t0) & (ts <= t1)
        ax.plot(ts[mask], xs[mask], color=col, lw=1.3,
                label=lbl if lbl else "_nolegend_")

    # Mark waypoints
    ax.axhline(0,    color="gray", lw=0.6, ls=":", zorder=1)
    ax.axhline(d_cm, color="gray", lw=0.6, ls=":", zorder=1)

    # Offset annotation
    ax.annotate("", xy=(T * 0.97, d_cm), xytext=(T * 0.97, 0),
                arrowprops=dict(arrowstyle="<->", color="gray",
                                lw=0.7, mutation_scale=6))
    ax.text(T * 0.96, d_cm / 2, f"{d_cm:.1f} cm",
            ha="right", va="center", fontsize=6, color="gray")

    ax.set_xlim(0, T)
    ax.set_ylim(-0.6, d_cm + 1.2)
    ax.set_yticks([0, d_cm])
    ax.set_yticklabels(["0", f"{d_cm:.1f}"])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$x$ [cm]")
    ax.set_title("(d) Ramp-Hold", fontsize=8, pad=4)
    ax.legend(fontsize=6, loc="center right", framealpha=0.85,
              handlelength=1.0, borderpad=0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(6.75, 1.8))
    fig.subplots_adjust(wspace=0.55)

    panel_sinusoidal(axes[0])
    panel_circle(    axes[1])
    panel_lissajous( axes[2])
    panel_ramphold(  axes[3])

    out = os.path.join(OUT_DIR, "trajectories_overview.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[TRAJ] Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
