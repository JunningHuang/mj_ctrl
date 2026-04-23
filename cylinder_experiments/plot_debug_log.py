import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "real", "debug_log.csv")

df = pd.read_csv(CSV_PATH)
t = df["t"]

groups = {
    "f_ext":        ["f_ext_x", "f_ext_y", "f_ext_z"],
    "f_ext_phi":    ["f_ext_phi"],
    "f_ext_motion": [c for c in df.columns if c.startswith("f_ext_motion_")],
    "tau_v":        [c for c in df.columns if c.startswith("tau_v_")],
    "tau_x":        [c for c in df.columns if c.startswith("tau_x_")],
    "tau_phi":      [c for c in df.columns if c.startswith("tau_phi_")],
    "tau":          [c for c in df.columns if c.startswith("tau_") and not any(
                        c.startswith(p) for p in ("tau_v_", "tau_x_", "tau_phi_"))],
}

for title, cols in groups.items():
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in cols:
        ax.plot(t, df[col], label=col)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()

plt.show()
