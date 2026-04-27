# tueplots-style

Apply the project's standard matplotlib/tueplots style to a plotting script.

## Rules

**Imports and rcParams — always at the top of the file:**

```python
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
```

**Figure size:** Default to `figsize=(3.25, 2)` (single-column width). For multi-row figures with a shared x-axis, scale height proportionally — e.g. `(3.25, 3.012)` for 3 rows.

**No grid:** Never add `ax.grid(...)`.

**Save at dpi=600:**
```python
fig.savefig(out_path, dpi=300)
```

## When invoked

1. Read the target plotting script(s).
2. Replace or add the rcParams block at the top (after imports).
3. Remove any `ax.grid(...)` calls.
4. Update all `fig.savefig(...)` calls to use `dpi=300`.
5. Update `figsize` to `(3.25, 2)` unless the figure has multiple rows — in that case scale proportionally and note the change.
6. Report what was changed.
