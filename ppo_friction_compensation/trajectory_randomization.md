# Random Trajectory Ranges (Training)

All parameters are sampled independently at the start of each episode when
`randomize_trajectory: true` (default). Position values are on the surface plane;
height is fixed by `slope_pos` and `size_z` in the controller config.

---

## Trajectory types (equal probability)

### Circle

| Parameter | Range | Unit |
|-----------|-------|------|
| Radius | uniform [0.05, 0.1] | m |
| Angular speed | uniform [3π, 4.5π] ≈ [9.42, 14.14] | rad/s |
| Center | `slope_pos` (fixed) | m |

Tangential speed = radius × angular_speed → **[0.47, 1.41] m/s**.

---

### Lissajous

| Parameter | Range | Unit |
|-----------|-------|------|
| Amplitude (x & y) | uniform [0.05, 0.1] | m |
| Base frequency | uniform [0.75, 1.5] | Hz |
| Frequency ratio (x:y) | choice of {1:1, 1:2, 2:3} | — |
| Phase offset | π/2 (fixed) | rad |
| Center | `slope_pos` (fixed) | m |

Peak speed ≈ `amplitude × 2π × base_freq × max(freq_ratio)`.
- Min ≈ 0.05 × 2π × 0.75 × 1 ≈ **0.24 m/s**
- Max ≈ 0.1 × 2π × 1.5 × 3 ≈ **2.83 m/s**

---

### Sinusoidal

| Parameter | Range | Unit |
|-----------|-------|------|
| Amplitude | uniform [0.05, 0.1] | m |
| Frequency | uniform [1.5, 2.25] | Hz |
| Direction | choice of {0°, 90°, 45°} (surface x, y, diagonal) | — |
| Start pos | `slope_pos` (fixed) | m |

Peak speed = `amplitude × 2π × frequency`.
- Min ≈ 0.05 × 2π × 1.5 ≈ **0.47 m/s**
- Max ≈ 0.1 × 2π × 2.25 ≈ **1.41 m/s**

---

### Ramp-Hold (stick-slip stress test)

| Parameter | Range | Unit |
|-----------|-------|------|
| Stroke (offset) | uniform [0.02, 0.05] | m |
| Direction | uniform [0°, 360°] | deg |
| Move duration | uniform [2.0, 4.0] | s |
| Hold duration | 2.0 (fixed) | s |

Uses minimum-jerk profile; peak speed ≈ `(15/8) × offset / move_duration`.
- Min ≈ (15/8) × 0.02 / 4.0 ≈ **0.009 m/s**
- Max ≈ (15/8) × 0.05 / 2.0 ≈ **0.047 m/s**

---

## Desired contact force

Sampled from a discrete set each episode:

| Values | Unit |
|--------|------|
| choice of {−5, −8, −12, −15} | N |

Configurable via `f_desired_choices` in `configs/experiment_config.yaml`:
```yaml
training:
  f_desired_choices: [-5.0, -8.0, -12.0, -15.0]
```

---

## Surface friction (optional)

Only active when `randomize_surface_friction: true`.

| Parameter | Range | Unit |
|-----------|-------|------|
| Sliding friction | uniform [0.3, 1.0] | — |

Rolling (0.02) and spinning (0.01) friction remain fixed.
