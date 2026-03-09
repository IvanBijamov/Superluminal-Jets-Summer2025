# Data Generator

`scripts/aniso_simulated_data_gen_v.py`

Generates synthetic transverse velocity data with configurable Lorentz violation parameters. Produces `generated_sources.csv` at the project root.

## Usage

```bash
# Standalone
python scripts/aniso_simulated_data_gen_v.py

# Also called automatically by the anisotropic model at startup
python scripts/PyMC_Pytensor_noise_v_main_varsig_aniso.py
```

## Simulation Parameters

Configured as constants at the top of `regenerate_data()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delta` ($\delta$) | `-1` | LV coefficient. Controls the sign/type of Lorentz violation. |
| `Bº` ($B_0$) | `0.8` | Scalar LV magnitude. |
| `B_vec` ($\vec{B}$) | `[0.5, 0.0, 0.2]` | 3-vector LV direction. |
| `N_SOURCES` | `1000` | Number of synthetic sources to generate. |

## Physics

### `solve_wc`

Solves the quadratic equation for the critical inverse velocity $w_c$ given a direction vector (either the source emission direction $\hat{v}$ for Cherenkov braking, or the observer direction $\hat{n}$):

$$
a\,w_c^2 + b\,w_c + c = 0
$$

where $a = \delta B_0^2 - 1$, $\;b = 2\delta B_0 (\vec{B}\cdot\hat{n})$, $\;c = \delta(\vec{B}\cdot\hat{n})^2 + 1$.

Returns all positive roots, or `None` if no positive solution exists. When $\vec{B} = 0$, reduces to $w_c = 1/\sqrt{1 - \delta B_0^2}$.

### Cherenkov braking

If a source's raw velocity $v$ exceeds the critical velocity $1/w_c$ in its emission direction, it is braked down to $1/w_c$. This simulates vacuum Cherenkov radiation losses.

### `w_true`

Computes the true inverse transverse velocity for a source with velocity $v$ in direction $\hat{v}$ as observed from direction $\hat{n}$:

$$
w_{\mathrm{true}} = \frac{1 + (v / v_c(\hat{n}))\,(\hat{n}\cdot\hat{v})}{v\,\sqrt{1 - (\hat{n}\cdot\hat{v})^2}}
$$

where $v_c(\hat{n}) = 1/w_c(\hat{n})$ is the critical velocity in the observer direction.

### Measurement noise

Each true velocity is perturbed by Gaussian noise: $v_{\mathrm{obs}} \sim \mathcal{N}(v_{\mathrm{true}},\, \sigma)$ where $\sigma \sim |\mathcal{N}(0, 0.1)|$. Negative observed velocities are reflected to ensure $v_{\mathrm{obs}} > 0$.

## Output CSV Format

The file `generated_sources.csv` has the following structure:

**Row 1 (header):** simulation parameters

```
delta, B0, B_x, B_y, B_z
```

**Rows 2+:** one row per source

| Column | Index | Description |
|--------|-------|-------------|
| `n_x`  | 0 | x-component of observer direction $\hat{n}$ |
| `n_y`  | 1 | y-component of observer direction $\hat{n}$ |
| `n_z`  | 2 | z-component of observer direction $\hat{n}$ |
| `v_obs` | 3 | Observed transverse velocity (with noise) |
| `v_sigma` | 4 | Measurement uncertainty |
| `v_true` | 5 | True transverse velocity (before noise) |
