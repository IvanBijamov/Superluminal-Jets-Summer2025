# Isotropic Model

`scripts/PyMC_Pytensor_noise_v_main_varsig.py`

Samples a single scalar parameter $q$ where $w_c = q + 1$. This is the direction-independent (isotropic) Lorentz violation model — the critical velocity is the same for all sources regardless of sky position.

## Usage

```bash
python scripts/PyMC_Pytensor_noise_v_main_varsig.py
```

## Configuration

Settings are configured by editing variables near the top of the file.

| Variable | Default | Description |
|----------|---------|-------------|
| `dataset` | `"generated_sources.csv"` | Input CSV. Comment/uncomment to switch to `"mojave_cleaned.csv"`. |
| `n_val_default` | `20` | Integration resolution for likelihood summation. |

MCMC settings are passed directly in the `pm.sample()` call:

| Parameter | Value |
|-----------|-------|
| `draws` | `1000` |
| `tune` | `1000` |
| `target_accept` | `0.95` |

!!! note "PyTensor compiler"
    The line `pytensor.config.cxx = "/usr/bin/clang++"` is hardcoded for a specific machine. Comment it out or adjust for your system.

## Model Structure

### Prior

- **$q$**: `TruncatedNormal(sigma=3, lower=-1)` — ensures $w_c = q + 1 > 0$.

### Likelihood

Uses the same numerical integration form as the anisotropic model, but with a single scalar $w_c$ shared across all observations (no per-source direction dependence):

$$
\mathcal{P}_{\mathrm{obs}}(v_t) \approx \sum_{v = v_t - m\sigma_v}^{v_t + m\sigma_v}
\frac{1}{\sqrt{2\pi}\,\sigma_v^{3}}
\left[
(v - v_t)\exp\!\left(-\frac{(v - v_t)^2}{2\sigma_v^{2}}\right)
+ (v + v_t)\exp\!\left(-\frac{(v + v_t)^2}{2\sigma_v^{2}}\right)
\right]
C_v(v)\,\Delta v
$$

The CDF function $C_v(v)$ is evaluated piecewise:

- If $v < 0$: $C_v = 0$
- If $w_c < 1$ (fast-light) and $w_c^2 + w_t^2 < 1$: $C_v = 1$
- If $w_c < 1$ (fast-light) and $w_c^2 + w_t^2 \geq 1$: uses $\arctan(\sqrt{w_c^2 + w_t^2 - 1})$ expression
- If $w_c \geq 1$ (slow-light): uses $\arctan(w_t / w_c)$ expression

## Outputs

| Output | Description |
|--------|-------------|
| **Trace plot** | Saved to `scripts/trace_iso.png`. Shows MCMC chains for $q$. |
| **Console summary** | ArviZ summary table with 25th, 50th, and 75th percentiles. |

## API Reference

::: PyMC_Pytensor_noise_v_main_varsig.loglike
    options:
      show_root_heading: true

::: PyMC_Pytensor_noise_v_main_varsig.main
    options:
      show_root_heading: true
