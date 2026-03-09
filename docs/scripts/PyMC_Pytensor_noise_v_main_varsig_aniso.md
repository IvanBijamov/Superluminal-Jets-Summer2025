# Anisotropic Model

`scripts/PyMC_Pytensor_noise_v_main_varsig_aniso.py`

Samples the scalar Lorentz-violation magnitude $B_0$ and the 3-vector $\vec{B}$ using a custom log-likelihood for transverse velocity data. The critical velocity $w_c$ varies per observation depending on the sky direction $\hat{n}$.

## Usage

```bash
python scripts/PyMC_Pytensor_noise_v_main_varsig_aniso.py
```

When using simulated data, the script calls `regenerate_data()` at startup to produce a fresh `generated_sources.csv` before sampling.

## Configuration

All settings are at the top of the file. Adjust before each run.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `"generated_sources.csv"` | Input CSV. Set to `"mojave_cleaned.csv"` for real data. |
| `N_VAL` | `20` | Integration resolution for likelihood summation. Higher = more accurate, slower. |
| `ENABLE_STRIP_TOP_10_PERCENT` | `False` | Remove top 10% highest-velocity sources before sampling. |
| `DRAWS` | `1000` | Number of posterior draws per chain. |
| `TUNE` | `1000` | Number of tuning (burn-in) samples per chain. |
| `TARGET_ACCEPT` | `0.93` | NUTS target acceptance rate. |
| `CHAINS` | `4` | Number of independent chains. |
| `CORES` | `4` | Number of CPU cores for parallel sampling. |
| `INIT_METHOD` | `"jitter+adapt_diag"` | PyMC initialization method. |
| `RANDOM_SEED` | `42` | Fixed seed for reproducibility. Set to `None` for production. |

!!! note "PyTensor compiler"
    The line `pytensor.config.cxx = "/usr/bin/clang++"` is hardcoded for a specific machine. Comment it out or adjust for your system.

## Model Structure

### Priors

- **$B_0$** (`Bº`): `HalfNormal(sigma=3)` — scalar LV magnitude, non-negative.
- **$\vec{B}$** (`B_vec`): Reparameterized to enforce $\|\vec{B}\| < 1$:
    1. `b_raw ~ Normal(0, 1, shape=3)` — unconstrained direction vector.
    2. $\hat{u} = \texttt{b\_raw} / \|\texttt{b\_raw}\|$ — normalized to a unit vector.
    3. $\rho \sim \text{Beta}(2, 2)$ — radial magnitude in $[0, 1)$.
    4. $\vec{B} = \rho \cdot \hat{u}$.

### Derived quantities

The per-observation inverse critical velocity $w_c$ comes from the positive root of:

$$
-(\,B_0^2 + 1)\,w_c^2 \;-\; 2\,B_0\,B_n\,w_c \;+\; (1 - B_n^2) \;=\; 0
$$

where $B_n = \hat{n} \cdot \vec{B}$ is the projection of $\vec{B}$ onto the source direction. The solution used is:

$$
w_c = \frac{-B_0\,B_n + \sqrt{1 + B_0^2 - B_n^2}}{1 + B_0^2}
$$

### Likelihood

The observed-velocity probability for each source is computed by numerical integration over a Gaussian measurement kernel:

$$
\mathcal{P}_{\mathrm{obs}}(v_t) \approx \sum_{v = v_t - m\sigma_v}^{v_t + m\sigma_v}
\frac{1}{\sqrt{2\pi}\,\sigma_v^{3}}
\left[
(v - v_t)\exp\!\left(-\frac{(v - v_t)^2}{2\sigma_v^{2}}\right)
+ (v + v_t)\exp\!\left(-\frac{(v + v_t)^2}{2\sigma_v^{2}}\right)
\right]
C_v(v)\,\Delta v
$$

where $C_v(v)$ is the cumulative distribution function of the underlying velocity distribution, evaluated piecewise depending on whether $w_c < 1$ (fast-light regime) or $w_c \geq 1$ (slow-light regime).

!!! warning "Current limitation"
    Only the fast-light case ($w_c < 1$) is implemented in the anisotropic code. The slow-light branch is not yet incorporated.

The total log-likelihood is $\sum_i \log \mathcal{P}_{\mathrm{obs}}(v_{t,i})$.

## Outputs

| Output | Description |
|--------|-------------|
| **Trace plot** | Saved to `scripts/trace.png`. Shows MCMC chains for $B_0$ and $\vec{B}$ components. |
| **Mollweide map** | Sky projection of sources with $v_t > 1$, colored by observed speed. |
| **Pair plot** | KDE joint posterior of $B_0$ and $\vec{B}$ with divergence markers. |
| **Posterior plot** | Marginal posterior densities with HDI intervals. |
| **Console summary** | ArviZ summary table with quartiles and sampler diagnostics (divergences, max tree depth). |

## API Reference

::: PyMC_Pytensor_noise_v_main_varsig_aniso.loglike
    options:
      show_root_heading: true

::: PyMC_Pytensor_noise_v_main_varsig_aniso.main
    options:
      show_root_heading: true
