# Lorentz Symmetry Violations of Superluminal Jets

Bayesian inference framework for constraining Lorentz violation parameters using superluminal astrophysical jet observations. Uses PyMC/PyTensor MCMC sampling to fit a custom log-likelihood model to transverse velocity data from the MOJAVE survey or synthetic datasets. The model constrains a scalar parameter ($B_0$) and a 3-vector ($\vec{B}$) that parameterize anisotropic Lorentz violation.

## Pipeline

```
 ┌──────────────────────┐     ┌───────────────────────┐
 │  MOJAVE HTML table   │     │  Simulation generator  │
 │  Mojave_html_to_csv  │     │  aniso_simulated_data  │
 └─────────┬────────────┘     └───────────┬────────────┘
           │                              │
           ▼                              ▼
   mojave_cleaned.csv           generated_sources.csv
           │                              │
           └──────────┬───────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │ simulationImport │
            │   importCSV()   │
            └────────┬────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
  ┌───────────────┐    ┌───────────────┐
  │  Anisotropic  │    │  Isotropic    │
  │  MCMC model   │    │  MCMC model   │
  │  (B0, B_vec)  │    │  (q)          │
  └───────┬───────┘    └───────┬───────┘
          │                    │
          ▼                    ▼
    trace plots           trace plots
    Mollweide map         posterior
    pair plot             summaries
    posterior
```

## Quick Start

```bash
# Install dependencies (Python 3.13+, managed with uv)
uv sync

# Run the anisotropic MCMC analysis
python scripts/PyMC_Pytensor_noise_v_main_varsig_aniso.py

# Run the isotropic MCMC analysis
python scripts/PyMC_Pytensor_noise_v_main_varsig.py

# Generate synthetic data only
python scripts/aniso_simulated_data_gen_v.py

# Fetch MOJAVE data
python scripts/Mojave_html_to_csv.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| [Anisotropic Model](scripts/PyMC_Pytensor_noise_v_main_varsig_aniso.md) | MCMC sampling of $B_0$ and $\vec{B}$ with direction-dependent $w_c$ |
| [Isotropic Model](scripts/iso_model.md) | MCMC sampling of a single scalar parameter $q$ ($w_c = q + 1$) |
| [Data Generator](scripts/data_gen.md) | Produces synthetic velocity data with configurable LV parameters |
| [Simulation Import](scripts/simulation_import.md) | Shared CSV loader for both datasets |
| [MOJAVE Scraper](scripts/mojave_scraper.md) | Fetches and cleans the MOJAVE velocity table |
