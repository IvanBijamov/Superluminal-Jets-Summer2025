# Scripts

All source code lives in the `scripts/` directory. The table below summarizes each script and its role in the pipeline.

| Script | File | Purpose |
|--------|------|---------|
| [Anisotropic Model](PyMC_Pytensor_noise_v_main_varsig_aniso.md) | `PyMC_Pytensor_noise_v_main_varsig_aniso.py` | MCMC sampling of $B_0$ and $\vec{B}$ for direction-dependent Lorentz violation |
| [Isotropic Model](iso_model.md) | `PyMC_Pytensor_noise_v_main_varsig.py` | MCMC sampling of scalar $q$ for direction-independent Lorentz violation |
| [Data Generator](data_gen.md) | `aniso_simulated_data_gen_v.py` | Generates synthetic velocity data with configurable LV parameters |
| [Simulation Import](simulation_import.md) | `simulationImport.py` | Shared CSV loader that normalizes both datasets into a common format |
| [MOJAVE Scraper](mojave_scraper.md) | `Mojave_html_to_csv.py` | Fetches the MOJAVE velocity HTML table and produces `mojave_cleaned.csv` |
