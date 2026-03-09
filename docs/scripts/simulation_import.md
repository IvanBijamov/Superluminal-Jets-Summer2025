# Simulation Import

`scripts/simulationImport.py`

Shared CSV loader that reads either simulated or MOJAVE data and returns a normalized list of per-source records.

## Usage

```python
from simulationImport import importCSV

data = importCSV("generated_sources.csv", filetype="Simulated")
data = importCSV("mojave_cleaned.csv", filetype="Mojave")
```

## `importCSV` function

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str` | Path to the CSV file. |
| `filetype` | `str` | `"Simulated"` for generated data, `"Mojave"` for MOJAVE survey data. |

### Return format

Returns a list of lists. Each row has the structure:

| Index | Field | Description |
|-------|-------|-------------|
| 0 | `r` | Radial distance (norm of position vector) |
| 1 | `theta` | Declination / polar angle |
| 2 | `phi` | Right ascension / azimuthal angle |
| 3 | `vt` | Observed transverse velocity |
| 4 | `sigma` | Velocity uncertainty |
| 5 | `n_x` | x-component of unit vector $\hat{n}$ |
| 6 | `n_y` | y-component of unit vector $\hat{n}$ |
| 7 | `n_z` | z-component of unit vector $\hat{n}$ |

### Column index mapping

The two filetypes read velocity and uncertainty from different CSV columns:

| Filetype | Velocity column | Uncertainty column |
|----------|----------------|--------------------|
| `"Simulated"` | 3 | 4 |
| `"Mojave"` | 13 | 14 |

For simulated data, columns 0-2 provide the Cartesian direction $(n_x, n_y, n_z)$ directly. For MOJAVE data, spherical coordinates and $\hat{n}$ are set to `NaN`.

!!! warning "MOJAVE n_hat"
    The MOJAVE branch currently returns `NaN` for all directional fields (indices 0-2, 5-7). The anisotropic model **will not work** with MOJAVE data until `n_hat` is derived.

## API Reference

::: simulationImport.importCSV
    options:
      show_root_heading: true
