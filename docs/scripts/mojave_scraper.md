# MOJAVE Scraper

`scripts/Mojave_html_to_csv.py`

Fetches the MOJAVE VLBA velocity table from NRAO, parses the HTML, splits `±` columns into separate value/error fields, and saves a cleaned CSV.

## Usage

```bash
python scripts/Mojave_html_to_csv.py
```

Requires network access. The output file `mojave_cleaned.csv` is written to the project root.

## Data Source

The script fetches the HTML table from:

```
https://www.cv.nrao.edu/MOJAVE/velocitytableXVIII.html
```

This is the MOJAVE XVIII velocity table containing proper motion measurements for AGN jet components observed with VLBA.

## Processing Steps

1. **Fetch** — Downloads the HTML page and unescapes HTML entities (e.g., `&plusmn;` to `±`).
2. **Parse** — Uses `pandas.read_html()` to extract the first table.
3. **Clean headers** — Drops the first 3 rows (multi-level headers) and the last row (footer).
4. **Split ±** — For every column containing `±`, creates two new columns: `<col>_val` and `<col>_err`.
5. **Convert** — Applies `pd.to_numeric` where possible.
6. **Save** — Writes to `mojave_cleaned.csv` at the project root.

## Output CSV

The cleaned CSV preserves all original columns from the MOJAVE table. Columns that originally contained `±` notation are split into separate value and error columns.

Key columns used by the MCMC models (via `simulationImport`):

| Column index | Content |
|-------------|---------|
| 13 | Transverse velocity (value) |
| 14 | Transverse velocity (uncertainty) |

The full CSV contains source names, component IDs, proper motions, position angles, and other VLBA measurements.
