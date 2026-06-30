#!/usr/bin/env python3
"""
Scrape the MOJAVE velocity table (Table XVIII) for B1950 source names,
follow each source's individual page, and extract the "R.A. and Dec.
(J2000)" field. Results are written to a CSV file.

Usage:
    pip install requests beautifulsoup4
    python mojave_radec_scraper.py

Output:
    mojave_radec.csv  (columns: B1950_Name, RA_Dec_J2000, Source_URL)
    
Written by Claude.AI at the behest of M. Seifert.  Caveat physicus. 
"""

import csv
import re
import time
import sys

import requests
from bs4 import BeautifulSoup

BASE = "https://www.cv.nrao.edu/MOJAVE"
VELOCITY_TABLE_URL = f"{BASE}/velocitytableXVIII.html"
SOURCEPAGE_URL = f"{BASE}/sourcepages/{{name}}.shtml"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MOJAVE-RA-Dec-Scraper/1.0)"
}

# Matches B1950-style source names like 0003+380, 0003-066, 1730-130, etc.
NAME_RE = re.compile(r"\b(\d{4}[+-]\d{2,3})\b")


def get_b1950_names():
    """Fetch the velocity table page and extract unique B1950 names."""
    resp = requests.get(VELOCITY_TABLE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    names = []
    seen = set()

    # The B1950 names appear as link text/hrefs pointing to sourcepages/
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "sourcepages/" in href and href.endswith(".shtml"):
            name = href.split("sourcepages/")[-1].replace(".shtml", "")
            if name not in seen:
                seen.add(name)
                names.append(name)

    # Fallback: if no links found, try parsing visible text in first column
    if not names:
        text = soup.get_text()
        for m in NAME_RE.finditer(text):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                names.append(name)

    return names


def get_ra_dec(name):
    """Fetch a source's page and extract the R.A. and Dec. (J2000) field."""
    url = SOURCEPAGE_URL.format(name=name)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        return None, url, f"ERROR: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the table row whose label cell contains "R.A. and Dec"
    ra_dec = None
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            if "R.A. and Dec" in label:
                ra_dec = cells[1].get_text(strip=True)
                break

    if ra_dec is None:
        # Fallback: regex search raw text for the typical format
        # e.g. "0h5m57.175s     +38d20'15.149\""
        m = re.search(r"\d+h\d+m[\d.]+s\s+[+-]\d+d\d+'[\d.]+\"", resp.text)
        if m:
            ra_dec = m.group(0)

    return ra_dec, url, None


def main():
    print("Fetching B1950 name list from velocity table...")
    names = get_b1950_names()
    print(f"Found {len(names)} unique B1950 source names.")

    if not names:
        print("No source names found — the page structure may have changed.")
        sys.exit(1)

    results = []
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] Fetching {name} ...", end=" ")
        ra_dec, url, err = get_ra_dec(name)
        if err:
            print(err)
        else:
            print(ra_dec if ra_dec else "NOT FOUND")
        results.append({
            "B1950_Name": name,
            "RA_Dec_J2000": ra_dec or "",
            "Source_URL": url,
        })
        time.sleep(0.5)  # be polite to the server

    out_path = "mojave_radec.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["B1950_Name", "RA_Dec_J2000", "Source_URL"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
