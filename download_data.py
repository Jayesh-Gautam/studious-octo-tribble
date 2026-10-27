"""
AeroFix data downloader
-----------------------
Downloads FAA Service Difficulty Report (SDR) yearly CSVs plus the free public
FAA maintenance reference documents used by the RAG layer.

Usage (run from the project root):
    python download_data.py                  # all SDR years 1995-2026 + docs
    python download_data.py --years 2015 2025 # only 2015..2025 (inclusive)
    python download_data.py --skip-docs

Files already downloaded are skipped, so you can re-run safely.
Requires: pip install requests tqdm pyyaml
"""
import argparse
import sys
import time
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

# URL and folders come from configs/config.yaml (single source of truth). The YAML
# is read directly so this script works before `pip install -e .`.
ROOT = Path(__file__).resolve().parent
with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)
SDR_URL = _CFG["download"]["sdr_url"]
SDR_DIR = ROOT / _CFG["paths"]["raw_sdr"]
DOCS_DIR = ROOT / _CFG["paths"]["raw_docs"]

# Free, public FAA documents. If a link 404s, the FAA has moved it:
# find the current link on faa.gov and add it here.
# Links verified Oct 2026. The "B" editions of the AMT handbooks are single combined
# volumes and supersede the 2018 split volumes (e.g. FAA-H-8083-31A Vol 2), so the
# old files are not downloaded: duplicate passages would skew RAG retrieval.
_FAA_HB = "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation"
PUBLIC_DOCS = {
    "FAA-H-8083-30B_AMT_General.pdf": f"{_FAA_HB}/amtg_handbook.pdf",
    "FAA-H-8083-31B_AMT_Airframe.pdf":
        f"{_FAA_HB}/FAA-H-8083-31B_Aviation_Maintenance_Technician_Handbook.pdf",
    "FAA-H-8083-32B_AMT_Powerplant.pdf": f"{_FAA_HB}/amt_powerplant_handbook.pdf",
    "AC_43.13-1B_w-chg1.pdf":
        "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_43.13-1B_w-chg1.pdf",
    # Download manually from drs.faa.gov (JavaScript site, no stable PDF URL) and
    # save into data/raw/docs/ under these names:
    #   TCDS_A28NM.pdf  (Airbus A318/A319/A320/A321)
    #   TCDS_A16WE.pdf  (Boeing 737)
}

HEADERS = {"User-Agent": "Mozilla/5.0 (AeroFix student project; research use)"}


def download(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    print(f"  404 not found: {url}")
                    return False
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or None
                with open(tmp, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.rename(dest)
            print(f"  ok: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return True
        except Exception as e:  # network hiccup -> retry
            print(f"  attempt {attempt} failed for {dest.name}: {e}")
            time.sleep(3 * attempt)
    if tmp.exists():
        tmp.unlink()
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs=2, type=int, default=_CFG["download"]["years"],
                   metavar=("START", "END"))
    p.add_argument("--skip-docs", action="store_true")
    a = p.parse_args()

    failed = []
    print(f"Downloading SDR CSVs {a.years[0]}-{a.years[1]} -> {SDR_DIR}/")
    for y in range(a.years[0], a.years[1] + 1):
        if not download(SDR_URL.format(year=y), SDR_DIR / f"SDR-{y}.csv"):
            failed.append(f"SDR-{y}")

    if not a.skip_docs:
        print(f"\nDownloading public FAA reference docs -> {DOCS_DIR}/")
        for name, url in PUBLIC_DOCS.items():
            if not download(url, DOCS_DIR / name):
                failed.append(name)

    print("\nDone." if not failed else f"\nDone with failures: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
