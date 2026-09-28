"""Download the public copies used by the leakage checks (about 2.4 GB in total).

Run from the repo root: python scripts/leakage/download.py [name ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import SOURCES, fetch  # noqa: E402

if __name__ == "__main__":
    names = sys.argv[1:] or list(SOURCES)
    for name in names:
        print(f"{name}: about {SOURCES[name][2]:g} MB", flush=True)
        for p in fetch(name):
            print(f"  {p} ({p.stat().st_size / 1e6:.1f} MB)", flush=True)
