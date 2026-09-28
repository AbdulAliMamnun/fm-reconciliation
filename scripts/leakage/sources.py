"""Public copies of training datasets used by the leakage checks.

Every file is pinned to a repository revision, so a later run downloads the
same bytes. Files go to data/leakage/ (gitignored) and are never committed.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_DIR = ROOT / "data" / "leakage"
LOCAL_DIR = ROOT / "data" / "hierarchical"   # our evaluation data, as downloaded by datasetsforecast
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

REVISIONS = {
    "autogluon/fev_datasets": "f71c0fff4cf81283a2c43e7f3a73aa4f9826aef8",
    "autogluon/chronos_datasets": "eeecad0b82a8c237e212ce6f8d1abecb513e2cec",
    "Salesforce/GiftEvalPretrain": "6830b624de7ed2b3d3e5b85bb6959d81dcc5d874",
}

# name -> (repo, files, approximate size in MB)
SOURCES = {
    "fev_australian_tourism": ("autogluon/fev_datasets",
                               ["australian_tourism/train-00000-of-00001.parquet"], 0.02),
    "m4_monthly": ("autogluon/chronos_datasets", ["m4_monthly/train-00000-of-00001.parquet"], 53),
    "m4_quarterly": ("autogluon/chronos_datasets", ["m4_quarterly/train-00000-of-00001.parquet"], 13),
    "monash_tourism_monthly": ("autogluon/chronos_datasets",
                               ["monash_tourism_monthly/train-00000-of-00001.parquet"], 0.3),
    "monash_tourism_quarterly": ("autogluon/chronos_datasets",
                                 ["monash_tourism_quarterly/train-00000-of-00001.parquet"], 0.2),
    "kaggle_web_traffic_weekly": ("Salesforce/GiftEvalPretrain",
                                  ["kaggle_web_traffic_weekly/data-00000-of-00001.arrow"], 71),
    "wiki_rolling": ("Salesforce/GiftEvalPretrain", ["wiki-rolling_nips/data-00000-of-00001.arrow"], 164),
    "wiki_daily_100k": ("autogluon/chronos_datasets",
                        [f"wiki_daily_100k/train-{i:05d}-of-00005.parquet" for i in range(5)], 593),
    "extended_web_traffic": ("Salesforce/GiftEvalPretrain",
                             [f"extended_web_traffic_with_missing/data-{i:05d}-of-00003.arrow"
                              for i in range(3)], 1488),
}


def fetch(name):
    """Download the files of one source if missing. Returns their local paths."""
    from huggingface_hub import hf_hub_download

    repo, files, _ = SOURCES[name]
    return [
        Path(hf_hub_download(repo, f, repo_type="dataset", revision=REVISIONS[repo],
                             local_dir=DOWNLOAD_DIR / repo.replace("/", "__")))
        for f in files
    ]
