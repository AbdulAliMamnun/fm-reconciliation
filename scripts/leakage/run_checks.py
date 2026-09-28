"""Leakage checks: is our evaluation data inside public copies of training datasets?

Compares the local evaluation files in data/hierarchical/ with public copies of
datasets that foundation models were trained on. The public copies are
downloaded at run time to data/leakage/ (about 2.4 GB, gitignored).

Writes to scripts/leakage/output/:
    summary.json            every number, date range and file hash
    summary.md              the same, readable; cited by docs/leakage.md
    labour_m4_matches.csv   Labour series <-> M4 Monthly series
    wiki2_matches.csv       Wiki2 bottom series <-> web-traffic items and pages
    tourism_best.csv        best candidate for each tourism series, per source

Run from the repo root: python scripts/leakage/run_checks.py
Takes a few minutes.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import DOWNLOAD_DIR, LOCAL_DIR, OUTPUT_DIR, REVISIONS, ROOT, SOURCES, fetch  # noqa: E402

# Criteria for "the same series in another data vintage", on the date-aligned overlap.
# Level correlation and a stable ratio are not enough: a state and the national
# total pass both. A match also needs the period-to-period changes to agree and
# the ratio to be a power of ten (a change of units), within 1%.
STRICT = {"corr": 0.999, "maxdev": 0.05}
LOOSE = {"corr": 0.995, "maxdev": 0.10}
MATCH = {"corr_diff": 0.98, "pow10_dev": 0.01}
MIN_OVERLAP = {"M": 120, "Q": 24}          # Labour
MIN_OVERLAP_TOURISM = {"M": 60, "Q": 20}   # the Monash tourism series end around 2007
M4_HOLDOUT = 18          # Chronos: last H = 18 observations of M4 Monthly are held out
M4_CUTOFF = "2017-01"    # question 2: test windows entirely after this month


# ------------------------------------------------------------------ helpers

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local(name):
    """Our evaluation data: frame (dates x series) and the bottom series ids."""
    df = pd.read_csv(LOCAL_DIR / name / "data.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    bottom = list(pd.read_csv(LOCAL_DIR / name / "agg_mat.csv", index_col=0).columns)
    return df.astype(np.float64), bottom


def read_parquet_series(name):
    """Rows of a chronos_datasets / fev_datasets parquet: id, timestamps, values."""
    out = []
    for p in fetch(name):
        t = pq.read_table(p).to_pandas()
        for row in t.itertuples(index=False):
            out.append((row.id, np.asarray(row.timestamp, dtype="datetime64[ms]"),
                        np.asarray(row.target, dtype=np.float64), getattr(row, "category", None)))
    return out


def arrow_batches(path):
    src = pa.memory_map(str(path))
    try:
        reader = ipc.open_stream(src)
    except pa.ArrowInvalid:
        reader = ipc.open_file(pa.memory_map(str(path)))
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)
        return
    yield from reader


def arrow_series(name):
    """Yield (item_ids, list of float32 arrays) per record batch of a GIFT-Eval Pretrain file."""
    for p in fetch(name):
        for b in arrow_batches(p):
            ids = b.column("item_id").to_pylist()
            col = b.column("target")
            values = col.values.to_numpy(zero_copy_only=False)
            offsets = col.offsets.to_numpy()
            yield ids, [values[offsets[i]:offsets[i + 1]] for i in range(len(ids))]


def period_index(dates, freq):
    d = pd.DatetimeIndex(dates)
    return d.year * 12 + d.month - 1 if freq == "M" else d.year * 4 + d.quarter - 1


def period_label(k, freq):
    return f"{k // 12}-{k % 12 + 1:02d}" if freq == "M" else f"{k // 4}Q{k % 4 + 1}"


def corr_columns(x, Y):
    """Correlation of vector x with every column of Y."""
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=0)
    den = np.sqrt((xc ** 2).sum() * (Yc ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, xc @ Yc / den, np.nan)


def aligned_candidates(ours, freq, public, source, keep_corr, min_overlap, stats=None):
    """Compare date-aligned series. Returns one row per (our series, public series)
    with level correlation above `keep_corr`, and the best correlation per our series."""
    cols = list(ours.columns)
    ok = period_index(ours.index, freq)
    O = ours.to_numpy()
    rows, best = [], {c: (-np.inf, None) for c in cols}
    for pid, ts, x, cat in public:
        pk = period_index(ts, freq)
        lo, hi = max(ok[0], pk[0]), min(ok[-1], pk[-1])
        n = hi - lo + 1
        if n < min_overlap or len(pk) != pk[-1] - pk[0] + 1 or np.isnan(x).any():
            continue
        if stats is not None:
            stats["public_series_compared"] = stats.get("public_series_compared", 0) + 1
        xs = x[lo - pk[0]: hi - pk[0] + 1]
        Os = O[lo - ok[0]: hi - ok[0] + 1]
        if xs.std() == 0:
            continue
        c = corr_columns(xs, Os)
        for j in np.flatnonzero(np.nan_to_num(c, nan=-1) > best_array(best, cols)):
            best[cols[j]] = (c[j], pid)
        for j in np.flatnonzero(np.nan_to_num(c, nan=-1) > keep_corr):
            o = Os[:, j]
            pos = (o > 0) & (xs > 0)
            if pos.sum() < 24:
                continue
            ratio = xs[pos] / o[pos]
            med = np.median(ratio)
            dc = np.corrcoef(np.diff(xs), np.diff(o))[0, 1]
            rows.append({
                "ours": cols[j], "source": source, "public_id": pid, "category": cat,
                "overlap": int(n), "from": period_label(lo, freq), "to": period_label(hi, freq),
                "public_from": period_label(pk[0], freq), "public_to": period_label(pk[-1], freq),
                "corr_level": c[j], "corr_diff": dc, "ratio_median": med,
                "ratio_maxdev": float(np.abs(ratio / med - 1).max()),
                "exact_equal_share": float((xs == o).mean()),
            })
    return pd.DataFrame(rows), best


def best_array(best, cols):
    return np.array([best[c][0] for c in cols])


def sliding_candidates(ours, public, min_overlap):
    """Compare series at every relative shift, ignoring the dates. Returns the
    number of matches under the match criteria and the best level correlation."""
    O = ours.to_numpy()
    T = O.shape[0]
    best, n_match, n_compared = -np.inf, 0, 0
    for pid, ts, x, cat in public:
        if np.isnan(x).any() or len(x) < min_overlap:
            continue
        n_compared += 1
        for shift in range(min_overlap - len(x), T - min_overlap + 1):
            a, b = max(shift, 0), min(shift + len(x), T)     # rows of ours
            xs, Os = x[a - shift: b - shift], O[a:b]
            if xs.std() == 0:
                continue
            c = np.nan_to_num(corr_columns(xs, Os), nan=-1.0)
            best = max(best, float(c.max()))
            for j in np.flatnonzero(c > STRICT["corr"]):
                o = Os[:, j]
                pos = (o > 0) & (xs > 0)
                if pos.sum() < 24:
                    continue
                ratio = xs[pos] / o[pos]
                med = np.median(ratio)
                pow10 = abs(10 ** (np.log10(med) - round(np.log10(med))) - 1)
                dc = np.corrcoef(np.diff(xs), np.diff(o))[0, 1]
                if (np.abs(ratio / med - 1).max() < STRICT["maxdev"] and dc > MATCH["corr_diff"]
                        and pow10 < MATCH["pow10_dev"]):
                    n_match += 1
    return {"public_series_compared": n_compared, "matches": n_match, "best_corr_level": best}


def classify(df):
    """levels_strict / levels_loose: level correlation and ratio stability only.
    match / match_loose: those, plus agreeing changes and a power-of-ten ratio."""
    df = df.copy()
    log_r = np.log10(df.ratio_median)
    df["pow10_dev"] = np.abs(10 ** (log_r - log_r.round()) - 1)
    same = (df.corr_diff > MATCH["corr_diff"]) & (df.pow10_dev < MATCH["pow10_dev"])
    df["levels_strict"] = (df.corr_level > STRICT["corr"]) & (df.ratio_maxdev < STRICT["maxdev"])
    df["levels_loose"] = (df.corr_level > LOOSE["corr"]) & (df.ratio_maxdev < LOOSE["maxdev"])
    df["match"] = df.levels_strict & same
    df["match_loose"] = df.levels_loose & same
    return df


def date_range(df):
    return [str(df.index[0].date()), str(df.index[-1].date()), int(len(df))]


# ------------------------------------------------------------------- checks

def check_tourism_small_vs_fev():
    ours, _ = local("TourismSmall")
    O = ours.to_numpy().T
    pub = read_parquet_series("fev_australian_tourism")
    matched = set()
    n_pub = 0
    for pid, ts, x, _ in pub:
        if len(x) != O.shape[1]:
            continue
        hit = np.flatnonzero(np.abs(O - x[None, :]).max(axis=1) == 0)
        n_pub += len(hit) > 0
        matched.update(ours.columns[hit])
    same_dates = bool((pd.DatetimeIndex(pub[0][1]) == ours.index).all())
    return {
        "ours": "TourismSmall", "public": "fev australian_tourism",
        "our_series": ours.shape[1], "public_series": len(pub),
        "public_series_with_identical_ours": int(n_pub),
        "our_series_with_identical_public": len(matched),
        "same_dates": same_dates, "our_range": date_range(ours),
        "public_range": [str(pd.Timestamp(pub[0][1][0]).date()), str(pd.Timestamp(pub[0][1][-1]).date()),
                         int(len(pub[0][1]))],
    }


def check_labour_vs_m4():
    ours, _ = local("Labour")
    pub = read_parquet_series("m4_monthly")
    cand, _ = aligned_candidates(ours, "M", pub, "m4_monthly", keep_corr=LOOSE["corr"],
                                 min_overlap=MIN_OVERLAP["M"])
    cand = classify(cand)
    cand = cand[cand.levels_loose].sort_values(["ours", "public_id"]).reset_index(drop=True)
    cand.to_csv(OUTPUT_DIR / "labour_m4_matches.csv", index=False, float_format="%.6f")
    strict = cand[cand.match]
    rejected = cand[cand.levels_strict & ~cand.match]
    other_series = rejected[rejected.pow10_dev >= MATCH["pow10_dev"]]   # ratio is not a change of units
    ambiguous = rejected[rejected.pow10_dev < MATCH["pow10_dev"]]       # units agree, changes do not

    # Which of our test-window months are inside the matched M4 series?
    from datasetsforecast.hierarchical import HierarchicalData
    Y_df, _, _ = HierarchicalData.load(str(ROOT / "data"), "Labour")
    loaded = pd.DatetimeIndex(sorted(pd.to_datetime(Y_df["ds"].unique())))
    h, k = 12, 5
    windows = [(loaded[-h * i], loaded[-h * i + h - 1]) for i in range(k, 0, -1)]
    test_months = loaded[-h * k:]

    def last_seen(row, holdout):
        end = pd.Period(row.public_to, "M") - holdout
        return end

    exposure = {}
    for label, holdout in [(f"last {M4_HOLDOUT} observations held out", M4_HOLDOUT), ("full series", 0)]:
        ends = strict.apply(lambda r: last_seen(r, holdout), axis=1)
        latest = strict.assign(end=ends).groupby("ours")["end"].max()
        in_test = {s: int((test_months.to_period("M") <= e).sum()) for s, e in latest.items()}
        touched = [w for w in windows if (latest.max() >= pd.Period(w[0], "M"))]
        exposure[label] = {
            "latest_value_in_corpus": str(latest.max()),
            "our_series_with_test_months_in_corpus": int(sum(v > 0 for v in in_test.values())),
            "max_test_months_in_corpus_per_series": int(max(in_test.values())),
            "test_windows_touched": [f"{a.date()} to {b.date()}" for a, b in touched],
        }

    # Question 2: do 5 yearly origins with h=12 fit entirely after M4_CUTOFF?
    cutoff = pd.Period(M4_CUTOFF, "M")
    after = loaded[loaded.to_period("M") > cutoff]
    raw_after = ours.index[ours.index.to_period("M") > cutoff]
    n_fit = len(after) // h
    first_needed = loaded[-h * k]
    fit = {
        "cutoff": M4_CUTOFF,
        "months_after_cutoff_as_loaded": int(len(after)),
        "months_after_cutoff_raw_file": int(len(raw_after)),
        "months_needed_for_5_origins_h12_spaced_12": h * k,
        "fits_as_loaded": bool(len(after) >= h * k),
        "fits_raw_file": bool(len(raw_after) >= h * k),
        "non_overlapping_windows_that_fit_as_loaded": int(n_fit),
        "windows_that_fit": [f"{loaded[-h * i].date()} to {loaded[-h * i + h - 1].date()}"
                             for i in range(n_fit, 0, -1)],
        "largest_spacing_that_fits_5_windows_as_loaded": int((len(after) - h) // (k - 1)),
        "context_months_at_first_registered_origin": int((loaded < first_needed).sum()),
        "context_months_at_first_origin_after_cutoff": int((loaded < after[0]).sum()) if len(after) else None,
        "min_context_required": 120,
    }
    return {
        "ours": "Labour", "public": "M4 Monthly (autogluon/chronos_datasets)",
        "our_series": ours.shape[1], "public_series": len(pub),
        "criteria": {"strict": STRICT, "loose": LOOSE, "match": MATCH,
                     "min_overlap_months": MIN_OVERLAP["M"]},
        "pairs_strict": int(len(strict)), "pairs_loose": int(cand.match_loose.sum()),
        "our_series_matched_strict": int(strict.ours.nunique()),
        "our_series_matched_loose": int(cand[cand.match_loose].ours.nunique()),
        "pairs_passing_levels_only": int(cand.levels_strict.sum()),
        "pairs_rejected": int(len(rejected)),
        "pairs_rejected_ratio_not_power_of_ten": int(len(other_series)),
        "pairs_rejected_changes_disagree": int(len(ambiguous)),
        "rejected_ratio_examples": [[r.ours, r.public_id, round(r.ratio_median, 3), round(r.corr_diff, 3)]
                                    for r in other_series.sort_values("corr_diff").head(4).itertuples()],
        "rejected_changes_examples": [[r.ours, r.public_id, round(r.ratio_median, 3), round(r.corr_diff, 3)]
                                      for r in ambiguous.itertuples()],
        "public_ids_matched_to_more_than_one_of_ours": int((strict.groupby("public_id").ours.nunique() > 1).sum()),
        "pairs_with_some_exactly_equal_values": int((strict.exact_equal_share > 0).sum()),
        "unmatched_strict": sorted(set(ours.columns) - set(strict.ours)),
        "overlap_months_strict": [int(strict.overlap.min()), int(strict.overlap.max())],
        "min_corr_of_monthly_changes_strict": float(strict.corr_diff.min()),
        "ratio_medians_strict": {str(k2): int(v) for k2, v in
                                 strict.ratio_median.round(2).value_counts().head(5).items()},
        "public_series_end_strict": {str(k2): int(v) for k2, v in strict.public_to.value_counts().items()},
        "public_categories_strict": {str(k2): int(v) for k2, v in strict.category.value_counts().items()},
        "raw_file_range": date_range(ours),
        "loaded_range": [str(loaded[0].date()), str(loaded[-1].date()), int(len(loaded))],
        "registered_test_windows": [f"{a.date()} to {b.date()}" for a, b in windows],
        "exposure_of_test_windows": exposure,
        "five_origins_after_cutoff": fit,
    }


def check_tourism():
    large, _ = local("TourismLarge")
    small, _ = local("TourismSmall")
    large_q = large.groupby(large.index.to_period("Q")).sum()
    large_q.index = large_q.index.to_timestamp(how="end").normalize()
    sets = {
        "M": [("TourismLarge", large)],
        "Q": [("TourismSmall", small), ("TourismLarge, quarterly sums", large_q)],
    }
    sources = {"M": ["monash_tourism_monthly", "m4_monthly"],
               "Q": ["monash_tourism_quarterly", "m4_quarterly"]}
    out, best_rows = [], []
    for freq in ["M", "Q"]:
        for src in sources[freq]:
            pub = read_parquet_series(src)
            for name, ours in sets[freq]:
                # series that are mostly zero correlate with anything that has one spike
                dense = ours.loc[:, (ours == 0).mean() <= 0.5]
                stats = {}
                cand, best = aligned_candidates(dense, freq, pub, src, keep_corr=LOOSE["corr"],
                                                min_overlap=MIN_OVERLAP_TOURISM[freq], stats=stats)
                cand = classify(cand) if len(cand) else cand
                sliding = (sliding_candidates(dense, pub, MIN_OVERLAP_TOURISM[freq])
                           if src.startswith("monash") else None)
                n_strict = int(cand.match.sum()) if len(cand) else 0
                n_loose = int(cand.match_loose.sum()) if len(cand) else 0
                n_levels = int(cand.levels_strict.sum()) if len(cand) else 0
                b = pd.Series({k2: v[0] for k2, v in best.items()})
                top = b.idxmax()
                out.append({
                    "ours": name, "public": src, "our_series_compared": int(dense.shape[1]),
                    "our_series_skipped_mostly_zero": int(ours.shape[1] - dense.shape[1]),
                    "public_series": len(pub),
                    "public_series_with_enough_date_overlap": int(stats.get("public_series_compared", 0)),
                    "min_overlap": MIN_OVERLAP_TOURISM[freq],
                    "pairs_strict": n_strict, "pairs_loose": n_loose,
                    "pairs_passing_levels_only": n_levels,
                    "best_corr_level": float(b.max()) if np.isfinite(b.max()) else None,
                    "best_pair": [top, best[top][1]],
                    "our_series_with_corr_above_0.99": int((b > 0.99).sum()),
                    "ignoring_dates": sliding,
                })
                for k2, v in best.items():
                    best_rows.append({"ours_dataset": name, "ours": k2, "source": src,
                                      "best_corr_level": v[0], "public_id": v[1]})
    pd.DataFrame(best_rows).to_csv(OUTPUT_DIR / "tourism_best.csv", index=False, float_format="%.6f")
    return out


def check_wiki2():
    ours, bottom = local("Wiki2")
    B = ours[bottom]
    year_start = pd.Timestamp("2016-01-01")
    result = {"ours": "Wiki2", "our_range": date_range(ours), "bottom_series": len(bottom),
              "registered_test_windows_h7": [f"{ours.index[-7 * i].date()} to {ours.index[-7 * i + 6].date()}"
                                             for i in range(5, 0, -1)]}
    match = pd.DataFrame({"wiki2": bottom}).set_index("wiki2")

    # --- daily: Extended Web Traffic, exact equality on every day of 2016
    first = None
    lookup = {}
    for ids, series in arrow_series("extended_web_traffic"):
        for pid, x in zip(ids, series):
            if first is None:
                first = (pid, len(x))
            lookup.setdefault(np.nan_to_num(x[184:184 + 366], nan=0.0).astype(np.float32).tobytes(), pid)
    n_items = len(lookup)
    start = pd.Timestamp("2015-07-01")
    assert (year_start - start).days == 184
    match["extended_item"] = [lookup.get(B[c].to_numpy(dtype=np.float32).tobytes()) for c in bottom]
    result["extended_web_traffic"] = {
        "public": "Extended Web Traffic (Salesforce/GiftEvalPretrain)",
        "public_start": str(start.date()), "public_length_days": int(first[1]),
        "distinct_2016_windows_in_public": int(n_items),
        "days_compared": 366,
        "bottom_series_equal_on_all_days": int(match.extended_item.notna().sum()),
        "distinct_public_items_matched": int(match.extended_item.nunique()),
    }

    # --- weekly: Kaggle Web Traffic Weekly, exact equality of weekly sums
    weekly = {}
    n_weekly_series = 0
    for ids, series in arrow_series("kaggle_web_traffic_weekly"):
        for pid, x in zip(ids, series):
            weekly[pid] = x
            n_weekly_series += 1
    by_offset = {}
    for offset in range(7):
        week0 = pd.Timestamp("2015-06-29") + pd.Timedelta(days=offset)
        n_weeks = len(next(iter(weekly.values())))
        starts = pd.date_range(week0, periods=n_weeks, freq="7D")
        keep = np.flatnonzero((starts >= ours.index[0]) & (starts + pd.Timedelta(days=6) <= ours.index[-1]))
        look = {}
        for pid, x in weekly.items():
            look.setdefault(np.nan_to_num(x[keep], nan=0.0).astype(np.float32).tobytes(), pid)
        hits = []
        for c in bottom:
            s = B[c]
            sums = np.array([s.loc[d:d + pd.Timedelta(days=6)].sum() for d in starts[keep]], dtype=np.float32)
            hits.append(look.get(sums.tobytes()))
        by_offset[offset] = (week0, len(keep), hits)
    best_offset = max(by_offset, key=lambda o: sum(hh is not None for hh in by_offset[o][2]))
    week0, n_weeks_cmp, hits = by_offset[best_offset]
    match["weekly_item"] = hits
    same_item = int((match.weekly_item == match.extended_item).sum())
    result["kaggle_web_traffic_weekly"] = {
        "public": "Kaggle Web Traffic Weekly (Salesforce/GiftEvalPretrain)",
        "public_series": int(n_weekly_series),
        "week_start_that_matches": f"{week0.date()} ({week0.day_name()})",
        "stated_start_in_file": "2015-06-29 (Monday)",
        "matches_by_week_start": {f"{by_offset[o][0].date()} ({by_offset[o][0].day_name()})":
                                  int(sum(hh is not None for hh in by_offset[o][2])) for o in by_offset},
        "full_weeks_of_2016_compared": int(n_weeks_cmp),
        "first_week_compared": str((pd.date_range(week0, periods=200, freq="7D")[
            pd.date_range(week0, periods=200, freq="7D") >= ours.index[0]][0]).date()),
        "bottom_series_with_equal_weekly_sums": int(match.weekly_item.notna().sum()),
        "same_item_id_as_daily_match": same_item,
    }

    # --- Wiki Daily (100k): near-matches by correlation of log counts over 2016
    P, names = [], []
    for p in fetch("wiki_daily_100k"):
        t = pq.read_table(p, columns=["timestamp", "target", "page_name"])
        ts0 = t.column("timestamp")
        tg = t.column("target")
        pn = t.column("page_name").to_pylist()
        for i in range(t.num_rows):
            d0 = pd.Timestamp(ts0[i].values[0].as_py())
            x = tg[i].values.to_numpy(zero_copy_only=False)
            lo = (year_start - d0).days
            if lo < 0 or lo + 366 > len(x):
                continue
            P.append(x[lo:lo + 366].astype(np.float32))
            names.append(pn[i])
    P = np.vstack(P)
    Pn = np.nan_to_num(P, nan=0.0)
    L = np.log1p(Pn)
    L = L - L.mean(axis=1, keepdims=True)
    norm = np.sqrt((L ** 2).sum(axis=1))
    norm[norm == 0] = np.inf
    L /= norm[:, None]
    rows = []
    for c in bottom:
        o = B[c].to_numpy()
        lo_ = np.log1p(o)
        lo_ = lo_ - lo_.mean()
        lo_ /= np.sqrt((lo_ ** 2).sum())
        corr = L @ lo_.astype(np.float32)
        j = int(np.argmax(corr))
        pos = (o > 0) & (Pn[j] > 0)
        ratio = Pn[j][pos] / o[pos]
        rows.append({"wiki2": c, "wiki100k_page": names[j], "wiki100k_log_corr": float(corr[j]),
                     "wiki100k_ratio_median": float(np.median(ratio)) if pos.any() else np.nan,
                     "wiki100k_days_equal": int((Pn[j] == o).sum())})
    w = pd.DataFrame(rows).set_index("wiki2")
    match = match.join(w)
    near = w[(w.wiki100k_log_corr > 0.999) & w.wiki100k_ratio_median.between(0.9, 1.1)]
    probable = w[(w.wiki100k_log_corr > 0.98) & ~w.index.isin(near.index)]
    result["wiki_daily_100k"] = {
        "public": "Wiki Daily (100k) (autogluon/chronos_datasets)",
        "public_series_covering_2016": int(len(names)),
        "criterion_near_identical": "log-count correlation > 0.999 and median ratio in [0.9, 1.1]",
        "criterion_probable": "log-count correlation > 0.98",
        "bottom_series_near_identical": int(len(near)),
        "bottom_series_probable": int(len(probable)),
        "bottom_series_equal_on_all_days": int((w.wiki100k_days_equal == 366).sum()),
        "near_identical": [[i, r.wiki100k_page, round(r.wiki100k_log_corr, 5),
                            round(r.wiki100k_ratio_median, 4)] for i, r in near.iterrows()],
        "probable": [[i, r.wiki100k_page, round(r.wiki100k_log_corr, 5),
                      round(r.wiki100k_ratio_median, 4)] for i, r in probable.iterrows()],
        "median_best_log_corr": float(w.wiki100k_log_corr.median()),
    }

    # --- Wiki-Rolling: dates in the file are not reliable, so search for exact 7-day runs
    win = 7
    mult = (np.uint64(1099511628211) ** np.arange(win, dtype=np.uint64))

    def hashes(x):
        v = np.lib.stride_tricks.sliding_window_view(x.astype(np.int64).astype(np.uint64), win)
        return (v * mult).sum(axis=1, dtype=np.uint64), v

    ours_hash = {}
    n_windows = 0
    for c in bottom:
        hv, v = hashes(B[c].to_numpy())
        informative = (v.max(axis=1) >= 100) & (np.array([len(set(r)) for r in v]) >= 5)
        n_windows += int(informative.sum())
        for i in np.flatnonzero(informative):
            ours_hash.setdefault(int(hv[i]), []).append((c, i))
    keys = np.sort(np.fromiter(ours_hash.keys(), dtype=np.uint64))
    n_pub, lengths, hits = 0, [], []
    for ids, series in arrow_series("wiki_rolling"):
        for pid, x in zip(ids, series):
            n_pub += 1
            lengths.append(len(x))
            x = np.nan_to_num(x, nan=0.0)
            if len(x) < win:
                continue
            hv, v = hashes(x)
            pos = np.minimum(np.searchsorted(keys, hv), len(keys) - 1)
            for i in np.flatnonzero(keys[pos] == hv):
                for c, k2 in ours_hash[int(hv[i])]:
                    if np.array_equal(v[i].astype(np.int64), B[c].to_numpy()[k2:k2 + win].astype(np.int64)):
                        hits.append((c, pid, int(k2), int(i)))
    result["wiki_rolling"] = {
        "public": "Wiki-Rolling (Salesforce/GiftEvalPretrain)",
        "public_series": int(n_pub), "public_length_days": [int(min(lengths)), int(max(lengths))],
        "method": f"exact equality of any run of {win} consecutive days; runs with a maximum below "
                  "100 or fewer than 5 distinct values are not used",
        "our_runs_searched": int(n_windows),
        "exact_runs_found": int(len(hits)),
        "bottom_series_with_a_run_found": int(len({hh[0] for hh in hits})),
    }

    match.reset_index().to_csv(OUTPUT_DIR / "wiki2_matches.csv", index=False, float_format="%.6f")
    return result


# ------------------------------------------------------------------ summary

def file_table():
    rows = []
    for name, (repo, files, _) in SOURCES.items():
        for f in files:
            p = DOWNLOAD_DIR / repo.replace("/", "__") / f
            rows.append({"source": name, "repo": repo, "revision": REVISIONS[repo], "file": f,
                         "bytes": p.stat().st_size, "sha256": sha256(p)})
    ours = []
    for ds in ["TourismSmall", "TourismLarge", "Labour", "Wiki2"]:
        for f in ["data.csv", "agg_mat.csv"]:
            p = LOCAL_DIR / ds / f
            ours.append({"dataset": ds, "file": f"data/hierarchical/{ds}/{f}", "bytes": p.stat().st_size,
                         "sha256": sha256(p)})
    return rows, ours


def md_table(rows, cols, headers=None):
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_markdown(s):
    t, lab, w = s["tourism_small_vs_fev"], s["labour_vs_m4"], s["wiki2"]
    ext, wk, w100, wr = (w["extended_web_traffic"], w["kaggle_web_traffic_weekly"],
                         w["wiki_daily_100k"], w["wiki_rolling"])
    fit = lab["five_origins_after_cutoff"]
    lines = [
        "# Leakage checks: summary",
        "",
        f"Generated by `scripts/leakage/run_checks.py` on {s['generated']}. Do not edit by hand.",
        "All numbers are also in `summary.json`. Full SHA-256 hashes are in `summary.json`; the tables "
        "here show the first 16 characters.",
        "",
        "## 1. Overlap counts",
        "",
        md_table([
            {"check": "TourismSmall vs fev-bench `australian_tourism`",
             "result": f"{t['our_series_with_identical_public']} of {t['our_series']} series identical",
             "period": f"{t['our_range'][0]} to {t['our_range'][1]} ({t['our_range'][2]} quarters)"},
            {"check": "Labour vs M4 Monthly, match",
             "result": f"{lab['our_series_matched_strict']} of {lab['our_series']} series "
                       f"({lab['pairs_strict']} pairs)",
             "period": f"overlap {lab['overlap_months_strict'][0]} to {lab['overlap_months_strict'][1]} months"},
            {"check": "Labour vs M4 Monthly, match with loose thresholds",
             "result": f"{lab['our_series_matched_loose']} of {lab['our_series']} series "
                       f"({lab['pairs_loose']} pairs)", "period": ""},
            {"check": "Wiki2 vs Extended Web Traffic (daily)",
             "result": f"{ext['bottom_series_equal_on_all_days']} of {w['bottom_series']} bottom series "
                       "equal on every day",
             "period": f"{w['our_range'][0]} to {w['our_range'][1]} ({ext['days_compared']} days)"},
            {"check": "Wiki2 vs Kaggle Web Traffic Weekly",
             "result": f"{wk['bottom_series_with_equal_weekly_sums']} of {w['bottom_series']} bottom series "
                       "with equal weekly sums",
             "period": f"{wk['full_weeks_of_2016_compared']} full weeks from {wk['first_week_compared']}"},
            {"check": "Wiki2 vs Wiki Daily (100k)",
             "result": f"{w100['bottom_series_near_identical']} near-identical, "
                       f"{w100['bottom_series_probable']} probable, "
                       f"{w100['bottom_series_equal_on_all_days']} equal on every day",
             "period": "2016, 366 days"},
            {"check": "Wiki2 vs Wiki-Rolling",
             "result": f"{wr['exact_runs_found']} exact 7-day runs found "
                       f"({wr['our_runs_searched']} runs searched)", "period": "dates not used"},
        ], ["check", "result", "period"], ["Check", "Result", "Period or scope"]),
        "",
        "Tourism. Series with more than half zeros are not compared.",
        "",
        md_table([
            {"ours": r["ours"], "public": r["public"], "n": r["our_series_compared"],
             "pub": f"{r['public_series_with_enough_date_overlap']} of {r['public_series']}",
             "m": r["pairs_strict"], "ml": r["pairs_loose"],
             "best": "n/a" if r["best_corr_level"] is None else f"{r['best_corr_level']:.3f}",
             "slide": "not run" if r["ignoring_dates"] is None else
                      f"{r['ignoring_dates']['matches']} matches, best correlation "
                      f"{r['ignoring_dates']['best_corr_level']:.3f}"}
            for r in s["tourism"]
        ], ["ours", "public", "n", "pub", "m", "ml", "best", "slide"],
            ["Ours", "Public copy", "Our series compared", "Public series with enough date overlap",
             "Matches", "Matches, loose", "Best level correlation, dates aligned",
             "Ignoring dates (every shift)"]),
        "",
        "A pair is a match (the same series in another data vintage) when, on the date-aligned "
        "overlap, all four hold:",
        "",
        f"- correlation of levels > {STRICT['corr']} (loose: {LOOSE['corr']})",
        f"- every ratio within {STRICT['maxdev']:.0%} of the median ratio (loose: {LOOSE['maxdev']:.0%})",
        f"- correlation of period-to-period changes > {MATCH['corr_diff']}",
        f"- the median ratio is within {MATCH['pow10_dev']:.0%} of a power of ten (a change of units)",
        "",
        f"Minimum overlap: {MIN_OVERLAP['M']} months for Labour; {MIN_OVERLAP_TOURISM['M']} months or "
        f"{MIN_OVERLAP_TOURISM['Q']} quarters for tourism.",
        "",
        f"The first two conditions alone are not enough. {lab['pairs_rejected']} Labour pairs pass them "
        "and are not counted as matches:",
        "",
        f"- Ratio not a power of ten: {lab['pairs_rejected_ratio_not_power_of_ten']} pairs. These are "
        "different series that move together, for example "
        f"{lab['rejected_ratio_examples'][0][0]} against M4 {lab['rejected_ratio_examples'][0][1]} "
        f"(ratio {lab['rejected_ratio_examples'][0][2]}).",
        f"- Power-of-ten ratio, but monthly changes agree less well: "
        f"{lab['pairs_rejected_changes_disagree']} pairs. These may be the same series in a more "
        "heavily revised vintage: "
        + "; ".join(f"{a} against M4 {b} (ratio {c}, change correlation {d})"
                    for a, b, c, d in lab["rejected_changes_examples"]) + ".",
        "",
        "## 2. Labour and M4 Monthly",
        "",
        md_table([
            {"k": "Labour, raw file", "v": f"{lab['raw_file_range'][0]} to {lab['raw_file_range'][1]} "
                                           f"({lab['raw_file_range'][2]} months)"},
            {"k": "Labour, as loaded (the loader drops 2020 onwards)",
             "v": f"{lab['loaded_range'][0]} to {lab['loaded_range'][1]} ({lab['loaded_range'][2]} months)"},
            {"k": "Registered test windows (5 origins, h = 12)", "v": "; ".join(lab["registered_test_windows"])},
            {"k": "Last month of the matched M4 series (pairs)",
             "v": ", ".join(f"{k2}: {v}" for k2, v in sorted(lab["public_series_end_strict"].items()))},
            {"k": "Median ratio M4 / Labour (pairs)",
             "v": ", ".join(f"{k2}: {v}" for k2, v in lab["ratio_medians_strict"].items())},
            {"k": "M4 categories (pairs)",
             "v": ", ".join(f"{k2}: {v}" for k2, v in lab["public_categories_strict"].items())},
            {"k": "Lowest correlation of monthly changes among matches",
             "v": f"{lab['min_corr_of_monthly_changes_strict']:.3f}"},
            {"k": "M4 series matched to more than one Labour series",
             "v": lab["public_ids_matched_to_more_than_one_of_ours"]},
            {"k": "Matched pairs with some exactly equal values",
             "v": f"{lab['pairs_with_some_exactly_equal_values']} of {lab['pairs_strict']}"},
            {"k": "Series without a match", "v": "; ".join(lab["unmatched_strict"])},
        ], ["k", "v"], ["Item", "Value"]),
        "",
        "Test-window months that are inside the matched M4 series:",
        "",
        md_table([
            {"scenario": k2, "latest": v["latest_value_in_corpus"],
             "n": f"{v['our_series_with_test_months_in_corpus']} of {lab['our_series']}",
             "months": v["max_test_months_in_corpus_per_series"],
             "windows": "; ".join(v["test_windows_touched"]) or "none"}
            for k2, v in lab["exposure_of_test_windows"].items()
        ], ["scenario", "latest", "n", "months", "windows"],
            ["Model trained on", "Latest Labour month in corpus", "Series with test months in corpus",
             "Most test months for one series", "Test windows touched"]),
        "",
        f"### Do 5 yearly origins with h = 12 fit entirely after {fit['cutoff']}?",
        "",
        md_table([
            {"k": "Months needed (5 windows of 12, spaced 12 apart)",
             "v": fit["months_needed_for_5_origins_h12_spaced_12"]},
            {"k": f"Months after {fit['cutoff']}, as loaded", "v": fit["months_after_cutoff_as_loaded"]},
            {"k": f"Months after {fit['cutoff']}, raw file", "v": fit["months_after_cutoff_raw_file"]},
            {"k": "Fits, as loaded", "v": "yes" if fit["fits_as_loaded"] else "no"},
            {"k": "Fits, raw file", "v": "yes" if fit["fits_raw_file"] else "no"},
            {"k": "Non-overlapping 12-month windows that fit, as loaded",
             "v": f"{fit['non_overlapping_windows_that_fit_as_loaded']}: " + "; ".join(fit["windows_that_fit"])},
            {"k": "Largest spacing at which 5 windows fit, as loaded",
             "v": f"{fit['largest_spacing_that_fits_5_windows_as_loaded']} months (windows overlap)"},
            {"k": "Context at the first window after the cutoff",
             "v": f"{fit['context_months_at_first_origin_after_cutoff']} months "
                  f"(required: {fit['min_context_required']})"},
        ], ["k", "v"], ["Item", "Value"]),
        "",
        "## 3. Wiki2",
        "",
        md_table([
            {"k": "Wiki2 range", "v": f"{w['our_range'][0]} to {w['our_range'][1]} ({w['our_range'][2]} days)"},
            {"k": "Registered test windows (5 origins, h = 7)", "v": "; ".join(w["registered_test_windows_h7"])},
            {"k": "Extended Web Traffic", "v": f"starts {ext['public_start']}, {ext['public_length_days']} days"},
            {"k": "Weekly copy: week start stated in the file", "v": wk["stated_start_in_file"]},
            {"k": "Weekly copy: week start at which the sums match", "v": wk["week_start_that_matches"]},
            {"k": "Weekly matches by assumed week start",
             "v": ", ".join(f"{k2}: {v}" for k2, v in wk["matches_by_week_start"].items())},
            {"k": "Weekly and daily matches point to the same item", "v": wk["same_item_id_as_daily_match"]},
            {"k": "Wiki Daily (100k), near-identical",
             "v": "; ".join(f"{a} = page {b} (log-corr {c}, ratio {d})" for a, b, c, d in w100["near_identical"])
                  or "none"},
            {"k": "Wiki Daily (100k), probable",
             "v": "; ".join(f"{a} = page {b} (log-corr {c}, ratio {d})" for a, b, c, d in w100["probable"])
                  or "none"},
            {"k": "Wiki Daily (100k), median best log-correlation", "v": f"{w100['median_best_log_corr']:.3f}"},
            {"k": "Wiki-Rolling", "v": f"{wr['public_series']} series, {wr['public_length_days'][0]} to "
                                       f"{wr['public_length_days'][1]} days; {wr['method']}"},
        ], ["k", "v"], ["Item", "Value"]),
        "",
        "## 4. Files and hashes",
        "",
        "Public copies, downloaded from Hugging Face at the pinned revision:",
        "",
        md_table([{**r, "sha": r["sha256"][:16], "rev": r["revision"][:10],
                   "mb": f"{r['bytes'] / 1e6:.1f}"} for r in s["public_files"]],
                 ["repo", "rev", "file", "mb", "sha"], ["Repository", "Revision", "File", "MB", "SHA-256"]),
        "",
        "Our evaluation files:",
        "",
        md_table([{**r, "sha": r["sha256"][:16], "kb": f"{r['bytes'] / 1e3:.0f}"} for r in s["our_files"]],
                 ["file", "kb", "sha"], ["File", "kB", "SHA-256"]),
        "",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    summary = {"generated": time.strftime("%Y-%m-%d"), "revisions": REVISIONS}
    for name in SOURCES:
        fetch(name)
    steps = [
        ("tourism_small_vs_fev", check_tourism_small_vs_fev),
        ("labour_vs_m4", check_labour_vs_m4),
        ("wiki2", check_wiki2),
        ("tourism", check_tourism),
    ]
    for key, fn in steps:
        t1 = time.perf_counter()
        summary[key] = fn()
        print(f"{key}: {time.perf_counter() - t1:.1f}s", flush=True)
    summary["public_files"], summary["our_files"] = file_table()
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_markdown(summary)
    print(f"done in {time.perf_counter() - t0:.1f}s; wrote {OUTPUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
