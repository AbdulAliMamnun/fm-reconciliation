"""Model registry and the code that runs each foundation model.

This module must import in every conda env that runs a model, so it depends
only on numpy and pandas. Model libraries are imported inside the functions.

Every model is pinned to a Hugging Face revision. `weights_sha256` and
`weights_date` are the hash of the weight file at that revision and the date it
was last changed. Each family has a loader and a predictor. A predictor returns either
    {"levels": [...], "quantiles": (n_series, h, n_levels)}   quantile models
    {"samples": (n_series, h, n_samples)}                     sample-path models
and `standardise` turns both into the same columns.

Point arms (PREREG Deviations log, mean-forecast arm):
    quantile models   median = the 0.5 quantile; mean = average of the native levels
    sample models     median = sample median;    mean = sample mean
"""
import sys
import time

import numpy as np
import pandas as pd

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILE_COLS = [f"q{int(round(q * 100))}" for q in QUANTILE_LEVELS]
COLUMNS = ["unique_id", "ds", "yhat", "mean", *QUANTILE_COLS]
NUM_SAMPLES = 100
SEED = 20260927

MODELS = {
    "chronos_bolt_tiny": {
        "family": "chronos_bolt", "repo": "amazon/chronos-bolt-tiny",
        "revision": "a0e552de83495b5c28c14c71c374f3e33280b340", "env": "fmrec", "output": "quantiles",
        "weights_file": "model.safetensors", "weights_date": "2024-11-13",
        "weights_sha256": "75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32",
    },
    "chronos_bolt_mini": {
        "family": "chronos_bolt", "repo": "amazon/chronos-bolt-mini",
        "revision": "251268337516a88e253628c43e1d26ec577b376b", "env": "fmrec", "output": "quantiles",
        "weights_file": "model.safetensors", "weights_date": "2024-11-13",
        "weights_sha256": "1a1a4297f132b808c5c7da24e3cce549d519c01ff5c2661fcad404baba018f24",
    },
    "chronos_bolt_small": {
        "family": "chronos_bolt", "repo": "amazon/chronos-bolt-small",
        "revision": "772f3d25d38aec6d914c8949dab4462e2d46f5d8", "env": "fmrec", "output": "quantiles",
        "weights_file": "model.safetensors", "weights_date": "2024-11-13",
        "weights_sha256": "06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd",
    },
    "chronos_t5_small": {
        "family": "chronos_t5", "repo": "amazon/chronos-t5-small",
        "revision": "a971ba21945c4f1796b17a91fe69214b5f4ad472", "env": "fmrec", "output": "samples",
        "weights_file": "model.safetensors", "weights_date": "2024-03-27",
        "weights_sha256": "9c8b6fde5300f72b01c173153bf9288fa0a200614275bf0585071ad71a6a3d43",
    },
    "chronos_2": {
        "family": "chronos_2", "repo": "amazon/chronos-2",
        "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c", "env": "fmrec", "output": "quantiles",
        "weights_file": "model.safetensors", "weights_date": "2025-10-30",
        "weights_sha256": "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42",
    },
    # The original checkpoint, not NX-AI/TiRex-1.1-gifteval (docs/leakage.md, section 2.4).
    "tirex": {
        "family": "tirex", "repo": "NX-AI/TiRex",
        "revision": "63c740922493f5fbe60b277609ec62babfba2762", "env": "fmrec", "output": "quantiles",
        "weights_file": "model.ckpt", "weights_date": "2025-05-26",
        "weights_sha256": "b8c3f5a036c63272ce4b91c00187e26922a394cb6cb49d4e16db070ad0422314",
    },
    "moirai_2_small": {
        "family": "moirai_2", "repo": "Salesforce/moirai-2.0-R-small",
        "revision": "30f43ff08c8494f4943ae1521e9d4e94a0fbb389", "env": "fmrec-moirai",
        "output": "quantiles",
        "weights_file": "model.safetensors", "weights_date": "2025-08-06",
        "weights_sha256": "fb5652a3db8ea572606221b7cb1e77bb8962b168e4d4cc752cf31ceb04074669",
    },
}

_LOADED = {}


def quantile_col(level):
    """Column name for a quantile level: 0.1 -> q10, 0.05 -> q5, 0.99 -> q99."""
    return f"q{level * 100:g}"


def check_openmp_clash():
    """hierarchicalforecast's compiled extension and torch each bundle their own
    OpenMP runtime. On macOS, loading both in one process hangs or segfaults at
    the first model call, so fail with a message instead."""
    if sys.platform == "darwin" and "hierarchicalforecast._lib" in sys.modules:
        raise RuntimeError(
            "hierarchicalforecast is loaded in this process. It cannot share a process with "
            "torch on macOS (two OpenMP runtimes). Run foundation-model forecasts in a process "
            "that does not import hierarchicalforecast."
        )


# ------------------------------------------------------------------ families

def _load_chronos(spec, device):
    check_openmp_clash()
    import torch
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(
        spec["repo"], revision=spec["revision"], device_map=device, torch_dtype=torch.float32
    )


def _predict_chronos_bolt(pipeline, contexts, h, seed):
    import torch

    out = []
    for i in range(0, len(contexts), 256):
        batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[i: i + 256]]
        q, _ = pipeline.predict_quantiles(batch, prediction_length=h, quantile_levels=QUANTILE_LEVELS)
        out.append(q.cpu().numpy())
    return {"levels": QUANTILE_LEVELS, "quantiles": np.concatenate(out).astype(np.float64)}


def _predict_chronos_t5(pipeline, contexts, h, seed):
    """Sample paths. The seed is set once per call, so a call is reproducible
    and does not depend on what ran before it."""
    import torch

    torch.manual_seed(seed)
    out = []
    # 8 series x 100 sample paths per batch. Larger batches run out of memory on
    # this machine and become many times slower.
    for i in range(0, len(contexts), 8):
        batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[i: i + 8]]
        s = pipeline.predict(batch, prediction_length=h, num_samples=NUM_SAMPLES)   # (b, S, h)
        out.append(s.cpu().numpy())
    samples = np.concatenate(out).astype(np.float64)
    return {"samples": np.swapaxes(samples, 1, 2)}                                 # (n, h, S)


def _predict_chronos_2(pipeline, contexts, h, seed):
    """Univariate mode: every series is its own task and cross_learning is off,
    so no information is shared between series."""
    levels = [float(q) for q in pipeline.quantiles]
    q, _ = pipeline.predict_quantiles(
        [np.asarray(c, dtype=np.float32) for c in contexts], prediction_length=h,
        quantile_levels=levels, cross_learning=False,
    )
    arr = np.stack([x.cpu().numpy()[0] for x in q])                                 # (n, h, L)
    return {"levels": levels, "quantiles": arr.astype(np.float64)}


def _load_tirex(spec, device):
    check_openmp_clash()
    from tirex import load_model

    return load_model(spec["repo"], device=device, backend="torch",
                      hf_kwargs={"revision": spec["revision"]})


def _predict_tirex(model, contexts, h, seed):
    q, _ = model.forecast(context=[np.asarray(c, dtype=np.float32) for c in contexts],
                          prediction_length=h, output_type="numpy")
    levels = [float(x) for x in model.config.quantiles] if hasattr(model, "config") else QUANTILE_LEVELS
    return {"levels": levels, "quantiles": np.asarray(q, dtype=np.float64)}


def _load_moirai_2(spec, device):
    check_openmp_clash()
    from uni2ts.model.moirai2 import Moirai2Module

    module = Moirai2Module.from_pretrained(spec["repo"], revision=spec["revision"])
    return module.to(device)


def _predict_moirai_2(module, contexts, h, seed):
    """uni2ts has no default context length: it is a required argument. It is
    set to the length of the longest history given, so nothing is cut off.
    Shorter series are left-padded by the library and flagged as padding."""
    from uni2ts.model.moirai2 import Moirai2Forecast

    device = next(module.parameters()).device
    model = Moirai2Forecast(
        module=module, prediction_length=h, context_length=max(len(c) for c in contexts),
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    ).to(device)
    out = []
    for i in range(0, len(contexts), 256):
        q = model.predict([np.asarray(c, dtype=np.float32) for c in contexts[i: i + 256]])
        out.append(np.asarray(q))                                                   # (b, L, h)
    levels = [float(x) for x in module.quantile_levels]
    return {"levels": levels, "quantiles": np.swapaxes(np.concatenate(out), 1, 2).astype(np.float64)}


FAMILIES = {
    "chronos_bolt": (_load_chronos, _predict_chronos_bolt),
    "chronos_t5": (_load_chronos, _predict_chronos_t5),
    "chronos_2": (_load_chronos, _predict_chronos_2),
    "tirex": (_load_tirex, _predict_tirex),
    "moirai_2": (_load_moirai_2, _predict_moirai_2),
}


# ------------------------------------------------------------ common output

def standardise(raw):
    """Arrays with the same meaning for every model.

    Returns a dict with median, mean, q (n_series, h, 9 levels 0.1..0.9),
    `native` (the native quantiles, or None for sample models) and their levels.
    """
    if "samples" in raw:
        s = np.asarray(raw["samples"], dtype=np.float64)
        q = np.moveaxis(np.quantile(s, QUANTILE_LEVELS, axis=-1), 0, -1)
        return {"median": np.median(s, axis=-1), "mean": s.mean(axis=-1), "q": q,
                "var": s.var(axis=-1, ddof=1), "levels": None, "native": None, "samples": s}
    levels = [round(float(x), 6) for x in raw["levels"]]
    native = np.asarray(raw["quantiles"], dtype=np.float64)
    if native.shape[-1] != len(levels):
        raise RuntimeError(f"{native.shape[-1]} quantiles for {len(levels)} levels.")
    missing = [x for x in QUANTILE_LEVELS if x not in levels]
    if missing:
        raise RuntimeError(f"The model does not output the quantile levels {missing}.")
    q = native[..., [levels.index(x) for x in QUANTILE_LEVELS]]
    return {"median": native[..., levels.index(0.5)], "mean": native.mean(axis=-1), "q": q,
            "levels": levels, "native": native, "samples": None}


def to_frame(ids, future, std):
    """Long frame: unique_id, ds, yhat (median), mean, q10..q90, then for sample
    models sample_var (the sample variance, ddof 1) and for quantile models any
    other native quantile levels (for example q1, q5, q95, q99)."""
    h = len(future)
    out = pd.DataFrame({
        "unique_id": np.repeat(np.asarray(ids, dtype=object), h),
        "ds": np.tile(np.asarray(future), len(ids)),
        "yhat": std["median"].reshape(-1),
        "mean": std["mean"].reshape(-1),
    })
    for j, col in enumerate(QUANTILE_COLS):
        out[col] = std["q"][:, :, j].reshape(-1)
    if std.get("var") is not None:
        out["sample_var"] = std["var"].reshape(-1)       # the predictive variance of PREREG §5
    if std["native"] is not None:
        for j, level in enumerate(std["levels"]):
            if quantile_col(level) not in out.columns:
                out[quantile_col(level)] = std["native"][:, :, j].reshape(-1)
    return out


def samples_frame(ids, future, samples):
    """Sample paths as a wide frame: unique_id, ds, s0..s{S-1}."""
    h, S = len(future), samples.shape[-1]
    out = pd.DataFrame({"unique_id": np.repeat(np.asarray(ids, dtype=object), h),
                        "ds": np.tile(np.asarray(future), len(ids))})
    flat = samples.reshape(-1, S).astype(np.float32)
    return pd.concat([out, pd.DataFrame(flat, columns=[f"s{i}" for i in range(S)])], axis=1)


# -------------------------------------------------------------------- running

def call_seed(cutoff):
    """One fixed seed per forecast call, derived from the cutoff date."""
    return SEED + pd.Timestamp(cutoff).toordinal()


def run_cutoffs(model, y_df, cutoffs, h, freq, device="cpu", keep_samples=False):
    """Forecast h steps from each cutoff, using only the data up to that cutoff.

    y_df: long frame with unique_id, ds, y.
    Returns (forecasts, info): forecasts has a `cutoff` column followed by the
    common columns; info holds the timing and, if asked, the sample paths.
    """
    spec = MODELS[model]
    load, predict = FAMILIES[spec["family"]]

    key = (model, device)
    if key not in _LOADED:
        t0 = time.perf_counter()
        handle = load(spec, device)
        _LOADED[key] = (handle, time.perf_counter() - t0)
    handle, load_seconds = _LOADED[key]        # the load time of the first load

    ids = list(dict.fromkeys(y_df["unique_id"]))      # the order in which the series are given
    y_df = y_df.sort_values(["unique_id", "ds"], kind="stable")
    frames, seconds, samples, levels = [], [], {}, None
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        ctx = y_df[y_df["ds"] <= cutoff]
        groups = ctx.groupby("unique_id", sort=False)["y"]
        contexts = [groups.get_group(u).to_numpy(dtype=np.float64) for u in ids]
        future = pd.date_range(cutoff, periods=h + 1, freq=freq)[1:]

        t0 = time.perf_counter()
        raw = predict(handle, contexts, h, call_seed(cutoff))
        seconds.append(time.perf_counter() - t0)

        std = standardise(raw)
        if std["q"].shape != (len(ids), h, len(QUANTILE_LEVELS)):
            raise RuntimeError(f"Unexpected forecast shape {std['q'].shape}.")
        levels = std["levels"]
        f = to_frame(ids, future, std)
        f.insert(0, "cutoff", cutoff)
        frames.append(f)
        if keep_samples and std["samples"] is not None:
            samples[cutoff] = samples_frame(ids, future, std["samples"])

    info = {
        "model": model, "repo": spec["repo"], "revision": spec["revision"], "device": device,
        "output": spec["output"], "native_levels": levels,
        "num_samples": NUM_SAMPLES if spec["output"] == "samples" else None,
        "load_seconds": load_seconds, "seconds": seconds, "samples": samples,
    }
    return pd.concat(frames, ignore_index=True), info
