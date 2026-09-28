import numpy as np
import pytest

from src import runners


@pytest.fixture
def register_model(monkeypatch):
    """Register a fake model that runs in-process.

    quantiles(contexts, h) -> (n_series, h, n_levels), or
    samples(contexts, h, seed) -> (n_series, h, n_samples).
    """
    def register(name, quantiles=None, samples=None, levels=runners.QUANTILE_LEVELS):
        if quantiles is not None:
            def predict(handle, contexts, h, seed):
                return {"levels": list(levels), "quantiles": np.asarray(quantiles(contexts, h))}
            output = "quantiles"
        else:
            def predict(handle, contexts, h, seed):
                return {"samples": np.asarray(samples(contexts, h, seed))}
            output = "samples"
        monkeypatch.setitem(runners.FAMILIES, name, (lambda spec, device: None, predict))
        monkeypatch.setitem(runners.MODELS, name, {
            "family": name, "repo": f"none/{name}", "revision": "none", "env": None, "output": output,
        })
        monkeypatch.setattr(runners, "_LOADED", {})

    return register
