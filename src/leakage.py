"""Leakage status of a (dataset, model) cell, read from docs/leakage.md.

The status table in docs/leakage.md is the single source. It sits between the
markers `<!-- status-table:start -->` and `<!-- status-table:end -->`.

    status(dataset, model)        is the dataset in the model's pretraining corpus?
    test_windows(dataset, model)  are the dataset's test windows in that corpus?

Classical models have no pretraining corpus, so both return "no" for them.
"""
import re
from functools import lru_cache
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "leakage.md"
START, END = "<!-- status-table:start -->", "<!-- status-table:end -->"
STATUS_VALUES = ("yes", "unclear", "no")
WINDOW_VALUES = ("yes", "partial", "unclear", "no")

CLASSICAL = ("autoets", "ets", "theta", "autotheta", "seasonalnaive", "naive")
# model family in the table -> prefix of the normalised model name
FAMILIES = {
    "Chronos-Bolt": "chronosbolt",
    "Chronos-T5": "chronost5",
    "Chronos-2": "chronos2",
    "TiRex": "tirex",
    "Moirai-2": "moirai2",
}


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def family(model):
    """Model family for a model name, e.g. chronos_bolt_small -> Chronos-Bolt.
    Returns None for classical models."""
    key = _norm(model)
    if key in CLASSICAL:
        return None
    for fam, prefix in FAMILIES.items():
        if key.startswith(prefix):
            return fam
    raise KeyError(f"Unknown model {model!r}. Known families: {sorted(FAMILIES)}")


@lru_cache(maxsize=None)
def load_table(path=DOC):
    """The status table as {(dataset, family): {"status", "test_windows", "basis"}}."""
    text = Path(path).read_text()
    if START not in text or END not in text:
        raise ValueError(f"Status table markers not found in {path}.")
    block = text.split(START, 1)[1].split(END, 1)[0]
    rows = [[c.strip() for c in line.strip().strip("|").split("|")]
            for line in block.splitlines() if line.strip().startswith("|")]
    header, body = [h.lower() for h in rows[0]], rows[2:]
    expected = ["dataset", "model", "status", "test windows", "basis"]
    if header != expected:
        raise ValueError(f"Status table columns must be {expected}, got {header}.")
    table = {}
    for dataset, model, status, windows, basis in body:
        if status not in STATUS_VALUES:
            raise ValueError(f"{dataset} / {model}: status {status!r} not in {STATUS_VALUES}.")
        if windows not in WINDOW_VALUES:
            raise ValueError(f"{dataset} / {model}: test windows {windows!r} not in {WINDOW_VALUES}.")
        if model not in FAMILIES:
            raise ValueError(f"{dataset} / {model}: unknown model family.")
        if (dataset, model) in table:
            raise ValueError(f"{dataset} / {model}: listed twice.")
        table[(dataset, model)] = {"status": status, "test_windows": windows, "basis": basis}
    return table


def _lookup(dataset, model, field):
    fam = family(model)
    table = load_table()
    datasets = {d for d, _ in table}
    if dataset not in datasets:
        raise KeyError(f"Unknown dataset {dataset!r}. Known: {sorted(datasets)}")
    if fam is None:
        return "no"
    return table[(dataset, fam)][field]


def status(dataset, model):
    """"yes", "unclear" or "no": is any of the dataset in the model's documented
    pretraining corpus?"""
    return _lookup(dataset, model, "status")


def test_windows(dataset, model):
    """"yes", "partial", "unclear" or "no": are the dataset's registered test
    windows in the model's pretraining corpus?"""
    return _lookup(dataset, model, "test_windows")


test_windows.__test__ = False  # not a pytest test
