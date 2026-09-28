import pytest

from src import leakage
from src.leakage import family, load_table, status, test_windows

DATASETS = ["TourismLarge", "TourismSmall", "Labour", "Wiki2", "M5"]
FAMILIES = ["Chronos-Bolt", "Chronos-T5", "Chronos-2", "TiRex", "Moirai-2"]


def test_table_covers_every_dataset_and_model():
    table = load_table()
    assert set(table) == {(d, m) for d in DATASETS for m in FAMILIES}
    for cell in table.values():
        assert cell["status"] in ("yes", "unclear", "no")
        assert cell["test_windows"] in ("yes", "partial", "unclear", "no")
        assert cell["basis"]


def test_table_is_internally_consistent():
    # Test windows cannot be in a corpus that does not contain the dataset,
    # and an unknown corpus cannot have a known answer for the windows.
    for (dataset, model), cell in load_table().items():
        if cell["status"] == "no":
            assert cell["test_windows"] == "no", (dataset, model)
        if cell["status"] == "unclear":
            assert cell["test_windows"] == "unclear", (dataset, model)


@pytest.mark.parametrize("name, expected", [
    ("chronos_bolt_small", "Chronos-Bolt"),
    ("chronos_bolt_tiny", "Chronos-Bolt"),
    ("Chronos-Bolt", "Chronos-Bolt"),
    ("chronos_t5_small", "Chronos-T5"),
    ("chronos_2", "Chronos-2"),
    ("Chronos-2", "Chronos-2"),
    ("tirex", "TiRex"),
    ("moirai_2_small", "Moirai-2"),
    ("Moirai-2", "Moirai-2"),
    ("AutoETS", None),
    ("Theta", None),
    ("SeasonalNaive", None),
])
def test_family(name, expected):
    assert family(name) == expected


def test_status_reads_the_documented_cells():
    assert status("TourismSmall", "chronos_bolt_small") == "no"
    assert status("TourismLarge", "chronos_bolt_small") == "unclear"
    assert status("TourismLarge", "chronos_t5_small") == "no"
    assert status("Labour", "chronos_t5_small") == "yes"
    assert status("Labour", "chronos_bolt_small") == "unclear"
    assert status("Wiki2", "tirex") == "yes"
    assert status("Wiki2", "moirai_2_small") == "unclear"
    assert status("M5", "chronos_2") == "no"
    assert status("M5", "moirai_2_small") == "unclear"


def test_test_windows_reads_the_documented_cells():
    assert test_windows("Labour", "chronos_t5_small") == "partial"
    assert test_windows("Wiki2", "tirex") == "yes"
    assert test_windows("TourismLarge", "chronos_2") == "no"
    assert test_windows("TourismLarge", "chronos_bolt_small") == "unclear"


def test_classical_models_have_no_pretraining_corpus():
    for dataset in DATASETS:
        assert status(dataset, "AutoETS") == "no"
        assert test_windows(dataset, "Theta") == "no"


def test_unknown_names_raise():
    with pytest.raises(KeyError, match="Unknown model"):
        status("Labour", "timesfm")
    with pytest.raises(KeyError, match="Unknown dataset"):
        status("Traffic", "tirex")
    with pytest.raises(KeyError, match="Unknown dataset"):
        status("Traffic", "AutoETS")


def _doc(tmp_path, rows, header="| Dataset | Model | Status | Test windows | Basis |"):
    p = tmp_path / "leakage.md"
    p.write_text("\n".join(["# x", leakage.START, header, "|---|---|---|---|---|", *rows, leakage.END, ""]))
    return p


def test_parser_reads_a_table(tmp_path):
    p = _doc(tmp_path, ["| Labour | TiRex | yes | unclear | some basis |"])
    assert load_table(p) == {("Labour", "TiRex"): {"status": "yes", "test_windows": "unclear",
                                                   "basis": "some basis"}}


@pytest.mark.parametrize("row", [
    "| Labour | TiRex | probably | no | x |",        # status not in the vocabulary
    "| Labour | TiRex | yes | maybe | x |",          # test windows not in the vocabulary
    "| Labour | TimesFM | yes | no | x |",           # unknown family
])
def test_parser_rejects_bad_cells(tmp_path, row):
    with pytest.raises(ValueError):
        load_table(_doc(tmp_path, [row]))


def test_parser_rejects_duplicates_and_missing_markers(tmp_path):
    row = "| Labour | TiRex | yes | no | x |"
    with pytest.raises(ValueError, match="twice"):
        load_table(_doc(tmp_path, [row, row]))
    p = tmp_path / "plain.md"
    p.write_text("# no table here\n")
    with pytest.raises(ValueError, match="markers"):
        load_table(p)
