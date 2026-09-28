# Leakage check: do the foundation models' pretraining corpora contain our evaluation data?

Checked 2026-09-27. The check was started before any foundation-model forecast was produced and finished after the Chronos-Bolt forecasts on TourismLarge had been scored. No forecast had been produced on Labour, Wiki2 or M5.

**Question.** For each zero-shot model in PREREG §4, does its documented pretraining corpus include any of our evaluation datasets (PREREG §3)?

**Verdicts** use three words. "yes": documented as trained on, or the data was found inside a documented training dataset. "no": the full corpus list is published and the dataset is absent, or the dataset is documented as held out. "unclear": the documentation does not settle it.

**Evidence labels.**

| Label | Meaning |
|---|---|
| [P] | Read in a primary source: paper, model card, dataset card, repository listing. Quotes are verbatim. |
| [S] | Secondary source: written by someone other than the model's authors. |
| [E] | Empirical: our data compared value by value with a public copy of a training dataset. Not documentation. |

## 1. Summary

This table is the single source for the leakage status. `src/leakage.py` reads it: `status(dataset, model)` returns the Status column and `test_windows(dataset, model)` the Test windows column. Keep the markers and the column order.

| Column | Question | Values |
|---|---|---|
| Status | Is any of the dataset in the model's documented pretraining corpus? | yes, unclear, no |
| Test windows | Are the dataset's registered test windows (PREREG §3) in that corpus? | yes, partial, unclear, no |

<!-- status-table:start -->
| Dataset | Model | Status | Test windows | Basis |
|---|---|---|---|---|
| TourismLarge | Chronos-Bolt | unclear | unclear | No dataset list is published. |
| TourismLarge | Chronos-T5 | no | no | Full list published, absent. No match in M4 or Monash Tourism. |
| TourismLarge | Chronos-2 | no | no | Full list published, absent. No match in M4. |
| TourismLarge | TiRex | no | no | Both lists published, absent. No match in M4. |
| TourismLarge | Moirai-2 | no | no | Absent from every public component. Internal data is described as CloudOps telemetry. |
| TourismSmall | Chronos-Bolt | no | no | Identical to a fev-bench task that the model's authors flag as not trained on. |
| TourismSmall | Chronos-T5 | no | no | Full list published, absent. |
| TourismSmall | Chronos-2 | no | no | Full list published, absent. fev-bench flag: not trained on. |
| TourismSmall | TiRex | no | no | Both lists published, absent. fev-bench flag: not trained on. |
| TourismSmall | Moirai-2 | no | no | Absent from every public component. fev-bench flag: not trained on. |
| Labour | Chronos-Bolt | unclear | unclear | No dataset list is published. |
| Labour | Chronos-T5 | yes | partial | 48 of 57 series are inside M4 Monthly. Last 18 observations held out, so values to 2015-07. |
| Labour | Chronos-2 | yes | unclear | 48 of 57 series are inside M4 Monthly. Size of the excluded test portion is not stated. |
| Labour | TiRex | yes | unclear | 48 of 57 series are inside M4 Monthly. Whether any observations were held out is not stated. |
| Labour | Moirai-2 | yes | unclear | 48 of 57 series are inside M4 Monthly, which is in the GIFT-Eval train split. Split point not checked. |
| Wiki2 | Chronos-Bolt | unclear | unclear | No dataset list is published. |
| Wiki2 | Chronos-T5 | unclear | unclear | Wiki Daily (100k) is in the corpus: same source, 1 or 2 of 150 bottom series found. |
| Wiki2 | Chronos-2 | unclear | unclear | Wiki Daily (100k) is in the corpus: same source, 1 or 2 of 150 bottom series found. |
| Wiki2 | TiRex | yes | yes | Kaggle Web Traffic Weekly is in the corpus and holds the weekly sums of all 150 bottom series for 2016. |
| Wiki2 | Moirai-2 | unclear | unclear | Extended Web Traffic holds all 150 bottom series for 2016. Whether it is in the training subset is not documented. |
| M5 | Chronos-Bolt | no | no | Documented as not seen in training. fev-bench flag: not trained on. |
| M5 | Chronos-T5 | no | no | Zero-shot benchmark dataset. |
| M5 | Chronos-2 | no | no | Full list published, absent. Zero-shot benchmark excluded. |
| M5 | TiRex | no | no | Both lists published, absent. Holds for the original checkpoint only. |
| M5 | Moirai-2 | unclear | unclear | M5 is in GIFT-Eval Pretrain. The training subset is not listed. fev-bench flags it as trained on. |
<!-- status-table:end -->

The PREREG leakage rule excludes a cell when the corpus is "confirmed to contain that dataset's test windows". That is the Test windows column. Status "yes" with Test windows "unclear" or "partial" means the model has seen the history that it is given as context, and may or may not have seen the values it is scored on.

Three points that the table cannot carry:

1. **Labour is not named in any corpus, but most of it is inside M4 Monthly.** Reading the dataset lists alone gives "no" for every model. The data says otherwise (section 3.2).
2. **Chronos-Bolt has no published dataset list.** Its "unclear" entries mean "not documented", not "probably clean". If it was trained on the Chronos-T5 corpus, its row equals the Chronos-T5 row. No primary source states that.
3. **The TiRex verdicts hold for the original checkpoint only.** `NX-AI/TiRex-1.1-gifteval` is documented as trained on a different corpus that, by its description, would include M5 and the Monash tourism data (section 2.4).

## 2. Per model

### 2.1 Chronos-T5

**Corpus [P].** Ansari et al. 2024, arXiv 2403.07815, Table 3 and Appendix B. 55 datasets in three groups, plus TSMixup augmentation and KernelSynth synthetic data. No internal data is mentioned.

Table 3 caption: "in-domain evalution data is used for training Chronos models and other task-specific baselines, except for the H observations that are held out for in-domain testing only; zero-shot evaluation data is not used in training Chronos models, but only for evaluation".

| Group | Relevant members |
|---|---|
| Pretraining only | Wiki Daily (100k), among 13 |
| In-domain: trained on, last H observations held out | M4 (Monthly), H = 18, among 15 |
| Zero-shot: not trained on | M5; Tourism (Monthly, Quarterly, Yearly), among 27 |

"Wiki Daily (100k) contains daily page views on the top-100k English Wikipedia articles between 2007 and 2022, ranked by number of observations (non-missing)."

| Dataset | Verdict | Evidence |
|---|---|---|
| TourismLarge, TourismSmall | no | Full list published, absent [P]. The nearest dataset, Monash Tourism, is zero-shot only [P] and does not match our series [E]. |
| Labour | yes, partial | M4 (Monthly) is a training dataset [P] and contains 48 of our 57 series [E]. The last 18 observations of each M4 series were held out [P]. |
| Wiki2 | unclear | Wiki2 is not listed. Wiki Daily (100k) comes from the same Wikipedia page-view source and covers 2016 [P]. 1 or 2 of our 150 bottom series were found in it [E]. |
| M5 | no | Zero-shot benchmark [P]. |

### 2.2 Chronos-Bolt

**Corpus: not published.**

- Model card [P]: "has been trained on nearly 100 billion time series observations."
- Model card [P], about the 27 zero-shot datasets of the Chronos paper: "despite having no prior exposure to these datasets during training".
- Chronos authors [P]: "We did not release/add support for training Chronos-Bolt here" (GitHub discussion 306).
- TiRex paper [S]: "Chronos-Bolt was selected due to its publicly available code and configuration, though the exact training data and procedure remain undisclosed."
- fev-bench results file for Chronos-Bolt, maintained by the same group [P]: `trained_on_this_dataset` is False on all 100 tasks, including `australian_tourism` and `m5_1D`.

| Dataset | Verdict | Evidence |
|---|---|---|
| TourismSmall | no | fev-bench `australian_tourism` is flagged not trained on [P], and that task is identical to TourismSmall [E]. |
| TourismLarge | unclear | No list. It is in no benchmark, so nothing documents it either way. |
| Labour | unclear | No list. "yes, partial" if the corpus includes M4 Monthly. |
| Wiki2 | unclear | No list. |
| M5 | no | Among the 27 datasets with "no prior exposure" [P]; fev-bench flag False [P]. |

Checkpoint used in this study: `amazon/chronos-bolt-small`, revision `772f3d25d38aec6d914c8949dab4462e2d46f5d8`.

### 2.3 Chronos-2

**Corpus [P].** Model card, "Training data": "Subset of Chronos Datasets (excluding test portion of datasets that overlap with GIFT-Eval)", "Subset of GIFT-Eval Pretrain", "Synthetic univariate and multivariate data". Paper, arXiv 2510.15821, sec. 4.1: "The full list of datasets is provided in Table 6 (Appendix)." No internal data is mentioned.

Table 6 lists, among others, M4 (Monthly) and "Wiki" (the Chronos Wiki Daily 100k data, also resampled to hourly and weekly). It does not list M5, any tourism dataset, the Kaggle web-traffic data or Wiki-Rolling.

On the Chronos zero-shot benchmark: "None of these datasets were included in the training corpus of Chronos-2."

| Dataset | Verdict | Evidence |
|---|---|---|
| TourismLarge, TourismSmall | no | Table 6 is stated to be the full list; absent [P]. fev-bench `australian_tourism` flagged False [P]. |
| Labour | yes, partial | M4 (Monthly) in Table 6 [P]; contains 48 of our 57 series [E]. How much of each M4 series was held out is described only as the "test portion". |
| Wiki2 | unclear | Same near-match as Chronos-T5. |
| M5 | no | Absent from Table 6; zero-shot benchmark excluded [P]. |

### 2.4 TiRex

**Corpus [P].** Auer et al. 2025, arXiv 2505.23719, sec. 4: "(1) We utilize the training datasets from Chronos ... (2) We enrich our training data with synthetic time series data ... (3) We add parts of the pre-training dataset proposed by GiftEval". Tables 5 and 6 list the datasets. No internal data is mentioned.

- Table 5 (from Chronos) includes Wiki Daily (100k) and M4 (Monthly).
- Table 6 (from GIFT-Eval Pretrain) includes "kaggle web traffic weekly", 145,063 series. It does not include M5, tourism, Wiki-Rolling or Extended Web Traffic.
- "TiRex's pre-training data has no overlap with Chronos-ZS benchmark", which contains M5 and Monash Tourism.

**Version warning [P].** Appendix E describes TiRex 1.1: "(1) All datasets present in the GiftEval benchmark were removed from our training corpus ... (2) Datasets from the Chronos-ZS benchmark that do not overlap with GiftEval were included". That would add M5 and Monash Tourism and remove M4. The checkpoint must be recorded when TiRex is run.

| Dataset | Verdict | Evidence |
|---|---|---|
| TourismLarge, TourismSmall | no | Both tables published; absent [P]. |
| Labour | yes, partial | M4 (Monthly) in Table 5 [P]; contains 48 of our 57 series [E]. The paper does not say whether the last observations were held out. |
| Wiki2 | yes at weekly frequency; daily unclear | "kaggle web traffic weekly" in Table 6 [P]. The weekly sums of all 150 of our bottom series are in it exactly [E]. |
| M5 | no | Absent; no overlap with the Chronos zero-shot benchmark [P]. Does not hold for TiRex 1.1. |

### 2.5 Moirai-2

**Corpus [P].** Model card `Salesforce/moirai-2.0-R-small`: "Subset of GIFT-Eval Pretrain, and Train datasets (Non-leaking historical context)", "Mixup data generated from non-leaking subsets of Chronos Dataset", "Synthetic time series produced via KernelSynth", "Internal Salesforce operational data". Paper, arXiv 2511.11698, sec. 4: "five complementary sources: (1) the non-leaking GIFT-EVAL PRETRAIN dataset, (2) train split of GIFT-EVAL TRAINTEST dataset (3) additional series generated via Chronos-Mixup, (4) KernelSynth data, and (5) anonymized internal Salesforce CloudOps telemetry data."

The subset is not listed anywhere. The card says "Subset of"; the paper does not.

GIFT-Eval Pretrain [P] (GIFT-Eval paper, arXiv 2410.10393, and the dataset repository) contains M5, Tourism (Monthly, Quarterly, Yearly), Wiki-Rolling, Kaggle Web Traffic Weekly and Extended Web Traffic. The GIFT-Eval train/test data contains M4 Monthly.

| Dataset | Verdict | Evidence |
|---|---|---|
| TourismLarge, TourismSmall | no for the public parts | Absent from every public component [P]. The internal data is described as CloudOps telemetry [P]. |
| Labour | yes, partial | The GIFT-Eval train split is used [P] and contains M4 Monthly [P]; M4 Monthly contains 48 of our 57 series [E]. |
| Wiki2 | unclear, likely yes | Extended Web Traffic contains all 150 of our bottom series exactly, for every day of 2016 [E]. Whether it is in the "subset" is not documented. |
| M5 | unclear, likely yes | M5 is in GIFT-Eval Pretrain [P]; the subset is not listed. The fev-bench results file flags Moirai-2.0 as trained on `m5_1D` [S]. |

## 3. Empirical checks

These compare our files in `data/hierarchical/` with public copies of the training datasets.

- **Script:** `scripts/leakage/run_checks.py`. It downloads the public copies at pinned revisions (about 2.4 GB, not committed) and runs in about one minute.
- **Results:** `scripts/leakage/output/summary.md`, with the same numbers in `summary.json`. Every figure in this section comes from that file, including the date ranges and the SHA-256 hash of every file compared.
- **Per-series matches:** `labour_m4_matches.csv`, `wiki2_matches.csv` and `tourism_best.csv` in the same folder.

| # | Check | Result |
|---|---|---|
| 1 | TourismSmall vs fev-bench `australian_tourism` | 89 of 89 series identical, 1998 Q1 to 2006 Q4 |
| 2 | Labour vs M4 Monthly (48,000 series) | 48 of 57 series matched, in 115 pairs; 52 of 57 with loose thresholds |
| 3 | Wiki2 vs Extended Web Traffic (daily) | 150 of 150 bottom series equal on all 366 days of 2016 |
| 4 | Wiki2 vs Kaggle Web Traffic Weekly | 150 of 150 bottom series: weekly sums equal on all 51 full weeks of 2016 |
| 5 | Wiki2 vs Wiki Daily (100k) | 1 near-identical series, 1 probable, 148 without a counterpart |
| 6 | Wiki2 vs Wiki-Rolling | 0 exact 7-day runs found, out of 38,953 searched |
| 7 | TourismLarge and TourismSmall vs Monash Tourism and M4 (monthly, quarterly) | 0 matches; best level correlation 0.944 with dates aligned, 0.977 at any shift |

A null result in checks 6 and 7 is evidence, not proof: a revised vintage of the same series could fall below the thresholds.

### 3.1 TourismSmall is a public benchmark task

fev-bench (arXiv 2509.26468) has a task "Australian Tourism": quarterly, 89 series, length 36, horizon 8. Its values equal TourismSmall exactly. fev-bench requires contributors to report "whether any part of the dataset at the same frequency was used during training". All five models are flagged False on it. TourismLarge is monthly and is not in fev-bench, the Chronos benchmark or GIFT-Eval.

### 3.2 Labour inside M4 Monthly

**Match criterion.** A pair counts as the same series when, on the date-aligned overlap of at least 120 months, all four hold:

| Condition | Threshold |
|---|---|
| Correlation of levels | above 0.999 |
| Every ratio M4 / Labour within this distance of the median ratio | 5% |
| Correlation of monthly changes | above 0.98 |
| Median ratio within this distance of a power of ten | 1% |

The first two conditions alone are not enough. 11 pairs pass them and are not counted: 10 have a ratio that is not a power of ten and are different series that move together (for example Queensland against a series 4.9 times its size), and 1 has a power-of-ten ratio but a change correlation of 0.971.

**Result.**

- **Matched:** 48 of 57 series, in 115 pairs. No M4 series matches more than one Labour series.
- **Same series, different vintage.** Ratios are 1, 10 or 100. 27 of the 115 pairs have some exactly equal values; none is identical throughout.
- **Overlap length:** 190 to 468 months per pair.
- **M4 series end** at 2007-10 (3 pairs), 2009-01 (15), 2015-07 (47) or 2017-01 (50).
- **Not matched:** 9 series, all at the state by sex or state by sex by employment-status level.

**Overlap with our test windows.** Labour runs from 1978-02 to 2020-11 in the raw file. Our loader drops 2020 onwards, so it ends at 2019-12 and the five registered test windows are the calendar years 2015 to 2019.

| Model trained on | Latest Labour month in corpus | Series with test months in corpus | Test windows touched |
|---|---|---|---|
| M4 Monthly with the last 18 observations held out (Chronos-T5) | 2015-07 | 39 of 57 | 2015 (January to July) |
| Full M4 Monthly series | 2017-01 | 47 of 57 | 2015, 2016 and January 2017 |

In both cases the history before 2015, which is the context the model is given, is in the training data.

**Do 5 yearly origins with h = 12 fit entirely after 2017-01?** No.

| Item | Value |
|---|---|
| Months needed | 60 |
| Months after 2017-01, as loaded (to 2019-12) | 35 |
| Months after 2017-01, raw file (to 2020-11) | 46 |
| Non-overlapping 12-month windows that fit, as loaded | 2: the years 2018 and 2019 |
| Largest spacing at which 5 windows fit, as loaded | 5 months, so the windows would overlap |
| Context at the first window after 2017-01 | 468 months, against 120 required |

The context requirement is met with room to spare. The limit is the number of months after the cutoff.

### 3.3 Wiki2 inside the Kaggle web-traffic data

Wiki2 was built from the Kaggle "Web Traffic Time Series Forecasting" data: Rangapuram et al. 2021 [P], "Wiki includes daily views for 145,000 Wikipedia articles starting from Jul. 2015 to Dec. 2016. We follow the procedure described by Ben Taieb & Koo (2019) to filter the dataset to 150 bottom series (199 total)." The Monash archive and GIFT-Eval Pretrain redistribute that same data as Kaggle Web Traffic Weekly and Extended Web Traffic.

- **Daily copy:** each of our 150 bottom series equals one item on every day of 2016.
- **Weekly copy:** the weekly sums of the same 150 items match exactly. The file states a week start of Monday 2015-06-29; the sums match only for weeks starting Wednesday 2015-07-01, the first day of the daily data.

Our five test windows are the last 35 days of 2016. They lie inside both copies. For a model trained on either, Wiki2 is not a zero-shot test.

## 4. Near-matches that are not our data

| Dataset in a corpus | Relation to ours |
|---|---|
| Monash Tourism (Monthly, Quarterly, Yearly) | The 2011 tourism forecasting competition, 1,311 series. Athanasopoulos et al. 2011 [P]: "They were supplied by both tourism bodies (such as Tourism Australia, the Hong Kong Tourism Board and Tourism New Zealand) and various academics"; series are published "under coded titles". So it holds some Australian tourism series of unknown identity. No match with our series was found [E]. |
| Wiki Daily (100k) | English Wikipedia page views from Wikimedia dumps. Same source as Wiki2, different extraction. |
| Wiki-Rolling | Wikipedia page views from the GluonTS collection. Provenance not confirmed in a primary source. |
| M4 Monthly | Contains the Labour series (section 3.2). |
| Hierarchical Sales, Favorita, M4 | Not M5. |

## 5. Not verified

1. The Chronos-Bolt training corpus. No primary source lists it.
2. Which part of GIFT-Eval Pretrain, and which Chronos datasets, Moirai-2 used.
3. How many observations of M4 Monthly were held out by Chronos-2, TiRex and Moirai-2. This decides their Test windows entry for Labour.
4. Who supplied the fev-bench flags for TiRex and Moirai-2.0. The files are maintained by the fev-bench team, not by those models' authors.
5. Whether the M5 copies in the corpora equal the Kaggle M5 data. Not compared.
6. The Kaggle competition files themselves (login required). The Wiki2 link rests on the redistributed copies.
7. The M4 Monthly copy inside GIFT-Eval. The check used the copy in `autogluon/chronos_datasets`.
8. Tourism checks did not cover M1 or M3, any yearly data, or FRED-MD. Tourism series with more than half zeros were not compared (51 of 555 in TourismLarge).
9. The published journal version of Athanasopoulos et al. 2011. The author's working paper was read.

## 6. Sources

| Source | Location |
|---|---|
| Chronos paper | https://arxiv.org/abs/2403.07815 |
| Chronos-T5 model card | https://huggingface.co/amazon/chronos-t5-small |
| Chronos-Bolt model card | https://huggingface.co/amazon/chronos-bolt-small |
| Chronos-Bolt training code statement | https://github.com/amazon-science/chronos-forecasting/discussions/306 |
| Chronos-2 paper | https://arxiv.org/abs/2510.15821 |
| Chronos-2 model card | https://huggingface.co/amazon/chronos-2 |
| Chronos datasets | https://huggingface.co/datasets/autogluon/chronos_datasets |
| TiRex paper | https://arxiv.org/abs/2505.23719 |
| TiRex model cards | https://huggingface.co/NX-AI/TiRex , https://huggingface.co/NX-AI/TiRex-1.1-gifteval |
| Moirai 2.0 paper | https://arxiv.org/abs/2511.11698 |
| Moirai 2.0 model card | https://huggingface.co/Salesforce/moirai-2.0-R-small |
| Moirai 1.0 and LOTSA | https://arxiv.org/abs/2402.02592 , https://huggingface.co/datasets/Salesforce/lotsa_data |
| GIFT-Eval paper and data | https://arxiv.org/abs/2410.10393 , https://huggingface.co/datasets/Salesforce/GiftEvalPretrain , https://huggingface.co/datasets/Salesforce/GiftEval |
| fev-bench paper, results, data | https://arxiv.org/abs/2509.26468 , https://github.com/autogluon/fev/tree/main/benchmarks/fev_bench/results , https://huggingface.co/datasets/autogluon/fev_datasets |
| Monash archive paper | https://arxiv.org/abs/2105.06643 |
| Rangapuram et al. 2021 | https://proceedings.mlr.press/v139/rangapuram21a/rangapuram21a.pdf |
| Ben Taieb and Koo 2019 | https://souhaib-bentaieb.com/papers/2019_kdd_hts_reg.pdf |
| Athanasopoulos et al. 2011 (working paper) | https://robjhyndman.com/papers/forecompijf.pdf |
