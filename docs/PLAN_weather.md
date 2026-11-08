# Does weather help? Analysis plan for the weather study (rung 2)

*Draft for the freeze · Andy Moran (Heaviside Analytics) · design: Brief 4, revision 11 (agreed 8 November 2026)*

> **Status at writing.** No forecaster has been given real weather-forecast inputs for a real day, and no weather-forecast error has been computed, in the backtest or the prospective window. Demand outturn to date is public, and rung 1's no-weather forecasts and errors are published (DOI [10.5281/zenodo.23962005](https://doi.org/10.5281/zenodo.23962005)). Before this plan the author has seen: rung 1's results; the synthetic test (`weather_study/SYNTHETIC_TEST.md`); feasibility checks comparing weather sources with each other, never with demand; national weather inputs built for the backtest from 15 April to 31 October 2026 and dry runs for 8 and 9 November 2026, with no forecast made from them; and the power check (Section 7), which reads only rung 1's published no-weather errors. The plan is committed and tagged before the first real weather forecast; the tag's time is the evidence of timing.

## 1. The questions

Rung 1 found that NESO's day-ahead forecast had less than half the error of every model. NESO uses weather; the models and the regression did not. This study gives all of them the same, honestly timed weather and asks:

| | Question | Status |
| --- | --- | --- |
| **Q1** | Does weather improve each forecaster: Chronos-2, Timestral, TimesFM 2.5 and the regression? | **Confirmatory**, in the backtest and the prospective window |
| Q2 | How much of the gap to NESO does weather close? | Estimate with interval |
| Q3 | Does rung 1's 7–9% edge over the regression survive when the regression has the same weather? | Estimate with interval |
| **Q4** | Is weather's help larger in the evening (Q4a), and smaller at midday (Q4b)? | **Confirmatory if powered**, prospective window only |

**What the answers cannot show.** Why NESO leads; what better weather inputs could do; freedom from training leakage in the backtest (the prospective window answers that); value in money. Brief 4, Section 9, lists the rest.

## 2. Data

- **Demand, timing and the no-weather arms:** rung 1's, unchanged and locked (`PLAN_LOCK.json`, `LOCK_AMENDMENT_1.json`): INDO outturn, the 09:30 UK issue time on D−1, the 08:30 input cut-off, the 2,688-half-hour context, the target day's half-hours, England and Wales bank holidays from the hashed GOV.UK file. In the backtest the no-weather forecasts are rung 1's saved forecasts (`results/attempt2/forecasts.csv`); in the prospective window, those made daily under the frozen prospective plan (`prospective/PLAN.md`).
- **NESO's forecast:** as rung 1 (Section 4.3 of its plan), from the 09:30 publication.
- **Weather:** national temperature, wind speed and solar radiation for every half-hour of each forecast's context and horizon, built by the frozen weather processing (Brief 4, Section 3): ECMWF's 00:00 UTC open-data runs at 0.25°, ERA5 for history, weighted to Great Britain by residential population (temperature, wind) and installed solar capacity (solar). The processing code is private until the results note and hashed in the lock; its outputs are public:
  - prospective: `weather_study/inputs/<D>.csv` and `.json`, pushed daily before 00:00 UK time on D;
  - backtest: `inputs_2024-03-16_2026-10-31.csv.gz` and its per-day record, built once, hashed in the lock, and published with the results.

## 3. The inputs each forecaster receives

All rules are Brief 4's, agreed before this plan; they are restated here for the reader, and the code implements them.

- **Target day:** from the 00:00 UTC run of D−1 (steps 21 to 48). If that run was not on a public copy by 09:30 UK time on D−1, from the run of D−2 (steps 45 to 72), flagged as a *fallback day* (186 such days in the backtest, 30 March to 1 October 2025; none expected in the prospective window).
- **History:** ERA5 up to UTC day D−11; each UTC day from D−10 to D−1 from its own run's first 24 hours. ERA5 never fills an ECMWF day.
- **Decision 13:** the context's ERA5 days are put on ECMWF's scale with factors from D−70 to D−11 (temperature added; wind and solar multiplied), at least 30 days per variable, else unadjusted and flagged. The regression's training is not adjusted.
- **Half-hours:** each takes its value at its midpoint; temperature and wind interpolated linearly, solar the mean of the interval containing the midpoint.
- **Dropped days:** a day the rules cannot build is dropped from every confirmatory analysis, for every forecaster, and listed with its reason.

## 4. The forecasters, with weather

Every setting not named here is rung 1's, unchanged.

| Forecaster | Weather enters as | Settings fixed here |
| --- | --- | --- |
| Chronos-2 (`chronos-forecasting` 2.3.2, rung 1's pinned revision) | `past_covariates` over the context and `future_covariates` over the horizon, each `{'t2m', 'wind', 'solar'}` | `cross_learning=False`; point forecast the 0.5 quantile |
| Timestral (`timestral` 1.0.0, pinned revision) | `known_covariates`, shape (1, 3, context + horizon), variables in the order t2m, wind, solar | `symmetric_averaging=True`, `clip_nonnegative=True`; point forecast the 0.5 quantile |
| TimesFM 2.5 (`timesfm` 3.0.2, pinned revision; JAX CPU build) | `forecast_with_covariates`, `dynamic_numerical_covariates` over context + horizon | `xreg_mode='xreg + timesfm'` (decision 11), `normalize_xreg_target_per_input=True`, `ridge=0.0`, `force_on_cpu=True`; compiled with rung 1's settings plus `return_backcast=True`; the forecast is the last *h* values of the returned point forecast |
| Regression | Rung 1's terms, plus for each half-hour slot: temperature centred on its training mean, its square, wind, temperature × wind, solar | Re-estimated daily on the 365 days to D−2; training weather by Section 3's splice, unadjusted; target day as Section 3 |

**The other TimesFM mode** (`timesfm + xreg`) is also run, on both windows, and reported descriptively in an appendix only.

**Environment:** `gbwx`, whose package list is in the lock. The no-weather prospective forecasts stay in `gbfm`, untouched.

## 5. Windows

| Window | Target days | Days | Confirmatory tests |
| --- | --- | --- | --- |
| **Backtest** | 16 March 2024 to 31 October 2026 | 960, less any dropped | Q1 (four tests) |
| **Prospective** | 20 November 2026 to 31 March 2027 | 132, less any dropped or not published on time | Q1 (four), Q4a and Q4b (three each), by the power branch in Section 7 |

Prospective days from 5 to 19 November 2026, before the weather processing was frozen, are reported descriptively. A prospective day counts only if its weather inputs reached GitHub before 00:00 UK time on D (by GitHub's record) and its no-weather forecast met the prospective plan's own rule.

## 6. Statistics

**Daily error:** for each forecaster, arm and day, the mean absolute error in MW over D's half-hours, against INDO outturn (rung 1's definition).

**Q1.** For each forecaster and window: *ratio* = mean daily error with weather ÷ mean daily error without, over the window's days. One-sided test of H0: ratio ≥ 1 against ratio < 1.

**Resampling, for every test and interval:** moving-block bootstrap over days, blocks of 7, **2,000 draws**, the same resampled days for both arms; seed **20261120** for the backtest and **20270401** for the prospective window (the scoring seed already fixed in the prospective plan). p-values by rung 1's percentile method: the share of bootstrap statistics on the null side (for Q1, ratios ≥ 1). Intervals: 90% percentile intervals, labelled descriptive.

**One study-wide error rate: 5%, by a graphical procedure** (Bretz and colleagues, 2009). Fourteen hypotheses: Q1 for four forecasters in two windows (Q1b for the backtest, Q1p for the prospective window), and Q4a and Q4b for each of the three models.

- **Initial weights:** each Q1 hypothesis 1/8 of 5% (0.00625); each Q4 hypothesis 0.
- **Transitions,** for each forecaster *f* and each model *m*:

| From | To | Weight |
| --- | --- | --- |
| Q1b(*f*) | Q1p(*f*) | 1 |
| Q1p(*m*) | Q1b(*m*) · Q4a(*m*) · Q4b(*m*) | 1/2 · 1/4 · 1/4 |
| Q1p(regression) | Q1b(regression) | 1 |
| Q4a(*m*) | Q4b(*m*) | 1 |
| Q4b(*m*) | Q4a(*m*) | 1 |

- **Algorithm:** reject any hypothesis whose p-value is at or below its current level; pass its level along its outgoing edges and update the graph (Bretz and colleagues, 2009, Algorithm 1); repeat until no further rejection. The result does not depend on the order of rejections.
- **Consequences:** a Q4 hypothesis can reject only after its model's prospective Q1 has rejected. Level never passes between forecasters.
- **Timing:** the backtest's hypotheses are tested when the backtest is scored. A rejection then is final. A non-rejection is provisional: when the prospective tests are scored, levels passed to it may let it reject. The backtest result is reported as "rejected" or "not rejected so far", and final only when the prospective tests are done.
- **If a prospective test is made descriptive** by the power branch (Section 7), its hypothesis and, for a model, its Q4 hypotheses are removed from the graph before any result, and its initial weight goes to the same forecaster's backtest Q1.

**Q2, the gap to NESO.** For each model and window: the reduction in mean daily error, without − with, in MW; and the share of the gap closed, (without − with) ÷ (without − NESO). The share is reported only where the gap without weather exceeds 100 MW, always beside the MW. Intervals as above.

**Q3, rung 1's edge.** For each model and window: mean daily error with weather ÷ the regression's with weather, with its interval, beside rung 1's ratio without weather. Read with Brief 4's caveat: the regression trains on ERA5 and forecasts with ECMWF, which in the cool months runs about 0.2 °C colder, roughly 100–200 MW of demand; part of any edge may be that mismatch.

**Q4, prospective window only.** *Blocks:* six four-hour blocks of UK clock time from midnight. For a block or set of blocks, the *fractional error reduction* FR = 1 − (mean absolute error with weather ÷ without), pooling every half-hour of the window's days in those blocks.

- **Q4a:** FR(16:00–20:00) − FR(00:00–04:00); H0 ≤ 0, one-sided.
- **Q4b:** FR(12:00–16:00) − FR(the other five blocks together); H0 ≥ 0, one-sided (midday smaller).

The bootstrap resamples days, as for Q1.

**Descriptive, outside the procedure:**
- every confirmatory quantity with all fallback days removed;
- Q1 on the unadjusted inputs (decision 13 off);
- Q1 with decision 13's minimum at 20 and 45 days;
- the other TimesFM mode;
- errors by block in MW, and by season;
- the bank-holiday check, as Brief 4, Section 6, with its reading fixed there; intervals from 2,000 draws resampling holidays and ordinary days separately, seed **20261121**;
- the five days from 5 to 19 November 2026.

**Missing data:** rung 1's rules for demand and NESO; Section 3's rules for weather. Every exclusion is listed.

## 7. Power

`weather_study/power.py`, 200 replications per cell, reading only rung 1's published no-weather forecasts. For each forecaster, the weather arm is simulated from the forecaster's own real errors: a share *c* of each half-hour's error is kept, the rest is the same forecaster's error from another day at least 30 days away, and the result is scaled to the true effect. The noise, *c*, comes from rung 1's similar forecasters: the half-hourly errors of the three models correlate at 0.82 to 0.95 over the backtest window. One model with and without weather should agree at least as closely as two different models, so the table uses 0.82, the lower bound, and 0.9. The prospective windows are stood in for by rung 1's two winters, 20 November to 31 March 2024–25 and 2025–26 (132 days each; 264 together).

*[Power table: inserted from `weather_study/power.json` when the check finishes.]*

**The branch for each prospective test,** by the brief's rule, evaluated at the level the test holds on its path (Q1: 0.00625; Q4: 0.00625, its share once both its model's Q1 tests have rejected) and the middle effect (Q1: a 5% cut; Q4: 6 points):

1. power at least 80% on 132 days at both values of *c*: **confirmatory this winter;**
2. otherwise, at least 80% on 264 days at both values: **confirmatory over two winters** (April 2028), with a descriptive interim read in April 2027;
3. otherwise: **descriptive.**

If power at the middle effect crosses 80% between the two values of *c*, the test cannot be called, and the default is **two winters**.

*[Branches: inserted from the table.]*

## 8. Gate checks

The official run is the first whose outputs pass every check; a failed run is a deviation, its outputs kept. Before the freeze every check runs on the real inputs with a stand-in forecaster and is shown to fail on corrupted input.

1. **Data:** every input matches its hash in the lock: demand, NESO, the weights, every ECMWF run and ERA5 file used, and the built backtest inputs.
2. **No look-ahead:** no demand input after 08:30 UK on D−1; every target-day weather value from a run public by 09:30 on D−1 (or the fallback run); every history value from before the cut-off.
3. **The weather inputs used are the published ones,** by hash, for every day in both windows.
4. **The archive matches the daily collection,** value for value, for every day since 31 October 2026.
5. **The no-weather forecasts equal rung 1's** on every shared backtest day, and the prospective plan's on every prospective day.
6. **Backtest and daily inputs are one method:** a sample of backtest days rebuilt with the daily push code agrees exactly.
7. **Model outputs:** finite, the exact horizon, identical on repetition for a sample of days; with weather, moving the horizon's temperature moves each model's forecast.
8. **Regression:** an independent implementation's forecast with weather for three sample days matches the pipeline's to 0.1 MW.
9. **Weights:** each model's weights file matches its recorded SHA-256; the spatial weights match theirs.

## 9. Results sentences, written now

Every sentence names the window, the ratio and its 90% interval.

| Result | Sentence |
| --- | --- |
| Q1 rejected | "Over [window], weather improved [forecaster]'s day-ahead forecast of GB demand: mean daily error fell by [x]% (error ratio r; 90% interval a–b, descriptive), confirmed under the study-wide 5% procedure." |
| Q1 not rejected | "Over [window], weather was not shown to improve [forecaster]'s forecast (error ratio r; 90% interval a–b, descriptive)." When the interval excludes 1 but the procedure does not reject, add: "Unadjusted, the interval excludes no improvement; after adjustment across the study's tests it is not confirmed." |
| Q1 backtest, provisional | "…not confirmed so far; the result is final when the prospective window is scored." |
| Q2 | "With weather, [model]'s gap to NESO fell by [x] MW (90% interval a–b, descriptive)[, [s]% of the gap]." |
| Q3 | "With the same weather, [model]'s error was r times the regression's (90% interval a–b, descriptive), against [r₁] without weather in rung 1. The regression trains on reanalysis and forecasts with ECMWF; in winter this mismatch is worth about 100–200 MW, so part of any edge may be the regression's." |
| Q4a / Q4b rejected | "In the evening, weather cut [model]'s error by [x] points more than overnight (90% interval a–b, descriptive), confirmed." / "At midday, weather cut [model]'s error by [x] points less than at other times…, confirmed." |
| Q4 not reached | "Q4 was not tested for [model], because weather was not confirmed to improve it in the prospective window; the block figures are descriptive." |

Every backtest sentence carries rung 1's leakage note: "The model may have seen GB demand from this period in training."

## 10. Procedure

1. Freeze the weather processing and this plan: commit; tag `weather-plan`; write `WEATHER_LOCK.json` (the tag, the hashes of this plan, the private code, the weights, the backtest inputs and every source file, `gbwx`'s package list, seeds, and the power branches). Publish this plan and the lock.
2. Start the daily push; first target day 20 November 2026.
3. Score the backtest: run until one attempt passes the gate; test the backtest hypotheses.
4. At the prospective window's end (or April 2028 on the two-winter branch): score it, complete the procedure, write the results note with Section 9's sentences, publish the private code the same day, archive with a DOI.

**Deviations:** after the tag, any change to the data, inputs, forecasters, settings, windows, statistics, procedure, gate or wording; a second passing run; a result outside Section 6 reported as anything but exploratory; a listed output not reported.

## 11. Decisions fixed in this plan

| # | Decision | Settled as |
| --- | --- | --- |
| 1 | Seeds | Backtest bootstrap 20261120; prospective 20270401; bank-holiday intervals 20261121; power check 20261108 |
| 2 | The graph | Section 6 |
| 3 | Model interfaces with weather | Section 4 |
| 4 | Power branches | Section 7, from the table |
