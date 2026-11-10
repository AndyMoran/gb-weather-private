# Does weather help? Analysis plan for the weather study (rung 2)

*Final text, agreed 8 November 2026, amended 9 November 2026 (model-cycle changes, decision 14); frozen with the weather processing (tag `weather-plan`) · Andy Moran (Heaviside Analytics) · design: Brief 4, revision 12*

> **Status at writing.** No forecaster has been given real weather-forecast inputs for a real day, and no weather-forecast error has been computed, in the backtest or the prospective window. Demand outturn to date is public, and rung 1's no-weather forecasts and errors are published (DOI [10.5281/zenodo.23962005](https://doi.org/10.5281/zenodo.23962005)). Before this plan the author has seen: rung 1's results; the synthetic test (`weather_study/SYNTHETIC_TEST.md`); feasibility checks comparing weather sources with each other, never with demand; national weather inputs built for the whole backtest (16 March 2024 to 31 October 2026), partial builds before it, and dry runs for 8 and 9 November 2026, with no forecast made from them; and the power check (Section 7, run 8 November 2026), which reads only rung 1's published no-weather errors. The plan is committed and tagged before the first real weather forecast; the tag's time is the evidence of timing.

## 1. The questions

Rung 1 found that NESO's day-ahead forecast had less than half the error of every model. NESO uses weather; the models and the regression did not. This study gives all of them the same, honestly timed weather and asks:

| | Question | Backtest | Prospective window |
| --- | --- | --- | --- |
| **Q1** | Does weather improve each forecaster: Chronos-2, Timestral, TimesFM 2.5 and the regression? | **Confirmatory** | Descriptive (Section 7) |
| Q2 | How much of the gap to NESO does weather close? | Estimate with interval | Estimate with interval |
| Q3 | Does rung 1's 7–9% edge over the regression survive when the regression has the same weather? | Estimate with interval | Estimate with interval |
| Q4 | Is weather's help larger in the evening (Q4a), and smaller at midday (Q4b)? | — | Descriptive (Section 7) |

Brief 4 designed the prospective Q1 and Q4 tests as confirmatory if the power check allowed. It did not: no prospective test reaches 80% power on one winter or two (Section 7). By the rule fixed in Brief 4's revision 7, before the power check was run, every prospective test is descriptive, and **the confirmatory result is the backtest's four Q1 tests.**

**What the answers cannot show.** Why NESO leads; what better weather inputs could do; freedom from training leakage in the backtest (Section 6); value in money. Brief 4, Section 9, lists the rest.

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
- **Decision 14:** decision 13's rule runs unchanged across a change of ECMWF's model cycle. A target day *straddles* a change if its factor window holds days from before it while its horizon run is from after it. ECMWF's changes are listed in `weather_study/cycle_changes.csv`, each dated by its first 00:00 UTC run: Cycle 49r1 (13 November 2024) and Cycle 50r1 (13 May 2026), so the backtest's straddling days are 14 November 2024 to 21 January 2025 and 14 May to 21 July 2026, 138 days. Any change ECMWF makes before 31 March 2027 is added to the list from its announcement, and the prospective window gets the same treatment.
- **Half-hours:** each takes its value at its midpoint; temperature and wind interpolated linearly, solar the mean of the interval containing the midpoint.
- **Dropped days:** a day the rules cannot build is dropped from every analysis, for every forecaster, and listed with its reason.

## 4. The forecasters, with weather

Every setting not named here is rung 1's, unchanged.

| Forecaster | Weather enters as | Settings fixed here |
| --- | --- | --- |
| Chronos-2 (`chronos-forecasting` 2.3.2, rung 1's pinned revision) | `past_covariates` over the context and `future_covariates` over the horizon, each `{'t2m', 'wind', 'solar'}` | `cross_learning=False`; point forecast the 0.5 quantile |
| Timestral (`timestral` 1.0.0, pinned revision) | `known_covariates`, shape (1, 3, context + horizon), variables in the order t2m, wind, solar | `symmetric_averaging=True`, `clip_nonnegative=True`; point forecast the 0.5 quantile |
| TimesFM 2.5 (`timesfm` 3.0.2, pinned revision; JAX CPU build) | `forecast_with_covariates`, `dynamic_numerical_covariates` over context + horizon | `xreg_mode='xreg + timesfm'` (decision 11), `normalize_xreg_target_per_input=True`, `ridge=0.0`, `force_on_cpu=True`; compiled with rung 1's settings plus `return_backcast=True`; the forecast is the last *h* values of the returned point forecast |
| Regression | Rung 1's terms, plus for each half-hour slot: temperature centred on its training mean, its square, wind, temperature × wind, solar | Re-estimated daily on the 365 days to D−2; training weather by Section 3's splice, unadjusted; target day as Section 3. A weather variable whose range over a slot's fit rows is below its tolerance (solar 1 W/m², temperature 0.01 °C, wind 0.01 m/s) is left out of that slot, with the terms built from it: in practice, solar at night |

**The other TimesFM mode** (`timesfm + xreg`) is also run, on both windows, and reported descriptively in an appendix only.

**Environment:** `gbwx`, whose package list is in the lock. The no-weather prospective forecasts stay in `gbfm`, untouched.

## 5. Windows

| Window | Target days | Days | Role |
| --- | --- | --- | --- |
| **Backtest** | 16 March 2024 to 31 October 2026 | 960, less any dropped | **Confirmatory:** Q1, four tests |
| **Prospective** | 20 November 2026 to 31 March 2027 | 132, less any dropped or not published on time | Descriptive, pre-specified in Section 6 |

**This supersedes the date stated in decision 7.** Decision 7 named 6 March 2024, the first run with solar radiation, as the start. The ten-day splice requires ten days of runs with solar radiation in every context, so the first target day with a complete context is 16 March 2024 (D−10 = 6 March), and the backtest is 960 days, not 970. This applies the agreed splice rule; it does not change it. Brief 4 records the correction in revision 11 (8 November 2026).

Prospective days from 5 to 19 November 2026, before the weather processing was frozen, are reported descriptively and kept apart from the window. A prospective day counts only if its weather inputs reached GitHub before 00:00 UK time on D (by GitHub's record) and its no-weather forecast met the prospective plan's own rule.

## 6. Statistics

**Daily error:** for each forecaster, arm and day, the mean absolute error in MW over D's half-hours, against INDO outturn (rung 1's definition).

**Resampling, for every test and interval:** moving-block bootstrap over days, blocks of 7, **2,000 draws**, the same resampled days for both arms; seed **20261120** for the backtest and **20270401** for the prospective window (the scoring seed already fixed in the prospective plan). Intervals: 90% percentile intervals, labelled descriptive.

### 6.1 The confirmatory tests: Q1 in the backtest

For each forecaster: *ratio* = mean daily error with weather ÷ mean daily error without, over the backtest's days. One-sided test of H0: ratio ≥ 1 against ratio < 1, with rung 1's percentile p-value: the share of bootstrap ratios at or above 1.

**The error rate.** Brief 4 set one study-wide rate of 5% under a graphical procedure (Bretz and colleagues, 2009) over fourteen hypotheses: Q1 for four forecasters in two windows, and Q4a and Q4b for each model. Each Q1 hypothesis started with 0.00625; a rejected backtest Q1 would pass its level to the same forecaster's prospective Q1, and a rejected prospective Q1 half back to the backtest and a quarter to each of the model's Q4 hypotheses; no level would pass between forecasters. The procedure removes any hypothesis made descriptive before results, passing its initial weight to the same forecaster's backtest Q1 (Brief 4, Section 7, and this plan's draft of 8 November). **Every prospective hypothesis is descriptive (Section 7), so the procedure reduces to four tests, one per forecaster, each at 0.05/4 = 0.0125,** with no further transitions. Each backtest result is final when the backtest is scored.

**The leakage caveat on the confirmatory result.** The backtest's days fall inside the period the models may have seen in training (rung 1, Section 7: the perturbation check raised no flag, which cannot rule memory out). A model that remembers GB demand would forecast well without weather and might gain less from it, or differently, than a model that does not. Both arms carry the same exposure, so the comparison is within each model, but the size of the gain could still differ on unseen data. Every confirmatory sentence carries the caveat (Section 9). **The prospective window is the check on it:** no day in it can be in any model's training data, and Section 6.3 fixes now what it will report and how it will be read against the backtest.

### 6.2 Further backtest estimates (with intervals, no tests)

- **Q2, the gap to NESO:** for each model, the reduction in mean daily error, without − with, in MW; and the share of the gap closed, (without − with) ÷ (without − NESO), reported only where the gap without weather exceeds 100 MW, always beside the MW.
- **Q3, rung 1's edge:** for each model, mean daily error with weather ÷ the regression's with weather, beside rung 1's ratio without weather. Read with Brief 4's caveat: the regression trains on ERA5 and forecasts with ECMWF, which in the cool months runs about 0.2 °C colder, roughly 100–200 MW of demand; part of any edge may be that mismatch.
- **Q4's measures** (as in 6.3), for the backtest too.

### 6.3 The prospective window, pre-specified (descriptive)

Reported for every forecaster, with 90% intervals from the prospective bootstrap (seed 20270401), and no formal test or p-value:

1. **Q1:** the ratio, with weather ÷ without. If ECMWF changes its model cycle before 31 March 2027, also without the window's straddling days (decision 14).
2. **Q2** and **Q3:** as in 6.2.
3. **Q4:** *blocks* are six four-hour blocks of UK clock time from midnight; for a block or set of blocks, the *fractional error reduction* FR = 1 − (mean absolute error with weather ÷ without), pooling every half-hour of the window's days in those blocks. Reported: FR in each block; Q4a's difference, FR(16:00–20:00) − FR(00:00–04:00); Q4b's difference, FR(12:00–16:00) − FR(the other five blocks together).
4. **Agreement with the backtest,** for each forecaster's Q1 ratio, read by a rule fixed now:
   - *consistent* if the prospective ratio lies inside the backtest's 90% interval, or the two intervals overlap;
   - *smaller prospective gain* if the prospective interval lies wholly above the backtest's (weather helped less on unseen days: consistent with, though not proof of, a backtest flattered by leakage);
   - *larger prospective gain* if it lies wholly below.
   The windows also differ in season (the prospective window is winter only), so the backtest's Q1 is also reported on its own winter days (20 November to 31 March in 2024–25 and 2025–26) beside the prospective result.
5. **Nothing else is reported as a prospective finding.** Any further prospective comparison is labelled exploratory.

### 6.4 Other descriptive results, outside the tests

- every Q1 result with all fallback days removed;
- every Q1 result without the days that straddle a model-cycle change (decision 14). It shows whether the result depends on those days; the confirmatory Q1 is computed on all days and is unaffected. With 822 days rather than 960 its intervals are a little wider, which matters little for a descriptive check;
- Q1's three readings side by side, for each forecaster: adjusted inputs on all days (the confirmatory figure), adjusted without the straddling days, and unadjusted inputs on all days. Together they show whether decision 13 matters, whether the cycle change matters, and whether the two point the same way;
- Q1 on the unadjusted inputs (decision 13 off), and with decision 13's minimum at 20 and 45 days;
- the other TimesFM mode;
- errors by block in MW, and by season;
- the bank-holiday check, as Brief 4, Section 6, with its reading fixed there; intervals from 2,000 draws resampling holidays and ordinary days separately, seed **20261121**;
- the days from 5 to 19 November 2026.

**Missing data:** rung 1's rules for demand and NESO; Section 3's rules for weather. Every exclusion is listed.

## 7. Power, and the branch for each prospective test

**Run 8 November 2026,** `weather_study/power.py`, 200 replications per cell, 1,000 bootstrap draws per replication, seed 20261108; output `weather_study/power.json`. It reads only rung 1's published no-weather forecasts and computes no weather-forecast error.

**Method.** For each forecaster, the weather arm is simulated from that forecaster's own real errors: a share *c* of each half-hour's simulated error is its real error, and the rest is the same forecaster's error on a day at least 30 days away; the result is scaled so that the true effect holds over the window. Each replication tests on a block-resampled set of days, as a real study would see them.

**The noise, *c*.** *c* is the correlation between the two arms' errors: how much of a forecaster's error with weather is the same error it made without. The brief requires it to be estimated from rung 1's similar forecasters, never from weather forecasts. Over the backtest window, the three models' half-hourly errors correlate at 0.82 (Chronos-2 with TimesFM 2.5) to 0.95 (Chronos-2 with Timestral). The table gives two values, as "cautious / optimistic": **0.82**, the lowest correlation between any two of rung 1's models, and **0.9**. The within-model correlation cannot be measured before any weather-forecast error exists, so a cross-model one stands in for it. This is a conservative proxy: the two arms of one model share the base model and the demand inputs and differ only in weather, so their true correlation is likely higher, which would give more power than the table shows. The conclusion below holds across the range used: even at 0.9, no prospective test exceeds 0.39. The rule is applied at these two values, set before the table was seen, and not re-run at higher ones.

**Windows.** The backtest is its 960 days. The prospective window is stood in for by rung 1's two winters, 20 November to 31 March in 2024–25 and 2025–26: 132 days each ("one winter", alternating), and 264 together ("two winters").

**Q1:** probability of rejecting H0 (ratio ≥ 1) at 0.00625, the level each Q1 test held under the full design; for the backtest, also at 0.0125, the level it now holds.

| Forecaster | Window | Weather cuts error by 5% (middle effect) | By 10% | Backtest at 0.0125, 5% cut |
| --- | --- | --- | --- | --- |
| Chronos-2 | backtest | 0.77 / 0.94 | 1.00 / 1.00 | 0.83 / 0.97 |
| | one winter | 0.14 / 0.24 | 0.46 / 0.83 | — |
| | two winters | 0.23 / 0.38 | 0.82 / 0.95 | — |
| Timestral | backtest | 0.70 / 0.96 | 1.00 / 1.00 | 0.75 / 0.98 |
| | one winter | 0.14 / 0.18 | 0.47 / 0.69 | — |
| | two winters | 0.28 / 0.39 | 0.77 / 0.96 | — |
| TimesFM 2.5 | backtest | 0.72 / 0.97 | 1.00 / 1.00 | 0.80 / 0.97 |
| | one winter | 0.19 / 0.19 | 0.57 / 0.69 | — |
| | two winters | 0.20 / 0.39 | 0.80 / 0.96 | — |
| Regression | backtest | 0.65 / 0.92 | 0.99 / 1.00 | 0.74 / 0.94 |
| | one winter | 0.17 / 0.23 | 0.42 / 0.60 | — |
| | two winters | 0.20 / 0.38 | 0.65 / 0.85 | — |

**Q4:** probability of rejecting at 0.00625, the share a Q4 test would have held once its model's Q1 tests had rejected in both windows.

| Model | Test | One winter, 6 points (middle) | One winter, 10 points | Two winters, 6 points | Two winters, 10 points |
| --- | --- | --- | --- | --- | --- |
| Chronos-2 | Q4a | 0.14 / 0.20 | 0.24 / 0.50 | 0.25 / 0.36 | 0.53 / 0.68 |
| | Q4b | 0.15 / 0.21 | 0.27 / 0.45 | 0.17 / 0.34 | 0.56 / 0.80 |
| Timestral | Q4a | 0.17 / 0.23 | 0.31 / 0.49 | 0.18 / 0.34 | 0.57 / 0.76 |
| | Q4b | 0.14 / 0.28 | 0.27 / 0.56 | 0.17 / 0.39 | 0.54 / 0.85 |
| TimesFM 2.5 | Q4a | 0.17 / 0.24 | 0.32 / 0.55 | 0.21 / 0.36 | 0.56 / 0.85 |
| | Q4b | 0.11 / 0.24 | 0.34 / 0.62 | 0.18 / 0.38 | 0.62 / 0.86 |

All effect sizes are in `power.json`. With 200 replications, each cell's standard error is at most about 0.035.

**The rule, fixed in Brief 4's revision 7 before this check was run,** for each prospective test, at its middle effect: (1) at least 80% power on 132 days: confirmatory this winter; (2) otherwise at least 80% on 264 days: confirmatory over two winters; (3) otherwise, descriptive. If 80% lies within the noise estimate's uncertainty, the default is two winters.

**Its application.** No prospective test reaches 80% power on either length at its middle effect. The highest value anywhere, at the optimistic noise, is 0.39. **Every prospective test, Q1 and Q4, is therefore descriptive (branch 3).** The decision does not depend on the uncertainty in *c*: even the optimistic value leaves every test far below 80% on both lengths, so 80% does not lie within the noise estimate's range and the two-winter default does not arise. The rule was applied as written; its thresholds, effect sizes and levels were not changed after the table was seen.

**Why the prospective window is weak.** Winter demand errors vary greatly from day to day, and 132 days is about 19 independent weeks; a second winter adds only as many again. The backtest's 960 days give the confirmatory tests adequate power at a 5% effect at the optimistic noise, and 0.74–0.83 at the cautious one.

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
| Backtest Q1 rejected | "Over the backtest (16 March 2024 to 31 October 2026), weather improved [forecaster]'s day-ahead forecast of GB demand: mean daily error fell by [x]% (error ratio r; 90% interval a–b, descriptive), confirmed at the study's 5% error rate." |
| Backtest Q1 not rejected | "Over the backtest, weather was not shown to improve [forecaster]'s forecast (error ratio r; 90% interval a–b, descriptive)." When the interval lies below 1 but the test does not reject: "Unadjusted, the interval excludes no improvement; at the adjusted level it is not confirmed." |
| Leakage note, after every backtest sentence about a model | "The backtest falls in a period the model may have seen in training. The prospective window, which it cannot have seen, is the descriptive check on this result." |
| Prospective Q1 | "Over the prospective window (20 November 2026 to 31 March 2027), with weather [forecaster]'s error ratio was r (90% interval a–b, descriptive; the window was too short for a confirmatory test). This is [consistent with / a smaller gain than / a larger gain than] the backtest (Section 6.3)." |
| Q2 | "With weather, [model]'s gap to NESO fell by [x] MW (90% interval a–b, descriptive)[, [s]% of the gap]." |
| Q3 | "With the same weather, [model]'s error was r times the regression's (90% interval a–b, descriptive), against [r₁] without weather in rung 1. The regression trains on reanalysis and forecasts with ECMWF; in winter this mismatch is worth about 100–200 MW, so part of any edge may be the regression's." |
| Q4 | "In the prospective window, weather cut [model]'s error by [x] points more in the evening than overnight, and [y] points [less/more] at midday than at other times (90% intervals a–b and c–d, descriptive; not tested, for lack of power)." |

## 10. Procedure

1. Freeze the weather processing and this plan: commit; tag `weather-plan`; write `WEATHER_LOCK.json` (the tag, the hashes of this plan, the private code, the weights, the backtest inputs and every source file, `gbwx`'s package list, seeds, and the power table's hash). Publish this plan and the lock.
2. Start the daily push; first target day 20 November 2026.
3. Score the backtest: run until one attempt passes the gate; test the four Q1 hypotheses; report Sections 6.1, 6.2 and 6.4.
4. After 31 March 2027: score the prospective window as Section 6.3; write the results note with Section 9's sentences; publish the private code the same day; archive with a DOI.

**Deviations:** after the tag, any change to the data, inputs, forecasters, settings, windows, statistics, gate or wording; a second passing run; a result outside Section 6 reported as anything but exploratory; a listed output not reported.

## 11. Decisions fixed in this plan

| # | Decision | Settled as |
| --- | --- | --- |
| 1 | Seeds | Backtest bootstrap 20261120; prospective 20270401; bank-holiday intervals 20261121; power check 20261108 |
| 2 | Confirmatory tests | Four backtest Q1 tests at 0.0125 each, after the procedure removed the descriptive prospective hypotheses (Section 6.1) |
| 3 | Power branches | Every prospective test descriptive, by Brief 4's rule, robust to the noise estimate (Section 7) |
| 4 | The prospective analysis | Pre-specified in Section 6.3, including how agreement with the backtest is read |
| 5 | Model interfaces with weather | Section 4 |
| 6 | Model-cycle changes | Decision 13's rule unchanged; Q1 also reported without the straddling days, descriptively (Sections 3 and 6.4) |
