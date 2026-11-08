# Brief 4: does weather close the gap?

Zero-shot forecasting models and a simple regression, given the same day-ahead weather forecasts, against NESO's forecast of GB electricity demand

Draft for agreement · revision 11 · 8 November 2026 · Andy Moran (Heaviside Analytics)

*Revision 11, 8 November 2026: the history is put on the forecast's scale (decision 13), after a monthly comparison of ECMWF and ERA5. Revision 10, amended 7 November 2026: the bank-holiday prediction becomes a descriptive check with its reading fixed in advance (Section 6). Revision 10 adds what the archive showed about release times: through summer time 2025, ECMWF released the 00:00 UTC run after the 09:30 issue, so on 186 days the target day takes the previous day's run (decision 12). Revision 9 writes decisions 7 to 11 as rules, after review: the splice and its seams, what happens when a run is missing or late, when a source may be switched, what counts as two ERA5 copies agreeing, and the threshold that would change TimesFM's mode. Nothing is left to be decided after a test. Revision 8 records the feasibility check (Section 12) and the changes it forces: the backtest starts on 6 March 2024, the first run with solar radiation; the last ten days of every context come from ECMWF's runs, in the backtest as in the prospective window; ECMWF's Google copy is the primary source; TimesFM's regression-first mode is proposed; and the weather study gets its own software environment. Revision 7 corrects the prospective window's bank holidays, sets one study-wide error rate for the confirmatory tests, adds a temperature-by-wind term to the regression, and states that no real weather-forecast error is computed before the freeze. Revision 6 recorded the decisions on openness and inputs. Revision 5 rewrote revision 4 for plain style, changing no decision. Revisions 2 to 4 answered three reviews; the main changes were a confirmatory test for the main question, reanalysis for weather history, ECMWF's own archive as the weather source, and a scale-free measure for the time-of-day tests.*

**Leakage first.** This study can offer what the demand study could not: a test window in which no day can be in any model's training data. A model cannot be trained on days that had not happened when its weights were fixed, and the pinned revisions of all three models predate November 2026: TimesFM 2.5, 2 October 2025; Chronos-2, 30 October 2025; Timestral, 15 October 2026 (Hugging Face commit dates, checked 5 November 2026; Chronos-2's pinned revision is dated 5 June 2026, but its weights file is unchanged, by hash, since it was added on 30 October 2025). The window runs from 20 November 2026 to March 2027. This rests on the pinned weights' own dates, not on any maker's account of its training data.

## 1. The questions

The demand study (rung 1, DOI [10.5281/zenodo.23962005](https://doi.org/10.5281/zenodo.23962005)) found that NESO's forecast had less than half the error of every model. NESO uses weather; the models and the regression did not. This study gives all of them the same weather and asks four questions, in order of the evidence each can give.

| | Question | Measure | Window | Status |
| --- | --- | --- | --- | --- |
| **Q1** | **Does weather improve each forecaster?** | Ratio of mean daily error with weather to without, for each model and the regression | Backtest; repeated on the prospective window | **Confirmatory** |
| Q2 | How much of the gap to NESO does weather close? | Reduction in the gap, in MW and as a share | Both | Estimate with interval |
| Q3 | Does rung 1's 7–9% edge survive when the regression gets the same weather? | Each model with weather against the regression with weather | Both | Estimate with interval |
| **Q4** | **Is weather's help concentrated in the evening, or weaker at midday?** | Each model's fractional error reduction in one block against others | Prospective | **Confirmatory if powered** |

**One error rate for the whole study.** Every confirmatory test, in both windows and in Q4, is tested together under one graphical multiple-testing procedure (Bretz and colleagues, 2009), with a **study-wide family-wise error rate of 5%**. Running Q1's two windows as separate families at 5% each would let the chance of a false claim about a model approach 10%; one procedure keeps it at 5%. In outline:

- the 5% is first shared equally among the eight Q1 tests: four forecasters (three models and the regression) in each window;
- when a test rejects, its share passes to that forecaster's test in the other window, and, for a model, on to its two Q4 tests (Q4a, evening; Q4b, midday);
- a Q4 test can therefore reject only after its model's Q1 test has rejected.

The exact weights and transitions are fixed in the plan.

**Q4 depends on Q1.** Asking where weather helps makes sense only for a model it helps, and the procedure enforces that: a model's Q4 tests receive no share of the error rate unless its Q1 test rejects. For any other model, Q4 is reported descriptively and cannot be read as evidence that weather helps somewhere.

**If the two windows disagree,** both results are reported side by side. A confirmatory prospective result takes precedence, because no day in it can be in training. A disagreement in sign, or an effect credible in one window and clearly smaller in the other, is itself a finding; the windows' differences in leakage and season are named as possible causes. The disagreement is not resolved by choosing the window that suits.

The backtest carries rung 1's leakage caveat: the models may have seen GB demand in training. A weather gain that appears in the backtest and holds in the prospective window is the strongest result this study can give.

## 2. What is already known

Searched 5 November 2026, on arXiv and Google Scholar, for "time series foundation models electricity load forecasting weather covariates", "foundation model day-ahead load forecasting weather forecast errors" and "Chronos-2 covariates load".

| Study | What it did | What bears on this study |
| --- | --- | --- |
| Kaljevic, Arzola and Zhang (UC Santa Cruz), arXiv 2610.07232, 2026 | TimesFM 2.5, Chronos-2, Moirai 2.0 and TiRex against four trained models; day-ahead load in three US regions; weather with synthetic noise added | With noisy weather, zero-shot Chronos-2 lost its lead to a conventional model (TimesNet) |
| Hertel et al. (KIT), arXiv 2607.15705, 2026 | Ten methods and Chronos-2 on German grid, feeder and customer loads | Chronos-2 "fails to accurately capture special events": large errors around holidays, as in rung 1 |
| Cheong, Jiang and Ng, arXiv 2602.05390 | Foundation models with external data, Singapore and Australia | Weather's benefit was mixed and model-dependent; a simple baseline often won in Singapore's stable climate |
| Virani et al., arXiv 2609.06656, 2026 (summary only; the abstract page could not be reached) | Chronos-2 with calendar and weather, US and European grids | Zero-shot Chronos-2 trailed purpose-built models; strong after fine-tuning |

**The gap.** None of the three studies read in full uses the forecasts actually available at a fixed issue time: they use actual weather, reanalysis or synthetic noise. None scores against a system operator's forecast, none is preregistered, and none covers GB. Kaljevic and colleagues' finding, that realistic forecast errors can reverse a ranking, is why this study uses archived forecasts.

## 3. Honest weather

**Three variables:** temperature at 2 m, wind speed at 10 m, and solar radiation reaching the ground (global horizontal irradiance). On winter evenings there is almost no sunlight, so any evening effect must come through temperature and wind chill. Without wind, the evening test would be weakened before it began.

**The target day's forecast is ECMWF's 00:00 UTC run of D−1,** the run available at the 09:30 issue time. It is the same open-data product the daily collector has saved since 31 October 2026.

- **Source.** ECMWF's portal keeps only its last 12 runs, but ECMWF publishes the same files to AWS, Azure and Google Cloud, which keep them. **The study reads Google's copy.** AWS has throttled requests on and off since November 2025 (ECMWF forum, 5 May 2026), and did so throughout the feasibility check; Google served every request. Each run is fetched whole from one source: Google, or AWS only if Google answers that the run is missing or keeps refusing after the downloader's retries. A source is never chosen by what its data contain. The source of every run is logged, with the hash of its cropped file. The study's own downloader reads each step's index and fetches only the four fields it needs; ECMWF's client library cannot reach runs before 29 February 2024, when the archive's folder layout changed. Licence: CC-BY-4.0, which ECMWF says allows commercial use with attribution.
- **Window.** The open data has been at 0.25° resolution since 1 February 2024, and every 00:00 UTC run from then to 31 October 2026 is in the archive, with steps 24 to 48. The runs before 6 March 2024, however, carry temperature and wind but not solar radiation. **The backtest runs from 6 March 2024 to 31 October 2026, 970 days,** every one with all three variables. Keeping February 2024 would give 34 days different inputs from the other 970; dropping them costs about 3% of the days. The 34 days are excluded from every analysis, confirmatory and descriptive. An earlier start would also mix resolutions. The start date is fixed at the freeze and never moved afterwards.
- **Resolution.** 0.25° (about 25 km) is coarser than the 9 km model named in revision 1. At national scale, averaged over many locations, the difference should be small, and the study's premise is honest, simple inputs, not the best available.
- **Steps.** The forecast hours that cover D in UK time: steps 24 to 48, or 23 to 47 on British Summer Time days, when midnight falls at 23:00 UTC. The data are 3-hourly, so interpolation needs the steps either side: 21 to 48 in all. On a fallback day (below) the run is a day older and the steps 24 hours later: 48 to 72, or 47 to 71 in summer time, so 45 to 72 in all. The rule for interpolating to half-hours is in the splice rule below.
- **Solar units.** ECMWF accumulates solar radiation from the start of the run, in J/m². Differencing between steps and dividing by the interval (10,800 s) gives mean W/m². ERA5 stores each hour's own total instead, so it is divided by 3,600 s without differencing. Mixing up the two conventions would make the splice jump by a factor of three; the synthetic pipeline test checks for it.
- **A trap avoided.** Open-Meteo's Previous Runs API defines `previous_day1` as "predicted 24 hours before valid time". For an evening half-hour of D, that comes from a run issued after 09:30 on D−1. It would leak, and is not used.

**Weather history** feeds the models' eight-week context and the regression's 365 training days.

- **Source: ERA5 reanalysis,** from the Copernicus Climate Data Store (licence allows commercial use with attribution), from January 2023 for the regression's training. It is downloaded one month per request, one request at a time, and every file is hashed. The whole backfill is complete and hashed before the freeze. Files are never downloaded again: recent months are the preliminary ERA5T, which ECMWF may revise, and the study uses the versions it hashed.
- **Fallback: Google's copy of ERA5 (ARCO-ERA5),** for technical failure only. It replaces the CDS for a month only if the CDS has not delivered that month by 14 November 2026. In the prospective window, the daily task needs ERA5 for the UTC day eleven days before the target day, which ERA5 publishes about three days before it is needed. The task asks the CDS for each new day as it appears; if the day is not on disk when the inputs are built, it reads Google's copy; if neither has it, no inputs are pushed and the day falls under Section 5's rule for missed days. The source of every ERA5 day is logged.
- **When two ERA5 copies agree.** Google's copy may stand in for the CDS only if, on the days compared, the grid, hours and units are identical, solar radiation is stored the same way (each hour's own total), and no value differs by more than 0.01 K, 0.01 m/s or 100 J/m² in an hour (0.03 W/m²). The differences are rounding: both copies are converted from ECMWF's original files. One day in each of four months is compared before the freeze: 3 January 2024 (done: at most 0.0009 K, 0.0004 m/s and 35 J/m²), and one day each from mid-2025, June 2026 (the last final ERA5 month) and late October 2026 (ERA5T). If any comparison fails, Google's copy is not used.
- **The last ten days.** ERA5 ran 6½ days behind on 6 November 2026. **In every forecast, backtest and prospective alike, the last ten days of history come from ECMWF's 00:00 UTC runs,** by the rule below. A fixed rule keeps the backtest honest: ERA5 for those days exists now, but did not at the issue time, and a rule that used whatever ERA5 had would give the backtest better history than any live forecast gets. Ten days leaves about three days of margin: at the 09:30 issue on D−1, the last ERA5 hour needed is 9.4 days old, against a lag of 6½. (Revision 8 proposed eight days; that left under one day of margin, too little for a lag that can vary.) If ERA5 falls further behind, the daily task stops and says so. All three models accept such a context (Section 12).
- **A stated limitation.** The history is reanalysis; the target day is a forecast. Every forecaster, NESO included, faces the same mix.

**The splice, defined.** For a target day D in UK time, the forecast is issued at 09:30 UK time on D−1, and its demand history ends at 08:30.

1. **UTC days, then one conversion.** Weather is assembled hour by hour in UTC, then converted to UK half-hours by one routine. A 23- or 25-hour UK day then needs no special case.
2. **Which source.** UTC days up to and including D−11 come from ERA5. The ten UTC days D−10 to D−1 come from ECMWF: each day X from the 00:00 UTC run of X, steps 0 to 24. History is used only up to the 08:30 cut-off.
3. **One run for the end of the history and the whole horizon.** UTC day D−1 and the target day D both come from the 00:00 UTC run of D−1: the history up to the 08:30 cut-off, then the whole horizon, from the cut-off to the end of D (steps 0 to 48). On a fallback day, both come from the run of D−2 (steps 24 to 72).
4. **Interpolation.** Each half-hour takes its value at its midpoint. Wind speed is computed from its two components at each grid point and source time, before weighting. Temperature and wind speed are interpolated linearly in time between each source's own times (hourly for ERA5, 3-hourly for ECMWF), including across the seam at 00:00 UTC on D−10. Solar radiation becomes mean W/m² over each source interval (ERA5: the hour's total ÷ 3,600 s; ECMWF: the difference between steps ÷ 10,800 s); a half-hour takes the mean of the interval that contains it.
5. **The regression's training** uses the same rule relative to D: training days to D−11 from ERA5, D−10 to D−2 from the ECMWF runs.
6. **The context's ERA5 days are put on ECMWF's scale** (decision 13, below). The regression's training is not.

**Putting the history on the forecast's scale (decision 13).** ECMWF's runs and ERA5 differ systematically. Comparing each run's first day (steps 0 to 24) with ERA5 at the same instants, every day from February to October 2026 (`feasibility/feas_6_bias_by_month.py`):

| Month | Temperature, ECMWF − ERA5 (90% interval) | Wind, ECMWF ÷ ERA5 | Solar, ECMWF ÷ ERA5, daylight (90% interval) |
| --- | --- | --- | --- |
| Feb 2026 | −0.29 °C (−0.36 to −0.22) | 0.925 | 1.040 (1.001–1.079) |
| Mar | −0.12 (−0.20 to −0.03) | 0.902 | 1.038 (1.020–1.059) |
| Apr | −0.13 (−0.20 to −0.06) | 0.902 | 1.014 (1.002–1.029) |
| May | +0.07 (−0.03 to +0.16) | 0.911 | 1.035 (1.017–1.053) |
| Jun | −0.01 (−0.10 to +0.08) | 0.909 | 1.079 (1.060–1.099) |
| Jul | +0.33 (+0.23 to +0.41) | 0.901 | 1.064 (1.043–1.090) |
| Aug | +0.04 (−0.05 to +0.14) | 0.906 | 1.065 (1.047–1.084) |
| Sep | −0.16 (−0.22 to −0.08) | 0.886 | 1.104 (1.077–1.129) |
| Oct | −0.19 (−0.24 to −0.12) | 0.887 | 1.110 (1.092–1.131) |

Wind is nearly steady, about a tenth lower in ECMWF (each month's interval about ±0.007). Temperature and solar follow an annual cycle well beyond their intervals (about ±0.08 °C and ±0.02): ECMWF is colder in the cool months and warmer in high summer, and its excess sunshine grows from a few percent in spring to a tenth in autumn. The day-two lead (the previous run's steps 24 to 48) agrees with the first day to within 0.01 °C and 0.005 in either ratio, so one set of factors serves both.

Without a correction, every context carries a step ten days before its end, where ERA5 gives way to ECMWF. The step is not conservative for every question: the solar difference falls at midday, where Q4b is tested, and could make weather look less useful there for a reason unrelated to behind-the-meter solar.

- **The rule.** For each target day D, factors are estimated from the UTC days D−70 to D−11 that both sources cover at the issue time: each run's first day (steps 0 to 24) against ERA5 at the same instants. Temperature: the mean difference, added. Wind: the ratio of totals, multiplied. Solar: the ratio of totals over ERA5-daylight 3-hour intervals (ERA5 mean above 10 W/m²), with ECMWF's 3-hour means from its accumulations and ERA5's from its hourly totals; it multiplies only ERA5 hours above 10 W/m², so night stays zero.
- **Applied to the context only.** The window lies in the same season as the context's ERA5 days (D−56 to D−11), so it follows the cycle above. The regression's training year is not adjusted: a 60-day factor would carry one season's difference across the whole year, and there is no archive before February 2024 from which to estimate 2023's. The regression therefore trains on ERA5 and forecasts with ECMWF; in the winter months this mismatch is about 0.2 °C, roughly 100 to 200 MW of demand. It is stated as a limitation and affects only a reference forecaster, not Q1's comparisons.
- **Minimum data.** At least 30 days per variable; with fewer, that variable is left unadjusted and the day flagged. Only the start of the backtest is affected: solar until about mid-April 2024 (solar radiation enters the runs on 6 March 2024), and temperature and wind briefly in early March, whose window is February 2024 alone. Results at 20 and 45 days are reported descriptively.
- **What it never uses:** demand, or anything from after the issue time.
- **Recording.** Each day's factors, and the inputs both adjusted and unadjusted, are published with the inputs. Q1 is also reported on the unadjusted inputs, descriptively.
- **Self-tested:** a synthetic day whose known 3-hour solar mean is recovered from both sources by their own conventions, with a deliberate mismatch of conventions shown to fail; and known offsets and ratios recovered and removed.

**When a run is missing or late.** Fixed now, applied identically in both windows, and every case listed in the results with its reason:

- **Late counts as missing.** A run is available at the issue time only if every file it needs was on at least one of the three public copies (AWS, Google, Azure) before 09:30 UK time on D−1, which is 08:30 UTC in summer time. Each copy records when each file arrived. The earliest copy is the evidence; the data are downloaded under the source rule above, since the copies hold the same files. A run that is not available at the issue time is treated, for that forecast, exactly as a missing run.
- **Target run missing or late: fall back one run.** D's weather comes from the 00:00 UTC run of D−2, steps 48 to 72 (47 to 71 in summer time; 45 to 72 with interpolation), the latest run public at the issue time. This is what a forecaster at 09:30 had. If that run is also missing or late, D is dropped from every confirmatory analysis.
- **Fallback days are flagged** in the published inputs and in the results. Q1 is reported on all days (confirmatory) and, descriptively, without the fallback days. A two-day-ahead forecast is worse than a one-day-ahead one, so fallback days can only weaken a weather benefit; the power calculation is not adjusted for them, because the direction is conservative.
- **History run missing:** that UTC day is filled from the previous day's 00:00 UTC run, steps 24 to 48, which was issued earlier still. ERA5 never fills it. If that run is missing too, D is dropped.
- **Regression training:** the same fill. A training day that cannot be filled is left out of that day's training set.
- **ERA5 day missing** from both copies: in the backtest, the freeze waits until the backfill is complete; in the prospective window, no inputs are pushed and Section 5's rule applies.
- **Dropped days are dropped for every forecaster,** so all arms are compared on the same days.

**From grid to Great Britain.** Over every 0.25° grid cell covering Great Britain, not a hand-picked list of places. One fixed set of weights serves the whole study; it is built once, hashed, and never tuned.

- **Temperature and wind: residential population.** UKCEH's *UK gridded population at 1 km resolution for 2021 based on Census 2021/2022 and Land Cover Map 2021* (Carnell, Tomlinson and Reis, 2025; doi:10.5285/7beefde9-c520-4ddf-897a-0167e8918595; Open Government Licence). It is built from the census and its total matches it, so it should beat global products such as WorldPop or GHS-POP for Great Britain. UKCEH's workday version was considered and rejected: heating and the evening peak follow where people live, and the workday version would shift weight toward city centres.
- **Solar radiation: installed solar capacity by local authority,** at the end of 2025, from DESNZ's *Renewable electricity by local authority, 2014 to 2025* (Regional Renewable Statistics, 29 September 2026; Open Government Licence). The sheet, column and publication date are recorded in the lock. If the workbook has no capacity sheet by local authority, DESNZ's regional installed-capacity file is used instead, in MW, never generation in MWh.
- **Onto the weather grid.** Each grid point is the centre of a 0.25° cell. Each 1 km population square goes to the cell containing its centre, after converting the centre from the British National Grid to latitude and longitude (EPSG:27700 to EPSG:4326, with pyproj; at 25 km cells, the conversion's metre-scale error cannot move a square between cells except at a cell's very edge). Each local authority's capacity is spread over its land area, and each cell receives the share of that area it contains. Spreading by area is crude: rooftop panels follow housing and solar farms follow fields. At 25 km it is acceptable, and it favours neither; it is stated as a limitation.
- **Northern Ireland is removed.** The population data cover the UK, and Northern Ireland lies inside the weather box but outside GB demand. Each 1 km square is assigned to the nearest of the four countries in ONS's full-resolution country boundaries, and squares nearest Northern Ireland are dropped. A rule of "keep only squares inside Great Britain" would lose populated coastal squares whose centres fall just offshore. Northern Ireland's local authorities are dropped from the solar weights.
- **Both weights are shares,** summing to one over Great Britain.

**Checking the archive.** From 31 October 2026, the archive's runs, cropped as the collector crops them, must equal the collector's saved files value for value. That ties the backtest's source to an independently timestamped record. (Hashes of the cropped files would differ on metadata such as creation times, so the check compares values.)

## 4. The forecasters, with weather

| Forecaster | How weather enters | Feasibility check confirms |
| --- | --- | --- |
| Chronos-2 | As known future inputs, natively | Confirmed in the pinned `chronos-forecasting` 2.3.2, with `cross_learning` still off |
| TimesFM 2.5 | Through XReg, a linear regression on the weather fitted within each forecast and combined with the model's output | Confirmed in the pinned `timesfm` 3.0.2, with JAX added (CPU build). Proposed mode: regression first (`xreg + timesfm`); see below |
| Timestral | As known future inputs (`known_covariates`, spanning context and horizon) | Confirmed in the pinned `timestral` 1.0.0 |
| Regression | Rung 1's terms (lags, day of week, bank holiday, Christmas period) plus, for each half-hour slot, centred temperature and its square, wind, temperature × wind (for wind chill), and solar radiation | Re-estimated daily on the 365 days to D−2, as before |
| NESO | Unchanged: its forecast as published at 09:30 | — |

**TimesFM's mode.** On synthetic demand with a known weather response, regression first moved the forecast by 2,580 MW when the horizon turned 4 °C colder and 4 m/s windier (true effect 2,720 MW). TimesFM first moved it by 112 MW. TimesFM's fit to the context absorbs slow weather-driven swings as history, leaving the regression little to learn. The synthetic response was linear, which favours regression first by construction. **The mode is therefore regression first, unless a second synthetic test, specified here, reverses it:**

- **Scenarios.** Synthetic GB-like demand (mean about 28 GW, daily swing about 10 GW, weekends lower) with a realistic weather response: 700 MW per °C below 15.5 °C and 100 MW per °C above it; wind adding 40 MW per m/s below 15.5 °C; solar radiation subtracting 6 MW per W/m² (behind-the-meter solar); and serially correlated noise (standard deviation 400 MW, correlation 0.9 between half-hours). Three scenarios: winter (temperatures mostly 0–10 °C) with forecast error; the shoulder season (5–18 °C, crossing the threshold) with forecast error; and winter with perfect weather. Forecast error adds noise with standard deviations of 1.0 K, 1.5 m/s and a quarter of the solar value to the target day's weather only.
- **Design.** In each scenario, 20 synthetic series (seeds 20261106 to 20261125), each forecast at 30 origins one day apart, through the study's own pipeline: eight-week context, the splice, the interpolation and the target-day horizon. Score: mean absolute error over D's half-hours.
- **The threshold.** TimesFM first replaces regression first only if, in every scenario, its mean error is at least 5% lower and its worst days (the 95th percentile of daily error) are no worse. Anything less, including a mixed or ambiguous result, leaves regression first.
- **Order.** The test code, scenarios, seeds and threshold are committed and hashed before it runs; the result is committed before any real weather-forecast error is computed. No real demand enters it.
- **The other mode** is run on the real data too and reported descriptively, in an appendix, outside the testing procedure. It never replaces the confirmatory mode, whatever it shows.

With regression first, TimesFM in this study is a hybrid of a linear weather regression and the model; its Q1 and Q3 results are reported with that caveat.

**A separate environment.** The weather study runs in `gbwx`, a clone of the demand study's `gbfm` with JAX and scikit-learn added and every existing package pinned; the install changed nothing already there. `gbfm` makes the prospective no-weather forecasts each morning and stays untouched. The full package list of `gbwx` goes in the lock.

All three models accept weather, so none is dropped. The regression gets a temperature-by-wind term so that it can represent wind chill, which the evening hypothesis invokes; without it, Q3 would be tilted against the regression in exactly the block the study cares about. TimesFM's route is partly a linear regression; its result is reported with that caveat.

**The no-weather forecasts already exist.** On the backtest they are rung 1's saved forecasts, reused rather than rerun; a gate check confirms they match rung 1 exactly over the overlap, 6 March 2024 to 31 October 2026. On the prospective window they are the forecasts already made each day under the frozen prospective plan. The weather forecasts use exactly the same demand inputs, including rung 1's one filled half-hour.

**Embedded solar is left out.** NESO publishes its own forecasts of solar and wind generation behind the meter, archived since 2019. Supplying them would add NESO's modelling, not just weather (decision 4).

## 5. When to freeze

The prospective no-weather forecasts are already being made, and outturn is public, so their errors could be computed at any time. An analyst who saw them could, in principle, shape the weather processing to suit. Three steps close that door.

1. **Freeze the weather processing first.** The code that turns ECMWF's runs and ERA5 into national weather inputs is frozen and hashed by about 19 November 2026, before any weather forecast is made.
2. **Publish the weather inputs daily.** From then on, a daily task computes each target day's national inputs on D−1 and pushes them, with hashes, at about 09:45, alongside the forecasts. A day counts only if its inputs reach GitHub before 00:00 UK time on D, by GitHub's server record: the same rule as the prospective forecasts, and for the same reason, since D's outturn cannot exist before D begins. **If a push misses that deadline,** the day is dropped from the confirmatory analysis; its inputs are rebuilt from the archive, logged as a deviation, and the day is reported descriptively. A rebuilt day is never counted as if it had been published on time.
3. **Start this study's prospective window the next day:** 20 November 2026 to 31 March 2027, **132 days, about 19 weekly blocks,** against 147 days and 21 blocks from 5 November. Earlier days are reported descriptively. The shorter window costs power (Section 7).

**No real weather-forecast error, in the backtest or the prospective window, is computed before the analysis plan is frozen.** Before the freeze the only computations are on synthetic data, with a stand-in forecaster, and the noise estimate for the power checks (Section 7). The analysis plan follows, frozen before any such error is computed. It states that outturn had been public but no forecast errors computed. The prospective weather forecasts themselves are computed later, from the archived inputs. Their protection is the public inputs and the freeze, not real-time computation, and the study says so.

## 6. Is weather's help concentrated? (Q4)

**What rung 1 shows.** From rung 1's published results (`results/attempt2/results.json`), the gap to NESO by block:

| Block (UK time) | NESO's error | Chronos-2's gap | Timestral's gap | TimesFM 2.5's gap | Models' error ÷ NESO's |
| --- | --- | --- | --- | --- | --- |
| 00:00–04:00 | 415 MW | 461 MW | 455 MW | 606 MW | 2.1–2.5 |
| 12:00–16:00 | 1,052 MW | 1,198 MW | 1,133 MW | 1,258 MW | 2.1–2.2 |
| 16:00–20:00 | 762 MW | 800 MW | 745 MW | 916 MW | 2.0–2.2 |

In megawatts the gap is largest at midday. As a ratio it barely changes: the models' error is about twice NESO's in every block. **That flat ratio matters.** If weather explained NESO's lead, and weather mattered more at some hours, the gap should be proportionally larger at those hours. It is not. Temperature affects demand at every hour, so this does not refute the weather explanation, but it lowers the odds of a strong time-of-day effect. Hence Q1 is primary and Q4 a refinement.

**Choosing the measure.** Three measures are possible, and each answers a different question.

| Measure, for each block | What it asks | Use |
| --- | --- | --- |
| **Fractional error reduction:** (model's error without weather − with) ÷ error without | Did weather cut the model's own error by a larger fraction here? | **Q4** |
| Megawatts: error without weather − with | Where does weather buy the most accuracy? The trader's number | Reported, all six blocks |
| Share of the gap closed: (error without − with) ÷ (error without − NESO's error) | How much of NESO's lead does weather remove here? | Reported (Q2) |

Megawatts cannot measure concentration: any proportional effect gains more megawatts where errors are larger. The share of the gap depends on NESO's error too, and ranks blocks like the fractional reduction only because rung 1's gap happens to be about half the model's error everywhere. The fractional reduction needs neither NESO nor that coincidence. Under it, "weather helps equally" means the same fraction of the error in every block, the natural baseline when the models' errors run from about 870 MW overnight to 2,250 MW at midday.

**Blocks.** Six four-hour blocks of UK clock time, from midnight, fixed by rule rather than from rung 1's curve. The overnight block, 00:00–04:00, holds the daily minimum, when weather matters least.

**Q4a, the evening.** Winter evening demand, 16:00–20:00, is the most weather-sensitive of the day: heating, lighting and the evening ramp all respond to temperature, wind chill and light. **Hypothesis:** weather cuts each model's error by a larger fraction in the evening than overnight. This comes from physics, not from rung 1, whose time-of-day finding concerned the regression, not weather.

**Q4b, midday.** Midday, 12:00–16:00, has the largest errors, but much of them probably comes from solar generation behind the meter, which the grid does not see and irradiance forecasts capture only roughly. **Hypothesis:** weather cuts each model's error by a smaller fraction at midday than in the other five blocks together. If weather helps as much or more at midday, that counts against the behind-the-meter explanation.

Each is one-sided, one test per model, inside the study-wide procedure (Section 1), so a model's Q4 tests count only if its Q1 test has rejected. Together they answer "why the evening and not midday?" before anyone asks: opposite predictions, for opposite physical reasons, and each can fail.

**Reported, not tested:**

- **Bank holidays.** 21 England and Wales bank holidays fall in the backtest (Scotland's differ), and five in the prospective window: 25 December, 28 December (the Boxing Day substitute, since 26 December 2026 is a Saturday), 1 January, and Easter (26 and 29 March 2027). The regression's calendar comes from the GOV.UK bank-holidays file, downloaded once and hashed in the demand study's lock, never from typed dates; the feasibility check confirms these five are in it. Too few and too varied for a confirmatory test, but the prediction can still be checked, descriptively and without giving the models a calendar.
  - **Prediction:** weather will not close the bank-holiday gap, because that gap is the calendar.
  - **The check.** For each foundation model, in each arm, the *holiday excess*: mean daily error on bank holidays ÷ mean daily error on ordinary days, minus one. Ordinary days are neither bank holidays nor in rung 1's Christmas period. Both arms use the same days, so weather is the only difference. The excess is a ratio, so it allows for weather lowering the error on ordinary days too. The backtest (21 holidays) and the prospective window (5) are reported separately and pooled; fallback days are included and flagged.
  - **References.** The same excess for the regression, which has a calendar, and for NESO. In rung 1 it was 65–80% for the models, 16% for the regression and 14% for NESO.
  - **The reading, fixed now.** The prediction holds for a model if its excess with weather is at least two-thirds of its excess without, and still at least twice the regression's. It fails if its excess with weather falls to the regression's or below. Anything between is reported as mixed.
  - **Intervals.** 90% intervals from resampling holidays and ordinary days separately (2,000 draws, seed fixed in the analysis plan), shown but not tested. The check sits outside the testing procedure and cannot affect any confirmatory result.
- **Season.** The prospective window is winter only, so Q4 measures winter effects and is described that way.
- **Share of the gap,** reported only where the gap without weather exceeds 100 MW, and always beside the change in MW.

## 7. Power

Every power check uses the block bootstrap, with 7-day blocks, on rung 1's errors. The size of any weather effect is unknown, so the checks assume a range. The noise between the two arms is estimated in the feasibility check, from synthetic inputs and from the spread between rung 1's similar forecasters, never from the backtest's weather forecasts. A first rough simulation proved too crude to report.

| Test | Window | Effect sizes in the power table |
| --- | --- | --- |
| Q1 | Backtest, 970 days | Weather cuts error by 2%, 3%, 5%, 7% and 10% |
| Q1, prospective | 132 days, and 264 days (two winters) | The same |
| Q4a, evening | Both prospective lengths | The evening's fractional reduction exceeds overnight's by 2, 4, 6, 8 and 10 points |
| Q4b, midday | Both prospective lengths | Midday's fractional reduction falls short of the other blocks' by 2, 4, 6, 8 and 10 points |

**The rule for each prospective test, fixed now:**

Evaluated in this order:

1. **One winter:** at least 80% power on 132 days at the middle effect (Q1: 5%; Q4: 6 points). Confirmatory this winter.
2. **Two winters:** otherwise, at least 80% power on 264 days. Confirmatory in April 2028, with a descriptive interim read in April 2027.
3. **Descriptive:** otherwise.

**The branch is chosen at the plan's freeze,** from the table computed then: not earlier, because the noise is not yet known, and never after a prospective error is seen. If the noise estimate is too uncertain to call (if 80% lies within its uncertainty), the default is **two winters,** not a choice made on weak numbers. The prospective bootstrap has only about 19 weekly blocks, so its intervals and power are coarse; that is why the two-winter option exists. Two winters cost only time, because the forecasts and the weather collection are already running. Choosing them would mean no confirmatory prospective result until April 2028: a legitimate outcome of the design, decided before any result, not a failure.

## 8. Gate checks

As in rung 1, the official run is the first to pass every check. Before the freeze, every check is run on the real inputs with a stand-in forecaster, and shown to fail on corrupted input. Rung 1's checks carry over: data hashes, no look-ahead, model outputs (finite, exact length, repeatable), responsiveness, an independent regression, and weights hashes. Four are added:

1. **The archive matches the daily collection,** value for value, for every day since 31 October 2026.
2. **The no-weather forecasts equal rung 1's** on every shared backtest day.
3. **The weather inputs used equal those published,** by hash, for every day.
4. **No weather input postdates its issue time:** every target-day value comes from the 00:00 UTC run of D−1, and every history value from before the cut-off.

## 9. What this study cannot establish

- **Freedom from leakage in the backtest.** As in rung 1; the prospective window answers it.
- **Why NESO leads.** Whatever weather leaves unexplained belongs to NESO's modelling, data and judgement, which this study cannot separate.
- **What better weather inputs could do.** One weather model, three variables, fixed weightings. Better choices might do better; this study tests honest, simple ones.
- **Weather within Great Britain.** The national inputs are fixed-weight averages. They do not capture changes in where people live or where panels are over the study period, and they reduce the weather's pattern within Great Britain to one number per variable.
- **The regression's weather scale.** The regression trains on ERA5 and forecasts with ECMWF, whose temperature differs by up to 0.3 °C with the season (Section 3). Its forecasts carry that mismatch; the foundation models' contexts do not.
- **Solar weights from a later year.** They use end-2025 capacity for every day, including 2024. That is later information about where panels are, not about how much they generated or about demand; the weights are shares, and change little from year to year. Weights for each year were considered and rejected as adding complexity for no material gain.
- **Weather as known at the time, for history.** Reanalysis is cleaner than real-time observation. It flatters every forecaster equally.
- **Higher resolution.** ECMWF moved its open data to 0.1° on 20 October 2026; the study keeps 0.25° throughout, to match the backtest.
- **Fine-tuning.** Zero-shot, by design.
- **More than dates can show about leakage.** The prospective window's protection rests on the pinned revisions' commit dates, which are checked. It needs nothing from the makers' training data, which they do not fully disclose, but it also says nothing about the backtest.
- **Value.** Accuracy, not money.

## 10. How open to be

A preregistration needs a verifiable timestamp, not a public method. Three options were considered:

| | Public at the freeze | Private until the results |
| --- | --- | --- |
| **A. Hybrid (chosen)** | The plan; the daily national weather inputs, with hashes; the forecasts; the results | The weather-processing code, hashed at the freeze so it cannot change |
| B. Sealed | Hashes of the plan and all code | Everything; "checkable" means checkable at the reveal |
| C. Embargoed | A Zenodo record with DOI and timestamp | The files, until the embargo date |

**A is the only option in which a reader can check, during the study, what the models actually received.** The national inputs are public as they are made, and so is the raw ECMWF archive they come from. The code that links the two is private only until the results, and its hash cannot change. At the reveal, code, hashes, raw data and published inputs close the loop exactly. B and C defer every check.

**Three commitments make A checkable:**

1. **The reveal date.** The weather-processing code is published with the results note, on the same day.
2. **The reason, stated.** The weather-processing code is the author's own method, which he may use in paid work. It is hashed at the freeze and published with the results.
3. **Inputs published before the target day.** Each day's inputs reach GitHub before 00:00 UK time on D, with the rule for missed days in Section 5.

**Before the plan commits to A,** the weather pipeline runs once end to end on a synthetic day and publishes its result to the public repository. If that fails, A's daily burden is real, and that is better known before the freeze.

## 11. Decisions

Taken 5 November 2026:

1. **Openness:** A, with the three commitments in Section 10.
2. **Weather variables:** temperature, wind and solar radiation.
3. **Spatial weighting:** a population- and capacity-weighted grid, with sources proposed in the feasibility check and fixed at the freeze.
4. **Embedded solar:** left out of the main design; a labelled extra arm remains possible.
5. **TimesFM's XReg mode:** chosen after the feasibility check, before any weather forecast, with its reason stated.
6. **Licences:** none outstanding. ECMWF's open data and ERA5 both allow commercial use with attribution. Open-Meteo only verifies the archive and never feeds the results.

Proposed in revisions 8 to 11, each as a fixed rule; 7 to 12 agreed on 6 November 2026, 13 proposed on 8 November:

7. **Backtest start:** 6 March 2024, the first run with solar radiation; 970 days, inclusive, to 31 October 2026. February 2024 is excluded from every analysis (Section 3).
8. **Splice:** a fixed ten days from ECMWF's runs, in every forecast and in the regression's training, with the seam, interpolation and missing-run rules in Section 3.
9. **ECMWF source:** Google first; AWS only on technical failure; one source per run; logged.
10. **ERA5 source:** the CDS; Google's copy only on technical failure, and only where the agreement check passes (Section 3).
11. **TimesFM's mode:** regression first, unless the specified synthetic test meets the reversal threshold in every scenario (Section 4). Settled on 6 November 2026: the test kept regression first (Section 12).
12. **Late runs:** a run released after the issue time counts as missing; the target day falls back to the previous day's run, is flagged, and stays in the confirmatory analysis. The 186 fallback days of summer 2025 are reported without, descriptively. The prospective window is unaffected: release at 07:34 UTC leaves 56 minutes even after the clocks change on 28 March 2027.
13. **History on the forecast's scale:** the context's ERA5 days are put on ECMWF's scale with factors from D−70 to D−11 (temperature added, wind and solar multiplied), at least 30 days each; the regression's training is not adjusted, a stated limitation (Section 3). Proposed 8 November 2026, after the monthly check the reviewer set as its gate.

The Copernicus account is registered and working.

## 12. Feasibility check and next steps

**Results,** 5–6 November 2026, with no model forecasting real demand. The scripts are in the private repository's `feasibility/` folder.

| Check | Result |
| --- | --- |
| Archive coverage, 1 Feb 2024 – 31 Oct 2026 | All 1,004 runs present, every step from 24 to 48. Folder layout changes on 29 February 2024 |
| Variables in each run | Temperature and wind throughout; solar radiation from the 6 March 2024 run, then every run: 970 days with all four fields |
| Depth: four sample runs (Feb 2024, Mar 2024, Jun 2025, Jan 2026) | Each file's own metadata names the run requested; every step present |
| Archive against the daily collector, 31 Oct – 4 Nov 2026 | Identical, value for value, on all five variables |
| Sources | Google served every request; AWS returned 503 "SlowDown" throughout |
| ERA5, January 2024 | Request works, about 3 minutes; 744 hours on the 49 × 47 box; nothing missing; units and ranges plausible (solar peak 359 W/m², near the clear-sky value for a January noon) |
| ERA5 lag | Ends 31 October 2026 00:00 UTC, read on 6 November: 6½ days |
| Chronos-2, synthetic | Uses the weather: forecast moves 2,759 MW against a true 2,720; error 267 MW with weather, 436 without |
| Timestral, synthetic | Moves 2,771 MW; error 247 MW with weather, 603 without |
| TimesFM 2.5, synthetic | Regression first moves 2,580 MW, error 282 MW against 679; TimesFM first moves 112 MW, error 690 |
| Spliced context | Runs in all three models |
| Steps 0–48, every run | Present in all 1,004 runs |
| Release time against the 09:30 issue | ECMWF releases the 00:00 UTC run to all three copies at the same minute each day: 06:59 UTC to March 2024, 07:55 from April 2024, 08:34 from January 2025, 07:34 from November 2025. In winter that beats the 09:30 UTC issue every day. In summer time 2025 (issue 08:30 UTC) it did not: 186 days, 30 March to 1 October 2025, take the fallback run. Two other days (18 September 2024, 25 February 2025) look late on AWS and Google, which were re-uploaded; Azure had both before the issue, so they count as on time |
| Fallback runs (00:00 UTC of D−2) for those 186 days | All 186 have steps 45–72 and were public before their issue time |
| Synthetic end-to-end test (`weather/synth_test.py`, run 6 Nov 2026 from commit 923192f; `results/synth_result.json`) | Pipeline: all five checks pass (ERA5 and ECMWF history equal the truth; a run whose accumulation is not differenced is caught; the horizon comes from the run of D−1, or of D−2 when D−1 is late; every half-hour filled across the clock change). TimesFM's mode, mean daily error over 600 forecasts, regression first / TimesFM first / no weather: winter with forecast error 1,127 / 1,409 / 1,534 MW; shoulder with forecast error 1,294 / 1,357 / 1,414; winter with perfect weather 1,098 / 1,407 / 1,534. TimesFM first was worse in every scenario, at the mean and the 95th percentile. **Decision 11 settled: regression first** (`xreg + timesfm`). On 7 November, writing the daily push exposed three faults in `national.py`, none reachable by the synthetic data: times in different units (pandas holds the periods in microseconds, files in nanoseconds), missing values able to pass as zero, and a solar total sometimes absent at step 0. All three were fixed; the pipeline checks reran with identical results to every decimal, and run again at the freeze on the final code |
| Spatial weights (`weather/make_weights.py`, 6 Nov 2026) | Population: UK 66.94 m, Northern Ireland 1.90 m dropped, GB 65.04 m on 706 cells; 2,447 coastal squares lie outside every outline, the furthest 658 m offshore. Solar: GB 21,352 MW in 350 authorities, all placed, on 740 cells; Barnsley and Sheffield recoded to ONS's 2025 codes; 164 MW "Unallocated (GB)" has no location and is not used. Largest cells: London, Manchester, Birmingham (population); Kent, Wiltshire, Cambridgeshire (solar). `weights.nc` SHA-256 c3af7721b673386ef31285ff55deab71617bff774b7cb5c13e448dee543102be; input hashes in `weights_meta.json` |
| ERA5: Google's copy against the CDS, 3 Jan 2024 | Same grid, hours, units and solar convention; at most 0.0009 K, 0.0004 m/s, 35 J/m². Google's store is chunked one global hour at a time: a GB day takes about 2 minutes to read |
| Bank holidays | The five prospective-window dates are in the hashed GOV.UK file |

The synthetic errors show that each model uses the weather. They do not predict how much weather will help on real demand: the synthetic response is simple and its noise known.

**Still to do before the freeze:**

1. **Download ERA5** from January 2023 (running since 6 November, newest month first), and run the three remaining agreement checks against Google's copy.
2. **Check Virani et al. (2609.06656)** on arxiv.org, and cross-check a few archive runs against Open-Meteo's Single Runs API, allowing for resolution.
3. **Publish the synthetic test's result** to the public repository (done privately on 6 November; the code stays private until the results note).
4. **Freeze the weather processing** and start the daily push of national weather inputs.
5. **Estimate the noise for the power checks,** then write and freeze the analysis plan, with both power checks and every gate check run on the real inputs with a stand-in forecaster, before any prospective error is computed.

**Sources,** read 5–6 November 2026: the four studies in Section 2; ECMWF's open data page and its 29 February 2024 announcement of the expanded open data; ECMWF's dissemination schedule; the ECMWF forum threads on 503 errors from the AWS bucket and on the Data Stores Service protection measures; the google-research/arco-era5 README; the Herbie documentation on ECMWF archive sources; Open-Meteo's Single Runs, Previous Runs and Historical Forecast API documentation; Mistral's Timestral announcement (15 October 2026); timecopilot-timestral on PyPI; the google-research/timesfm README; NESO's Embedded Wind and Solar Forecasts page. Rung 1's figures by block come from `results/attempt2/results.json` in the demand study's repository.
