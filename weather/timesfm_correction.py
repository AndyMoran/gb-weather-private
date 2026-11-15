"""
Correction to the weather study (DOI 10.5281/zenodo.24161799): TimesFM 2.5's forecasts with weather, rerun one day per
call. Specified and committed on 15 November 2026, before it was run.

The error. TimesFM 2.5 with covariates fits one linear in-context regression across every input in a call
(timesfm 3.0.2, utils/xreg_lib.py, BatchedInContextXRegLinear.fit: the targets of all inputs are flattened into one
vector and one coefficient vector is solved). The scored backtest called it with up to 64 consecutive days at a time
(run_weather.model_forecasts, RUN_BATCH). So each day's forecast depended on the other days in its call, and the
regression for day D was fitted partly on the contexts of later days, which contain demand after D's issue time,
including D's own outturn: look-ahead. Found on 15 November 2026 by checkpoint 0's cost check (gb-foundation-model-tests,
checkpoint0/REVISIONS.md), where one-day calls differed from the scored forecasts by up to 5,643 MW on a half-hour.
Chronos-2 (cross_learning=False) and Timestral reproduce one day per call to within 0.01 MW, and rung 1's TimesFM had
no covariates, so neither is affected.

The correction, fixed now:
  1. Check: TimesFM rerun in the scored batches must reproduce the scored forecasts (within 1 MW) on all 960 days, and
     the weather study's evaluation, rerun on the scored forecasts, must reproduce the published Q1-Q3 for every
     forecaster (within 1e-9). Either failing stops the run.
  2. TimesFM 2.5 with weather (the scored configuration, 'xreg + timesfm') is rerun one day per call on every backtest
     day. One day per call is how the forecast would be issued, and no call sees another day's context.
  3. The weather study's evaluation (run_weather.evaluate, unchanged: seed, bootstrap and level as scored) is rerun with
     TimesFM's corrected forecasts and every other forecaster's scored forecasts. Reported side by side for TimesFM:
     Q1, Q2, Q3, Q4 and the scored run's descriptive readings (without fallback days, without cycle-straddle days,
     winter days, bank holidays, mean errors).
  4. Withdrawn, not rerun: TimesFM's figures from batched sensitivity runs (the other xreg mode, unadjusted inputs,
     decision 13 at 20 and 45 days) and its exploratory figures (E1-E3). They are listed in the output as withdrawn.
Direction is not predicted: pooling across days gave the regression more data, which can help, and included later
days, which can only flatter.

Run in gbcp0 (about three hours; resumable: finished days are kept in single_day.jsonl):
    JAX_PLATFORMS=cpu python weather/timesfm_correction.py
Writes ~/gb-weather-data/results/timesfm_correction/: timesfm25_single_day_forecasts.csv, correction.json, correction.md
"""
import json, os, sys, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_weather as R
import models_weather as MW

OUT = os.path.join(R.DATA, 'results', 'timesfm_correction')
WITHDRAWN = ['descriptive.q1_timesfm_other_mode', 'descriptive.q1_unadjusted_inputs.timesfm25',
             'descriptive.q1_decision13_min20.timesfm25', 'descriptive.q1_decision13_min45.timesfm25',
             'exploratory E1 (release split), E2 (block length), E3: TimesFM 2.5 rows']


def close(a, b, tol):
    if isinstance(a, dict): return all(close(a[k], b[k], tol) for k in a if k in b)
    if isinstance(a, list): return all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float): return abs(a - b) <= tol
    return a == b


def main():
    os.makedirs(OUT, exist_ok=True)
    pub = os.path.join(R.PUBLIC, 'weather_study', 'results')
    saved = pd.read_csv(os.path.join(pub, 'forecasts.csv'))
    published = json.load(open(os.path.join(pub, 'results.json')))['results']
    series, pubs, hol, days_inputs, meta, era5_hh, built = R.load_real(full=True)
    rec, excl = R.build(series, hol, pubs, days_inputs, meta, R.load_rung1(), era5_hh, list(pd.date_range(R.FIRST, R.LAST)))
    if len(rec) != 960 or excl: sys.exit(f'expected the scored 960 days; built {len(rec)}, excluded {len(excl)}')
    sv = {pd.Timestamp(d): g for d, g in saved.groupby('day', sort=False)}
    col = lambda c: {D: sv[D][c].to_numpy(float) for D in rec}
    fw_scored = {'chronos2': col('chronos2_weather'), 'timestral': col('timestral_weather'),
                 'timesfm25': col('timesfm25_weather'), 'regression': col('regression_weather')}

    # 1. checks
    res0, _ = R.evaluate(rec, fw_scored, R.SEED_BT, hol)
    for q in ('q1', 'q2', 'q3'):
        if not close(published[q], json.loads(json.dumps(res0[q])), 1e-9):
            sys.exit(f'Check failed: the evaluation rerun on the scored forecasts does not reproduce the published {q}')
    R.say('check: the evaluation reproduces the published Q1-Q3 from the scored forecasts')
    m = MW.get('timesfm25'); t = time.time()
    batched = R.model_forecasts(m, rec)
    dev = max(float(np.max(np.abs(batched[D] - fw_scored['timesfm25'][D]))) for D in rec)
    if dev > 1.0: sys.exit(f'Check failed: TimesFM in the scored batches differs from the scored forecasts by {dev:.3f} MW')
    R.say(f'check: TimesFM in the scored batches reproduces the scored forecasts (max {dev:.4f} MW; {time.time() - t:.0f} s)')

    # 2. one day per call, resumable
    jl = os.path.join(OUT, 'single_day.jsonl'); done = {}
    if os.path.exists(jl):
        for line in open(jl): r = json.loads(line); done[pd.Timestamp(r['day'])] = np.array(r['fc'], float)
    t = time.time(); todo = [D for D in sorted(rec) if D not in done]
    with open(jl, 'a') as fh:
        for k, D in enumerate(todo):
            fc = R.model_forecasts(m, {D: rec[D]})[D]
            if not (np.isfinite(fc).all() and len(fc) == len(rec[D]['y'])): sys.exit(f'non-finite or wrong length on {D:%Y-%m-%d}')
            done[D] = fc; fh.write(json.dumps(dict(day=f'{D:%Y-%m-%d}', fc=[float(x) for x in fc])) + '\n'); fh.flush()
            if (k + 1) % 20 == 0: R.say(f'one day per call: {len(done)} of 960 days ({time.time() - t:.0f} s this session)')
    rows = [dict(day=f'{D:%Y-%m-%d}', start_utc=ts.isoformat(), timesfm25_weather_single_day=float(v))
            for D in sorted(rec) for ts, v in zip(rec[D]['idx'], done[D])]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'timesfm25_single_day_forecasts.csv'), index=False)

    # 3. evaluation with the corrected forecasts
    res1, _ = R.evaluate(rec, {**fw_scored, 'timesfm25': done}, R.SEED_BT, hol)
    res1 = json.loads(json.dumps(res1))
    diff = {D: float(np.max(np.abs(done[D] - fw_scored['timesfm25'][D]))) for D in rec}
    pick = lambda r: dict(
        q1=r['q1']['timesfm25'], q2=r['q2'].get('timesfm25'), q3=r['q3']['timesfm25'], q4=r['q4']['timesfm25'],
        mae_mw=r['descriptive']['mae_mw']['timesfm25'],
        q1_without_fallback_days=r['descriptive'].get('q1_without_fallback_days', {}).get('timesfm25'),
        q1_without_cycle_straddle=r['descriptive'].get('q1_without_cycle_straddle', {}).get('timesfm25'),
        q1_winter_days=r['descriptive'].get('q1_winter_days', {}).get('timesfm25'),
        bank_holidays=(r['descriptive'].get('bank_holidays') or {}).get('timesfm25'))
    out = dict(note='correction to the weather study, specified and committed 15 November 2026 before it was run',
               published=pick(json.loads(json.dumps(res0))), corrected=pick(res1), withdrawn=WITHDRAWN,
               max_abs_difference_mw=dict(max=max(diff.values()), median_of_daily_max=float(np.median(list(diff.values())))),
               others_unchanged=all(close(res0['q1'][k], res1['q1'][k], 0) for k in ('chronos2', 'timestral', 'regression')))
    json.dump(out, open(os.path.join(OUT, 'correction.json'), 'w'), indent=1)
    p, c = out['published'], out['corrected']
    f = lambda r: f"{r['ratio']:.3f} ({r['ci90'][0]:.3f}–{r['ci90'][1]:.3f})"
    L = ['# Correction: TimesFM 2.5 with weather, one day per call', '',
         '| Measure | Published (batched) | Corrected (one day per call) |', '|---|---|---|',
         f"| Mean daily error with weather (MW) | {p['mae_mw']['with_weather']:,.0f} | {c['mae_mw']['with_weather']:,.0f} |",
         f"| Q1: with / without weather | {f(p['q1'])}, p = {p['q1']['p_one_sided']:.4f} | {f(c['q1'])}, p = {c['q1']['p_one_sided']:.4f} |",
         f"| Q2: gap to NESO closed (MW; share) | {p['q2']['mw']:,.0f}; {p['q2']['share_of_gap']:.0%} | {c['q2']['mw']:,.0f}; {c['q2']['share_of_gap']:.0%} |",
         f"| Q3: with weather / regression with weather | {f(p['q3']['with_weather'])} | {f(c['q3']['with_weather'])} |",
         f"| Q1 without fallback days | {f(p['q1_without_fallback_days'])} | {f(c['q1_without_fallback_days'])} |",
         f"| Q1 on winter days | {f(p['q1_winter_days'])} | {f(c['q1_winter_days'])} |", '',
         f"Largest difference between the two forecasts on any half-hour: {out['max_abs_difference_mw']['max']:,.0f} MW; "
         f"median over days of each day's largest difference: {out['max_abs_difference_mw']['median_of_daily_max']:,.0f} MW. "
         f"Other forecasters' results unchanged: {out['others_unchanged']}.", '',
         'Withdrawn, not rerun: ' + '; '.join(WITHDRAWN) + '.', '']
    open(os.path.join(OUT, 'correction.md'), 'w').write('\n'.join(L)); print('\n'.join(L))


if __name__ == '__main__':
    main()
