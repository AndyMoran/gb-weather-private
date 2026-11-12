"""
Exploratory check E5: the foundation models given the calendar. Specified and committed on 12 November 2026, before it
was run; exploratory, not in the plan, and it changes no confirmatory result.

Why. The scored backtest gave the regression a calendar (day of week, England and Wales bank holidays, 24 December to
2 January) and the models none. E4 showed that this asymmetry accounts for part of Q3: on the 35 calendar days the
regression gains far more from weather than the models. E5 gives Chronos-2 and Timestral the regression's calendar and
asks how the comparison then reads.

Design, fixed before any forecast is made:
  Models      Chronos-2 and Timestral, at the frozen revisions and settings (models_weather.py, unchanged).
              TimesFM 2.5 is left out: it estimates covariate effects by a linear regression on its own eight-week
              context, which rarely contains a bank holiday and never a Christmas, so it cannot learn a calendar.
  Calendar    the regression's own terms, as covariates for every half-hour of context and horizon, by the local day
              the half-hour falls in: six day-of-week indicators (Monday the base), bank holiday (the study's GOV.UK
              file), and 24 December to 2 January. Known in advance, so they are given over the horizon too.
  Arms        A  calendar + weather (the adjusted weather inputs, as scored), the main arm;
              B  calendar only, no weather.
  Days        the 960 backtest days, as scored; the forecasts without weather and the regression's are the scored
              run's, read from its forecasts.csv.
  Statistics  ratios of mean daily error with the plan's bootstrap (blocks of 7 days, 2,000 draws, seed 20261120),
              90% percentile intervals; all descriptive. Reported for all days, ordinary days (E4's 925) and the
              35 calendar days:
                - the calendar's gain:    A / the scored model with weather;
                - Q3 with the calendar:   A / the regression with weather;
                - Q3 without weather:     B / the regression without weather;
                - Q1 with the calendar:   A / the model without weather.
  Check       before the arms run, each model is rerun with the scored weather inputs alone on 6 days, and must
              reproduce the scored forecasts to within 1 MW (batching can differ in the last digits).

Expectation, stated before running: on the 35 calendar days the models' errors fall towards the regression's; on
ordinary days little changes, since the models already infer the weekly cycle from their context. If so, Chronos-2's
all-days ratio to the regression with weather moves from 0.966 towards its ordinary-day 0.940.

Run in `gbwx` (about an hour or two on CPU):   JAX_PLATFORMS=cpu python weather/calendar_study.py
Writes, beside the scored run: calendar_forecasts.csv, calendar.json, calendar.md
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_weather as R                                       # loading, building, batching and the bootstrap, unchanged
import models_weather as MW

WX = tuple(R.VARS)
CAL = ('dow_tue', 'dow_wed', 'dow_thu', 'dow_fri', 'dow_sat', 'dow_sun', 'bank_holiday', 'christmas')
MODELS = ('chronos2', 'timestral')
NAMES = {'chronos2': 'Chronos-2', 'timestral': 'Timestral', 'regression': 'Regression'}


def calendar(periods, hol):
    loc = pd.DatetimeIndex(periods).tz_convert('Europe/London'); day = loc.normalize().tz_localize(None)
    dow = loc.dayofweek.to_numpy()
    c = {f: (dow == k).astype(float) for k, f in zip(range(1, 7), CAL[:6])}
    c['bank_holiday'] = np.array([d in hol for d in day], float)
    c['christmas'] = np.array([(d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 2) for d in day], float)
    return c


def run_arm(model, rec, names, cov_key):
    R.VARS = names; MW.VARS = names                           # the wrappers and the batcher read the covariate list
    try:
        return R.model_forecasts(model, rec, cov_key=cov_key)
    finally:
        R.VARS = WX; MW.VARS = WX


def ratio(a, b, idx):
    return R.stat_q1(a, b, idx)


def main():
    run = os.path.join(R.OUT, 'backtest_attempt1')
    saved = pd.read_csv(os.path.join(run, 'forecasts.csv'))
    sv = {d: g for d, g in saved.groupby('day', sort=False)}
    series, pubs, hol, days_inputs, meta, era5_hh, built = R.load_real(full=True)
    hol = {pd.Timestamp(h).normalize() for h in hol}
    days = list(pd.date_range(R.FIRST, R.LAST))
    rec, excl = R.build(series, hol, pubs, days_inputs, meta, R.load_rung1(), era5_hh, days)
    if len(rec) != 960 or excl: sys.exit(f'expected the scored 960 days; built {len(rec)}, excluded {len(excl)}')
    for D, r in rec.items():
        cal = calendar(r['wx'].index, hol)
        r['cov_cw'] = {**r['cov'], **cal}; r['cov_c'] = cal

    models = {name: MW.get(name) for name in MODELS}
    # the check: the scored weather-only forecasts are reproduced
    probe = {D: rec[D] for D in list(rec)[::160]}
    for name, m in models.items():
        fc = R.model_forecasts(m, probe)
        dev = max(np.max(np.abs(fc[D] - sv[f'{D:%Y-%m-%d}'][f'{name}_weather'].to_numpy())) for D in probe)
        if dev > 1.0: sys.exit(f'Check failed: {name} rerun differs from the scored forecasts by {dev:.3f} MW')
        R.say(f'check: {name} reproduces the scored forecasts on {len(probe)} days (max difference {dev:.4f} MW)')

    fcs = {}
    for name, m in models.items():
        fcs[name, 'A'] = run_arm(m, rec, WX + CAL, 'cov_cw'); R.say(f'{name}: arm A (calendar + weather) done')
        fcs[name, 'B'] = run_arm(m, rec, CAL, 'cov_c'); R.say(f'{name}: arm B (calendar only) done')
        for arm in ('A', 'B'):
            bad = [D for D in rec if not (np.isfinite(fcs[name, arm][D]).all() and len(fcs[name, arm][D]) == len(rec[D]['y']))]
            if bad: sys.exit(f'{name} arm {arm}: non-finite or wrong length on {len(bad)} days')

    ds = list(rec); n = len(ds); keys = [f'{D:%Y-%m-%d}' for D in ds]
    y = {k: sv[k]['outturn'].to_numpy() for k in keys}
    if not all(np.allclose(y[k], rec[D]['y']) for k, D in zip(keys, ds)): sys.exit('outturn differs from the scored run')
    mae = lambda f: np.array([np.abs(y[k] - f(D, k)).mean() for D, k in zip(ds, keys)])
    dm = {(name, arm): mae(lambda D, k: fcs[name, arm][D]) for name in MODELS for arm in ('A', 'B')}
    for col in ('chronos2_weather', 'timestral_weather', 'regression_weather', 'chronos2_no_weather',
                'timestral_no_weather', 'regression_no_weather'):
        dm[col] = mae(lambda D, k: sv[k][col].to_numpy())
    special = np.array([(D.normalize() in hol) or (D.month == 12 and D.day >= 24) or (D.month == 1 and D.day <= 2) for D in ds])
    parts = {'all days': np.arange(n), 'ordinary days': np.where(~special)[0], 'calendar days': np.where(special)[0]}
    idx = {p: R.boot_idx(len(ii), R.SEED_BT) for p, ii in parts.items()}

    out = dict(note='exploratory check E5, specified and committed before it was run (12 November 2026)',
               calendar_days=int(special.sum()), results={}, mae_mw={})
    for name in MODELS:
        comps = {'calendar gain (A / model with weather)': (dm[name, 'A'], dm[f'{name}_weather']),
                 'Q3 with calendar (A / regression with weather)': (dm[name, 'A'], dm['regression_weather']),
                 'Q3 without weather (B / regression without weather)': (dm[name, 'B'], dm['regression_no_weather']),
                 'Q1 with calendar (A / model without weather)': (dm[name, 'A'], dm[f'{name}_no_weather'])}
        out['results'][name] = {c: {p: ratio(a[ii], b[ii], idx[p]) for p, ii in parts.items()} for c, (a, b) in comps.items()}
        out['mae_mw'][name] = {lab: {p: float(v[ii].mean()) for p, ii in parts.items()} for lab, v in
                               (('A calendar + weather', dm[name, 'A']), ('B calendar only', dm[name, 'B']),
                                ('weather (scored)', dm[f'{name}_weather']), ('no weather (scored)', dm[f'{name}_no_weather']))}
    out['mae_mw']['regression'] = {lab: {p: float(dm[c][ii].mean()) for p, ii in parts.items()}
                                   for lab, c in (('weather (scored)', 'regression_weather'), ('no weather (scored)', 'regression_no_weather'))}
    json.dump(out, open(os.path.join(run, 'calendar.json'), 'w'), indent=1)
    rows = [dict(day=k, start_utc=t.isoformat(), **{f'{name}_{arm}': fcs[name, arm][D][i] for name in MODELS for arm in ('A', 'B')})
            for D, k in zip(ds, keys) for i, t in enumerate(rec[D]['idx'])]
    pd.DataFrame(rows).to_csv(os.path.join(run, 'calendar_forecasts.csv'), index=False, float_format='%.3f')

    fmt = lambda r: f"{r['ratio']:.3f} ({r['ci90'][0]:.3f}–{r['ci90'][1]:.3f})"
    md = ['# Exploratory check E5: the models given the calendar', '',
          f'Specified before running (weather/calendar_study.py). Calendar days: {int(special.sum())}. Ratios of mean daily error, 90% intervals.', '']
    for name in MODELS:
        md += [f'## {NAMES[name]}', '', '| Comparison | All days | Ordinary days | Calendar days |', '| --- | --- | --- | --- |']
        for c, v in out['results'][name].items():
            md.append(f"| {c} | {fmt(v['all days'])} | {fmt(v['ordinary days'])} | {fmt(v['calendar days'])} |")
        md += ['', '| Mean daily error (MW) | All days | Ordinary days | Calendar days |', '| --- | --- | --- | --- |']
        for lab, v in list(out['mae_mw'][name].items()) + [('regression, ' + l, v) for l, v in out['mae_mw']['regression'].items()]:
            md.append(f"| {lab} | {v['all days']:,.0f} | {v['ordinary days']:,.0f} | {v['calendar days']:,.0f} |")
        md.append('')
    open(os.path.join(run, 'calendar.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))


if __name__ == '__main__':
    main()
