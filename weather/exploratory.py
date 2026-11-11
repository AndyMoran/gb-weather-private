"""
Exploratory analyses of the scored backtest, prompted by review of the draft results note (11 November 2026). None was
in the plan; every figure is labelled exploratory, and none changes a confirmatory result. Nothing is scored again:
E1 and E2 read the run's saved forecasts; E3 runs the regression once more, with one input changed.

  E1  Leakage: Q1 before and after each model's public release. A model's weights cannot hold days after they were
      fixed, so the post-release days are unseen by construction (release dates as the demand study's code,
      common.POST_RELEASE). Timestral's 17 post-release days are too few to read.
  E2  Bootstrap block length: Q1's interval and one-sided p with blocks of 7 (the plan's), 14 and 28 days.
  E3  The regression's mismatch: the regression trains on raw ERA5 and forecasts with raw ECMWF. Here its ECMWF
      inputs (the forecast day and the ten days before it) are put on ERA5's scale with the inverse of decision 13's
      factors for that day, its training unchanged. Reported: the regression's Q1, and Q3 (each model's error over
      this regression's). A check first reruns the regression unchanged on 10 days and requires the saved forecasts
      to within 1e-6 MW.

Bootstrap as the plan: 2,000 draws, seed 20261120, the same draws for both arms.

Run in `gbwx`:   python weather/exploratory.py [results/backtest_attempt1]
Writes exploratory.json and exploratory.md beside forecasts.csv.
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_weather as R                                       # loading, the regression and the paths, unchanged
import regression_weather as RW
ST, C = R.ST, R.C

BOOT, SEED = 2000, 20261120
FORECASTERS = ('chronos2', 'timestral', 'timesfm25', 'regression')
NAMES = {'chronos2': 'Chronos-2', 'timestral': 'Timestral', 'timesfm25': 'TimesFM 2.5', 'regression': 'Regression'}


def boot_idx(n, block, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.stack([ST.block_indices(n, rng, block) for _ in range(BOOT)])


def ratio(a, b, idx):
    bs = a[idx].mean(1) / b[idx].mean(1)
    return dict(ratio=float(a.mean() / b.mean()), ci90=[float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))],
                p_one_sided=float((bs >= 1).mean()), n_days=int(len(a)))


def on_era5_scale(w, f):
    """The ECMWF rows of a day's unadjusted inputs put on ERA5's scale: the inverse of decision 13's factors."""
    w = w.copy(); ifs = w['source'].astype(str).str.startswith('ifs').to_numpy()
    w.loc[ifs, 't2m_unadjusted'] -= f['t2m_add']
    w.loc[ifs, 'wind_unadjusted'] /= f['wind_mult']
    s = w['solar_unadjusted'].to_numpy().copy()
    day = ifs & (s > R.N.DAYLIGHT_WM2 * f['solar_mult'])
    s[day] = s[day] / f['solar_mult']; w['solar_unadjusted'] = s
    return w


def main(run):
    f = pd.read_csv(os.path.join(run, 'forecasts.csv'))
    days = list(dict.fromkeys(f['day'])); n = len(days)
    g = {d: x for d, x in f.groupby('day', sort=False)}
    dm = {}
    for name in FORECASTERS:
        for arm in ('weather', 'no_weather'):
            dm[name, arm] = np.array([np.abs(g[d]['outturn'] - g[d][f'{name}_{arm}']).mean() for d in days])
    out = {'note': 'exploratory; not in the plan; prompted by review of the draft results note, 11 November 2026'}
    idx7 = boot_idx(n, 7)

    # E1: before and after each model's release
    rel = {k: pd.Timestamp(v) for k, v in C.POST_RELEASE.items()}
    dts = pd.to_datetime(pd.Series(days)).to_numpy()
    out['E1_release'] = {}
    for name in ('chronos2', 'timestral', 'timesfm25'):
        e = {'release': f'{rel[name]:%Y-%m-%d}'}
        for part, ii in (('before', np.where(dts < rel[name])[0]), ('after', np.where(dts >= rel[name])[0])):
            if len(ii) < 14: e[part] = dict(n_days=int(len(ii)), note='too few days'); continue
            e[part] = ratio(dm[name, 'weather'][ii], dm[name, 'no_weather'][ii], boot_idx(len(ii), 7))
        out['E1_release'][name] = e

    # E2: block length
    out['E2_block_length'] = {str(b): {k: ratio(dm[k, 'weather'], dm[k, 'no_weather'], boot_idx(n, b)) for k in FORECASTERS}
                              for b in (7, 14, 28)}
    res = json.load(open(os.path.join(run, 'results.json')))['results']['q1']
    for k in FORECASTERS:                                     # blocks of 7 must reproduce the scored Q1
        if abs(out['E2_block_length']['7'][k]['ci90'][0] - res[k]['ci90'][0]) > 1e-9:
            sys.exit(f'Check failed: E2 with 7-day blocks does not reproduce the scored Q1 for {k}')
    print('check: 7-day blocks reproduce the scored Q1 intervals')

    # E3: the regression with its ECMWF inputs on ERA5's scale
    series, pubs, hol, days_inputs, meta, era5_hh, built = R.load_real(full=True)
    reg = RW.RegressionW(series, hol)
    ts = [pd.Timestamp(d) for d in days]
    for D in ts[::96][:10]:                                   # the check: unchanged inputs reproduce the saved forecasts
        got = reg.forecast(D, days_inputs[D], era5_hh); want = g[f'{D:%Y-%m-%d}']['regression_weather'].to_numpy()
        if np.max(np.abs(got - want)) > 1e-6: sys.exit(f'Check failed: regression rerun differs on {D:%Y-%m-%d}')
    print('check: the regression rerun reproduces the saved forecasts on 10 days')
    alt = []
    for i, D in enumerate(ts):
        fc = reg.forecast(D, on_era5_scale(days_inputs[D], meta[D]['history_factors']), era5_hh)
        alt.append(np.abs(g[days[i]]['outturn'].to_numpy() - fc).mean())
        if i % 100 == 0: print(f'  E3 {i}/{n}', flush=True)
    alt = np.array(alt)
    out['E3_regression_on_era5_scale'] = {
        'mae_mw': float(alt.mean()), 'mae_mw_as_scored': float(dm['regression', 'weather'].mean()),
        'q1': ratio(alt, dm['regression', 'no_weather'], idx7),
        'vs_scored_regression': ratio(alt, dm['regression', 'weather'], idx7),
        'q3': {k: ratio(dm[k, 'weather'], alt, idx7) for k in ('chronos2', 'timestral', 'timesfm25')}}
    pd.DataFrame({'day': days, 'regression_era5_scale_mae': alt}).to_csv(os.path.join(run, 'exploratory_e3_daily.csv'), index=False)

    json.dump(out, open(os.path.join(run, 'exploratory.json'), 'w'), indent=1)
    fmt = lambda r: f"{r['ratio']:.3f} ({r['ci90'][0]:.3f}–{r['ci90'][1]:.3f}; p {r['p_one_sided']:.4f}; {r['n_days']} days)"
    md = ['# Exploratory analyses of the backtest (not in the plan)', '', '## E1: before and after each model\'s release (Q1 ratio)', '']
    for k, e in out['E1_release'].items():
        md.append(f"- {NAMES[k]}, released {e['release']}: before {fmt(e['before']) if 'ratio' in e['before'] else e['before']}; "
                  f"after {fmt(e['after']) if 'ratio' in e['after'] else e['after']}")
    md += ['', '## E2: bootstrap block length (Q1 ratio, 90% interval, p)', '']
    for b, v in out['E2_block_length'].items():
        md.append(f'- {b}-day blocks: ' + '; '.join(f'{NAMES[k]} {fmt(r)}' for k, r in v.items()))
    e3 = out['E3_regression_on_era5_scale']
    md += ['', '## E3: the regression with its ECMWF inputs on ERA5\'s scale', '',
           f"- Mean daily error {e3['mae_mw']:,.1f} MW, against {e3['mae_mw_as_scored']:,.1f} MW as scored; ratio {fmt(e3['vs_scored_regression'])}",
           f"- Q1 for this regression: {fmt(e3['q1'])}"] + \
          [f"- Q3, {NAMES[k]} ÷ this regression: {fmt(r)}" for k, r in e3['q3'].items()]
    open(os.path.join(run, 'exploratory.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))


if __name__ == '__main__':
    main(os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(R.OUT, 'backtest_attempt1'))
