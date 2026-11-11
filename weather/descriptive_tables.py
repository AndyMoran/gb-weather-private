"""
Section 6.4's remaining descriptive tables, from the saved forecasts of the scored backtest run: errors by four-hour
block in MW, and by season. Descriptive only: no test, no p-value. Reads only the run's own output
(forecasts.csv) and scores nothing again.

  Blocks:  six four-hour blocks of UK clock time from midnight, as in Q4. For each block, mean absolute error in MW
           pooled over every half-hour of the backtest in that block, with weather and without, and the reduction.
  Seasons: meteorological seasons (winter December-February, spring March-May, summer June-August, autumn
           September-November), pooled across years. For each, mean daily error with weather and without, the
           reduction in MW and the error ratio. The plan's winter window (20 November to 31 March) is in results.json.

Intervals: 90% percentile intervals from the plan's resampling, a moving-block bootstrap over days (blocks of 7, 2,000
draws, seed 20261120, the same draws for both arms). Within a season, days are resampled within that season.

Check: each forecaster's mean daily error, recomputed here, must equal results.json's `mae_mw` to 0.001 MW, or the
script stops.

Run in `gbwx`:   python weather/descriptive_tables.py [results/backtest_attempt1]
Writes, beside forecasts.csv:  descriptive_blocks.csv, descriptive_seasons.csv, descriptive_tables.md
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.expanduser(os.environ.get('GBFM_PUBLIC', '~/gb-foundation-model-tests'))
sys.path.insert(0, os.path.join(PUBLIC, 'code'))
import stats as ST                                             # rung 1's block_indices, as run_weather.py uses

DATA = os.path.expanduser('~/gb-weather-data')
FORECASTERS = ('chronos2', 'timestral', 'timesfm25', 'regression')
NAMES = {'chronos2': 'Chronos-2', 'timestral': 'Timestral', 'timesfm25': 'TimesFM 2.5', 'regression': 'Regression'}
BLOCKS = ['00:00-04:00', '04:00-08:00', '08:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-24:00']
SEASONS = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
           6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn', 11: 'autumn'}
BOOT, BLOCK, SEED = 2000, 7, 20261120


def boot_idx(n, seed):
    rng = np.random.default_rng(seed)
    return np.stack([ST.block_indices(n, rng, BLOCK) for _ in range(BOOT)])


def ci(bs):
    return float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))


def main(run):
    f = pd.read_csv(os.path.join(run, 'forecasts.csv'))
    res = json.load(open(os.path.join(run, 'results.json')))
    t = pd.to_datetime(f['start_utc'], utc=True).dt.tz_convert('Europe/London')
    f['block'] = t.dt.hour // 4
    days = sorted(f['day'].unique()); di = {d: i for i, d in enumerate(days)}
    f['di'] = f['day'].map(di)
    n = len(days); idx = boot_idx(n, SEED)

    # daily MAE per arm, and per-day sums and counts of absolute error per block
    daily, s, k = {}, {}, np.zeros((n, 6))
    np.add.at(k, (f['di'].to_numpy(), f['block'].to_numpy()), 1)
    for name in FORECASTERS:
        for arm in ('weather', 'no_weather'):
            e = (f['outturn'] - f[f'{name}_{arm}']).abs()
            daily[name, arm] = e.groupby(f['di']).mean().reindex(range(n)).to_numpy()
            a = np.zeros((n, 6)); np.add.at(a, (f['di'].to_numpy(), f['block'].to_numpy()), e.to_numpy()); s[name, arm] = a

    # the check against the scored run
    for name in FORECASTERS:
        for arm, key in (('weather', 'with_weather'), ('no_weather', 'without')):
            want = res['results']['descriptive']['mae_mw'][name][key]; got = daily[name, arm].mean()
            if abs(got - want) > 1e-3: sys.exit(f'Check failed: {name} {key} mean daily error {got:.4f} MW, results.json {want:.4f}')
    print(f'check: mean daily errors equal results.json for all {len(FORECASTERS)} forecasters, both arms')

    rows = []
    for name in FORECASTERS:
        w, o = s[name, 'weather'], s[name, 'no_weather']
        for j, b in enumerate(BLOCKS):
            mw, mo = w[:, j].sum() / k[:, j].sum(), o[:, j].sum() / k[:, j].sum()
            bw = w[idx, j].sum(1) / k[idx, j].sum(1); bo = o[idx, j].sum(1) / k[idx, j].sum(1)
            rows.append(dict(forecaster=name, block=b, half_hours=int(k[:, j].sum()), mae_with_mw=mw, mae_without_mw=mo,
                             reduction_mw=mo - mw, reduction_ci90=ci(bo - bw), fr=1 - mw / mo))
    blocks = pd.DataFrame(rows)
    # Q4's FR by block must match results.json too (same pooling)
    for name in ('chronos2', 'timestral', 'timesfm25'):
        for j, b in enumerate(BLOCKS):
            want = res['results']['q4'][name][f'fr_{b[:2]}-{b[6:8]}']
            got = blocks[(blocks.forecaster == name) & (blocks.block == b)].fr.iloc[0]
            if abs(got - want) > 1e-9: sys.exit(f'Check failed: {name} {b} FR {got} vs results.json {want}')
    print('check: block error reductions equal results.json Q4')

    month = pd.to_datetime(pd.Series(days)).dt.month.to_numpy()
    rows = []
    for season in ('winter', 'spring', 'summer', 'autumn'):
        ii = np.where(np.vectorize(SEASONS.get)(month) == season)[0]
        if len(ii) < BLOCK: continue
        sidx = ii[boot_idx(len(ii), SEED)]
        for name in FORECASTERS:
            a, b = daily[name, 'weather'], daily[name, 'no_weather']
            rows.append(dict(forecaster=name, season=season, days=len(ii), mae_with_mw=a[ii].mean(), mae_without_mw=b[ii].mean(),
                             reduction_mw=b[ii].mean() - a[ii].mean(), reduction_ci90=ci(b[sidx].mean(1) - a[sidx].mean(1)),
                             ratio=a[ii].mean() / b[ii].mean(), ratio_ci90=ci(a[sidx].mean(1) / b[sidx].mean(1))))
    seasons = pd.DataFrame(rows)

    def flat(df):
        df = df.copy()
        for c in [c for c in df if c.endswith('_ci90')]:
            df[c[:-5] + '_lo'] = df[c].map(lambda x: x[0]); df[c[:-5] + '_hi'] = df[c].map(lambda x: x[1]); del df[c]
        return df
    flat(blocks).to_csv(os.path.join(run, 'descriptive_blocks.csv'), index=False, float_format='%.4f')
    flat(seasons).to_csv(os.path.join(run, 'descriptive_seasons.csv'), index=False, float_format='%.4f')

    md = ['# Descriptive tables (Section 6.4), backtest', '',
          f'From `{os.path.basename(run)}/forecasts.csv`, {n} days. Descriptive: 90% intervals, no tests.', '',
          '## Mean absolute error by four-hour block, UK clock time (MW)', '',
          '| Forecaster | Block | Without weather | With weather | Reduction | 90% interval | Error cut |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    for r in blocks.itertuples():
        md.append(f'| {NAMES[r.forecaster]} | {r.block} | {r.mae_without_mw:,.0f} | {r.mae_with_mw:,.0f} | {r.reduction_mw:,.0f} | '
                  f'{r.reduction_ci90[0]:,.0f} to {r.reduction_ci90[1]:,.0f} | {r.fr:.0%} |')
    md += ['', '## Mean daily error by season (MW)', '',
           'Meteorological seasons, pooled across years.', '',
           '| Forecaster | Season | Days | Without weather | With weather | Reduction | 90% interval | Error ratio | 90% interval |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in seasons.itertuples():
        md.append(f'| {NAMES[r.forecaster]} | {r.season} | {r.days} | {r.mae_without_mw:,.0f} | {r.mae_with_mw:,.0f} | '
                  f'{r.reduction_mw:,.0f} | {r.reduction_ci90[0]:,.0f} to {r.reduction_ci90[1]:,.0f} | {r.ratio:.3f} | '
                  f'{r.ratio_ci90[0]:.3f}–{r.ratio_ci90[1]:.3f} |')
    open(os.path.join(run, 'descriptive_tables.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'\nwritten: descriptive_blocks.csv, descriptive_seasons.csv, descriptive_tables.md in {run}')


if __name__ == '__main__':
    main(os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(DATA, 'results', 'backtest_attempt1'))
