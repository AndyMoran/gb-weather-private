"""
Feasibility check 6: is the ECMWF-ERA5 difference stable across the year? (The gate for proposed decision 13.)

For four sample days a month (the 2nd, 9th, 16th and 23rd), on national (weighted) values, compares ECMWF with ERA5
at two lead times:
  short  the day's own run, steps 0 to 24: the lead the context's last ten days use
  day2   the previous day's run, steps 24 to 48: the lead the target day uses
Per month and lead, with decision 13's conventions:
  temperature  mean difference, ECMWF - ERA5 (deg C)
  wind         ratio of totals, ECMWF / ERA5
  solar        ratio of totals over ERA5-daylight intervals (ERA5 3-hour mean above 10 W/m2); ECMWF 3-hour means
               from its accumulations, ERA5 3-hour means from its hourly totals
Only months the downloaded ERA5 files cover are processed; rerun as more arrive (runs are cached, and the backtest
needs them anyway). From March 2024, the first month with solar radiation in the runs.

Run in `wx`:   python feas_6_bias_by_month.py [--all]  (writes ~/gb-weather-data/feasibility/bias_by_month[_all].csv)
  --all uses every day of each month, with 90% intervals from resampling days (500 draws, seed 20261107).
"""
import calendar, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'weather'))
PUBLIC = os.path.expanduser(os.environ.get('GBFM_PUBLIC', '~/gb-foundation-model-tests'))
sys.path.insert(0, os.path.join(PUBLIC, 'code'))
import national as N, sources as S

SAMPLE_DAYS = (2, 9, 16, 23)
ALL_DAYS = '--all' in sys.argv            # every day of each month, with 90% intervals from resampling days
rng = np.random.default_rng(20261107)
w_pop, w_pv = N.load_weights(os.path.expanduser('~/gb-weather-data/weights/weights.nc'))
manifest = S.collector_manifest(PUBLIC)
cache = {}


def run_nat(X):
    if X not in cache:
        ds, r = S.run(X, PUBLIC, manifest)
        cache[X] = None if ds is None else N.ecmwf_national(ds, w_pop, w_pv)[1:]
    return cache[X]


def compare(p, i, ep, ei):
    """Sums for one day: temperature differences, wind totals, daylight solar totals (3-hour means)."""
    e = ep.reindex(p.index)
    es = np.array([ei.loc[(ei['end'] > a) & (ei['end'] <= b), 'solar'].mean() for a, b in zip(i['start'], i['end'])])
    day = es > 10
    return dict(dT=float((p['t2m'] - e['t2m']).sum()), nT=len(p), w_ifs=float(p['wind'].sum()), w_era=float(e['wind'].sum()),
                s_ifs=float(i['solar'].to_numpy()[day].sum()), s_era=float(es[day].sum()), n_s=int(day.sum()))


rows, out = [], os.path.expanduser('~/gb-weather-data/feasibility/bias_by_month' + ('_all' if ALL_DAYS else '') + '.csv')
for (y, m), path in sorted(S.cds_files().items()):
    if (y, m) < (2024, 3): continue
    last_day = calendar.monthrange(y, m)[1] if os.path.basename(path) == f'era5_{y}_{m:02d}.nc' else int(path[-5:-3])
    days = [pd.Timestamp(y, m, d) for d in (range(1, last_day) if ALL_DAYS else SAMPLE_DAYS) if d < last_day]
    if not days: continue
    e_ds, _ = S.era5(days[0].tz_localize('UTC'), (days[-1] + N.DAY).tz_localize('UTC'))
    ep, ei = N.era5_national(e_ds, w_pop, w_pv)
    acc = {'short': [], 'day2': []}
    for X in days:
        lo, hi = X.tz_localize('UTC'), (X + N.DAY).tz_localize('UTC')
        for lead, run_day in (('short', X), ('day2', X - N.DAY)):
            r = run_nat(run_day)
            if r is None: continue
            p, i = r
            p = p[(p.index >= lo) & (p.index <= hi)]; i = i[(i['start'] >= lo) & (i['end'] <= hi)]
            acc[lead].append(compare(p, i, ep, ei))
    for lead, a in acc.items():
        if not a: continue
        t = pd.DataFrame(a)
        stat = lambda u: (u.dT.sum() / u.nT.sum(), u.w_ifs.sum() / u.w_era.sum(),
                          u.s_ifs.sum() / u.s_era.sum() if u.s_era.sum() > 0 else np.nan)
        T, W, Sr = stat(t)
        boot = np.array([stat(t.iloc[rng.integers(0, len(t), len(t))]) for _ in range(500)])
        lo, hi = np.nanpercentile(boot, 5, axis=0), np.nanpercentile(boot, 95, axis=0)
        rows.append(dict(month=f'{y}-{m:02d}', lead=lead, days=len(t), t2m_diff=T, t2m_lo=lo[0], t2m_hi=hi[0],
                         wind_ratio=W, wind_lo=lo[1], wind_hi=hi[1], solar_ratio=Sr, solar_lo=lo[2], solar_hi=hi[2],
                         wind_era5_mean=t.w_era.sum() / t.nT.sum()))
    sh = [r for r in rows if r['month'] == f'{y}-{m:02d}']
    print(f'{y}-{m:02d}: ' + ' | '.join(
        f"{r['lead']}: T {r['t2m_diff']:+.2f} [{r['t2m_lo']:+.2f},{r['t2m_hi']:+.2f}] C, wind x{r['wind_ratio']:.3f} "
        f"[{r['wind_lo']:.3f},{r['wind_hi']:.3f}], solar x{r['solar_ratio']:.3f} [{r['solar_lo']:.3f},{r['solar_hi']:.3f}] "
        f"({r['days']} d)" for r in sh), flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)

t = pd.DataFrame(rows)
for lead in ('short', 'day2'):
    s = t[t.lead == lead]
    if len(s):
        hw = lambda c: ((s[c + '_hi'] - s[c + '_lo']) / 2).median() if c + '_hi' in s else np.nan
        print(f'\n{lead}: typical half-width of a month\'s 90% interval: T {hw("t2m"):.2f} C, wind {hw("wind"):.3f}, solar {hw("solar"):.3f}')
        print(f'{lead}: {len(s)} months. Temperature {s.t2m_diff.min():+.2f} to {s.t2m_diff.max():+.2f} C (sd {s.t2m_diff.std():.2f}); '
              f'wind x{s.wind_ratio.min():.3f} to x{s.wind_ratio.max():.3f} (sd {s.wind_ratio.std():.3f}); '
              f'solar x{s.solar_ratio.min():.3f} to x{s.solar_ratio.max():.3f} (sd {s.solar_ratio.std():.3f})')
both = t.pivot(index='month', columns='lead', values=['t2m_diff', 'wind_ratio', 'solar_ratio']).dropna()
if len(both):
    print('\nshort vs day2, mean difference across months: '
          f"T {(both['t2m_diff']['day2'] - both['t2m_diff']['short']).mean():+.3f} C; "
          f"wind {(both['wind_ratio']['day2'] - both['wind_ratio']['short']).mean():+.4f}; "
          f"solar {(both['solar_ratio']['day2'] - both['solar_ratio']['short']).mean():+.4f}")
print('written to', out)
