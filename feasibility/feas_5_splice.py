"""
Feasibility check 5: do the splice's two sources agree? (Brief 4, Section 3, "The last ten days".)

The last ten days of every context come from ECMWF runs (each day from its own run's first 24 hours); the days
before come from ERA5. If the two differ systematically, every context carries a step ten days before its end.
For each UTC day both cover, this compares, on national (weighted) values:
  temperature and wind speed at the run's own times (steps 0, 3, ..., 21) against ERA5 at the same instants;
  solar radiation as means over the run's 3-hour intervals against ERA5's three hourly values averaged.
Reports the mean difference (ECMWF minus ERA5) and the mean absolute difference, per day and overall.
Runs come from the collector or, before it, from the archive (cached), exactly as the daily push reads them.

Run in `wx`:   python feas_5_splice.py [first_day] [last_day]      (default: the last 30 days ERA5 covers)
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'weather'))
PUBLIC = os.path.expanduser(os.environ.get('GBFM_PUBLIC', '~/gb-foundation-model-tests'))
sys.path.insert(0, os.path.join(PUBLIC, 'code'))
import national as N, sources as S

w_pop, w_pv = N.load_weights(os.path.expanduser('~/gb-weather-data/weights/weights.nc'))
files = S.cds_files()
last_cds = max(files)
if len(sys.argv) > 2:
    first, last = pd.Timestamp(sys.argv[1]), pd.Timestamp(sys.argv[2])
else:
    import calendar
    y, m = last_cds; full = os.path.basename(files[last_cds]) == f'era5_{y}_{m:02d}.nc'
    last = pd.Timestamp(y, m, calendar.monthrange(y, m)[1] if full else int(files[last_cds][-5:-3])) - N.DAY
    first = last - 29 * N.DAY
print(f'days {first:%Y-%m-%d} .. {last:%Y-%m-%d}', flush=True)
e_ds, rec = S.era5(first.tz_localize('UTC'), (last + N.DAY).tz_localize('UTC'))
ep, ei = N.era5_national(e_ds, w_pop, w_pv)
manifest = S.collector_manifest(PUBLIC)
rows = []
for X in pd.date_range(first, last, freq='D'):
    ds, r = S.run(X, PUBLIC, manifest)
    if ds is None: print(f'{X:%Y-%m-%d}: no run'); continue
    _, p, i = N.ecmwf_national(ds, w_pop, w_pv)
    p = p[p['step_h'] <= 21]; i = i[i['step_h'] <= 24]
    e = ep.reindex(p.index)
    es = np.array([ei.loc[(ei['end'] > a) & (ei['end'] <= b), 'solar'].mean() for a, b in zip(i['start'], i['end'])])
    d = dict(day=f'{X:%Y-%m-%d}', source=r['copy'],
             t2m_bias=float((p['t2m'] - e['t2m']).mean()), t2m_mae=float((p['t2m'] - e['t2m']).abs().mean()),
             wind_bias=float((p['wind'] - e['wind']).mean()), wind_mae=float((p['wind'] - e['wind']).abs().mean()),
             solar_bias=float(np.mean(i['solar'].to_numpy() - es)), solar_mae=float(np.mean(np.abs(i['solar'].to_numpy() - es))),
             era5_wind=float(e['wind'].mean()), era5_solar=float(np.mean(es)))
    rows.append(d)
    print(f"{d['day']} ({d['source'][:9]}): t2m {d['t2m_bias']:+.2f} (|{d['t2m_mae']:.2f}|) C; wind {d['wind_bias']:+.2f} "
          f"(|{d['wind_mae']:.2f}|) of {d['era5_wind']:.1f} m/s; solar {d['solar_bias']:+.1f} (|{d['solar_mae']:.1f}|) "
          f"of {d['era5_solar']:.0f} W/m2", flush=True)
t = pd.DataFrame(rows)
out = os.path.expanduser('~/gb-weather-data/feasibility/splice_agreement.csv'); t.to_csv(out, index=False)
print(f"\nALL {len(t)} days: temperature {t.t2m_bias.mean():+.2f} C (abs {t.t2m_mae.mean():.2f}); "
      f"wind {t.wind_bias.mean():+.2f} m/s (abs {t.wind_mae.mean():.2f}) against a mean of {t.era5_wind.mean():.1f}; "
      f"solar {t.solar_bias.mean():+.1f} W/m2 (abs {t.solar_mae.mean():.1f}) against a mean of {t.era5_solar.mean():.0f}")
print('written to', out)
