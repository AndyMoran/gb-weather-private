"""
Checkpoint 0, WP2: cost (gb-foundation-model-tests, checkpoint0/PLAN.md, frozen at tag checkpoint0-plan).

Eight forecaster set-ups, timed on the 20 days in checkpoint0/wp2_days.csv, on this machine, CPU only, with the same
thread count (4) for every forecaster, one forecaster per process:
    chronos2_A          Chronos-2, calendar and weather (E5 arm A: calendar_study.py's covariates)
    timesfm25_weather   TimesFM 2.5, weather (the scored configuration)
    timestral_A         Timestral, calendar and weather (E5 arm A)
    timestral_weather   Timestral, weather
    regression_weather  the weather regression (re-fitted each day)
    lightgbm_default, lightgbm_s1, lightgbm_s2   (re-fitted each day)
Timed per forecast day: from that day's built inputs to its forecasts, including any fit (time.perf_counter). Days are
forecast one at a time (a batch of one): an operational forecaster issues one day each morning. The scored backtest
batched 64 days per model call, which changes foundation-model forecasts only in the last digits.
Model loading is timed once per forecaster, separately. One untimed warm-up forecast (the first sample day) precedes the
timed days. Peak resident memory per process is reported, with the resident memory after the data had loaded, so the
model's own share can be read as the difference.
Check: each forecaster's timed forecasts must match its saved backtest forecasts within 1 MW on every half-hour
(LightGBM exactly), confirming the timed configuration is the one scored.

Run in gbcp0, with nothing else heavy running:
    python weather/cost_wp2.py --dry     plumbing only: 2 days per forecaster; nothing kept
    python weather/cost_wp2.py           the timed run; writes ~/gb-weather-data/results/checkpoint0/cost.json and cost.md
"""
import argparse, json, os, platform, resource, subprocess, sys, time

THREADS = '4'
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[k] = THREADS
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
NAMES = ('chronos2_A', 'timesfm25_weather', 'timestral_A', 'timestral_weather', 'regression_weather',
         'lightgbm_default', 'lightgbm_s1', 'lightgbm_s2')
OUT = os.path.join(os.path.expanduser('~/gb-weather-data'), 'results', 'checkpoint0')


def rss_mb():
    for line in open('/proc/self/status'):
        if line.startswith('VmRSS:'): return int(line.split()[1]) / 1024
    return float('nan')


def machine():
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?')
    mem = next((int(l.split()[1]) / 1024 / 1024 for l in open('/proc/meminfo') if l.startswith('MemTotal')), float('nan'))
    import torch, lightgbm
    return dict(cpu=cpu, logical_cpus=os.cpu_count(), ram_gb=round(mem, 1), os=platform.platform(),
                python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                torch=torch.__version__, lightgbm=lightgbm.__version__, threads=int(THREADS))


def saved_forecasts(public):
    f = pd.read_csv(os.path.join(public, 'weather_study', 'results', 'forecasts.csv'))
    cal = pd.read_csv(os.path.join(public, 'weather_study', 'results', 'calendar_forecasts.csv'))
    lg = pd.read_csv(os.path.join(public, 'checkpoint0', 'lightgbm_forecasts.csv'))
    s = f.set_index('start_utc')
    for c in ('chronos2_A', 'timestral_A'): s[c] = cal.set_index('start_utc')[c]
    for c in ('lightgbm_default', 'lightgbm_s1', 'lightgbm_s2'): s[c] = lg.set_index('start_utc')[c]
    return s


def one(name, dry):
    """Time one forecaster in this process."""
    import torch; torch.set_num_threads(int(THREADS))
    import run_weather as R, calendar_study as CS, regression_weather as RW, lightgbm_weather as LW, models_weather as MW
    days = [pd.Timestamp(d) for d in pd.read_csv(os.path.join(R.PUBLIC, 'checkpoint0', 'wp2_days.csv'))['day']]
    if dry: days = days[:2]
    series, pubs, hol, days_inputs, meta, era5_hh, built = R.load_real(full=True)
    rec, excl = R.build(series, hol, pubs, days_inputs, meta, R.load_rung1(), era5_hh, list(pd.date_range(R.FIRST, R.LAST)))
    holn = {pd.Timestamp(h).normalize() for h in hol}
    for D in days:
        r = rec[D]; r['cov_cw'] = {**r['cov'], **CS.calendar(r['wx'].index, holn)}
    saved = saved_forecasts(R.PUBLIC)
    rss_data = rss_mb()

    t0 = time.perf_counter()
    if name in ('chronos2_A', 'timestral_A', 'timesfm25_weather', 'timestral_weather'):
        m = MW.get(name.split('_')[0])
        names, key = (CS.WX + CS.CAL, 'cov_cw') if name.endswith('_A') else (CS.WX, 'cov')
        f = lambda D: CS.run_arm(m, {D: rec[D]}, names, key)[D]
    elif name == 'regression_weather':
        m = RW.RegressionW(series, hol); f = lambda D: m.forecast(D, rec[D]['wx'], era5_hh)
    else:
        m = LW.LightGBMW(series, hol); s = {'lightgbm_default': 'default', 'lightgbm_s1': 'S1', 'lightgbm_s2': 'S2'}[name]
        f = lambda D: m.forecast_all(D, rec[D]['wx'], era5_hh, (s,))[s]
    load_s = time.perf_counter() - t0

    f(days[0])                                                # warm-up, untimed
    times, devs = [], []
    for D in days:
        t = time.perf_counter(); fc = f(D); times.append(time.perf_counter() - t)
        ref = saved.loc[[x.isoformat() for x in rec[D]['idx']], name].to_numpy(float)
        devs.append(float(np.max(np.abs(np.asarray(fc, float) - ref))))
    tol = 0.0 if name.startswith('lightgbm') else 1.0
    return dict(name=name, days=[f'{D:%Y-%m-%d}' for D in days], seconds=[round(x, 4) for x in times],
                median_s=round(float(np.median(times)), 4), min_s=round(float(np.min(times)), 4),
                max_s=round(float(np.max(times)), 4), load_s=round(load_s, 2),
                rss_after_data_mb=round(rss_data, 0), peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0),
                max_abs_diff_mw=round(max(devs), 6), tolerance_mw=tol, matches_saved=max(devs) <= tol)


def markdown(res):
    L = ['# Checkpoint 0: cost (WP2)', '',
         f"Machine: {res['machine']['cpu']}, {res['machine']['logical_cpus']} logical CPUs, {res['machine']['ram_gb']} GB RAM; "
         f"{res['machine']['os']}. Every forecaster on CPU with {res['machine']['threads']} threads, one per process, "
         'one day per call. Times are wall-clock per forecast day, including any daily re-fit; median (range) over the 20 '
         'sample days after one untimed warm-up. Memory: peak resident memory of the process, and the part above the '
         'memory held once the backtest data had loaded.', '',
         '| Forecaster | Seconds per day, median (range) | Model load (s) | Peak memory (MB) | Above data (MB) | Matches scored forecasts |',
         '|---|---|---|---|---|---|']
    for n in NAMES:
        r = res['forecasters'][n]
        L.append(f"| {n} | {r['median_s']:.2f} ({r['min_s']:.2f}–{r['max_s']:.2f}) | {r['load_s']:.1f} | {r['peak_rss_mb']:,.0f} | "
                 f"{r['peak_rss_mb'] - r['rss_after_data_mb']:,.0f} | {'yes' if r['matches_saved'] else 'NO'} (max {r['max_abs_diff_mw']:.3g} MW) |")
    L += ['', 'Cost is reported in time and memory only; no £ figure is computed in checkpoint 0 (PLAN.md WP2).', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--one'); ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    if a.one:
        print('RESULT ' + json.dumps(one(a.one, a.dry)), flush=True); return
    res = dict(note='checkpoint 0 WP2' + (' (dry run: plumbing only, not kept)' if a.dry else ''),
               machine=machine(), forecasters={})
    for n in NAMES:
        print(f'{time.strftime("%H:%M:%S")} {n} ...', flush=True)
        p = subprocess.run([sys.executable, __file__, '--one', n] + (['--dry'] if a.dry else []),
                           capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
        if p.returncode or not line:
            sys.exit(f'{n} failed:\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}')
        r = json.loads(line[-1][7:]); res['forecasters'][n] = r
        print(f"   median {r['median_s']:.2f} s/day, load {r['load_s']:.1f} s, peak {r['peak_rss_mb']:,.0f} MB, "
              f"matches saved: {r['matches_saved']} (max {r['max_abs_diff_mw']:.3g} MW)", flush=True)
    md = markdown(res); print(md)
    if a.dry: print('dry run: nothing written'); return
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'cost.json'), 'w'), indent=1)
    open(os.path.join(OUT, 'cost.md'), 'w').write(md)
    print(f'written {OUT}/cost.json and cost.md')
    if not all(r['matches_saved'] for r in res['forecasters'].values()):
        print('WARNING: at least one forecaster did not reproduce its scored forecasts; report it (PLAN.md WP2)')


if __name__ == '__main__':
    main()
