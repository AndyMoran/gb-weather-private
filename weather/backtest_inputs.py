"""
Build the backtest's national weather inputs: every target day from 16 March 2024 to 31 October 2026 (Brief 4,
Section 3), with the same code as the daily push (national.py, sources.py, archive.py), so backtest and prospective
inputs come from one method.

Phases, each resumable and cached under ~/gb-weather-data/backtest/:
  runs          fetch every 00 UTC run from 1 February 2024 to 30 October 2026 that the collector does not hold
                (Google first; `--workers` at a time, default 3), reduced once to national series
  availability  when each run was on a public copy, against the 09:30 UK issue time on its date (cached CSV)
  era5          ERA5 reduced to national hourly series, month by month (from the CDS files; Google's copy fills
                missing hours, as in the daily push)
  days          for each target day: decision 13's factors, the adjusted and unadjusted inputs, the horizon run and
                any fallback, or the reason the rules drop the day
Outputs:
  inputs_<first>_<last>.csv.gz   one row per target day and half-hour (long format)
  era5_halfhourly_<first>_<last>.csv.gz   raw ERA5 on every half-hour from a year before the first day: the
                                 regression's training weather before D-10
  days_<first>_<last>.jsonl      per day: factors, fallback, the runs used and their hashes, or the drop reason
  build.json                     hashes of the code and weights, counts, and every output's SHA-256

Run in `wx`:
    python backtest_inputs.py                                  # all 960 days (needs ERA5 from 15 March 2023)
    python backtest_inputs.py --from 2026-04-15 --to 2026-10-31  # a partial build on the months downloaded
    python backtest_inputs.py --phase runs --workers 3        # only prefetch runs (long: run overnight)
"""
import argparse, concurrent.futures as cf, csv, datetime as dt, gzip, json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PUBLIC = os.path.expanduser(os.environ.get('GBFM_PUBLIC', '~/gb-foundation-model-tests'))
sys.path.insert(0, os.path.join(PUBLIC, 'code'))
import national as N, sources as S, archive as A
import common as C

FIRST_DAY, LAST_DAY = dt.date(2024, 3, 16), dt.date(2026, 10, 31)
FIRST_RUN = dt.date(2024, 2, 1)                              # the archive at 0.25 deg
OUT = os.path.expanduser('~/gb-weather-data/backtest'); os.makedirs(OUT, exist_ok=True)
WEIGHTS = os.path.expanduser('~/gb-weather-data/weights/weights.nc')
H30 = pd.Timedelta('30min')


def say(*a): print(f'{dt.datetime.now():%H:%M:%S}', *a, flush=True)


def run_dates(first_day, last_day):
    lo = max(FIRST_RUN, first_day - dt.timedelta(days=N.FACTOR_WINDOW[0]))
    return [lo + dt.timedelta(days=k) for k in range((last_day - dt.timedelta(days=1) - lo).days + 1)]


def phase_runs(dates, workers):
    manifest = S.collector_manifest(PUBLIC)
    def one(d):
        t = time.time()
        try: ds, rec = S.run(pd.Timestamp(d), PUBLIC, manifest)
        except Exception as e: return d, f'ERROR {type(e).__name__}: {str(e)[:120]}', time.time() - t
        return d, (rec['copy'] if ds is not None else 'absent'), time.time() - t
    todo = [d for d in dates if not os.path.exists(os.path.join(PUBLIC, 'prospective', 'weather', 'data', f'ifs_0p25_{d:%Y%m%d}_00z_gb.nc'))
            and not os.path.exists(os.path.join(S.RUNS_DIR, f'ifs_0p25_{d:%Y%m%d}_00z_gb.nc'))]
    say(f'runs: {len(dates)} needed, {len(todo)} to fetch, {workers} at a time')
    with cf.ThreadPoolExecutor(workers) as ex:
        for k, (d, what, sec) in enumerate(ex.map(one, todo), 1):
            if k % 20 == 0 or what.startswith(('ERROR', 'absent')): say(f'  {k}/{len(todo)} {d}: {what} ({sec:.0f} s)')
    held = [d for d in dates if any(os.path.exists(os.path.join(p, f'ifs_0p25_{d:%Y%m%d}_00z_gb.nc'))
                                    for p in (os.path.join(PUBLIC, 'prospective', 'weather', 'data'), S.RUNS_DIR))]
    missing = sorted(set(dates) - set(held))
    say(f'runs: {len(held)} of {len(dates)} held; missing: {", ".join(map(str, missing)) if missing else "none"}')


def phase_availability(dates, workers):
    path = os.path.join(OUT, 'availability.csv')
    have = {r['run_date']: r for r in csv.DictReader(open(path))} if os.path.exists(path) else {}
    todo = [d for d in dates if f'{d}' not in have]
    def one(d):
        issue = C.issue_time(pd.Timestamp(d) + N.DAY)          # the issue for target day d+1 is on d
        t, src = A.public_by(d, list(range(0, 49, 3)), by=issue)
        return dict(run_date=f'{d}', public_by_utc=t.isoformat() if t else '', copy=src,
                    issue_utc=issue.isoformat(), on_time=bool(t is not None and t <= issue))
    say(f'availability: {len(todo)} runs to check')
    rows = list(have.values())
    with cf.ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(one, todo), 1):
            rows.append(r)
            if k % 50 == 0: say(f'  {k}/{len(todo)}')
    rows.sort(key=lambda r: r['run_date'])
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run_date', 'public_by_utc', 'copy', 'issue_utc', 'on_time']); w.writeheader(); w.writerows(rows)
    late = [r['run_date'] for r in rows if str(r['on_time']) not in ('True', 'true')]
    say(f'availability: {len(rows)} runs; {len(late)} not public by the issue time')
    return {pd.Timestamp(d) for d in late}


def national_runs(dates, w_pop, w_pv):
    manifest = S.collector_manifest(PUBLIC); runs, recs = {}, {}
    for d in dates:
        ds, rec = S.run(pd.Timestamp(d), PUBLIC, manifest)
        recs[f'{d}'] = rec
        if ds is not None:
            _, p, i = N.ecmwf_national(ds, w_pop, w_pv); runs[pd.Timestamp(d)] = (p, i)
    return runs, recs


def national_era5(first, last, w_pop, w_pv):
    pts, ivs, rec = [], [], {}
    for m0 in pd.date_range(first.normalize().replace(day=1), last, freq='MS'):
        a = max(first, m0.tz_localize('UTC') if m0.tzinfo is None else m0)
        b = min(last, (m0 + pd.offsets.MonthBegin(1)).tz_localize('UTC') if m0.tzinfo is None else m0 + pd.offsets.MonthBegin(1))
        ds, r = S.era5(a, b); p, i = N.era5_national(ds, w_pop, w_pv)
        pts.append(p); ivs.append(i); rec.update(r)
    p = pd.concat(pts); i = pd.concat(ivs)
    p = p[~p.index.duplicated()].sort_index(); i = i.drop_duplicates('end').sort_values('end').reset_index(drop=True)
    return (p, i), rec


def periods_for(D):
    end = C.input_end(D); h, i0 = C.horizon(D)
    ctx = pd.date_range(end - C.CONTEXT_LEN * H30, end - H30, freq='30min')
    hor = pd.date_range(end, end + (h - 1) * H30, freq='30min')
    return ctx, hor, h, i0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='first', default=f'{FIRST_DAY}'); ap.add_argument('--to', dest='last', default=f'{LAST_DAY}')
    ap.add_argument('--phase', choices=['runs', 'availability', 'all'], default='all'); ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--check', help='a built target day: rebuild it with the daily push code and compare')
    ap.add_argument('--no-regression-history', action='store_true',
                    help='skip the year of raw ERA5 for the regression (partial builds before ERA5 has fully downloaded)')
    a = ap.parse_args()
    if a.check: return check(a.check)
    first, last = dt.date.fromisoformat(a.first), dt.date.fromisoformat(a.last)
    if first < FIRST_DAY or last > LAST_DAY: sys.exit(f'the backtest is {FIRST_DAY} to {LAST_DAY}')
    dates = run_dates(first, last)
    say(f'=== backtest inputs {first} .. {last}: {len(dates)} runs ({dates[0]} .. {dates[-1]})')
    phase_runs(dates, a.workers)
    if a.phase == 'runs': return
    late = phase_availability(dates, a.workers)
    if a.phase == 'availability': return
    w_pop, w_pv = N.load_weights(WEIGHTS)
    say('reducing runs to national series'); runs, run_recs = national_runs(dates, w_pop, w_pv)
    D0 = pd.Timestamp(first)
    reg_first = pd.Timestamp(D0 - pd.Timedelta(days=0 if a.no_regression_history else 367)).tz_localize('UTC')
    e_first = min(periods_for(D0)[0][0].floor('1h') - pd.Timedelta(hours=1), pd.Timestamp(D0 - N.FACTOR_WINDOW[0] * N.DAY).tz_localize('UTC'),
                  reg_first - pd.Timedelta(hours=1))
    e_last = pd.Timestamp(pd.Timestamp(last) - N.SPLICE_DAYS * N.DAY).tz_localize('UTC')
    say(f'ERA5 national, {e_first:%Y-%m-%d} .. {e_last:%Y-%m-%d}'); era5, era5_rec = national_era5(e_first, e_last, w_pop, w_pv)

    tag = f'{first}_{last}'
    hh_path = os.path.join(OUT, f'era5_halfhourly_{tag}.csv.gz')
    if not a.no_regression_history:
        hh_periods = pd.date_range(reg_first, e_last - pd.Timedelta(minutes=30), freq='30min')
        hh = N.era5_halfhourly(era5, hh_periods)               # raw ERA5 for the regression's training weather
        hh.index.name = 'period_start_utc'
        hh.to_csv(hh_path, float_format='%.4f', date_format='%Y-%m-%dT%H:%M:%SZ', compression='gzip')
        say(f'ERA5 half-hourly for the regression: {len(hh)} half-hours, {hh.index[0]:%Y-%m-%d} .. {hh.index[-1]:%Y-%m-%d}')
    csv_path = os.path.join(OUT, f'inputs_{tag}.csv.gz'); jl_path = os.path.join(OUT, f'days_{tag}.jsonl')
    n_ok = n_drop = n_fb = 0
    with gzip.open(csv_path, 'wt', newline='') as fc, open(jl_path, 'w') as fj:
        fc.write('target_day,period_start_utc,part,t2m_c,wind_ms,solar_wm2,t2m_c_unadjusted,wind_ms_unadjusted,'
                 'solar_wm2_unadjusted,source\n')
        for D in pd.date_range(first, last, freq='D'):
            ctx, hor, h, i0 = periods_for(D)
            rec = dict(target_day=f'{D:%Y-%m-%d}', issue_time_utc=C.issue_time(D).isoformat())
            try:
                f = N.history_factors(D.date(), era5, runs)
                lt = {x for x in late if x == D - N.DAY}
                out, rD = N.assemble(D.date(), ctx.append(hor), era5, runs, late=lt, factors=f)
                raw, _ = N.assemble(D.date(), ctx.append(hor), era5, runs, late=lt)
            except N.Dropped as e:
                rec.update(dropped=str(e)); fj.write(json.dumps(rec) + '\n'); n_drop += 1; continue
            fb = rD != D - N.DAY; n_ok += 1; n_fb += fb
            rec.update(horizon_run=f'{rD:%Y-%m-%d}', fallback=bool(fb), history_factors=f, horizon_steps=h,
                       first_target_step=i0, runs_used=sorted({s for s in out['source'] if s != 'era5'}))
            fj.write(json.dumps(rec) + '\n')
            block = pd.DataFrame({'target_day': f'{D:%Y-%m-%d}', 'period_start_utc': out.index.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                  'part': ['context'] * len(ctx) + ['horizon'] * len(hor),
                                  't2m_c': out['t2m'].to_numpy(), 'wind_ms': out['wind'].to_numpy(), 'solar_wm2': out['solar'].to_numpy(),
                                  't2m_c_unadjusted': raw['t2m'].to_numpy(), 'wind_ms_unadjusted': raw['wind'].to_numpy(),
                                  'solar_wm2_unadjusted': raw['solar'].to_numpy(), 'source': out['source'].to_numpy()})
            block.to_csv(fc, header=False, index=False, float_format='%.4f')
            if D.day == 1: say(f'  {D:%Y-%m-%d}: built ({n_ok} so far, {n_drop} dropped, {n_fb} fallback)')
    build = dict(built_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), first=f'{first}', last=f'{last}',
                 days_built=n_ok, days_dropped=n_drop, fallback_days=n_fb, runs_late=sorted(f'{x:%Y-%m-%d}' for x in late),
                 code_sha256={f: S.sha256(os.path.join(HERE, f)) for f in ('national.py', 'sources.py', 'archive.py', 'backtest_inputs.py')},
                 weights_sha256=S.sha256(WEIGHTS), outputs={os.path.basename(p): S.sha256(p) for p in (csv_path, jl_path, hh_path) if os.path.exists(p)},
                 runs={k: v.get('sha256') for k, v in run_recs.items()}, era5_days=era5_rec)
    json.dump(build, open(os.path.join(OUT, f'build_{tag}.json'), 'w'), indent=1)
    say(f'=== done: {n_ok} days built, {n_drop} dropped, {n_fb} on the fallback run; outputs in {OUT}')


def check(day):
    """The backtest and the daily push are one method: rebuild a built day with push_daily.build and compare."""
    import glob, push_daily as P
    D = pd.Timestamp(day)
    files = [f for f in glob.glob(os.path.join(OUT, 'inputs_*.csv.gz'))
             if f.split('inputs_')[1][:10] <= day <= f.split('inputs_')[1][11:21]]
    if not files: sys.exit(f'{day} is in no built inputs file')
    t = pd.read_csv(files[-1]); t = t[t.target_day == day]
    out, rec = P.build(D)
    same = len(t) == len(out) and np.allclose(t[['t2m_c', 'wind_ms', 'solar_wm2']].to_numpy(),
                                              out[['t2m', 'wind', 'solar']].to_numpy(), atol=6e-5) and \
           np.allclose(t[['t2m_c_unadjusted', 'wind_ms_unadjusted', 'solar_wm2_unadjusted']].to_numpy(),
                       out[['t2m_raw', 'wind_raw', 'solar_raw']].to_numpy(), atol=6e-5)
    say(f'{day}: backtest builder and daily push {"AGREE" if same else "DIFFER"} on {len(out)} half-hours '
        f'(horizon run {rec["horizon_run"]}, fallback {rec["fallback"]})')


if __name__ == '__main__':
    main()
