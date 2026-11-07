"""
Build and publish the national weather inputs for one target day (Brief 4, Section 5, item 2; Section 10, option A).

On D-1, after the 09:30 collection, this builds every half-hour of D's context and horizon with national.py, writes
  weather_study/inputs/<D>.csv    the inputs exactly as the models receive them
  weather_study/inputs/<D>.json   where each piece came from, every input's SHA-256, and the hashes of this code
  weather_study/inputs/manifest.csv
in the public repository, pushes them, and confirms GitHub has them. A day counts only if its inputs reach GitHub
before 00:00 UK time on D; after that the script refuses to push, and the day is rebuilt descriptively instead.

The run of D-1 counts as available only if every file it needs was on a public copy before 09:30 UK time on D-1
(archive.public_by); otherwise it is treated as missing and D falls back to the run of D-2, flagged.

Usage, in `wx`, from anywhere:
    python push_daily.py                      # tomorrow's inputs (UK date), pushed: needs WEATHER_LOCK.json
    python push_daily.py --day 2026-11-21     # a given target day
    python push_daily.py --dry-run [--day D]  # build and check, write to ~/gb-weather-data/push_test/, no git
    python push_daily.py --self-test          # synthetic files only: no network, no git
"""
import argparse, csv, datetime as dt, hashlib, json, os, subprocess, sys, tempfile, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); PRIVATE = os.path.dirname(HERE)
PUBLIC = os.path.expanduser(os.environ.get('GBFM_PUBLIC', '~/gb-foundation-model-tests'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(PUBLIC, 'code'))
import national as N, sources as S, archive as A
import common as C

WINDOW = (dt.date(2026, 11, 20), dt.date(2027, 3, 31))
WEIGHTS = os.path.expanduser('~/gb-weather-data/weights/weights.nc')
WEIGHTS_SHA = 'c3af7721b673386ef31285ff55deab71617bff774b7cb5c13e448dee543102be'
LOCK = os.path.join(PRIVATE, 'WEATHER_LOCK.json')
CODE = ('national.py', 'sources.py', 'archive.py', 'push_daily.py')
OUT_REL = os.path.join('weather_study', 'inputs')
H30 = pd.Timedelta('30min')
ATTRIBUTION = ('Contains modified Copernicus Climate Change Service information (ERA5) and modified '
               'Copernicus/ECMWF data (open-data forecasts). Neither ECMWF nor the European Commission is '
               'responsible for any use of this information.')


def say(*a): print(*a, flush=True)


def periods_for(D):
    end = C.input_end(D); h, i0 = C.horizon(D)
    ctx = pd.date_range(end - C.CONTEXT_LEN * H30, end - H30, freq='30min')
    hor = pd.date_range(end, end + (h - 1) * H30, freq='30min')
    return ctx, hor, h, i0


def build(D, public_root=PUBLIC, late_check=True):
    """Everything for target day D: (table, record). Raises N.Dropped if the rules drop the day."""
    D = pd.Timestamp(D)
    w_pop, w_pv = N.load_weights(WEIGHTS)
    ctx, hor, h, i0 = periods_for(D)
    manifest = S.collector_manifest(public_root)
    runs, run_rec = {}, {}
    for k in range(N.SPLICE_DAYS + 1, 0, -1):               # D-11 (a fallback for D-10) .. D-1
        X = D - k * N.DAY
        ds, rec = S.run(X, public_root, manifest)
        run_rec[f'{X:%Y-%m-%d}'] = rec
        if ds is not None:
            _, p, i = N.ecmwf_national(ds, w_pop, w_pv); runs[X] = (p, i)
    issue = C.issue_time(D)
    late, avail = set(), None
    if late_check:
        t, src = A.public_by((D - N.DAY).date(), list(range(0, 49, 3)))
        avail = dict(run=f'{D - N.DAY:%Y-%m-%d}', public_by_utc=t.isoformat() if t else None, copy=src,
                     issue_utc=issue.isoformat(), on_time=bool(t is not None and t <= issue))
        if not avail['on_time']: late.add(D - N.DAY)
    first = (ctx[0] + pd.Timedelta(minutes=15)).floor('1h') - pd.Timedelta(hours=1)
    last = pd.Timestamp(D - (N.SPLICE_DAYS) * N.DAY).tz_localize('UTC')
    e_ds, era5_rec = S.era5(first, last)
    era5 = N.era5_national(e_ds, w_pop, w_pv)
    out, rD = N.assemble(D.date(), ctx.append(hor), era5, runs, late=late)
    out.insert(0, 'part', ['context'] * len(ctx) + ['horizon'] * len(hor))
    record = dict(target_day=f'{D:%Y-%m-%d}', issue_time_utc=issue.isoformat(),
                  input_cut_utc=C.input_end(D).isoformat(), horizon_steps=h, first_target_step=i0,
                  context_halfhours=len(ctx), horizon_run=f'{rD:%Y-%m-%d}', fallback=bool(rD != D - N.DAY),
                  run_d_minus_1_availability=avail, runs=run_rec, era5_days=era5_rec,
                  weights_sha256=S.sha256(WEIGHTS),
                  code_sha256={f: S.sha256(os.path.join(HERE, f)) for f in CODE})
    return out, record


def write(out, record, folder):
    os.makedirs(folder, exist_ok=True)
    D = record['target_day']; p = os.path.join(folder, f'{D}.csv')
    t = out.copy(); t.index.name = 'period_start_utc'
    t.insert(1, 'period_start_uk', t.index.tz_convert('Europe/London').strftime('%Y-%m-%d %H:%M'))
    t.rename(columns={'t2m': 't2m_c', 'wind': 'wind_ms', 'solar': 'solar_wm2'}).to_csv(
        p, float_format='%.4f', date_format='%Y-%m-%dT%H:%M:%SZ')
    record = dict(record, built_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                  csv=os.path.basename(p), csv_sha256=S.sha256(p), attribution=ATTRIBUTION)
    json.dump(record, open(os.path.join(folder, f'{D}.json'), 'w'), indent=1)
    m = os.path.join(folder, 'manifest.csv'); new = not os.path.exists(m)
    with open(m, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(['target_day', 'built_utc', 'horizon_run', 'fallback', 'csv_sha256'])
        w.writerow([D, record['built_utc'], record['horizon_run'], record['fallback'], record['csv_sha256']])
    return record


def git(*a):
    r = subprocess.run(['git', '-C', PUBLIC, *a], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def push(D):
    files = [os.path.join(OUT_REL, f'{D}.csv'), os.path.join(OUT_REL, f'{D}.json'), os.path.join(OUT_REL, 'manifest.csv')]
    c, o = git('add', *files)
    if c: raise RuntimeError(f'git add: {o}')
    c, o = git('-c', 'user.name=Andy Moran', '-c', 'user.email=andrewgmoran@gmail.com', 'commit', '-q', '-m',
               f'Weather inputs for {D}')
    if c: raise RuntimeError(f'git commit: {o}')
    for k in range(5):
        c1, o1 = git('pull', '--rebase', '-q'); c2, o2 = git('push', '-q')
        if c1 == 0 and c2 == 0:
            git('fetch', '-q', 'origin')
            c3, _ = git('merge-base', '--is-ancestor', 'HEAD', 'origin/main')
            if c3 == 0:
                return git('rev-parse', '--short', 'HEAD')[1]
        say(f'  push attempt {k + 1} failed: {o1} {o2}'); time.sleep(60 * (k + 1))
    raise RuntimeError('push did not reach GitHub after 5 attempts')


def check_lock():
    if not os.path.exists(LOCK): sys.exit('No WEATHER_LOCK.json: the weather processing is not frozen. Use --dry-run.')
    lock = json.load(open(LOCK))
    for f in CODE:
        h = S.sha256(os.path.join(HERE, f))
        if lock['code_sha256'].get(f) != h: sys.exit(f'{f} differs from the frozen version ({h[:12]})')
    if S.sha256(WEIGHTS) != lock['weights_sha256']: sys.exit('weights.nc differs from the frozen version')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day'); ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--test', action='store_true', help='allow a day outside the window (dry runs only)')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test: return self_test()
    now = pd.Timestamp.now(tz='UTC')
    D = pd.Timestamp(a.day) if a.day else (now.tz_convert('Europe/London').normalize() + N.DAY).tz_localize(None)
    say(f'=== start {now.isoformat(timespec="seconds")} target {D:%Y-%m-%d}{" (dry run)" if a.dry_run else ""}')
    if S.sha256(WEIGHTS) != WEIGHTS_SHA: sys.exit('weights.nc is not the agreed file')
    if not a.dry_run:
        check_lock()
        if not (WINDOW[0] <= D.date() <= WINDOW[1]): sys.exit(f'{D:%Y-%m-%d} is outside the prospective window')
        if now >= C.day_bounds(D)[0]:
            sys.exit(f'deadline passed (00:00 UK on {D:%Y-%m-%d}): rebuild descriptively, never push as on time')
        js = os.path.join(PUBLIC, OUT_REL, f'{D:%Y-%m-%d}.json')
        git('pull', '--rebase', '-q')
        if os.path.exists(js) and git('ls-files', '--error-unmatch', js)[0] == 0:
            say(f'{D:%Y-%m-%d}: already published; nothing to do'); return
    elif not a.test and not (WINDOW[0] - dt.timedelta(days=30) <= D.date() <= WINDOW[1]):
        say('note: dry run outside the window')
    try:
        out, rec = build(D)
    except N.Dropped as e:
        say(f'DROPPED under the rules: {e}'); sys.exit(2)
    folder = os.path.expanduser('~/gb-weather-data/push_test') if a.dry_run else os.path.join(PUBLIC, OUT_REL)
    rec = write(out, rec, folder)
    say(f'{rec["target_day"]}: {len(out)} half-hours ({rec["context_halfhours"]} context, {rec["horizon_steps"]} horizon); '
        f'horizon run {rec["horizon_run"]}{" FALLBACK" if rec["fallback"] else ""}; csv {rec["csv_sha256"][:12]}')
    say(f'  ERA5 days: {sum(v["copy"] == "cds" for v in rec["era5_days"].values())} from CDS, '
        f'{sum(v["copy"] != "cds" for v in rec["era5_days"].values())} from Google; runs: '
        f'{sorted({v["copy"] for v in rec["runs"].values()}, key=str)}')
    if a.dry_run: say(f'dry run: written to {folder}'); return
    sha = push(rec['target_day'])
    say(f'pushed {sha}; GitHub has it')
    say(f'=== end {pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}: ok')


# ------------------------------------------------------------------ self-test ------------------------------------------
def self_test():
    """Synthetic sources in the real formats, built through every path: collector files checked against their manifest;
    a tampered collector file refused; ERA5 hours missing from the CDS files taken from Google's copy (mocked); the
    run of D-1 late, so D falls back to D-2; the output equal to national.assemble's; the CSV read back exactly."""
    import zipfile, xarray as xr
    sys.path.insert(0, HERE); import synth_test as T
    tmp = tempfile.mkdtemp(); pub = os.path.join(tmp, 'public'); data = os.path.join(pub, 'prospective', 'weather', 'data')
    os.makedirs(data); S.ERA5_DIR = os.path.join(tmp, 'era5'); S.ARCO_DIR = os.path.join(tmp, 'arco'); S.RUNS_DIR = os.path.join(tmp, 'runs')
    os.makedirs(S.ERA5_DIR)
    global WEIGHTS
    if not os.path.exists(WEIGHTS):
        w, _ = T.weights(); WEIGHTS = os.path.join(tmp, 'w.nc')
        xr.Dataset({'w_pop': w, 'w_pv': w}).to_netcdf(WEIGHTS)
    w_pop, w_pv = N.load_weights(WEIGHTS)
    D = pd.Timestamp('2025-12-10'); ctx, hor, h, i0 = periods_for(D)
    t0 = (ctx[0] - 3 * N.DAY).normalize(); t1 = C.day_bounds(D)[1].normalize() + 4 * N.DAY
    tr = T.Truth(T.SEEDS[0], T.SCENARIOS['winter_error'], t0, t1)
    rows = []
    for k in range(N.SPLICE_DAYS + 1, 0, -1):               # collector files, with a manifest
        X = (D - k * N.DAY); name = f'ifs_0p25_{X:%Y%m%d}_00z_gb.nc'
        ds = T.run_ds(tr, X.tz_localize('UTC'), np.random.default_rng(k))
        ds.drop_vars('ssrd').merge(ds[['ssrd']].where(ds.step > pd.Timedelta(0))).to_netcdf(os.path.join(data, name))
        rows.append(dict(run_date=f'{X:%Y-%m-%d}', run_utc='00', resol='0p25', source='ecmwf', downloaded_utc='x',
                         file=name, sha256=S.sha256(os.path.join(data, name)), bytes=0, raw_sha256='', variables=''))
    with open(os.path.join(pub, 'prospective', 'weather', 'manifest.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    first = (ctx[0] + pd.Timedelta(minutes=15)).floor('1h') - pd.Timedelta(hours=1)
    last = pd.Timestamp(D - N.SPLICE_DAYS * N.DAY).tz_localize('UTC')
    full = T.era5_ds(tr, first, last)
    cut = pd.Timestamp(last.tz_localize(None)) - 2 * N.DAY  # the CDS files stop two days short; Google fills the rest
    months = pd.DatetimeIndex(full.valid_time.values)
    for ym in sorted({(t.year, t.month) for t in months if t < cut}):
        sel = full.sel(valid_time=[t for t in months if (t.year, t.month) == ym and t < cut])
        p = os.path.join(S.ERA5_DIR, f'era5_{ym[0]}_{ym[1]:02d}.nc'); d = p[:-3] + '_parts'; os.makedirs(d)
        sel[['t2m', 'u10', 'v10']].to_netcdf(os.path.join(d, 'data_stream-oper_stepType-instant.nc'))
        sel[['ssrd']].to_netcdf(os.path.join(d, 'data_stream-oper_stepType-accum.nc'))
        with zipfile.ZipFile(p, 'w') as z:
            for n in os.listdir(d): z.write(os.path.join(d, n), n)
    arco_calls = []
    def fake_arco(day):
        arco_calls.append(day); path = os.path.join(tmp, f'arco_{day:%Y%m%d}.nc')
        full.sel(valid_time=[t for t in months if t.normalize() == day]).to_netcdf(path)
        return xr.open_dataset(path).load(), path
    S._arco_day = fake_arco
    A.public_by = lambda date, steps: (pd.Timestamp(f'{date} 07:34', tz='UTC').to_pydatetime(), 'test')
    res = {}
    out, rec = build(D, public_root=pub)
    era5 = N.era5_national(full, w_pop, w_pv)
    runs = {}
    for k in range(N.SPLICE_DAYS + 1, 0, -1):
        X = D - k * N.DAY; _, p, i = N.ecmwf_national(xr.open_dataset(os.path.join(data, f'ifs_0p25_{X:%Y%m%d}_00z_gb.nc')), w_pop, w_pv)
        runs[X] = (p, i)
    direct, _ = N.assemble(D.date(), ctx.append(hor), era5, runs)
    res['W1 output equals national.assemble'] = bool(np.allclose(out[['t2m', 'wind', 'solar']], direct[['t2m', 'wind', 'solar']], atol=1e-9))
    res['W2 step-0 solar missing in the file is treated as zero'] = bool(np.isfinite(out['solar']).all())
    res['W3 ERA5 hours missing from CDS taken from Google'] = bool(len(arco_calls) >= 2 and any(v['copy'] != 'cds' for v in rec['era5_days'].values()))
    res['W4 every run from the collector, hash-checked'] = all(v['copy'] == 'collector' for v in rec['runs'].values())
    A.public_by = lambda date, steps: (pd.Timestamp(f'{date} 09:40', tz='UTC').to_pydatetime(), 'test')
    out2, rec2 = build(D, public_root=pub)
    res['W5 run of D-1 after the issue time: fallback to D-2, flagged'] = bool(rec2['fallback'] and rec2['horizon_run'] == f'{D - 2 * N.DAY:%Y-%m-%d}')
    folder = os.path.join(tmp, 'out'); r = write(out, rec, folder)
    back = pd.read_csv(os.path.join(folder, f'{rec["target_day"]}.csv'))
    res['W6 CSV reads back (4 decimals) and its hash is recorded'] = bool(
        np.allclose(back[['t2m_c', 'wind_ms', 'solar_wm2']].to_numpy(), out[['t2m', 'wind', 'solar']].to_numpy(), atol=6e-5)
        and r['csv_sha256'] == S.sha256(os.path.join(folder, f'{rec["target_day"]}.csv')) and len(back) == C.CONTEXT_LEN + h)
    victim = os.path.join(data, f'ifs_0p25_{D - 3 * N.DAY:%Y%m%d}_00z_gb.nc')
    with open(victim, 'ab') as f: f.write(b'x')
    try: build(D, public_root=pub); res['W7 tampered collector file refused'] = False
    except RuntimeError as e: res['W7 tampered collector file refused'] = 'does not match' in str(e)
    for k, v in res.items(): say(('PASS ' if v else 'FAIL ') + k)
    say('self-test: ' + ('PASS' if all(res.values()) else 'FAIL')); sys.exit(0 if all(res.values()) else 1)


if __name__ == '__main__':
    main()
