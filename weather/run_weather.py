"""
Run the weather study (analysis plan, Sections 6 and 8).

Modes:
  models-selftest      the three models with weather on synthetic series only: finite, exact horizon, repeatable,
                       and moved by the horizon's temperature. No real demand.
  dry [--corrupt]      before the freeze: the real backtest inputs (the days built so far), the stand-in forecaster
                       with weather and the regression with weather; every gate check that does not need the lock.
                       NO forecast is compared with outturn: no error is computed or printed. --corrupt then corrupts
                       one input per check and confirms each check fails.
  synthetic            the whole scoring pipeline (forecasts, gates, Section 6's statistics) on synthetic demand and
                       weather, with the stand-in for every model: tests the statistics before they are frozen.
  score                after the freeze only (WEATHER_LOCK.json and the tag): the backtest, every forecaster, gates,
                       Section 6. Refused before the freeze, after a passing attempt, or with changed files.

Run from anywhere, in `gbwx`:   python run_weather.py <mode>
"""
import datetime as dt, glob, gzip, hashlib, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); PRIVATE = os.path.dirname(HERE)
PUBLIC = os.path.expanduser(os.environ.get('GBFM_PUBLIC', '~/gb-foundation-model-tests'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(PUBLIC, 'code'))
import common as C, inputs as I, baselines as B, stats as ST
import national as N, models_weather as MW, regression_weather as RW

DATA = os.path.expanduser('~/gb-weather-data')
BT = os.path.join(DATA, 'backtest')
OUT = os.path.join(DATA, 'results')
FIRST, LAST = pd.Timestamp('2024-03-16'), pd.Timestamp('2026-10-31')
MODELS = ('chronos2', 'timestral', 'timesfm25')
FORECASTERS = MODELS + ('regression',)
VARS = ('t2m', 'wind', 'solar')
LEVEL = 0.05 / 4                                             # Section 6.1: four tests, 0.0125 each
BOOT, BLOCK, SEED_BT, SEED_PR, SEED_HOL = 2000, 7, 20261120, 20270401, 20261121
BLOCKS = {'00-04': 0, '04-08': 1, '08-12': 2, '12-16': 3, '16-20': 4, '20-24': 5}
GATE_DAYS = ('2024-03-31', '2025-01-15', '2026-10-25')         # rung 1's: a spring change, a winter day, an autumn change
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def say(*a): print(f'{dt.datetime.now():%H:%M:%S}', *a, flush=True)


# ------------------------------------------------------------------ loading --------------------------------------------
def load_inputs(path_csv, path_jl):
    t = pd.read_csv(path_csv, parse_dates=['period_start_utc'])
    t = t.rename(columns={'t2m_c': 't2m', 'wind_ms': 'wind', 'solar_wm2': 'solar', 't2m_c_unadjusted': 't2m_unadjusted',
                          'wind_ms_unadjusted': 'wind_unadjusted', 'solar_wm2_unadjusted': 'solar_unadjusted'})
    days = {pd.Timestamp(d): g.set_index('period_start_utc').drop(columns='target_day') for d, g in t.groupby('target_day')}
    meta = {pd.Timestamp(json.loads(l)['target_day']): json.loads(l) for l in open(path_jl)}
    return days, meta


def load_rung1():
    f = pd.read_csv(os.path.join(PUBLIC, 'results', 'attempt2', 'forecasts.csv'), parse_dates=['start_utc'])
    return {pd.Timestamp(d): g.sort_values('period') for d, g in f.groupby('day')}


def build(series, hol, pubs, days_inputs, meta, rung1, era5_hh, days):
    """Per target day: targets, demand context, weather (adjusted for the models, both for the regression), the
    no-weather forecasts, NESO; or the reason the day is excluded."""
    rec, excl = {}, []
    for D in days:
        r = dict(day=D)
        m = meta.get(D, {})
        if 'dropped' in m: excl.append(dict(day=f'{D:%Y-%m-%d}', scope='all', reason='weather: ' + m['dropped'])); continue
        if D not in days_inputs: excl.append(dict(day=f'{D:%Y-%m-%d}', scope='all', reason='weather inputs not built')); continue
        try:
            r['y'], r['idx'] = I.targets(series, D)
            r['ctx'], cidx, r['filled'] = I.context(series, D)
        except ValueError as e:
            excl.append(dict(day=f'{D:%Y-%m-%d}', scope='all', reason=str(e))); continue
        w = days_inputs[D]; h, i0 = C.horizon(D)
        if len(w) != C.CONTEXT_LEN + h or not w.index[:C.CONTEXT_LEN].equals(cidx):
            excl.append(dict(day=f'{D:%Y-%m-%d}', scope='all', reason='weather periods differ from the demand context')); continue
        r['wx'] = w; r['cov'] = {v: w[v].to_numpy() for v in VARS}; r['meta'] = m
        r['fallback'] = bool(m.get('fallback', False))
        if rung1 is not None:
            g = rung1.get(D)
            if g is None or len(g) != len(r['y']):
                excl.append(dict(day=f'{D:%Y-%m-%d}', scope='all', reason='no rung 1 forecast')); continue
            r['nw'] = {k: g[k].to_numpy(float) for k in FORECASTERS}
            r['rung1_outturn'] = g['outturn'].to_numpy(float)
            r['neso'] = g['neso'].to_numpy(float) if g['neso'].notna().all() else None
        rec[D] = r
    return rec, excl


# ------------------------------------------------------------------ forecasting ----------------------------------------
def model_forecasts(model, rec, cov_key='cov', cov_override=None):
    """Point forecasts for each day's target periods, batching days that share a horizon (rung 1's RUN_BATCH)."""
    out, groups = {}, {}
    for D in rec: groups.setdefault(C.horizon(D), []).append(D)
    for (h, first), ds in groups.items():
        for i in range(0, len(ds), C.RUN_BATCH):
            b = ds[i:i + C.RUN_BATCH]
            cov = {v: [(cov_override or {}).get(D, rec[D][cov_key])[v] for D in b] for v in VARS}
            F = model.forecast([rec[D]['ctx'] for D in b], cov, h)
            for D, f in zip(b, F): out[D] = f[first:first + len(rec[D]['y'])]
    return out


# ------------------------------------------------------------------ gates ----------------------------------------------
def gates(rec, models, reg, series, hol, era5_hh, mode, lock=None, built=None):
    g, info = {}, {}
    if mode == 'score':
        g['1_data_hashes'] = all(sha(p) == h for p, h in lock['files_sha256'].items() if os.path.exists(p)) and \
            all(os.path.exists(p) for p in lock['files_sha256'])
    # 2: no look-ahead. Demand: rung 1's context asserts its cut-off. Weather: every target value from the run of D-1
    # (or D-2 on a recorded fallback); every history value from ERA5 before D-10 or a run of a day before D.
    ok = True
    for D, r in rec.items():
        m, w = r['meta'], r['wx']
        hr = pd.Timestamp(m['horizon_run'])
        ok &= hr == D - pd.Timedelta(days=2 if m.get('fallback') else 1)
        src = w['source'].to_numpy(); per = w.index
        era = src == 'era5'
        ok &= bool((per[era] + pd.Timedelta(minutes=15) < pd.Timestamp(D - pd.Timedelta(days=N.SPLICE_DAYS)).tz_localize('UTC')).all())
        runs = [pd.Timestamp(x.split()[1]) for x in src[~era]]
        ok &= all(x <= D - pd.Timedelta(days=1) for x in runs)
        ok &= bool((w['part'] == 'context').sum() == C.CONTEXT_LEN)
    g['2_no_lookahead'] = bool(ok)
    # 3: the weather inputs used are the published ones (backtest: the built files' hashes, as recorded at build)
    if built is not None:
        g['3_inputs_are_the_published_ones'] = all(sha(os.path.join(BT, f)) == h for f, h in built['outputs'].items())
    # 5: the no-weather forecasts are rung 1's, and rung 1's outturn is the outturn used here
    g['5_no_weather_are_rung1'] = all(np.array_equal(r['rung1_outturn'], r['y']) for r in rec.values() if 'rung1_outturn' in r)
    # 7: model outputs
    for name in [n for n in models if not n.endswith('_fc')]:
        fc = models[name + '_fc']; m = models[name]
        g[f'7_{name}_finite_and_length'] = all(np.isfinite(fc[D]).all() and len(fc[D]) == len(rec[D]['y']) for D in rec)
        groups = {}
        for D in rec: groups.setdefault(C.horizon(D), []).append(D)
        (h, first), ds = next(iter(groups.items())); batch = ds[:C.RUN_BATCH]
        again = model_forecasts(m, {D: rec[D] for D in batch})
        g[f'7_{name}_repeatable'] = all(np.array_equal(again[D], fc[D]) for D in batch)
        D = batch[0]; warm = {v: rec[D]['cov'][v].copy() for v in VARS}; warm['t2m'][C.CONTEXT_LEN:] += 2.0
        moved = model_forecasts(m, {D: rec[D]}, cov_override={D: warm})[D]
        g[f'7_{name}_moved_by_weather'] = bool(np.abs(moved - fc[D]).max() > 1e-6)
    # 8: the regression with weather, by an independent implementation, on rung 1's gate days that are present
    worst, n = 0.0, 0
    for d in GATE_DAYS:
        D = pd.Timestamp(d)
        if D not in rec: continue
        a = reg.forecast(D, rec[D]['wx'], era5_hh)
        b = RW.regression_weather_independent(series, D, hol, rec[D]['wx'], era5_hh)
        worst = max(worst, float(np.abs(a - b).max())); n += 1
    if n == 0:                                                 # none built yet: the first and last built days
        for D in (min(rec), max(rec)):
            a = reg.forecast(D, rec[D]['wx'], era5_hh); b = RW.regression_weather_independent(series, D, hol, rec[D]['wx'], era5_hh)
            worst = max(worst, float(np.abs(a - b).max())); n += 1
    g['8_regression_independent'] = bool(worst <= 0.1); info['8_days'] = n; info['8_max_abs_mw'] = worst
    # 9: spatial weights
    if mode != 'synthetic':
        g['9_spatial_weights'] = sha(os.path.join(DATA, 'weights', 'weights.nc')) == 'c3af7721b673386ef31285ff55deab71617bff774b7cb5c13e448dee543102be'
    if mode == 'score':
        from huggingface_hub import hf_hub_download
        import models as M
        for name, (repo, rev) in M.REVISIONS.items():
            g[f'9_{name}_weights'] = sha(hf_hub_download(repo, 'model.safetensors', revision=rev)) == lock['weights_sha256'][name]
    return g, info


# ------------------------------------------------------------------ statistics -----------------------------------------
def boot_idx(n, seed):
    rng = np.random.default_rng(seed)
    return np.stack([ST.block_indices(n, rng, BLOCK) for _ in range(BOOT)])


def stat_q1(a, b, idx):
    r = a.mean() / b.mean(); bs = a[idx].mean(1) / b[idx].mean(1)
    return dict(ratio=float(r), ci90=[float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))],
                p_one_sided=float((bs >= 1.0).mean()), n_days=len(a))


def stat_diff(a, b, idx):
    d = b.mean() - a.mean(); bs = b[idx].mean(1) - a[idx].mean(1)
    return dict(mw=float(d), ci90=[float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))])


def blocks_of(rec, D):
    """Six four-hour blocks of UK clock time from midnight, for each target half-hour of D."""
    return np.array([t.tz_convert(C.UK).hour // 4 for t in rec[D]['idx']])


def q4(rec, ds, fw, fn, idx):
    """Fractional error reductions by block, and Q4a's and Q4b's differences, pooling half-hours over days."""
    s1 = np.zeros((len(ds), 6)); s0 = np.zeros((len(ds), 6)); k = np.zeros((len(ds), 6))
    for i, D in enumerate(ds):
        b = blocks_of(rec, D); y = rec[D]['y']
        np.add.at(s1[i], b, np.abs(y - fw[D])); np.add.at(s0[i], b, np.abs(y - fn[D])); np.add.at(k[i], b, 1)
    def fr(cols, ii=None):
        a, z = (s1, s0) if ii is None else (s1[ii], s0[ii])
        ax = (-2, -1) if ii is not None else None
        return 1 - a[..., cols].sum(axis=ax) / z[..., cols].sum(axis=ax)
    out = {f'fr_{name}': float(fr([j])) for name, j in BLOCKS.items()}
    for name, num, den in (('q4a_evening_minus_overnight', [4], [0]), ('q4b_midday_minus_other_five', [3], [0, 1, 2, 4, 5])):
        d = float(fr(num) - fr(den)); bs = fr(num, idx) - fr(den, idx)
        out[name] = dict(points=d, ci90=[float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))])
    return out


def holiday_check(rec, ds, dm_w, dm_n, hol, seed):
    """Brief 4, Section 6: holiday excess = mean daily error on bank holidays / on ordinary days - 1, per arm."""
    is_hol = np.array([D in hol for D in ds])
    xmas = np.array([(D.month == 12 and D.day >= 24) or (D.month == 1 and D.day <= 2) for D in ds])
    ordinary = ~is_hol & ~xmas
    if is_hol.sum() == 0: return None
    rng = np.random.default_rng(seed); H, O = np.where(is_hol)[0], np.where(ordinary)[0]
    out = {}
    for name in dm_w:
        res = {}
        for arm, dm in (('with_weather', dm_w[name]), ('without', dm_n[name])):
            ex = dm[H].mean() / dm[O].mean() - 1
            bs = [dm[rng.choice(H, len(H))].mean() / dm[rng.choice(O, len(O))].mean() - 1 for _ in range(BOOT)]
            res[arm] = dict(excess=float(ex), ci90=[float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))])
        out[name] = res
    reg = out.get('regression', {}).get('with_weather', {}).get('excess')
    for name in MODELS:
        if name not in out or reg is None: continue
        w, n = out[name]['with_weather']['excess'], out[name]['without']['excess']
        out[name]['reading'] = ('holds' if (w >= 2 / 3 * n and w >= 2 * reg) else 'fails' if w <= reg else 'mixed')
    out['n_holidays'] = int(is_hol.sum()); out['n_ordinary'] = int(ordinary.sum())
    return out


def evaluate(rec, fw, seed, hol, other_timesfm=None, unadjusted=None):
    ds = sorted(rec); n = len(ds); idx = boot_idx(n, seed)
    dm = lambda fc: np.array([ST.daily_mae(rec[D]['y'], fc[D]) for D in ds])
    fn = {k: {D: rec[D]['nw'][k] for D in ds} for k in FORECASTERS}
    dm_w = {k: dm(fw[k]) for k in FORECASTERS}; dm_n = {k: dm(fn[k]) for k in FORECASTERS}
    res = dict(n_days=n, level=LEVEL, q1={}, q2={}, q3={}, q4={}, descriptive={})
    for k in FORECASTERS:
        s = stat_q1(dm_w[k], dm_n[k], idx); s['rejected'] = bool(s['p_one_sided'] <= LEVEL); res['q1'][k] = s
    neso_days = [i for i, D in enumerate(ds) if rec[D].get('neso') is not None]
    if neso_days:
        ii = np.array(neso_days); idn = boot_idx(len(ii), seed + 1)
        dneso = np.array([ST.daily_mae(rec[ds[i]]['y'], rec[ds[i]]['neso']) for i in ii])
        for k in MODELS:
            red = stat_diff(dm_w[k][ii], dm_n[k][ii], idn); gap = dm_n[k][ii].mean() - dneso.mean()
            red['gap_without_mw'] = float(gap); red['share_of_gap'] = float(red['mw'] / gap) if gap > 100 else None
            res['q2'][k] = red
    for k in MODELS:
        res['q3'][k] = dict(with_weather=stat_q1(dm_w[k], dm_w['regression'], idx),
                            without_weather_rung1=stat_q1(dm_n[k], dm_n['regression'], idx))
    for k in MODELS:
        res['q4'][k] = q4(rec, ds, fw[k], fn[k], idx)
    keep = [i for i, D in enumerate(ds) if not rec[D]['fallback']]
    if len(keep) < n:
        idk = boot_idx(len(keep), seed + 2)
        res['descriptive']['q1_without_fallback_days'] = {k: stat_q1(dm_w[k][keep], dm_n[k][keep], idk) for k in FORECASTERS}
    if other_timesfm is not None:
        res['descriptive']['q1_timesfm_other_mode'] = stat_q1(dm(other_timesfm), dm_n['timesfm25'], idx)
    if unadjusted is not None:
        res['descriptive']['q1_unadjusted_inputs'] = {k: stat_q1(dm(unadjusted[k]), dm_n[k], idx) for k in unadjusted}
    winter = [i for i, D in enumerate(ds) if (D.month, D.day) >= (11, 20) or (D.month, D.day) <= (3, 31)]
    if winter:
        idw = boot_idx(len(winter), seed + 3)
        res['descriptive']['q1_winter_days'] = {k: stat_q1(dm_w[k][winter], dm_n[k][winter], idw) for k in FORECASTERS}
    res['descriptive']['bank_holidays'] = holiday_check(rec, ds, dm_w, dm_n, hol, SEED_HOL)
    res['descriptive']['mae_mw'] = {k: dict(with_weather=float(dm_w[k].mean()), without=float(dm_n[k].mean())) for k in FORECASTERS}
    return res, pd.DataFrame({f'{k}_{arm}': (dm_w if arm == 'w' else dm_n)[k] for k in FORECASTERS for arm in ('w', 'n')},
                             index=[f'{D:%Y-%m-%d}' for D in ds])


# ------------------------------------------------------------------ modes ----------------------------------------------
def mode_models_selftest():
    sys.path.insert(0, os.path.join(PRIVATE, 'feasibility'))
    rng = np.random.default_rng(7); T, h = C.CONTEXT_LEN, 79; n = T + h; t = np.arange(n)
    temp = 8 + 6 * np.sin(2 * np.pi * (t / 48 - 0.4)) + np.cumsum(rng.normal(0, 0.15, n))
    wind = np.abs(6 + np.cumsum(rng.normal(0, 0.2, n)))
    sun = np.clip(np.sin(2 * np.pi * ((t % 48) / 48 - 0.25)), 0, None) * 300
    dem = 30000 + 5000 * np.sin(np.pi * ((t % 48) / 48)) - 600 * (temp - 10) + rng.normal(0, 300, n)
    ctxs = [dem[:T], dem[:T] * 1.01]; cov = {'t2m': [temp, temp], 'wind': [wind, wind], 'solar': [sun, sun]}
    res = {}
    for name in ('stand-in',) + MODELS + ('timesfm25-other',):
        m = MW.get(name); F = m.forecast(ctxs, cov, h); F2 = m.forecast(ctxs, cov, h)
        warm = {**cov, 't2m': [temp + np.r_[np.zeros(T), np.full(h, 2.0)]] * 2}
        F3 = m.forecast(ctxs, warm, h)
        res[name] = dict(shape_ok=F.shape == (2, h), finite=bool(np.isfinite(F).all()), repeatable=bool(np.array_equal(F, F2)),
                         moved_by_weather_mw=float(np.abs(F3 - F).max()))
        say(name, res[name])
    ok = all(v['shape_ok'] and v['finite'] and v['repeatable'] and v['moved_by_weather_mw'] > 1 for v in res.values())
    say('models-selftest: ' + ('PASS' if ok else 'FAIL')); return ok


def load_real(first=None, last=None):
    sys.path.insert(0, os.path.join(PUBLIC, 'code')); import run as R1
    series, pubs, hol, perf, nd = R1.load_data()
    files = sorted(glob.glob(os.path.join(BT, 'inputs_*.csv.gz')))
    if not files: sys.exit('no built backtest inputs in ' + BT)
    f = files[-1]; tag = os.path.basename(f)[len('inputs_'):-len('.csv.gz')]
    days_inputs, meta = load_inputs(f, os.path.join(BT, f'days_{tag}.jsonl'))
    hh_path = os.path.join(BT, f'era5_halfhourly_{tag}.csv.gz')
    era5_hh = pd.read_csv(hh_path, index_col=0, parse_dates=True) if os.path.exists(hh_path) else None
    built = json.load(open(os.path.join(BT, f'build_{tag}.json')))
    return series, pubs, hol, days_inputs, meta, era5_hh, built


def mode_dry(corrupt):
    series, pubs, hol, days_inputs, meta, era5_hh, built = load_real()
    if era5_hh is None: sys.exit('dry run needs the regression\'s ERA5 history: rebuild without --no-regression-history')
    days = sorted(days_inputs)
    rec, excl = build(series, hol, pubs, days_inputs, meta, load_rung1(), era5_hh, days)
    say(f'dry run on {len(rec)} built days ({min(rec):%Y-%m-%d} .. {max(rec):%Y-%m-%d}); {len(excl)} excluded. No error is computed.')
    stand = MW.get('stand-in'); models = {'stand-in': stand, 'stand-in_fc': model_forecasts(stand, rec)}
    reg = RW.RegressionW(series, hol)
    G, info = gates(rec, models, reg, series, hol, era5_hh, 'dry', built=built)
    for k, v in G.items(): say(('PASS ' if v else 'FAIL ') + k)
    say(f'gate info: {info}')
    if not corrupt: return all(G.values())
    # each check, shown to fail on corrupted input
    D0 = sorted(rec)[len(rec) // 2]; res = {}
    def rerun(rec2, key, built2=built, models2=None, reg2=reg):
        g, _ = gates(rec2, models2 or models, reg2, series, hol, era5_hh, 'dry', built=built2)
        return g[key]
    r2 = {D: dict(r) for D, r in rec.items()}; r2[D0] = dict(rec[D0], meta=dict(rec[D0]['meta'], horizon_run=f'{D0:%Y-%m-%d}'))
    res['2 (horizon from a run of D itself)'] = not rerun(r2, '2_no_lookahead')
    b2 = dict(built, outputs={k: '0' * 64 for k in built['outputs']})
    res['3 (a published file altered)'] = not rerun(rec, '3_inputs_are_the_published_ones', built2=b2)
    r3 = {D: dict(r) for D, r in rec.items()}; r3[D0] = dict(rec[D0], rung1_outturn=rec[D0]['rung1_outturn'] + 1)
    res['5 (rung 1 outturn differs)'] = not rerun(r3, '5_no_weather_are_rung1')
    class Deaf:
        name = 'deaf'
        def forecast(self, c, cov, h): return MW.M.StandIn().forecast(c, h)
    deaf = Deaf(); m2 = {'deaf': deaf, 'deaf_fc': model_forecasts(deaf, rec)}
    res['7 (a forecaster that ignores the weather)'] = not rerun(rec, '7_deaf_moved_by_weather', models2=m2)
    class Off(RW.RegressionW):
        def forecast(self, *a): return super().forecast(*a) + 0.5
    res['8 (regression off by 0.5 MW)'] = not rerun(rec, '8_regression_independent', reg2=Off(series, hol))
    for k, v in res.items(): say(('PASS ' if v else 'FAIL ') + 'fails on corrupted input: ' + k)
    return all(G.values()) and all(res.values())


def mode_synthetic():
    """The scoring pipeline end to end on synthetic data: the stand-in with weather against the stand-in without."""
    import synth
    series = synth.demand(); hol = synth.holidays()
    idx = pd.date_range(series.index[0], series.index[-1], freq='30min'); n = len(idx); h_ = np.arange(n)
    rng = np.random.default_rng(3)
    T = 10 + 6 * np.sin(2 * np.pi * h_ / (48 * 365)) + 3 * np.sin(2 * np.pi * (h_ % 48) / 48) + np.cumsum(rng.normal(0, 0.05, n))
    wx = pd.DataFrame({'t2m': T, 'wind': 5 + np.abs(np.cumsum(rng.normal(0, 0.1, n))) % 6,
                       'solar': np.clip(300 * np.sin(2 * np.pi * ((h_ % 48) / 48 - 0.25)), 0, None)}, index=idx)
    series = series - 300 * (wx['t2m'].reindex(series.index) - 10).fillna(0).values      # demand that weather explains
    days = list(pd.date_range('2025-01-10', '2025-06-30'))
    days_inputs, meta, rung1 = {}, {}, {}
    stand = MW.M.StandIn()
    for D in days:
        end = C.input_end(D); hz, i0 = C.horizon(D)
        per = pd.date_range(end - C.CONTEXT_LEN * C.HALF_HOUR, end + (hz - 1) * C.HALF_HOUR, freq='30min')
        w = wx.reindex(per).copy()
        for v in VARS: w[v + '_unadjusted'] = w[v]
        w['part'] = ['context'] * C.CONTEXT_LEN + ['horizon'] * hz
        w['source'] = ['era5' if p + pd.Timedelta(minutes=15) < pd.Timestamp(D - pd.Timedelta(days=10)).tz_localize('UTC')
                       else f'ifs {D - pd.Timedelta(days=1):%Y-%m-%d}' for p in per]
        fb = D.day % 15 == 0                                   # a few fallback days, labelled consistently
        if fb: w.loc[w['part'] == 'horizon', 'source'] = f'ifs {D - pd.Timedelta(days=2):%Y-%m-%d}'
        days_inputs[D] = w; meta[D] = dict(horizon_run=f'{D - pd.Timedelta(days=2 if fb else 1):%Y-%m-%d}', fallback=fb)
        try:
            y, tidx = I.targets(series, D); ctx, _, _ = I.context(series, D)
        except ValueError:
            continue
        f = stand.forecast([ctx], hz)[0][i0:i0 + len(y)]
        rung1[D] = pd.DataFrame({'period': np.arange(1, len(y) + 1), 'outturn': y, 'neso': y + rng.normal(0, 300, len(y)),
                                 **{k: f for k in FORECASTERS}})
    rec, excl = build(series, hol, None, days_inputs, meta, rung1, wx, days)
    models = {'stand-in': MW.get('stand-in')}; models['stand-in_fc'] = model_forecasts(models['stand-in'], rec)
    reg = RW.RegressionW(series, hol)
    G, info = gates(rec, models, reg, series, hol, wx, 'synthetic')
    fw = {k: models['stand-in_fc'] for k in MODELS}
    fw['regression'] = {D: reg.forecast(D, rec[D]['wx'], wx) for D in rec}
    res, dm = evaluate(rec, fw, SEED_BT, hol, other_timesfm=fw['timesfm25'])
    for k, v in G.items(): say(('PASS ' if v else 'FAIL ') + k)
    say('Q1:', {k: (round(v['ratio'], 3), v['rejected']) for k, v in res['q1'].items()})
    say('Q4 (stand-in):', {k: round(v, 3) for k, v in res['q4']['chronos2'].items() if k.startswith('fr_')})
    q1r = res['q1']['regression']
    ok = all(G.values()) and q1r['rejected'] and q1r['ratio'] < 0.9 and res['n_days'] > 100 and \
        'q1_without_fallback_days' in res['descriptive'] and res['q2'] and res['q3'] and res['q4']
    say('synthetic pipeline: ' + ('PASS' if ok else 'FAIL') + f' ({res["n_days"]} days; synthetic demand responds to '
        f'temperature, so the regression with weather must beat it without: ratio {q1r["ratio"]:.3f}, rejected {q1r["rejected"]})')
    return ok


def mode_score():
    lockf = os.path.join(PRIVATE, 'WEATHER_LOCK.json')
    if not os.path.exists(lockf): sys.exit('Scoring refused: WEATHER_LOCK.json not found (the plan is not frozen).')
    lock = json.load(open(lockf))
    changed = [p for p, h in lock['code_sha256'].items() if sha(os.path.join(PRIVATE, p)) != h]
    if changed: sys.exit('Scoring refused: changed since the freeze (a deviation; log it first): ' + ', '.join(changed))
    done = sorted(glob.glob(os.path.join(OUT, 'backtest_attempt*')))
    for d in done:
        f = os.path.join(d, 'results.json')
        if not os.path.exists(f): sys.exit(f'Scoring refused: {d} has no results.json; log it first.')
        if json.load(open(f)).get('gate_passed'): sys.exit(f'Scoring refused: {os.path.basename(d)} already passed.')
    out = os.path.join(OUT, f'backtest_attempt{len(done) + 1}'); os.makedirs(out)
    series, pubs, hol, days_inputs, meta, era5_hh, built = load_real()
    days = list(pd.date_range(FIRST, LAST))
    rec, excl = build(series, hol, pubs, days_inputs, meta, load_rung1(), era5_hh, days)
    models, fw = {}, {}
    for name in MODELS:
        m = MW.get(name); models[name] = m; models[name + '_fc'] = model_forecasts(m, rec); fw[name] = models[name + '_fc']
        say(f'{name}: {len(fw[name])} days forecast')
    reg = RW.RegressionW(series, hol); fw['regression'] = {D: reg.forecast(D, rec[D]['wx'], era5_hh) for D in rec}
    other = model_forecasts(MW.get('timesfm25-other'), rec)
    unadj = {}
    for D in rec: rec[D]['cov_unadj'] = {v: rec[D]['wx'][v + '_unadjusted'].to_numpy() for v in VARS}
    for name in MODELS: unadj[name] = model_forecasts(models[name], rec, cov_key='cov_unadj')
    G, info = gates(rec, models, reg, series, hol, era5_hh, 'score', lock=lock, built=built)
    res, dm = evaluate(rec, fw, SEED_BT, hol, other_timesfm=other, unadjusted=unadj)
    rows = []
    for D in rec:
        for k, t in enumerate(rec[D]['idx']):
            row = dict(day=f'{D:%Y-%m-%d}', start_utc=t.isoformat(), outturn=rec[D]['y'][k], fallback=rec[D]['fallback'])
            for name in FORECASTERS: row[name + '_weather'] = fw[name][D][k]; row[name + '_no_weather'] = rec[D]['nw'][name][k]
            row['timesfm25_other_mode'] = other[D][k]; rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, 'forecasts.csv'), index=False)
    dm.to_csv(os.path.join(out, 'daily_mae.csv'), index_label='day')
    pd.DataFrame(excl, columns=['day', 'scope', 'reason']).to_csv(os.path.join(out, 'exclusions.csv'), index=False)
    json.dump(dict(run_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), n_days=len(rec),
                   n_excluded=len(excl), gates=G, gate_info=info, gate_passed=bool(all(G.values())), results=res,
                   lock_sha256=sha(lockf)), open(os.path.join(out, 'results.json'), 'w'), indent=1, default=str)
    say(json.dumps({'gates': G, 'gate_passed': all(G.values())}, indent=1))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    ok = {'models-selftest': mode_models_selftest, 'synthetic': mode_synthetic, 'score': mode_score,
          'dry': lambda: mode_dry('--corrupt' in sys.argv)}.get(mode, lambda: sys.exit(__doc__))()
    sys.exit(0 if ok in (True, None) else 1)
