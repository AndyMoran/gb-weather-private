"""
Checkpoint 0 (gb-foundation-model-tests: checkpoint0/PLAN.md, frozen at tag checkpoint0-plan): LightGBM given exactly
the weather regression's information (weather/regression_weather.py).

Information, built from RegressionW's own arrays and weather functions, so both models see the same values:
    clock slot (0-47); day of week (categorical, Monday 0), bank holiday, 24 Dec-2 Jan, all from the regression's
    calendar rows; demand at D-2 (else D-9) and D-7 (else D-14), the regression's L2 and L7; temperature, wind and solar
    for the half-hour, unadjusted, by the regression's splice rule (weather_for).
Training: one model across all slots, re-fitted for each target day D on every half-hour of the complete days
D-366 .. D-2 whose target, lags and weather are finite. A target-day half-hour with any non-finite input gets no
forecast, as in the regression.
Settings (PLAN.md section 5): 'default' (LightGBM's defaults; objective l1), 'S1' (Misiurek et al., Energies 2026),
'S2' (Uzzaman et al., Electricity 2026). Every run: objective l1, seed 20261115, deterministic, a fixed thread count,
bagging_freq 1 where row sampling is set, no early stopping, no search.

Modes (run in environment gbcp0):
    --synthetic   every check below on synthetic data (development only; no real data read)
    --checks      WP1 checks 1-5 on the real backtest. Check 5 is the only one that touches outturn, and only with
                  corrupted demand. No forecast is saved.
    --run         the scored run: all three settings on every backtest day, written to
                  ~/gb-weather-data/results/checkpoint0/lightgbm_forecasts.csv. Errors are not computed here.
"""
import argparse, datetime as dt, hashlib, json, os, sys, time
import numpy as np, pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_weather as RWX                                    # load_real, build, load_rung1, paths
import regression_weather as RW
import common as C, baselines as B

SEED, THREADS = 20261115, 4
WX = RW.WX                                                   # ('t2m', 'wind', 'solar')
FEATURES = ['slot', 'dow', 'holiday', 'xmas', 'lag2', 'lag7', 't2m', 'wind', 'solar']
CAT = ['dow']
BASE = dict(objective='l1', seed=SEED, deterministic=True, force_row_wise=True, num_threads=THREADS, verbosity=-1)
SETTINGS = {
    'default': dict(num_boost_round=100, params=dict(BASE)),
    'S1': dict(num_boost_round=500, params=dict(BASE, learning_rate=0.05, num_leaves=31, max_depth=8,
                                                 min_data_in_leaf=20, bagging_fraction=0.8, bagging_freq=1,
                                                 feature_fraction=0.8)),
    'S2': dict(num_boost_round=500, params=dict(BASE, learning_rate=0.03, num_leaves=31)),
}
OUT = os.path.join(os.path.expanduser('~/gb-weather-data'), 'results', 'checkpoint0')
WP2_DAYS = os.path.join(RWX.PUBLIC, 'checkpoint0', 'wp2_days.csv')


def say(*a): print(f'{dt.datetime.now():%H:%M:%S}', *a, flush=True)


def lag_offsets(first, fallback):
    """The only lags allowed: at least two days back, as the regression. The leakage check calls this with 0."""
    assert first >= 2 and fallback >= 2, f'lag of {first}/{fallback} days reads demand later than D-2'
    return first, fallback


class LightGBMW(RW.RegressionW):
    """RegressionW's data (Y, L2, L7, CAL, complete, pos) and weather functions; a LightGBM fit in place of the
    per-slot least squares."""

    def __init__(self, series, holidays, lags=(2, 9, 7, 14)):
        super().__init__(series, holidays)
        a, b = lag_offsets(*lags[:2]); c, d = lag_offsets(*lags[2:])
        if (a, b, c, d) != (2, 9, 7, 14):                    # rebuild L2/L7 only if asked for other lags
            lag = lambda k: np.vstack([np.full((k, 48), np.nan), self.Y[:-k]]) if k else self.Y.copy()
            self.L2 = np.where(np.isfinite(lag(a)), lag(a), lag(b))
            self.L7 = np.where(np.isfinite(lag(c)), lag(c), lag(d))
        self.lags = (a, b, c, d)

    def training_days(self, D):
        tr = [self.pos[d] for d in pd.date_range(D - pd.Timedelta(days=C.TRAIN_DAYS + 1), D - pd.Timedelta(days=2))
              if d in self.pos and self.complete[self.pos[d]]]
        assert all(self.days[i] <= D - pd.Timedelta(days=2) for i in tr), 'training day after D-2'
        return tr

    def features(self, D, inputs_day, era5_hh):
        """Training matrix X, targets y, and the target day's 48 rows x0, in slot order. Rows with any non-finite
        input are dropped from training; x0 keeps them (as NaN) so the caller can refuse to forecast them."""
        D = pd.Timestamp(D).normalize()
        wx = RW._by_day_slot(RW.weather_for(D, inputs_day, era5_hh))
        get = lambda v, d: wx[v].reindex(index=[d], columns=B.SLOTS).to_numpy()[0] if d in wx[v].index else np.full(48, np.nan)
        tr = self.training_days(D)
        rows, ys = [], []
        for i in tr:
            cal = self.CAL[i]; dow = sum(k * cal[k] for k in range(1, 7))
            W = {v: get(v, self.days[i]) for v in WX}
            for j in range(48):
                r = [j, dow, cal[7], cal[8], self.L2[i, j], self.L7[i, j], W['t2m'][j], W['wind'][j], W['solar'][j]]
                if np.isfinite(self.Y[i, j]) and np.isfinite(r).all():
                    rows.append(r); ys.append(self.Y[i, j])
        a, b, c, d = self.lags
        def at(off):
            dd = D - pd.Timedelta(days=off)
            return self.Y[self.pos[dd]] if dd in self.pos else np.full(48, np.nan)
        x2 = np.where(np.isfinite(at(a)), at(a), at(b)); x7 = np.where(np.isfinite(at(c)), at(c), at(d))
        cal = B.calendar(D, self.holidays); dow = sum(k * cal[k] for k in range(1, 7))
        W0 = {v: get(v, D) for v in WX}
        x0 = np.array([[j, dow, cal[7], cal[8], x2[j], x7[j], W0['t2m'][j], W0['wind'][j], W0['solar'][j]]
                       for j in range(48)], float)
        return np.array(rows, float), np.array(ys, float), x0

    def fit_predict(self, X, y, x0, setting):
        s = SETTINGS[setting]
        ds = lgb.Dataset(pd.DataFrame(X, columns=FEATURES).astype({'dow': int}), y, categorical_feature=CAT,
                         free_raw_data=True)
        m = lgb.train(s['params'], ds, num_boost_round=s['num_boost_round'])
        ok = np.isfinite(x0).all(1)
        out = np.full(48, np.nan)
        if ok.any():
            out[ok] = m.predict(pd.DataFrame(x0[ok], columns=FEATURES).astype({'dow': int}))
        return out

    def forecast_all(self, D, inputs_day, era5_hh, settings=('default',)):
        """Forecasts for D's settlement periods, one array per setting, mapped from local slots as the regression."""
        X, y, x0 = self.features(D, inputs_day, era5_hh)
        D = pd.Timestamp(D).normalize()
        starts = C.target_starts(D)
        return {s: self._map(self.fit_predict(X, y, x0, s), starts) for s in settings}

    @staticmethod
    def _map(fc, starts):
        return np.array([fc[B.SLOTS.index(C.clock(t))] for t in starts])


# ------------------------------------------------------------------ independent feature build (check 4) ---------------
def features_independent(series, D, holidays, inputs_day, era5_hh):
    """A second build of the same matrices from raw half-hours, with plain loops and dictionaries, sharing no code with
    LightGBMW beyond the time rules and the calendar (as regression_weather_independent)."""
    D = pd.Timestamp(D).normalize()
    s = series.dropna()
    by, count = {}, {}
    for ts, v in zip(s.index, s.values):
        L = ts.tz_convert(C.UK); d = L.normalize().tz_localize(None); k = (d, L.strftime('%H:%M'))
        if k not in by: by[k] = v
        count[d] = count.get(d, 0) + 1
    own = {ts: tuple(r) for ts, r in zip(inputs_day.index, inputs_day[[v + '_unadjusted' for v in WX]].to_numpy())}
    lo = (pd.Timestamp(D).tz_localize(C.UK) - pd.Timedelta(days=367)).tz_convert('UTC')
    wx = {}
    for ts, r in zip(era5_hh.index, era5_hh[list(WX)].to_numpy()):
        if ts < lo.floor('30min'): continue
        L = ts.tz_convert(C.UK); k = (L.normalize().tz_localize(None), L.strftime('%H:%M'))
        if k not in wx: wx[k] = own.get(ts, tuple(r))
    for ts, r in own.items():
        if ts < lo.floor('30min'): continue
        L = ts.tz_convert(C.UK); k = (L.normalize().tz_localize(None), L.strftime('%H:%M'))
        if k not in wx: wx[k] = r
    def look(d, slot, lag):
        for extra in (0, 7):
            k = (d - pd.Timedelta(days=lag + extra), slot)
            if k in by and np.isfinite(by[k]): return by[k]
        return np.nan
    def cal_row(d):
        c = B.calendar(d, holidays)
        return [sum(k * c[k] for k in range(1, 7)), c[7], c[8]]
    rows, ys = [], []
    for td in pd.date_range(D - pd.Timedelta(days=366), D - pd.Timedelta(days=2), freq='D'):
        if count.get(td, 0) != len(C.target_starts(td)): continue
        for j, slot in enumerate(B.SLOTS):
            if (td, slot) not in by: continue
            w = wx.get((td, slot), (np.nan,) * 3)
            r = [j] + cal_row(td) + [look(td, slot, 2), look(td, slot, 7)] + list(w)
            if np.isfinite(by[(td, slot)]) and all(np.isfinite(r)):
                rows.append(r); ys.append(by[(td, slot)])
    x0 = [[j] + cal_row(D) + [look(D, slot, 2), look(D, slot, 7)] + list(wx.get((D, slot), (np.nan,) * 3))
          for j, slot in enumerate(B.SLOTS)]
    return np.array(rows, float), np.array(ys, float), np.array(x0, float)


# ------------------------------------------------------------------ checks ---------------------------------------------
def regression_covered():
    """Half-hours the weather regression forecast in the published backtest (forecasts.csv)."""
    f = pd.read_csv(os.path.join(RWX.PUBLIC, 'weather_study', 'results', 'forecasts.csv'),
                    usecols=['day', 'start_utc', 'regression_weather'])
    f = f[np.isfinite(f['regression_weather'])]
    return {pd.Timestamp(d): set(g['start_utc']) for d, g in f.groupby('day')}


def check_coverage(model, rec, era5_hh, covered, settings):
    missing, n = [], 0
    for D in sorted(rec):
        fc = model.forecast_all(D, rec[D]['wx'], era5_hh, settings)
        starts = [t.isoformat() for t in rec[D]['idx']]
        want = covered.get(D, set())
        for s, f in fc.items():
            for t, v in zip(starts, f):
                if t in want and not np.isfinite(v): missing.append((f'{D:%Y-%m-%d}', t, s))
        n += 1
        if n % 50 == 0: say(f'coverage: {n} of {len(rec)} days')
    lacking = [f'{D:%Y-%m-%d}' for D in covered if D not in rec]
    return not missing and not lacking, dict(days=n, missing=missing[:50], n_missing=len(missing),
                                            regression_days_not_built=lacking)


def check_leakage(series, hol, rec, era5_hh, days):
    """The training-window assertion holds on every day; a lag of 0 days (the target day's own demand) is refused."""
    m = LightGBMW(series, hol)
    for D in days: m.training_days(D)                         # raises if any training day is after D-2
    try:
        LightGBMW(series, hol, lags=(0, 7, 7, 14))
        refused = False
    except AssertionError:
        refused = True
    return refused, dict(days_checked=len(days), same_day_lag_refused=refused)


def check_determinism(model, rec, era5_hh, days):
    diffs = []
    for D in days:
        a = model.forecast_all(D, rec[D]['wx'], era5_hh, tuple(SETTINGS))
        b = model.forecast_all(D, rec[D]['wx'], era5_hh, tuple(SETTINGS))
        for s in SETTINGS:
            if not np.array_equal(a[s], b[s], equal_nan=True): diffs.append((f'{D:%Y-%m-%d}', s))
    return not diffs, dict(days=len(days), differing=diffs)


def check_feature_build(model, series, hol, rec, era5_hh, days):
    bad = []
    for D in days:
        X1, y1, x01 = model.features(D, rec[D]['wx'], era5_hh)
        X2, y2, x02 = features_independent(series, D, hol, rec[D]['wx'], era5_hh)
        same = (X1.shape == X2.shape and np.array_equal(X1, X2) and np.array_equal(y1, y2)
                and np.array_equal(x01, x02, equal_nan=True))
        if not same: bad.append(dict(day=f'{D:%Y-%m-%d}', shapes=[list(X1.shape), list(X2.shape)]))
    return not bad, dict(days=len(days), mismatches=bad)


def check_corruption(series, hol, rec, era5_hh, days):
    """Permute complete days of demand (seed 20261115) before any feature is built; LightGBM (defaults) must do no
    better than 0.9 x the per-slot median of the corrupted training targets, on true outturn."""
    m = LightGBMW(series, hol)
    rng = np.random.default_rng(SEED)
    comp = np.where(m.complete)[0]
    perm = comp.copy(); rng.shuffle(perm)
    Y = m.Y.copy(); Y[comp] = m.Y[perm]; m.Y = Y
    lag = lambda k: np.vstack([np.full((k, 48), np.nan), m.Y[:-k]])
    m.L2 = np.where(np.isfinite(lag(2)), lag(2), lag(9)); m.L7 = np.where(np.isfinite(lag(7)), lag(7), lag(14))
    e_lgb, e_med = [], []
    for D in days:
        X, y, x0 = m.features(D, rec[D]['wx'], era5_hh)
        starts = C.target_starts(pd.Timestamp(D))
        f = LightGBMW._map(m.fit_predict(X, y, x0, 'default'), starts)
        med = np.array([np.median(y[X[:, 0] == j]) if (X[:, 0] == j).any() else np.nan for j in range(48)])
        g = LightGBMW._map(med, starts)
        truth = rec[D]['y']
        e_lgb.append(np.nanmean(np.abs(truth - f))); e_med.append(np.nanmean(np.abs(truth - g)))
    a, b = float(np.mean(e_lgb)), float(np.mean(e_med))
    return a >= 0.9 * b, dict(days=len(days), lightgbm_corrupted_mae=round(a, 1), median_mae=round(b, 1),
                              ratio=round(a / b, 4), threshold=0.9)


def run_checks(series, hol, rec, era5_hh, covered, wp2, ten):
    model = LightGBMW(series, hol)
    res = {}
    say('check 2: leakage'); res['2_leakage'] = check_leakage(series, hol, rec, era5_hh, sorted(rec))
    say('check 4: feature build'); res['4_feature_build'] = check_feature_build(model, series, hol, rec, era5_hh, ten)
    say('check 3: determinism'); res['3_determinism'] = check_determinism(model, rec, era5_hh, wp2)
    say('check 5: corruption'); res['5_corruption'] = check_corruption(series, hol, rec, era5_hh, wp2)
    say('check 1: coverage (all settings, every day)')
    res['1_coverage'] = check_coverage(model, rec, era5_hh, covered, tuple(SETTINGS))
    for k, (ok, info) in res.items(): say(('PASS ' if ok else 'FAIL ') + k, json.dumps(info)[:400])
    return all(ok for ok, _ in res.values()), {k: dict(passed=ok, **info) for k, (ok, info) in res.items()}


# ------------------------------------------------------------------ data ----------------------------------------------
def real_data():
    series, pubs, hol, days_inputs, meta, era5_hh, built = RWX.load_real(full=True)
    days = list(pd.date_range(RWX.FIRST, RWX.LAST))
    rec, excl = RWX.build(series, hol, pubs, days_inputs, meta, RWX.load_rung1(), era5_hh, days)
    return series, hol, rec, era5_hh


def synthetic_data():
    """As run_weather.mode_synthetic: synthetic demand that responds to synthetic temperature."""
    import synth
    series = synth.demand(); hol = synth.holidays()
    idx = pd.date_range(series.index[0], series.index[-1], freq='30min'); n = len(idx); h_ = np.arange(n)
    rng = np.random.default_rng(3)
    T = 10 + 6 * np.sin(2 * np.pi * h_ / (48 * 365)) + 3 * np.sin(2 * np.pi * (h_ % 48) / 48) + np.cumsum(rng.normal(0, 0.05, n))
    wx = pd.DataFrame({'t2m': T, 'wind': 5 + np.abs(np.cumsum(rng.normal(0, 0.1, n))) % 6,
                       'solar': np.clip(300 * np.sin(2 * np.pi * ((h_ % 48) / 48 - 0.25)), 0, None)}, index=idx)
    series = series - 300 * (wx['t2m'].reindex(series.index) - 10).fillna(0).values
    rec = {}
    for D in pd.date_range('2025-03-01', '2025-04-15'):
        end = C.input_end(D); hz, i0 = C.horizon(D)
        per = pd.date_range(end - C.CONTEXT_LEN * C.HALF_HOUR, end + (hz - 1) * C.HALF_HOUR, freq='30min')
        w = wx.reindex(per).copy()
        for v in WX: w[v + '_unadjusted'] = w[v]
        try:
            import inputs as I
            y, tidx = I.targets(series, D)
        except ValueError:
            continue
        rec[D] = dict(wx=w, y=y, idx=tidx)
    return series, hol, rec, wx


def sha(p):
    h = hashlib.sha256(); h.update(open(p, 'rb').read()); return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--synthetic', action='store_true'); g.add_argument('--checks', action='store_true')
    g.add_argument('--run', action='store_true')
    a = ap.parse_args()
    say(f'lightgbm {lgb.__version__}, numpy {np.__version__}, pandas {pd.__version__}, threads {THREADS}, '
        f'script sha256 {sha(__file__)[:16]}')
    if a.synthetic:
        series, hol, rec, era5_hh = synthetic_data()
        days = sorted(rec)
        covered = {D: {t.isoformat() for t in rec[D]['idx']} for D in days}
        ok, res = run_checks(series, hol, rec, era5_hh, covered, days[::3][:6], days[:3])
        say('synthetic: ' + ('PASS' if ok else 'FAIL')); return
    series, hol, rec, era5_hh = real_data()
    say(f'{len(rec)} backtest days built')
    if a.checks:
        wp2 = [pd.Timestamp(d) for d in pd.read_csv(WP2_DAYS)['day']]
        assert all(D in rec for D in wp2), 'a WP2 day is not in the backtest'
        ten = wp2[::2]
        ok, res = run_checks(series, hol, rec, era5_hh, regression_covered(), wp2, ten)
        os.makedirs(OUT, exist_ok=True)
        json.dump(dict(passed=ok, script_sha256=sha(__file__), lightgbm=lgb.__version__, checks=res),
                  open(os.path.join(OUT, 'checks.json'), 'w'), indent=1, default=str)
        say('WP1 checks: ' + ('ALL PASS' if ok else 'FAILED') + f'; written to {OUT}/checks.json')
        return
    if a.run:
        chk = os.path.join(OUT, 'checks.json')
        if not os.path.exists(chk) or not json.load(open(chk))['passed']:
            sys.exit('Run refused: WP1 checks have not passed (checks.json).')
        if json.load(open(chk))['script_sha256'] != sha(__file__):
            sys.exit('Run refused: the script changed after the checks passed; rerun --checks.')
        model = LightGBMW(series, hol); rows = []; t0 = time.time()
        for k, D in enumerate(sorted(rec)):
            fc = model.forecast_all(D, rec[D]['wx'], era5_hh, tuple(SETTINGS))
            for i, t in enumerate(rec[D]['idx']):
                rows.append(dict(day=f'{D:%Y-%m-%d}', start_utc=t.isoformat(),
                                 **{f'lightgbm_{s.lower()}': fc[s][i] for s in SETTINGS}))
            if (k + 1) % 50 == 0: say(f'{k + 1} of {len(rec)} days, {time.time() - t0:.0f} s')
        f = os.path.join(OUT, 'lightgbm_forecasts.csv')
        pd.DataFrame(rows).to_csv(f, index=False)
        say(f'written {f}: {len(rows)} half-hours; sha256 {sha(f)}')


if __name__ == '__main__':
    main()
