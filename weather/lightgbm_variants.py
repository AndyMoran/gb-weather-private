"""
LightGBM variants (gb-foundation-model-tests: lgbm_variants/PLAN.md, frozen at tag lgbm-variants-plan): five changes
to LightGBM S1, each aimed at how it learns the heating response, plus V2's control and the two-year regression.
Built on lightgbm_weather.py and regression_weather.py without changing either. Committed before it runs.

Forecasters (PLAN.md §4). Every one is re-fitted for each target day D, one day per call (draft rule 4.8), with S1's
settings (lightgbm_weather.SETTINGS['S1']: 500 rounds, learning rate 0.05, 31 leaves, depth 8, min 20 rows a leaf,
row and column sampling 0.8, bagging frequency 1, seed 20261115, deterministic, row-wise, 4 threads) unless stated:
    V1    the weather regression is fitted on the training days; LightGBM S1 is fitted to its in-sample residuals with
          S1's features; the forecast is the regression's forecast plus LightGBM's forecast of the residual
    V2    linear_tree=True, objective 'regression' (L2: LightGBM 4.7.0 refuses L1 with linear trees); slot and day of
          week categorical; every other feature standardised by its mean and standard deviation over the fit's rows
    V2c   V2 without linear_tree: the control that isolates the linear leaves
    V3    48 models, one per local clock slot, on that slot's rows; slot dropped as a feature
    V4    three features added: Tc (temperature minus its mean over the slot's training rows), Tc squared, Tc x wind,
          built as the weather regression builds them
    V5    training days D-731 to D-2 (two winters), on INDO extended back to March 2022 and ERA5 to March 2022
    REG2  the weather regression with training days D-731 to D-2 (shown beside V5, not compared)
Training rows, features and the splice rule for weather are LightGBM S1's (lightgbm_weather.LightGBMW.features), with
the window as a parameter; at 365 days they must equal S1's exactly (check 2).

Checks (PLAN.md §8), each written to lgbm_variants/checks.json in the public repository:
    1 leakage       every training day of every variant is on or before D-2, and on or after D-366 (D-731 for V5 and
                    REG2), on every backtest day; a same-day lag is refused
    2 independent   each variant's training matrix, targets and target-day rows rebuilt by a second function that shares
                    no code beyond the time rules and the calendar, on 10 sample days: exact for the features; V2's
                    standardisation and V4's terms to 1e-9 (relative and absolute); V1's and REG2's in-sample regression
                    fits within 0.1 MW of the normal equations (gate check 8's tolerance); the window function at 365
                    days equals S1's features exactly, and the regression at 365 days equals RegressionW exactly
    3 synthetic     on invented demand with a known quadratic heating curve and little noise: every variant runs
                    without error; V1's mean absolute error is at most 1.10 x the weather regression's (it keeps the curve
                    the regression is given) and at most a quarter of the heating term's mean absolute size (it recovers
                    the curve, rather than missing it)
    4 determinism   two runs of each variant on the 20 sample days (checkpoint0/wp2_days.csv) agree exactly
    5 call mode     each sample day forecast alone equals the same day forecast after every other sample day, by the
                    same object: no forecast depends on another target day
    6 reproduction  (added; REVISIONS.md) this script's S1, built with the window function at 365 days, equals
                    checkpoint 0's saved S1 forecasts exactly on the 20 sample days
    7 coverage      (added) every variant forecasts every half-hour the weather regression forecast, on every day:
                    checked at the end of --run, which refuses to write forecasts that miss one

Modes (run in gbcp0):
    --synthetic   checks 3, and 1, 2, 4 and 5 on synthetic data (development only; no real data read)
    --checks      checks 1-2 and 4-7 on the real backtest; writes lgbm_variants/checks.json
    --run         refused unless checks.json passed with this script's hash; every forecaster on every backtest day,
                  written to lgbm_variants/forecasts.csv in the public repository. No error is computed here.
    --cost        time per forecast day and peak memory, one forecaster per process, on the 20 sample days (PLAN.md §5)
    --drop-v5     with --checks, --run or --cost: run without V5 and REG2. Allowed only if V5's data are absent;
                  PLAN.md §4 drops V5 if its data cannot be fetched and checked by 3 December 2026.
"""
import argparse, datetime as dt, hashlib, json, os, resource, subprocess, sys, time
import numpy as np, pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_weather as RWX                                    # noqa: E402
import regression_weather as RW                              # noqa: E402
import lightgbm_weather as LW                                # noqa: E402
import common as C, baselines as B                           # noqa: E402

PUBLIC = RWX.PUBLIC
OUTDIR = os.path.join(PUBLIC, 'lgbm_variants')
V5_MANIFEST = os.path.join(OUTDIR, 'v5_data_manifest.json')
V5_INDO = os.path.join(PUBLIC, 'data', 'indo_2022.csv')
V5_ERA5 = os.path.expanduser('~/gb-weather-data/backtest/era5_halfhourly_v5_2022-03-15_2023-03-14.csv.gz')
S1 = LW.SETTINGS['S1']
WX = RW.WX
BASE_F = list(LW.FEATURES)                                    # slot, dow, holiday, xmas, lag2, lag7, t2m, wind, solar
V4_F = BASE_F + ['tc', 'tc2', 'tcw']
V3_F = [f for f in BASE_F if f != 'slot']
NUMERIC = ['holiday', 'xmas', 'lag2', 'lag7', 't2m', 'wind', 'solar']
VARIANTS = ('V1', 'V2', 'V2c', 'V3', 'V4', 'V5', 'REG2')
NEEDS_V5 = ('V5', 'REG2')
WIN = {'V5': 730, 'REG2': 730}                                # training days: D-(win+1) .. D-2
TOL_MW, TOL_REL = 1e-9, 1e-9                                # arithmetic: V2's scaling, V4's terms
TOL_FIT_MW = 0.1                                             # least squares against normal equations (gate check 8's)


def say(*a): print(f'{dt.datetime.now():%H:%M:%S}', *a, flush=True)


def sha(p):
    h = hashlib.sha256(); h.update(open(p, 'rb').read()); return h.hexdigest()


# ------------------------------------------------------------------ the window, as a parameter ------------------------
def weather_window(D, inputs_day, era5_hh, back_days):
    """RW.weather_for with its 367 days as a parameter: unadjusted weather from D-back_days to the end of D, D's inputs
    where they cover a half-hour, else ERA5."""
    lo = pd.Timestamp(D).tz_localize(C.UK) - pd.Timedelta(days=back_days)
    hi = C.day_bounds(D)[1]
    idx = pd.date_range(lo.tz_convert('UTC').floor('30min'), hi - C.HALF_HOUR, freq='30min')
    w = era5_hh.reindex(idx)[list(WX)].copy()
    own = inputs_day[[v + '_unadjusted' for v in WX]].rename(columns=lambda c: c.replace('_unadjusted', ''))
    w.loc[w.index.isin(own.index)] = own.reindex(w.index[w.index.isin(own.index)]).to_numpy()
    return w


class Model(LW.LightGBMW):
    """LightGBMW's data and rules, with the training window as a parameter."""

    def __init__(self, series, holidays, window=C.TRAIN_DAYS):
        super().__init__(series, holidays)
        self.window = window

    def training_days(self, D):
        lo, hi = D - pd.Timedelta(days=self.window + 1), D - pd.Timedelta(days=2)
        tr = [self.pos[d] for d in pd.date_range(lo, hi) if d in self.pos and self.complete[self.pos[d]]]
        assert all(lo <= self.days[i] <= hi for i in tr), 'training day outside D-(window+1) .. D-2'
        return tr

    def _wx(self, D, inputs_day, era5_hh):
        return RW._by_day_slot(weather_window(D, inputs_day, era5_hh, self.window + 2))

    def rows(self, D, inputs_day, era5_hh):
        """As LightGBMW.features, with the window: X, y, x0, and each training row's (day index, slot)."""
        D = pd.Timestamp(D).normalize()
        wx = self._wx(D, inputs_day, era5_hh)
        get = lambda v, d: wx[v].reindex(index=[d], columns=B.SLOTS).to_numpy()[0] if d in wx[v].index else np.full(48, np.nan)
        rows, ys, meta = [], [], []
        for i in self.training_days(D):
            cal = self.CAL[i]; dow = sum(k * cal[k] for k in range(1, 7))
            W = {v: get(v, self.days[i]) for v in WX}
            for j in range(48):
                r = [j, dow, cal[7], cal[8], self.L2[i, j], self.L7[i, j], W['t2m'][j], W['wind'][j], W['solar'][j]]
                if np.isfinite(self.Y[i, j]) and np.isfinite(r).all():
                    rows.append(r); ys.append(self.Y[i, j]); meta.append((i, j))
        a, b, c, d = self.lags
        def at(off):
            dd = D - pd.Timedelta(days=off)
            return self.Y[self.pos[dd]] if dd in self.pos else np.full(48, np.nan)
        x2 = np.where(np.isfinite(at(a)), at(a), at(b)); x7 = np.where(np.isfinite(at(c)), at(c), at(d))
        cal = B.calendar(D, self.holidays); dow = sum(k * cal[k] for k in range(1, 7))
        W0 = {v: get(v, D) for v in WX}
        x0 = np.array([[j, dow, cal[7], cal[8], x2[j], x7[j], W0['t2m'][j], W0['wind'][j], W0['solar'][j]]
                       for j in range(48)], float)
        return np.array(rows, float).reshape(-1, len(BASE_F)), np.array(ys, float), x0, meta

    def regression(self, D, inputs_day, era5_hh):
        """RegressionW.forecast with the window as a parameter. Returns the slot-order forecast and the in-sample fitted
        value of every training row it fits, keyed by (day index, slot)."""
        D = pd.Timestamp(D).normalize()
        wx = self._wx(D, inputs_day, era5_hh)
        get = lambda v, d: wx[v].reindex(index=[d], columns=B.SLOTS).to_numpy()[0] if d in wx[v].index else np.full(48, np.nan)
        tr = self.training_days(D)
        Wtr = {v: np.array([get(v, self.days[i]) for i in tr]).reshape(len(tr), 48) for v in WX}
        W0 = {v: get(v, D) for v in WX}
        cal = np.array(B.calendar(D, self.holidays))
        def at(off):
            d = D - pd.Timedelta(days=off)
            return self.Y[self.pos[d]] if d in self.pos else np.full(48, np.nan)
        x2 = np.where(np.isfinite(at(2)), at(2), at(9)); x7 = np.where(np.isfinite(at(7)), at(7), at(14))
        fc, fitted = np.full(48, np.nan), {}
        for j in range(48):
            T, Wd, S = Wtr['t2m'][:, j], Wtr['wind'][:, j], Wtr['solar'][:, j]
            base = np.column_stack([self.CAL[tr, :1], self.L2[tr, j], self.L7[tr, j], self.CAL[tr, 1:]])
            y = self.Y[tr, j]
            ok = np.isfinite(y) & np.isfinite(base).all(1) & np.isfinite(T) & np.isfinite(Wd) & np.isfinite(S)
            if ok.sum() < 20: continue
            tbar = T[ok].mean(); Tc = T - tbar
            X = np.column_stack([base, Tc, Tc ** 2, Wd, Tc * Wd, S])
            t0 = W0['t2m'][j] - tbar
            x0 = np.r_[cal[:1], x2[j], x7[j], cal[1:], t0, t0 ** 2, W0['wind'][j], t0 * W0['wind'][j], W0['solar'][j]]
            keep = np.r_[np.ones(base.shape[1], bool), RW.kept_terms(T[ok], Wd[ok], S[ok])]
            X, x0 = X[:, keep], x0[keep]
            b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            if np.isfinite(x0).all(): fc[j] = x0 @ b
            for k in np.nonzero(ok)[0]: fitted[(tr[k], j)] = float(X[k] @ b)
        return fc, fitted


# ------------------------------------------------------------------ the variants ---------------------------------------
def _frame(X, cols, cat):
    return pd.DataFrame(X, columns=cols).astype({c: int for c in cat})


def fit(X, y, x0, cols, cat=('dow',), params=None, rounds=None):
    p = params or S1['params']; n = rounds or S1['num_boost_round']
    out = np.full(len(x0), np.nan)
    if not len(y): return out
    m = lgb.train(p, lgb.Dataset(_frame(X, cols, cat), y, categorical_feature=list(cat), free_raw_data=True), num_boost_round=n)
    ok = np.isfinite(x0).all(1)
    if ok.any(): out[ok] = m.predict(_frame(x0[ok], cols, cat))
    return out


def standardise(X, x0, cols):
    """Numerical columns to (x - mean) / sd over the fit's rows; a column with sd 0 is centred only."""
    X, x0 = X.copy(), x0.copy()
    for c in NUMERIC:
        k = cols.index(c); mu = X[:, k].mean(); sd = X[:, k].std()
        sd = sd if sd > 0 else 1.0
        X[:, k] = (X[:, k] - mu) / sd; x0[:, k] = (x0[:, k] - mu) / sd
    return X, x0


def v4_terms(X, x0):
    """Tc, Tc squared and Tc x wind, Tc centred on the slot's mean temperature over its training rows."""
    j, T, W = X[:, 0].astype(int), X[:, 6], X[:, 7]
    tbar = np.full(48, np.nan)
    for s in range(48):
        if (j == s).any(): tbar[s] = T[j == s].mean()
    tc = T - tbar[j]; t0 = x0[:, 6] - tbar[x0[:, 0].astype(int)]
    return (np.column_stack([X, tc, tc ** 2, tc * W]), np.column_stack([x0, t0, t0 ** 2, t0 * x0[:, 7]]))


V2_PARAMS = dict(S1['params'], objective='regression')


def forecast(name, model, D, inputs_day, era5_hh):
    """One forecaster, one target day: a forecast for each of D's settlement periods."""
    starts = C.target_starts(pd.Timestamp(D).normalize())
    if name == 'REG2':
        fc, _ = model.regression(D, inputs_day, era5_hh)
        return LW.LightGBMW._map(fc, starts)
    X, y, x0, meta = model.rows(D, inputs_day, era5_hh)
    if name in ('S1', 'V5'):
        fc = fit(X, y, x0, BASE_F)
    elif name == 'V1':
        reg, fitted = model.regression(D, inputs_day, era5_hh)
        keep = np.array([m in fitted for m in meta], bool)
        resid = y[keep] - np.array([fitted[m] for m, k in zip(meta, keep) if k])
        fc = reg + fit(X[keep], resid, x0, BASE_F)
    elif name in ('V2', 'V2c'):
        Xs, x0s = standardise(X, x0, BASE_F)
        fc = fit(Xs, y, x0s, BASE_F, cat=('slot', 'dow'), params=dict(V2_PARAMS, linear_tree=(name == 'V2')))
    elif name == 'V3':
        fc = np.full(48, np.nan)
        for s in range(48):
            m = X[:, 0] == s
            fc[s] = fit(X[m][:, 1:], y[m], x0[s:s + 1, 1:], V3_F)[0] if m.any() else np.nan
    elif name == 'V4':
        X4, x04 = v4_terms(X, x0)
        fc = fit(X4, y, x04, V4_F)
    else:
        raise KeyError(name)
    return LW.LightGBMW._map(fc, starts)


def models_for(names, series, hol, series2=None):
    m365 = Model(series, hol)
    m730 = Model(series2, hol, window=730) if series2 is not None else None
    return {n: (m730 if n in NEEDS_V5 else m365) for n in names}


# ------------------------------------------------------------------ the independent build (check 2) -------------------
def indep(series, D, holidays, inputs_day, era5_hh, window):
    """lightgbm_weather.features_independent with the window as a parameter, plus each row's (day, slot): plain loops
    and dictionaries over raw half-hours, sharing no code with Model beyond the time rules and the calendar."""
    D = pd.Timestamp(D).normalize()
    s = series.dropna()
    by, count = {}, {}
    for ts, v in zip(s.index, s.values):
        L = ts.tz_convert(C.UK); d = L.normalize().tz_localize(None); k = (d, L.strftime('%H:%M'))
        if k not in by: by[k] = v
        count[d] = count.get(d, 0) + 1
    own = {ts: tuple(r) for ts, r in zip(inputs_day.index, inputs_day[[v + '_unadjusted' for v in WX]].to_numpy())}
    lo = (pd.Timestamp(D).tz_localize(C.UK) - pd.Timedelta(days=window + 2)).tz_convert('UTC')
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
        return [sum(k * c[k] for k in range(1, 7)), c[7], c[8]], c
    rows, ys, keys, regrows = [], [], [], []
    for td in pd.date_range(D - pd.Timedelta(days=window + 1), D - pd.Timedelta(days=2), freq='D'):
        if count.get(td, 0) != len(C.target_starts(td)): continue
        cr, c = cal_row(td)
        for j, slot in enumerate(B.SLOTS):
            if (td, slot) not in by: continue
            w = wx.get((td, slot), (np.nan,) * 3)
            r = [j] + cr + [look(td, slot, 2), look(td, slot, 7)] + list(w)
            if np.isfinite(by[(td, slot)]) and all(np.isfinite(r)):
                rows.append(r); ys.append(by[(td, slot)]); keys.append((td, j))
                regrows.append([c[0], r[4], r[5]] + list(c[1:]) + list(w))
    cr, c = cal_row(D)
    x0 = [[j] + cr + [look(D, slot, 2), look(D, slot, 7)] + list(wx.get((D, slot), (np.nan,) * 3))
          for j, slot in enumerate(B.SLOTS)]
    return (np.array(rows, float).reshape(-1, len(BASE_F)), np.array(ys, float), np.array(x0, float), keys,
            np.array(regrows, float))


def indep_regression_fitted(keys, ys, regrows):
    """In-sample fits of the weather regression, per slot, by the normal equations on the independent rows. Returns
    {(day, slot): fitted}. Slots with fewer than 20 rows are not fitted, as the regression."""
    out = {}
    for j in range(48):
        idx = [k for k, (_, s) in enumerate(keys) if s == j]
        if len(idx) < 20: continue
        R = regrows[idx]; y = ys[idx]
        T, W, S = R[:, -3], R[:, -2], R[:, -1]
        tbar = sum(T) / len(T); tc = T - tbar
        X = np.column_stack([R[:, :-3], tc, tc ** 2, W, tc * W, S])
        nb = R.shape[1] - 3
        kt = (max(T) - min(T) >= 0.01, max(W) - min(W) >= 0.01, max(S) - min(S) >= 1.0)
        keep = [True] * nb + [kt[0], kt[0], kt[1], kt[0] and kt[1], kt[2]]
        X = X[:, keep]
        b = np.linalg.solve(X.T @ X, X.T @ y)
        for k, f in zip(idx, X @ b): out[keys[k]] = float(f)
    return out


def close(a, b, atol=TOL_MW):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= atol + TOL_REL * np.abs(b)))


# ------------------------------------------------------------------ checks ---------------------------------------------
def check_leakage(models, days):
    seen = set()
    for n, m in models.items():
        if id(m) in seen: continue
        seen.add(id(m))
        for D in days: m.training_days(D)                     # raises on a day outside D-(window+1) .. D-2
    try:
        LW.lag_offsets(0, 7); refused = False
    except AssertionError:
        refused = True
    return refused, dict(days=len(days), windows=sorted({m.window for m in models.values()}), same_day_lag_refused=refused)


def check_independent(models, series, series2, hol, rec, era5, era52, days):
    bad = []
    for D in days:
        wx = rec[D]['wx']
        # the window function at 365 days equals S1's features, and the regression at 365 equals RegressionW
        m = models['V1']
        X, y, x0, meta = m.rows(D, wx, era5)
        Xs1, ys1, x0s1 = LW.LightGBMW.features(m, D, wx, era5)
        if not (np.array_equal(X, Xs1) and np.array_equal(y, ys1) and np.array_equal(x0, x0s1, equal_nan=True)):
            bad.append((f'{D:%Y-%m-%d}', 'window(365) != S1 features'))
        reg, fitted = m.regression(D, wx, era5)
        starts = C.target_starts(D)
        if not np.array_equal(LW.LightGBMW._map(reg, starts), RW.RegressionW.forecast(m, D, wx, era5), equal_nan=True):
            bad.append((f'{D:%Y-%m-%d}', 'regression(365) != RegressionW'))
        for win, mm, s, e in ((365, m, series, era5), (730, models.get('V5'), series2, era52)):
            if mm is None: continue
            Xa, ya, x0a, meta_a = mm.rows(D, wx, e)
            Xb, yb, x0b, keys, regrows = indep(s, D, hol, wx, e, win)
            if not (np.array_equal(Xa, Xb) and np.array_equal(ya, yb) and np.array_equal(x0a, x0b, equal_nan=True)):
                bad.append((f'{D:%Y-%m-%d}', f'rows, window {win}'))
                continue
            if win == 365:
                # V1: in-sample regression fits
                fb = indep_regression_fitted(keys, yb, regrows)
                fa = {(mm.days[i], j): v for (i, j), v in fitted.items()}
                common = sorted(set(fa) & set(fb))
                if set(fa) != set(fb) or not close([fa[k] for k in common], [fb[k] for k in common], atol=TOL_FIT_MW):
                    bad.append((f'{D:%Y-%m-%d}', 'V1 in-sample fits'))
                # V2: standardisation
                Xs, x0s = standardise(Xa, x0a, BASE_F)
                ok = True
                for c in NUMERIC:
                    k = BASE_F.index(c); col = [r[k] for r in Xb]
                    mu = sum(col) / len(col); sd = (sum((v - mu) ** 2 for v in col) / len(col)) ** 0.5 or 1.0
                    ok &= close(Xs[:, k], [(v - mu) / sd for v in col]) and close(x0s[:, k], [(r[k] - mu) / sd for r in x0b])
                if not ok: bad.append((f'{D:%Y-%m-%d}', 'V2 standardisation'))
                # V4: the regression's terms
                X4, x04 = v4_terms(Xa, x0a)
                tb = {}
                for r in Xb: tb.setdefault(int(r[0]), []).append(r[6])
                tb = {s_: sum(v) / len(v) for s_, v in tb.items()}
                t4 = [[r[6] - tb[int(r[0])], (r[6] - tb[int(r[0])]) ** 2, (r[6] - tb[int(r[0])]) * r[7]] for r in Xb]
                t04 = [[r[6] - tb.get(int(r[0]), np.nan), (r[6] - tb.get(int(r[0]), np.nan)) ** 2,
                        (r[6] - tb.get(int(r[0]), np.nan)) * r[7]] for r in x0b]
                if not (close(X4[:, -3:], t4) and np.allclose(x04[:, -3:], t04, rtol=TOL_REL, atol=TOL_MW, equal_nan=True)):
                    bad.append((f'{D:%Y-%m-%d}', 'V4 terms'))
                # V3: each slot's rows are the slot's rows of the independent matrix
                for s_ in range(48):
                    if not np.array_equal(Xa[Xa[:, 0] == s_][:, 1:], Xb[Xb[:, 0] == s_][:, 1:]):
                        bad.append((f'{D:%Y-%m-%d}', f'V3 slot {s_}')); break
            else:
                # REG2: in-sample fits at 730 days against the independent normal equations
                _, fit2 = mm.regression(D, wx, e)
                fb = indep_regression_fitted(keys, yb, regrows)
                fa = {(mm.days[i], j): v for (i, j), v in fit2.items()}
                common = sorted(set(fa) & set(fb))
                if set(fa) != set(fb) or not close([fa[k] for k in common], [fb[k] for k in common], atol=TOL_FIT_MW):
                    bad.append((f'{D:%Y-%m-%d}', 'REG2 in-sample fits'))
    return not bad, dict(days=len(days), mismatches=bad)


def check_determinism(names, models, rec, era5s, days):
    diffs = []
    for n in names:
        for D in days:
            a = forecast(n, models[n], D, rec[D]['wx'], era5s[n]); b = forecast(n, models[n], D, rec[D]['wx'], era5s[n])
            if not np.array_equal(a, b, equal_nan=True): diffs.append((n, f'{D:%Y-%m-%d}'))
    return not diffs, dict(days=len(days), differing=diffs)


def check_call_mode(names, models, rec, era5s, days, fresh):
    """Each day alone, by a fresh object, against the same day after every other sample day, by one object."""
    diffs = []
    for n in names:
        shared = models[n]
        for D in days[:-1]: forecast(n, shared, D, rec[D]['wx'], era5s[n])
        last = days[-1]
        a = forecast(n, shared, last, rec[last]['wx'], era5s[n])
        b = forecast(n, fresh(n), last, rec[last]['wx'], era5s[n])
        if not np.array_equal(a, b, equal_nan=True): diffs.append(n)
    return not diffs, dict(days=len(days), differing=diffs)


def check_reproduction(models, rec, era5, days):
    f = pd.read_csv(os.path.join(PUBLIC, 'checkpoint0', 'lightgbm_forecasts.csv'), usecols=['day', 'lightgbm_s1'])
    saved = {d: g['lightgbm_s1'].to_numpy(float) for d, g in f.groupby('day')}
    bad = []
    for D in days:
        a = forecast('S1', models['V1'], D, rec[D]['wx'], era5)
        if not np.array_equal(a, saved[f'{D:%Y-%m-%d}'], equal_nan=True): bad.append(f'{D:%Y-%m-%d}')
    return not bad, dict(days=len(days), differing=bad)


# ------------------------------------------------------------------ data ----------------------------------------------
def v5_data(series, era5_hh):
    """INDO and ERA5 extended back to March 2022, after checking both against lgbm_variants/v5_data_manifest.json.
    The frozen files' values are kept wherever the two overlap (the manifest's checks show they agree)."""
    import inputs as I
    m = json.load(open(V5_MANIFEST))
    if sha(V5_INDO) != m['indo']['sha256'] or sha(V5_ERA5) != m['era5']['sha256']:
        sys.exit('Refused: V5 data differ from lgbm_variants/v5_data_manifest.json')
    s22, _ = I.load_indo(V5_INDO)
    s2 = series.combine_first(s22)
    full = pd.date_range(s2.index.min(), s2.index.max(), freq='30min')
    s2 = s2.reindex(full)
    e22 = pd.read_csv(V5_ERA5, index_col=0, parse_dates=True)
    e2 = pd.concat([e22[e22.index < era5_hh.index.min()], era5_hh]).sort_index()
    assert not e2.index.has_duplicates
    return s2, e2


def real_data(drop_v5):
    series, pubs, hol, days_inputs, meta, era5_hh, built = RWX.load_real(full=True)
    rec, excl = RWX.build(series, hol, pubs, days_inputs, meta, RWX.load_rung1(), era5_hh, list(pd.date_range(RWX.FIRST, RWX.LAST)))
    if drop_v5:
        if os.path.exists(V5_MANIFEST): sys.exit('Refused: --drop-v5 while V5\'s data manifest exists (PLAN.md §4).')
        return series, hol, rec, era5_hh, None, None
    if not os.path.exists(V5_MANIFEST): sys.exit('Refused: no V5 data manifest; fetch V5\'s data or pass --drop-v5.')
    s2, e2 = v5_data(series, era5_hh)
    return series, hol, rec, era5_hh, s2, e2


def synthetic_data(quiet=False):
    """Invented demand with a known quadratic heating curve on invented temperature: synth.demand plus
    h(T) = -300 (T - 10) + 15 (T - 10)^2, so the curve is one the regression can represent. quiet=True replaces
    synth.demand's persistent noise (sd about 1,000 MW, which no model can forecast) with independent noise of sd 50 MW,
    so that check 3 measures how well the curve is recovered, not how much noise is added."""
    import synth, inputs as I
    series = synth.demand(); hol = synth.holidays()
    if quiet:
        loc = series.index.tz_convert(C.UK); h = np.asarray(loc.hour + loc.minute / 60, float); doy = np.asarray(loc.dayofyear)
        clean = (28000 + 6000 * np.cos(2 * np.pi * (doy - 15) / 365.25) + 5000 * np.sin(np.pi * (h - 6) / 15).clip(0)
                 + 2500 * np.exp(-((h - 18) ** 2) / 4) + np.where(np.asarray(loc.dayofweek) >= 5, -3000, 0.0))
        series = pd.Series(clean + np.random.default_rng(5).normal(0, 50, len(series)), index=series.index)
    idx = pd.date_range(series.index[0], series.index[-1], freq='30min'); n = len(idx); h_ = np.arange(n)
    rng = np.random.default_rng(3)
    T = 10 + 6 * np.sin(2 * np.pi * h_ / (48 * 365)) + 3 * np.sin(2 * np.pi * (h_ % 48) / 48) + np.cumsum(rng.normal(0, 0.05, n))
    wx = pd.DataFrame({'t2m': T, 'wind': 5 + np.abs(np.cumsum(rng.normal(0, 0.1, n))) % 6,
                       'solar': np.clip(300 * np.sin(2 * np.pi * ((h_ % 48) / 48 - 0.25)), 0, None)}, index=idx)
    t = wx['t2m'].reindex(series.index).to_numpy() - 10
    series = series + (-300 * t + 15 * t ** 2)
    rec = {}
    for D in pd.date_range('2025-03-01', '2025-03-30'):
        end = C.input_end(D); hz, i0 = C.horizon(D)
        per = pd.date_range(end - C.CONTEXT_LEN * C.HALF_HOUR, end + (hz - 1) * C.HALF_HOUR, freq='30min')
        w = wx.reindex(per).copy()
        for v in WX: w[v + '_unadjusted'] = w[v]
        try: y, tidx = I.targets(series, D)
        except ValueError: continue
        rec[D] = dict(wx=w, y=y, idx=tidx)
    return series, hol, rec, wx


# ------------------------------------------------------------------ modes ----------------------------------------------
def run_checks(names, series, hol, rec, era5, s2, e2, sample, ten, covered, synthetic=False):
    models = models_for(names, series, hol, s2)
    era5s = {n: (e2 if n in NEEDS_V5 else era5) for n in names}
    fresh = lambda n: (Model(s2, hol, window=730) if n in NEEDS_V5 else Model(series, hol))
    res = {}
    say('check 1: leakage'); res['1_leakage'] = check_leakage(models, sorted(rec))
    say('check 2: independent build'); res['2_independent'] = check_independent(models, series, s2, hol, rec, era5, e2, ten)
    say('check 4: determinism'); res['4_determinism'] = check_determinism(names, models, rec, era5s, sample)
    say('check 5: call mode'); res['5_call_mode'] = check_call_mode(names, models, rec, era5s, sample, fresh)
    if not synthetic:
        say('check 6: reproduction of S1'); res['6_reproduction'] = check_reproduction(models, rec, era5, sample)
    for k, (ok, info) in res.items(): say(('PASS ' if ok else 'FAIL ') + k, json.dumps(info, default=str)[:400])
    return all(ok for ok, _ in res.values()), {k: dict(passed=ok, **info) for k, (ok, info) in res.items()}


def synthetic():
    series, hol, rec, wx = synthetic_data()
    days = sorted(rec)
    names = VARIANTS
    ok, res = run_checks(names, series, hol, rec, wx, series, wx, days[::10][:3], days[:2], None, synthetic=True)
    # check 3: every variant runs; V1 recovers the curve the regression is given, on the quiet series
    series, hol, rec, wx = synthetic_data(quiet=True)
    days = sorted(rec)[::3]
    models = models_for(names, series, hol, series)
    reg = Model(series, hol)
    e = {n: [] for n in names + ('REG',)}; heat = []
    for D in days:
        truth = rec[D]['y']
        t = wx['t2m'].reindex(rec[D]['idx']).to_numpy() - 10; heat.append(np.mean(np.abs(-300 * t + 15 * t ** 2)))
        for n in names: e[n].append(np.nanmean(np.abs(truth - forecast(n, models[n], D, rec[D]['wx'], wx))))
        e['REG'].append(np.nanmean(np.abs(truth - LW.LightGBMW._map(reg.regression(D, rec[D]['wx'], wx)[0], C.target_starts(D)))))
    mae = {n: float(np.mean(v)) for n, v in e.items()}
    ran = all(np.isfinite(mae[n]) for n in names)
    h = float(np.mean(heat))
    rec_ok = mae['V1'] <= 1.10 * mae['REG'] and mae['V1'] <= 0.25 * h
    res['3_synthetic'] = dict(passed=bool(ran and rec_ok), mae_mw={k: round(v, 1) for k, v in mae.items()},
                              heating_term_mean_abs_mw=round(h, 1),
                              rule='every variant runs; V1 MAE <= 1.10 x the regression\'s, and <= 0.25 x the mean '
                                   'absolute heating term')
    say(('PASS ' if res['3_synthetic']['passed'] else 'FAIL ') + '3_synthetic', json.dumps(res['3_synthetic']))
    ok &= res['3_synthetic']['passed']
    json.dump(dict(passed=bool(ok), script_sha256=sha(__file__), lightgbm=lgb.__version__, checks=res),
              open(os.path.join(OUTDIR, 'synthetic.json'), 'w'), indent=1, default=str)
    say('synthetic: ' + ('PASS' if ok else 'FAIL'))
    return ok


def sample_days(rec):
    wp2 = [pd.Timestamp(d) for d in pd.read_csv(os.path.join(PUBLIC, 'checkpoint0', 'wp2_days.csv'))['day']]
    assert all(D in rec for D in wp2), 'a sample day is not in the backtest'
    return wp2, wp2[::2]


def checks(drop_v5):
    series, hol, rec, era5, s2, e2 = real_data(drop_v5)
    names = tuple(n for n in VARIANTS if not (drop_v5 and n in NEEDS_V5))
    sample, ten = sample_days(rec)
    ok, res = run_checks(names, series, hol, rec, era5, s2, e2, sample, ten, None)
    syn = os.path.join(OUTDIR, 'synthetic.json')
    syn_ok = os.path.exists(syn) and json.load(open(syn)).get('passed') and json.load(open(syn)).get('script_sha256') == sha(__file__)
    res['3_synthetic'] = dict(passed=bool(syn_ok), note='from lgbm_variants/synthetic.json, this script')
    ok &= bool(syn_ok)
    json.dump(dict(passed=bool(ok), script_sha256=sha(__file__), lightgbm=lgb.__version__, forecasters=names,
                   v5_dropped=drop_v5, checks=res), open(os.path.join(OUTDIR, 'checks.json'), 'w'), indent=1, default=str)
    say('checks: ' + ('ALL PASS' if ok else 'FAILED') + '; written to lgbm_variants/checks.json')


def run(drop_v5):
    chk = os.path.join(OUTDIR, 'checks.json')
    if not os.path.exists(chk) or not json.load(open(chk))['passed']: sys.exit('Run refused: checks have not passed.')
    c = json.load(open(chk))
    if c['script_sha256'] != sha(__file__): sys.exit('Run refused: the script changed after the checks passed.')
    if c['v5_dropped'] != drop_v5: sys.exit('Run refused: --drop-v5 differs from the checked run.')
    series, hol, rec, era5, s2, e2 = real_data(drop_v5)
    names = tuple(c['forecasters'])
    models = models_for(names, series, hol, s2)
    era5s = {n: (e2 if n in NEEDS_V5 else era5) for n in names}
    rows, t0 = [], time.time()
    for k, D in enumerate(sorted(rec)):
        fc = {n: forecast(n, models[n], D, rec[D]['wx'], era5s[n]) for n in names}
        for i, t in enumerate(rec[D]['idx']):
            rows.append(dict(day=f'{D:%Y-%m-%d}', start_utc=t.isoformat(), **{f'lgbm_{n.lower()}': fc[n][i] for n in names}))
        if (k + 1) % 20 == 0: say(f'{k + 1} of {len(rec)} days, {time.time() - t0:.0f} s')
    out = pd.DataFrame(rows)
    covered = LW.regression_covered()
    miss = [(n, r.day, r.start_utc) for n in names for r in out.itertuples()
            if r.start_utc in covered.get(pd.Timestamp(r.day), set()) and not np.isfinite(getattr(r, f'lgbm_{n.lower()}'))]
    if miss: sys.exit(f'Run refused to write: {len(miss)} half-hours the regression forecast have no variant forecast, '
                      f'e.g. {miss[:5]} (check 7)')
    f = os.path.join(OUTDIR, 'forecasts.csv')
    out.to_csv(f, index=False)
    json.dump(dict(forecasts_sha256=sha(f), script_sha256=sha(__file__), half_hours=len(rows), forecasters=names,
                   v5_dropped=drop_v5, seconds=round(time.time() - t0)), open(os.path.join(OUTDIR, 'forecasts.json'), 'w'), indent=1)
    say(f'written {f}: {len(rows)} half-hours; sha256 {sha(f)}')


def rss_mb():
    for line in open('/proc/self/status'):
        if line.startswith('VmRSS'): return int(line.split()[1]) / 1024
    return float('nan')


def cost_one(name, drop_v5):
    series, hol, rec, era5, s2, e2 = real_data(drop_v5)
    sample, _ = sample_days(rec)
    m = models_for((name,), series, hol, s2)[name]; e = e2 if name in NEEDS_V5 else era5
    rss_data = rss_mb()
    forecast(name, m, sample[0], rec[sample[0]]['wx'], e)               # warm-up, untimed
    times = []
    for D in sample:
        t = time.perf_counter(); forecast(name, m, D, rec[D]['wx'], e); times.append(time.perf_counter() - t)
    print(json.dumps(dict(name=name, seconds_per_day_median=round(float(np.median(times)), 3),
                          seconds_per_day_mean=round(float(np.mean(times)), 3), rss_after_data_mb=round(rss_data),
                          peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))))


def cost(drop_v5):
    names = [n for n in VARIANTS if not (drop_v5 and n in NEEDS_V5)]
    out = []
    for n in names:
        say(f'cost: {n}')
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--cost-one', n] + (['--drop-v5'] if drop_v5 else []),
                           capture_output=True, text=True, check=True)
        out.append(json.loads(r.stdout.strip().splitlines()[-1]))
    json.dump(dict(script_sha256=sha(__file__), threads=LW.THREADS, days=20, rows=out),
              open(os.path.join(OUTDIR, 'cost.json'), 'w'), indent=1)
    for r in out: say(json.dumps(r))


def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    for k in ('--synthetic', '--checks', '--run', '--cost'): g.add_argument(k, action='store_true')
    g.add_argument('--cost-one'); ap.add_argument('--drop-v5', action='store_true')
    a = ap.parse_args()
    say(f'lightgbm {lgb.__version__}, numpy {np.__version__}, pandas {pd.__version__}, script sha256 {sha(__file__)[:16]}')
    if a.synthetic: sys.exit(0 if synthetic() else 1)
    if a.checks: return checks(a.drop_v5)
    if a.run: return run(a.drop_v5)
    if a.cost: return cost(a.drop_v5)
    if a.cost_one: return cost_one(a.cost_one, a.drop_v5)


if __name__ == '__main__':
    main()
