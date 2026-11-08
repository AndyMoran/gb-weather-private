"""
The regression with weather (analysis plan, Section 4): rung 1's weekly-lag regression (code/baselines.py), plus, in
each local clock slot's least-squares fit, five weather terms for that half-hour:
    Tc = temperature minus its mean over the slot's fit rows, Tc squared, wind, Tc x wind (wind chill), solar.
Re-estimated for each target day D on the complete days D-366 .. D-2, exactly as rung 1.

Weather, unadjusted (decision 13 does not touch the regression): for each half-hour, from D's built inputs (their
`_unadjusted` columns) where they cover it, so ECMWF's runs for D-10 .. D-2 and the target day itself; otherwise from
raw ERA5 on the same half-hours (national.era5_halfhourly). That is the splice rule relative to D.

Fit rows for a slot: training days whose demand, both lags and all three weather values are finite. The temperature
mean is taken over the fit rows. A weather variable whose range across a slot's fit rows is below its tolerance
(solar 1 W/m2, temperature 0.01 C, wind 0.01 m/s) is left out of that slot's fit, with the terms built from it
(temperature: Tc, Tc squared, Tc x wind; wind: wind, Tc x wind; solar: solar). In practice only solar at night is left
out. The fit then never depends on how a solver treats a column that is zero or nearly so. Clock-time rules (spring gap, autumn repeat: first occurrence) are rung 1's.

`regression_weather_independent` is the second implementation for gate check 8: plain loops and normal equations,
sharing no code with RegressionW beyond the time rules and the calendar in rung 1's code.
"""
import numpy as np, pandas as pd
import common as C, baselines as B

WX = ('t2m', 'wind', 'solar')
TOL = {'t2m': 0.01, 'wind': 0.01, 'solar': 1.0}                 # a variable with less range is left out of the slot


def kept_terms(T, W, S):
    """Which of the five weather terms (Tc, Tc^2, wind, Tc x wind, solar) a slot keeps, from the fit rows' ranges."""
    t, w, s = (max(x) - min(x) >= TOL[v] for x, v in ((T, 't2m'), (W, 'wind'), (S, 'solar')))
    return [t, t, w, t and w, s]


def weather_for(D, inputs_day, era5_hh):
    """Unadjusted weather on every half-hour from D-367 to the end of D: D's inputs where they cover it, else ERA5."""
    lo = pd.Timestamp(D).tz_localize(C.UK) - pd.Timedelta(days=367)
    hi = C.day_bounds(D)[1]
    idx = pd.date_range(lo.tz_convert('UTC').floor('30min'), hi - C.HALF_HOUR, freq='30min')
    w = era5_hh.reindex(idx)[list(WX)].copy()
    own = inputs_day[[v + '_unadjusted' for v in WX]].rename(columns=lambda c: c.replace('_unadjusted', ''))
    w.loc[w.index.isin(own.index)] = own.reindex(w.index[w.index.isin(own.index)]).to_numpy()
    return w


def _by_day_slot(w):
    loc = w.index.tz_convert(C.UK)
    df = w.assign(day=loc.normalize().tz_localize(None), slot=loc.strftime('%H:%M'))
    df = df[~df.duplicated(['day', 'slot'], keep='first')]
    return {v: df.pivot(index='day', columns='slot', values=v) for v in WX}


class RegressionW(B.Regression):
    def forecast(self, day, inputs_day, era5_hh):
        D = pd.Timestamp(day).normalize()
        wx = _by_day_slot(weather_for(D, inputs_day, era5_hh))
        get = lambda v, d: wx[v].reindex(index=[d], columns=B.SLOTS).to_numpy()[0] if d in wx[v].index else np.full(48, np.nan)
        tr = [self.pos[d] for d in pd.date_range(D - pd.Timedelta(days=C.TRAIN_DAYS + 1), D - pd.Timedelta(days=2))
              if d in self.pos and self.complete[self.pos[d]]]
        assert all(self.days[i] <= D - pd.Timedelta(days=2) for i in tr), 'training day after D-2'
        Wtr = {v: np.array([get(v, self.days[i]) for i in tr]) for v in WX}
        W0 = {v: get(v, D) for v in WX}
        cal = np.array(B.calendar(D, self.holidays))
        def at(off):
            d = D - pd.Timedelta(days=off)
            return self.Y[self.pos[d]] if d in self.pos else np.full(48, np.nan)
        x2 = np.where(np.isfinite(at(2)), at(2), at(9)); x7 = np.where(np.isfinite(at(7)), at(7), at(14))
        fc = np.full(48, np.nan)
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
            nb = base.shape[1]
            keep = np.r_[np.ones(nb, bool), kept_terms(T[ok], Wd[ok], S[ok])]
            X, x0 = X[:, keep], x0[keep]
            b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            if np.isfinite(x0).all(): fc[j] = x0 @ b
        return np.array([fc[B.SLOTS.index(C.clock(t))] for t in C.target_starts(D)])


def regression_weather_independent(series, day, holidays, inputs_day, era5_hh):
    """Second implementation for the gate: loops over raw half-hours and solves the normal equations."""
    D = pd.Timestamp(day).normalize()
    s = series[series.index + C.HALF_HOUR <= C.input_end(D)].dropna()
    by, count = {}, {}
    for ts, v in zip(s.index, s.values):
        L = ts.tz_convert(C.UK); d = L.normalize().tz_localize(None); k = (d, L.strftime('%H:%M'))
        if k not in by: by[k] = v
        count[d] = count.get(d, 0) + 1
    own = {ts: tuple(r) for ts, r in zip(inputs_day.index, inputs_day[[v + '_unadjusted' for v in WX]].to_numpy())}
    wx = {}
    for ts, r in zip(era5_hh.index, era5_hh[list(WX)].to_numpy()):
        L = ts.tz_convert(C.UK); k = (L.normalize().tz_localize(None), L.strftime('%H:%M'))
        if k not in wx: wx[k] = own.get(ts, tuple(r))
    for ts, r in own.items():                                  # the target day itself, and anything ERA5 lacks
        L = ts.tz_convert(C.UK); k = (L.normalize().tz_localize(None), L.strftime('%H:%M'))
        if k not in wx: wx[k] = r
    def look(d, slot, lag):
        for extra in (0, 7):
            k = (d - pd.Timedelta(days=lag + extra), slot)
            if k in by: return by[k]
        return np.nan
    out = []
    for t in C.target_starts(D):
        slot = C.clock(t); rows, y = [], []
        for td in pd.date_range(D - pd.Timedelta(days=366), D - pd.Timedelta(days=2), freq='D'):
            if count.get(td, 0) != len(C.target_starts(td)) or (td, slot) not in by: continue
            w = wx.get((td, slot), (np.nan,) * 3)
            cal = B.calendar(td, holidays); r = [cal[0], look(td, slot, 2), look(td, slot, 7)] + cal[1:] + list(w)
            if all(np.isfinite(r)): rows.append(r); y.append(by[(td, slot)])
        tbar = sum(r[-3] for r in rows) / len(rows)
        X = [r[:-3] + [r[-3] - tbar, (r[-3] - tbar) ** 2, r[-2], (r[-3] - tbar) * r[-2], r[-1]] for r in rows]
        cal = B.calendar(D, holidays); w = wx[(D, slot)]; t0 = w[0] - tbar
        x0 = [cal[0], look(D, slot, 2), look(D, slot, 7)] + cal[1:] + [t0, t0 ** 2, w[1], t0 * w[1], w[2]]
        nb = len(x0) - 5
        kt = [rr[-3] for rr in rows], [rr[-2] for rr in rows], [rr[-1] for rr in rows]
        tmin, tmax = min(kt[0]), max(kt[0]); wmin, wmax = min(kt[1]), max(kt[1]); smin, smax = min(kt[2]), max(kt[2])
        keep_t, keep_w, keep_s = tmax - tmin >= 0.01, wmax - wmin >= 0.01, smax - smin >= 1.0
        drop = [nb + k for k, keep in enumerate((keep_t, keep_t, keep_w, keep_t and keep_w, keep_s)) if not keep]
        X = np.array([[v for i, v in enumerate(r) if i not in drop] for r in X])
        x0 = np.array([v for i, v in enumerate(x0) if i not in drop])
        b = np.linalg.solve(X.T @ X, X.T @ np.array(y))
        out.append(float(x0 @ b))
    return np.array(out)
