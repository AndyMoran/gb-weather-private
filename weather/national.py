"""
National weather inputs for one target day (Brief 4, Section 3: "The splice, defined" and "When a run is missing or
late"). This is the processing frozen and hashed before any weather forecast is made.

Sources, as xarray datasets on the 0.25 deg GB box (latitude 61..49, longitude -9..2.5):
  ERA5 (CDS NetCDF)  dims (valid_time, latitude, longitude); t2m K, u10 and v10 m/s, ssrd J/m2 accumulated over the
                     hour ENDING at valid_time
  ECMWF run          the collector's cropped file: dims (step, latitude, longitude); t2m K, u10, v10, ssrd J/m2
                     accumulated from the start of the run; coordinate `time` is the run time (00:00 UTC)

Each is first reduced to Great Britain with the fixed weights (weights.nc): temperature and wind speed by population,
solar radiation by solar capacity. Wind speed is computed from u and v at each grid point and time, before weighting.
A source then becomes two series:
  points     instantaneous temperature (deg C) and wind speed (m/s) at the source's own times
  intervals  mean solar radiation (W/m2) over each source interval: ERA5's hour (the hour's total / 3,600 s), or the
             ECMWF interval between consecutive steps (the difference of the accumulations / the seconds between)

For target day D, every half-hour of the context and the horizon takes its value at the half-hour's MIDPOINT:
temperature and wind by linear interpolation between the assembled points, solar radiation from the interval that
contains the midpoint. Points are assembled per UTC day:
  - UTC days up to and including D-11: ERA5;
  - each UTC day X from D-10 to D-1: the 00:00 UTC run of X, steps 0 to 21 (its own day), or, if that run is missing
    or (for X = D-1) was not public at the issue time, the run of X-1, steps 24 to 45;
  - UTC day D and the first instant of D+1 (the horizon's remainder): the same run that gave day D-1, steps 24 to 48
    (or 48 to 72 from the run of D-2). So the whole horizon, from the cut-off to the end of D, comes from one run.
ERA5 never fills an ECMWF day. Interpolation runs across the seam at 00:00 UTC on D-10.

Decision 13: before assembly, the ERA5 days can be put on the runs' scale with history_factors (estimated from
D-70 to D-11) and apply_factors. The daily inputs use it; the unadjusted inputs are published alongside.
"""
import numpy as np, pandas as pd, xarray as xr

SPLICE_DAYS = 10
DAY = pd.Timedelta(days=1)


class Dropped(Exception):
    """The day cannot be built under the rules: it is dropped from every confirmatory analysis."""


def load_weights(path):
    w = xr.open_dataset(path)
    return _rounded(w['w_pop']), _rounded(w['w_pv'])


def _rounded(x):
    return x.assign_coords(latitude=np.round(x.latitude.values, 2), longitude=np.round(x.longitude.values, 2))


def _weighted(field, w):
    f, ww = xr.align(_rounded(field), w, join='exact')       # the grids must be identical, point for point
    # A field missing anywhere the weights are positive gives NaN (skipna off), so a gap can never pass as a value.
    return (f * ww).sum(('latitude', 'longitude'), skipna=False).values


def era5_national(ds, w_pop, w_pv):
    tc = 'valid_time' if 'valid_time' in ds.dims else 'time'
    t = pd.DatetimeIndex(ds[tc].values).tz_localize('UTC')
    points = pd.DataFrame({'t2m': _weighted(ds['t2m'], w_pop) - 273.15,
                           'wind': _weighted(np.hypot(ds['u10'], ds['v10']), w_pop)}, index=t)
    intervals = pd.DataFrame({'start': t - pd.Timedelta(hours=1), 'end': t,
                              'solar': _weighted(ds['ssrd'], w_pv) / 3600.0})
    bad = points.isna().any(axis=1).to_numpy() | intervals['solar'].isna().to_numpy()
    if bad.any(): raise Dropped(f'ERA5 missing values at {list(t[bad][:5])}')
    return points.sort_index(), intervals.sort_values('end').reset_index(drop=True)


def ecmwf_national(ds, w_pop, w_pv):
    run = pd.Timestamp(np.asarray(ds['time'].values).ravel()[0]).tz_localize('UTC')
    steps = pd.to_timedelta(ds['step'].values)
    order = np.argsort(steps); steps = steps[order]
    vt = run + steps
    temp = _weighted(ds['t2m'], w_pop)[order] - 273.15
    wind = _weighted(np.hypot(ds['u10'], ds['v10']), w_pop)[order]
    # Runs before 6 March 2024 have no solar radiation: their solar is missing (NaN), never zero. They serve
    # decision 13's temperature and wind factors; a half-hour that would take solar from them drops the day.
    has_solar = 'ssrd' in ds
    acc = _weighted(ds['ssrd'], w_pv)[order] if has_solar else np.full(len(steps), np.nan)   # J/m2 since the start
    # The accumulation at step 0 is zero by definition; some files omit it, which reads as missing. Set it to zero.
    # Any other missing value stops the build.
    if has_solar and np.isnan(acc[0]) and steps[0] == pd.Timedelta(0): acc[0] = 0.0
    for name, v in (('t2m', temp), ('wind', wind)) + ((('ssrd', acc),) if has_solar else ()):
        if np.isnan(v).any(): raise Dropped(f'run {run:%Y-%m-%d}: {name} missing at steps '
                                            f'{list((steps[np.isnan(v)] / pd.Timedelta(hours=1)).astype(int))}')
    points = pd.DataFrame({'t2m': temp, 'wind': wind, 'step_h': steps / pd.Timedelta(hours=1)}, index=vt)
    dt = np.diff(steps / pd.Timedelta(seconds=1))
    # Differences of rounded running totals can dip a little below zero at night; irradiance cannot: set them to zero.
    solar = np.diff(acc) / dt
    intervals = pd.DataFrame({'start': vt[:-1], 'end': vt[1:], 'solar': np.clip(solar, 0, None),
                              'step_h': (steps / pd.Timedelta(hours=1))[1:], 'solar_raw': solar})
    return run, points, intervals


FACTOR_WINDOW = (70, 11)                                     # UTC days D-70 .. D-11
MIN_DAYS = 30
DAYLIGHT_WM2 = 10.0


def history_factors(target_day, era5, runs, min_days=MIN_DAYS):
    """Decision 13: factors that put ERA5 on the ECMWF runs' scale, from the UTC days D-70 to D-11 that both cover.
    Each run's first day (steps 0 to 24) is compared with ERA5 at the same instants: temperature by the mean
    difference (added), wind by the ratio of totals (multiplied), solar by the ratio of totals over ERA5-daylight
    3-hour intervals (ERA5 3-hour mean above 10 W/m2; ECMWF 3-hour means from its accumulations, ERA5's from its
    hourly totals). A variable with fewer than `min_days` days is left unadjusted and flagged. Uses no demand."""
    D = pd.Timestamp(target_day).normalize().tz_localize(None)
    ep, ei = era5
    sol_end = pd.Series(ei['solar'].to_numpy(), index=pd.DatetimeIndex(ei['end']))
    dT = nT = w_ifs = w_era = s_ifs = s_era = 0.0; days_tw = days_s = 0
    for k in range(FACTOR_WINDOW[0], FACTOR_WINDOW[1] - 1, -1):
        X = D - k * DAY
        if X not in runs: continue
        p, i = runs[X]
        lo, hi = pd.Timestamp(X).tz_localize('UTC'), pd.Timestamp(X + DAY).tz_localize('UTC')
        pp = p[(p.index >= lo) & (p.index <= hi)]
        e = ep.reindex(pp.index)
        if len(pp) and not e[['t2m', 'wind']].isna().any().any() and not pp[['t2m', 'wind']].isna().any().any():
            dT += float((pp['t2m'] - e['t2m']).sum()); nT += len(pp)
            w_ifs += float(pp['wind'].sum()); w_era += float(e['wind'].sum()); days_tw += 1
        ii = i[(i['start'] >= lo) & (i['end'] <= hi)]
        if len(ii) and not ii['solar'].isna().any():
            es = np.array([sol_end[(sol_end.index > a) & (sol_end.index <= b)].mean()
                           if ((sol_end.index > a) & (sol_end.index <= b)).sum() == round((b - a) / pd.Timedelta(hours=1))
                           else np.nan for a, b in zip(ii['start'], ii['end'])])
            if not np.isnan(es).any():
                day = es > DAYLIGHT_WM2
                s_ifs += float(ii['solar'].to_numpy()[day].sum()); s_era += float(es[day].sum()); days_s += 1
    f = dict(window=f"{D - FACTOR_WINDOW[0] * DAY:%Y-%m-%d}..{D - FACTOR_WINDOW[1] * DAY:%Y-%m-%d}",
             days_temperature_wind=days_tw, days_solar=days_s, min_days=min_days, flagged=[])
    if days_tw >= min_days and nT and w_era > 0:
        f['t2m_add'] = dT / nT; f['wind_mult'] = w_ifs / w_era
    else:
        f['t2m_add'], f['wind_mult'] = 0.0, 1.0; f['flagged'] += ['temperature', 'wind']
    if days_s >= min_days and s_era > 0:
        f['solar_mult'] = s_ifs / s_era
    else:
        f['solar_mult'] = 1.0; f['flagged'] += ['solar']
    return f


def apply_factors(era5, f):
    """ERA5 on the runs' scale: temperature shifted, wind scaled, solar scaled in ERA5 daylight only (night stays zero)."""
    p, i = era5
    p = p.copy(); i = i.copy()
    p['t2m'] = p['t2m'] + f['t2m_add']; p['wind'] = p['wind'] * f['wind_mult']
    i['solar'] = np.where(i['solar'] > DAYLIGHT_WM2, i['solar'] * f['solar_mult'], i['solar'])
    return p, i


def _day(ts):
    return pd.Timestamp(ts.date())


def assemble(target_day, periods, era5, runs, late=(), factors=None):
    """National weather for every half-hour period of one forecast.

    target_day  the UK date D
    periods     UTC start times of the context and horizon half-hours (the demand study's periods), in order
    era5        (points, intervals) from era5_national, covering at least the context's ERA5 days
    runs        {UTC run date (Timestamp, midnight): (points, intervals) from ecmwf_national}
    late        run dates whose files were not public before the issue time (counted as missing for D)
    factors     history_factors(...) to put the ERA5 days on the runs' scale (decision 13), or None for raw ERA5
    Returns a DataFrame indexed by period start: t2m, wind, solar, source; and the run used for the horizon.
    """
    D = pd.Timestamp(target_day).normalize().tz_localize(None)
    late = {pd.Timestamp(x).normalize() for x in late}
    have = lambda r: r in runs and r not in late
    utc0 = lambda x: pd.Timestamp(x).tz_localize('UTC')

    pts, ivs, src_of_day = [], [], {}
    e_pts, e_ivs = apply_factors(era5, factors) if factors else era5
    seam = utc0(D - SPLICE_DAYS * DAY)
    pts.append(e_pts[e_pts.index < seam]); ivs.append(e_ivs[e_ivs['end'] <= seam])

    def take(run, lo, hi, last_point=False):
        p, i = runs[run]
        pp = p[(p.index >= lo) & ((p.index <= hi) if last_point else (p.index < hi))]
        ii = i[(i['start'] >= lo) & (i['end'] <= hi)]
        return pp, ii

    for k in range(SPLICE_DAYS, 0, -1):                       # UTC days D-10 .. D-1
        X = D - k * DAY
        if have(X): r = X
        elif have(X - DAY): r = X - DAY
        else: raise Dropped(f'{D:%Y-%m-%d}: no run for history day {X:%Y-%m-%d} or the day before')
        p, i = take(r, utc0(X), utc0(X + DAY))
        pts.append(p); ivs.append(i); src_of_day[X] = r
    rD = src_of_day[D - DAY]                                  # the run that gave D-1 gives the whole horizon
    p, i = take(rD, utc0(D), utc0(D + DAY), last_point=True)
    pts.append(p); ivs.append(i); src_of_day[D] = rD

    P = pd.concat(pts).sort_index(); I = pd.concat(ivs).sort_values('start').reset_index(drop=True)
    if P.index.has_duplicates: raise ValueError('duplicate point times in the assembled series')
    mids = pd.DatetimeIndex(periods) + pd.Timedelta(minutes=15)
    if mids[0] < P.index[0] or mids[-1] > P.index[-1]:
        raise Dropped(f'{D:%Y-%m-%d}: weather points do not span the periods ({P.index[0]} .. {P.index[-1]})')
    # One time unit throughout: pandas may hold the periods in microseconds and file times in nanoseconds,
    # and raw integers in different units never match.
    ns = lambda ix: pd.DatetimeIndex(ix).as_unit('ns').asi8
    x = ns(P.index); m = ns(mids)
    out = pd.DataFrame({'t2m': np.interp(m, x, P['t2m'].to_numpy()), 'wind': np.interp(m, x, P['wind'].to_numpy())},
                       index=pd.DatetimeIndex(periods))
    st = ns(I['start']); en = ns(I['end'])
    k = np.searchsorted(st, m, side='right') - 1
    ok = (k >= 0) & (m < en[np.clip(k, 0, None)])
    if not ok.all(): raise Dropped(f'{D:%Y-%m-%d}: {int((~ok).sum())} half-hours have no solar interval')
    if (st[1:] < en[:-1]).any():
        raise ValueError('overlapping solar intervals in the assembled series')
    out['solar'] = I['solar'].to_numpy()[k]
    if out['solar'].isna().any():
        raise Dropped(f'{D:%Y-%m-%d}: {int(out["solar"].isna().sum())} half-hours would take solar radiation from a run '
                      'that has none (before 6 March 2024)')
    days = mids.tz_convert('UTC').tz_localize(None).normalize()
    out['source'] = ['era5' if d < D - SPLICE_DAYS * DAY else f"ifs {src_of_day[min(d, D)]:%Y-%m-%d}"
                     for d in days]
    return out, rD
