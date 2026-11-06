"""
Synthetic end-to-end test of the weather pipeline, and TimesFM's mode test (Brief 4, Section 4). No real demand.

Part 1, the pipeline (pass or fail):
  P1  ERA5-only history reproduces the true national weather at ERA5's hours, and its solar means exactly.
  P2  ECMWF history (no forecast error before step 24) reproduces the truth at its 3-hourly points, and its solar
      interval means: the accumulation convention is right.
  P3  The same check FAILS on a deliberately corrupted run, whose solar accumulation is not differenced.
  P4  The horizon comes from the run of D-1; with that run marked late, from the run of D-2, and the day survives.
  P5  Every half-hour of context and horizon is filled, in the demand study's own periods (77, 79 or 81 steps),
      across the clock change (the shoulder scenario's target days span 26 October 2025).

Part 2, TimesFM's mode (Brief 4, Section 4, fixed before this runs):
  Three scenarios: winter with forecast error; shoulder season with forecast error; winter with perfect weather.
  Each: 20 synthetic series (seeds 20261106 to 20261125), each forecast at 30 target days one day apart, through this
  pipeline: eight-week context, the splice, the interpolation and the horizon. Score: mean absolute error over D.
  TimesFM first replaces regression first only if, in EVERY scenario, its mean error is at least 5% lower AND its
  95th-percentile daily error is no higher. Anything else leaves regression first.

The test code, scenarios, seeds and threshold are committed and pushed before it runs: the script refuses to run
from an uncommitted or unpushed tree, and records the commit in its result.

Run from the root of the PUBLIC repository (for the demand study's locked `common` and `models`), in `gbwx`:
    JAX_PLATFORMS=cpu python ~/gb-weather-private/weather/synth_test.py            # both parts
    JAX_PLATFORMS=cpu python ~/gb-weather-private/weather/synth_test.py pipeline   # part 1 only
Result: ~/gb-weather-data/synth/result.json
"""
import datetime as dt, hashlib, json, os, subprocess, sys, time
import numpy as np, pandas as pd, xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.getcwd(), 'code'))
import national as N
import common as C

OUT = os.path.expanduser('~/gb-weather-data/synth'); os.makedirs(OUT, exist_ok=True)
WEIGHTS = os.path.expanduser('~/gb-weather-data/weights/weights.nc')
LATS = np.round(np.arange(61, 48.99, -0.25), 2); LONS = np.round(np.arange(-9, 2.51, 0.25), 2)
SEEDS = list(range(20261106, 20261126))                      # 20 series
ORIGINS = 30
SCENARIOS = {                                                  # target days, temperature climate, forecast error
    'winter_error':   dict(first=dt.date(2025, 12, 1), t_mean=5.0, t_diurnal=2.0, solar_peak=250, error=True),
    'shoulder_error': dict(first=dt.date(2025, 10, 5), t_mean=11.5, t_diurnal=4.0, solar_peak=550, error=True),
    'winter_perfect': dict(first=dt.date(2025, 12, 1), t_mean=5.0, t_diurnal=2.0, solar_peak=250, error=False),
}
ERR_SD = dict(t2m=1.0, wind=1.5, solar_frac=0.25)              # added to the target day's weather only (lead >= 24 h)
MARGIN, RULE = 0.05, 'TimesFM first only if, in every scenario, mean error >= 5% lower and p95 daily error no higher'
H30 = pd.Timedelta('30min')


# ---------------------------------------------------------------- synthetic truth ------------------------------------
class Truth:
    """True weather on the grid, as smooth national series times fixed spatial patterns. Values are available at any
    instant (5-minute resolution), so ERA5, ECMWF runs and the demand can all be drawn from the same truth."""
    def __init__(self, seed, sc, t0, t1):
        rng = np.random.default_rng(seed)
        self.t = pd.date_range(t0, t1, freq='5min', tz='UTC'); n = len(self.t)
        hrs = (self.t - self.t[0]) / pd.Timedelta(hours=1)
        def ar(sd, tau_h):                                     # an AR(1) process at 5-minute steps
            phi = np.exp(-(5 / 60) / tau_h); e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n); x = np.empty(n)
            x[0] = rng.normal(0, sd)
            for k in range(1, n): x[k] = phi * x[k - 1] + e[k]
            return x
        local_h = np.asarray(self.t.tz_convert('Europe/London').hour + self.t.tz_convert('Europe/London').minute / 60)
        self.T = sc['t_mean'] + sc['t_diurnal'] * np.sin(2 * np.pi * (local_h - 9) / 24) + ar(3.0, 72)
        self.W = np.abs(7 + ar(3.0, 36)); self.dir = np.cumsum(rng.normal(0, 0.02, n))
        self.cloud = np.clip(0.6 + ar(0.25, 12), 0.1, 1.0)
        doy = np.asarray(self.t.dayofyear); utc_h = np.asarray(hrs % 24)
        decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
        self.decl, self.utc_h, self.peak = decl, utc_h, sc['solar_peak']
        g = np.random.default_rng(seed + 7)
        la = LATS[:, None] * np.ones((1, len(LONS)))
        self.gT = -0.6 * (la - 54) + g.normal(0, 0.5, la.shape)          # colder north, local variation
        self.gW = np.clip(0.3 * g.normal(0, 1, la.shape), -0.8, None)    # speed multiplier 1 + gW stays positive
        self.gS = 0.1 * g.normal(0, 1, la.shape)

    def clear(self, k):                                        # clear-sky GHI by latitude row, at time indices k
        lat = np.radians(LATS)[None, :]; ha = np.radians(15 * (self.utc_h[k] - 12))[:, None]
        el = np.sin(lat) * np.sin(self.decl[k])[:, None] + np.cos(lat) * np.cos(self.decl[k])[:, None] * np.cos(ha)
        return np.clip(el, 0, None) * self.peak / 0.55          # (time, lat)

    def idx(self, times):
        k = self.t.get_indexer(pd.DatetimeIndex(times))
        if (k < 0).any(): raise ValueError('time outside the truth')
        return k

    def fields(self, times):                                    # instantaneous fields at times: (time, lat, lon)
        k = self.idx(times)
        T = self.T[k][:, None, None] + self.gT[None]
        S = self.W[k][:, None, None] * (1 + self.gW[None]); d = self.dir[k][:, None, None]
        return T, S * np.cos(d), S * np.sin(d)

    def ghi(self, k):                                           # (time, lat, lon) W/m2
        return (self.clear(k) * self.cloud[k][:, None])[:, :, None] * (1 + self.gS[None])

    def energy(self, a, b):                                     # J/m2 over [a, b) per grid point, from 5-minute GHI
        k = np.arange(self.idx([a])[0], self.idx([b])[0])
        return self.ghi(k).sum(0) * 300.0

    def national(self, times, w_pop, w_pv):                     # the true GB values at instants
        k = self.idx(times); wp = w_pop.values; ws = w_pv.values
        T = self.T[k] + (wp * self.gT).sum(); Wn = self.W[k] * (wp * (1 + self.gW)).sum()
        S = (self.ghi(k) * ws[None]).sum((1, 2))
        return T, Wn, S


def era5_ds(tr, a, b):
    t = pd.date_range(a, b, freq='1h', tz='UTC'); T, U, V = tr.fields(t)
    ssrd = np.stack([tr.energy(x - pd.Timedelta(hours=1), x) for x in t])
    co = dict(valid_time=t.tz_localize(None), latitude=LATS, longitude=LONS)
    dims = ('valid_time', 'latitude', 'longitude')
    return xr.Dataset({'t2m': (dims, T + 273.15), 'u10': (dims, U), 'v10': (dims, V), 'ssrd': (dims, ssrd)}, coords=co)


def run_ds(tr, run, error_rng=None, corrupt=False):
    steps = np.arange(0, 73, 3); vt = run + pd.to_timedelta(steps, 'h')
    T, U, V = tr.fields(vt); acc = np.zeros((len(steps), len(LATS), len(LONS)))
    eT = np.zeros(len(steps)); eW = np.zeros(len(steps)); eS = np.zeros(len(steps))
    if error_rng is not None:                                   # errors from lead 24 h on, correlated along the run
        z = lambda sd: np.convolve(error_rng.normal(0, sd, len(steps) + 2), np.ones(3) / np.sqrt(3), 'valid')
        lead = steps >= 24
        eT = np.where(lead, z(ERR_SD['t2m']), 0); eW = np.where(lead, z(ERR_SD['wind']), 0)
        eS = np.where(lead, z(ERR_SD['solar_frac']), 0)
    T = T + eT[:, None, None]
    sp = np.hypot(U, V); f = np.clip(sp + eW[:, None, None], 0, None) / np.where(sp > 0, sp, 1)
    U, V = U * f, V * f
    for k in range(1, len(steps)):
        e = tr.energy(vt[k - 1], vt[k]) * max(0.0, 1 + eS[k])
        acc[k] = acc[k - 1] + e
    if corrupt: acc = np.stack([tr.energy(vt[k] - pd.Timedelta(hours=3), vt[k]) if k else acc[0] for k in range(len(steps))])
    dims = ('step', 'latitude', 'longitude')
    return xr.Dataset({'t2m': (dims, T + 273.15), 'u10': (dims, U), 'v10': (dims, V), 'ssrd': (dims, acc)},
                      coords=dict(step=pd.to_timedelta(steps, 'h'), latitude=LATS, longitude=LONS,
                                  time=run.tz_localize(None)))


def demand(tr, periods, w_pop, w_pv, rng):
    mids = pd.DatetimeIndex(periods) + pd.Timedelta(minutes=15)
    T, W, S = tr.national(mids, w_pop, w_pv)
    loc = mids.tz_convert('Europe/London'); h = np.asarray(loc.hour + loc.minute / 60)
    shape = 0.5 * (1 - np.cos(2 * np.pi * (h - 4) / 24)) + 0.35 * np.exp(-((h - 18) / 2) ** 2)
    base = 19000 + 9000 * shape - 2000 * np.isin(loc.dayofweek, [5, 6])
    weather = 700 * np.clip(15.5 - T, 0, None) - 100 * np.clip(T - 15.5, 0, None) + 40 * W * (T < 15.5) - 6 * S
    e = np.empty(len(periods)); e[0] = rng.normal(0, 400)
    for k in range(1, len(e)): e[k] = 0.9 * e[k - 1] + rng.normal(0, 400 * np.sqrt(1 - 0.81))
    return base + weather + e


# ---------------------------------------------------------------- one synthetic series --------------------------------
def weights():
    if os.path.exists(WEIGHTS): return N.load_weights(WEIGHTS)
    print('  (weights.nc not found: using uniform synthetic weights for a dry run)')
    w = xr.DataArray(np.ones((len(LATS), len(LONS))) / (len(LATS) * len(LONS)), coords=dict(latitude=LATS, longitude=LONS),
                     dims=('latitude', 'longitude'))
    return w, w


def series(seed, sc, w_pop, w_pv, n_origins=ORIGINS):
    """Truth, ERA5, runs and demand for one series; yields (D, periods, assembled weather, demand, truth on periods)."""
    days = [pd.Timestamp(sc['first']) + k * N.DAY for k in range(n_origins)]
    first_period = C.input_end(days[0]) - C.CONTEXT_LEN * H30
    t0 = (first_period - 2 * N.DAY).normalize(); t1 = C.day_bounds(days[-1])[1].normalize() + 4 * N.DAY
    tr = Truth(seed, sc, t0, t1)
    era5 = N.era5_national(era5_ds(tr, t0 + pd.Timedelta(hours=1), C.input_end(days[-1]) - 8 * N.DAY), w_pop, w_pv)
    err = np.random.default_rng(seed + 1000) if sc['error'] else None
    runs = {}
    r = (first_period - N.SPLICE_DAYS * N.DAY - 2 * N.DAY).normalize()
    last_run = pd.Timestamp(days[-1]).tz_localize('UTC') - N.DAY
    while r <= last_run:
        if r >= t0 + N.DAY:
            run, p, i = N.ecmwf_national(run_ds(tr, r, err), w_pop, w_pv)
            runs[r.tz_localize(None)] = (p, i)
        r += N.DAY
    all_periods = pd.date_range(first_period, C.day_bounds(days[-1])[1] - H30, freq='30min')
    dem = pd.Series(demand(tr, all_periods, w_pop, w_pv, np.random.default_rng(seed + 2000)), index=all_periods)
    return tr, era5, runs, dem, days


def periods_for(D):
    end = C.input_end(D); h, i0 = C.horizon(D)
    ctx = pd.date_range(end - C.CONTEXT_LEN * H30, end - H30, freq='30min')
    hor = pd.date_range(end, end + (h - 1) * H30, freq='30min')
    return ctx, hor, h, i0


# ---------------------------------------------------------------- part 1: the pipeline --------------------------------
def part_pipeline(w_pop, w_pv):
    res = {}
    sc = SCENARIOS['shoulder_error']
    tr, era5, runs, dem, days = series(SEEDS[0], sc, w_pop, w_pv, n_origins=30)
    # P1: ERA5 points and intervals equal the truth
    p, i = era5
    T, W, _ = tr.national(p.index, w_pop, w_pv)
    s_true = np.array([(tr.energy(a, b) * w_pv.values).sum() / 3600 for a, b in zip(i['start'][:200], i['end'][:200])])
    res['P1'] = bool(np.allclose(p['t2m'], T, atol=1e-6) and np.allclose(p['wind'], W, atol=1e-6)
                     and np.allclose(i['solar'][:200], s_true, rtol=1e-9, atol=1e-6))
    # P2 and P3: a run with no early error reproduces the truth; a corrupted one does not
    r0 = sorted(runs)[len(runs) // 2]; run_utc = pd.Timestamp(r0).tz_localize('UTC')
    def check(ds):
        _, p, i = N.ecmwf_national(ds, w_pop, w_pv)
        early = p[p['step_h'] <= 21]; T, W, _ = tr.national(early.index, w_pop, w_pv)
        ie = i[i['step_h'] <= 24]
        s_true = np.array([(tr.energy(a, b) * w_pv.values).sum() / 10800 for a, b in zip(ie['start'], ie['end'])])
        return np.allclose(early['t2m'], T, atol=1e-6) and np.allclose(ie['solar'], s_true, rtol=1e-6, atol=1e-3)
    res['P2'] = bool(check(run_ds(tr, run_utc)))
    res['P3_corrupted_fails'] = bool(not check(run_ds(tr, run_utc, corrupt=True)))
    # P4 and P5 over every target day
    fills, falls, shapes = [], [], []
    for D in days:
        ctx, hor, h, i0 = periods_for(D)
        out, rD = N.assemble(D.date(), ctx.append(hor), era5, runs)
        fills.append(out[['t2m', 'wind', 'solar']].notna().all().all() and len(out) == C.CONTEXT_LEN + h)
        shapes.append(h)
        out2, rD2 = N.assemble(D.date(), ctx.append(hor), era5, runs, late={D - N.DAY})
        falls.append(rD == D - N.DAY and rD2 == D - 2 * N.DAY and not np.allclose(out2['t2m'][-h:], out['t2m'][-h:]))
    res['P4_horizon_run_and_fallback'] = bool(all(falls))
    res['P5_every_halfhour_filled'] = bool(all(fills))
    res['P5_horizon_lengths_seen'] = sorted(set(shapes))
    # interpolation error against the truth at midpoints, for the record
    D = days[10]; ctx, hor, h, i0 = periods_for(D); out, _ = N.assemble(D.date(), ctx.append(hor), era5, runs)
    T, W, S = tr.national(out.index[:C.CONTEXT_LEN] + pd.Timedelta(minutes=15), w_pop, w_pv)
    res['history_mae_vs_truth'] = dict(t2m=float(np.abs(out['t2m'][:C.CONTEXT_LEN] - T).mean()),
                                       wind=float(np.abs(out['wind'][:C.CONTEXT_LEN] - W).mean()),
                                       solar=float(np.abs(out['solar'][:C.CONTEXT_LEN] - S).mean()))
    res['pass'] = all(res[k] for k in ('P1', 'P2', 'P3_corrupted_fails', 'P4_horizon_run_and_fallback',
                                        'P5_every_halfhour_filled')) and 81 in shapes   # the clock change was crossed
    print('pipeline:', json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- part 2: TimesFM's mode ------------------------------
def part_timesfm(w_pop, w_pv):
    import timesfm, models as M
    repo, rev = M.REVISIONS['timesfm25']
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo, revision=rev)
    s = dict(M.SETTINGS['timesfm25']); s['return_backcast'] = True
    m.compile(timesfm.ForecastConfig(**s))
    xreg_args = dict(normalize_xreg_target_per_input=True, ridge=0.0, force_on_cpu=True)
    scores = {}
    for name, sc in SCENARIOS.items():
        daily = {'xreg + timesfm': [], 'timesfm + xreg': [], 'no weather': []}
        for seed in SEEDS:
            t = time.time(); tr, era5, runs, dem, days = series(seed, sc, w_pop, w_pv)
            by_h = {}
            for D in days:
                ctx, hor, h, i0 = periods_for(D)
                wx, _ = N.assemble(D.date(), ctx.append(hor), era5, runs)
                by_h.setdefault(h, []).append((D, dem[ctx].to_numpy(), wx, dem[hor].to_numpy(), i0))
            for h, items in by_h.items():
                inputs = [x[1] for x in items]
                cov = {v: [x[2][v].to_numpy() for x in items] for v in ('t2m', 'wind', 'solar')}
                for mode in ('xreg + timesfm', 'timesfm + xreg'):
                    out = m.forecast_with_covariates(inputs=inputs, dynamic_numerical_covariates=cov, xreg_mode=mode,
                                                     **xreg_args)
                    for k, x in enumerate(items):
                        f = np.asarray(out[0][k]).ravel()[-h:]
                        daily[mode].append(float(np.abs(f[x[4]:] - x[3][x[4]:]).mean()))
                _, q = m.forecast(horizon=h, inputs=inputs); q = np.asarray(q)
                for k, x in enumerate(items):
                    daily['no weather'].append(float(np.abs(q[k, -h:, 5][x[4]:] - x[3][x[4]:]).mean()))
            print(f'  {name} seed {seed}: {time.time() - t:.0f} s', flush=True)
        scores[name] = {k: dict(mean=float(np.mean(v)), p95=float(np.percentile(v, 95)), n=len(v))
                        for k, v in daily.items()}
        print(name, json.dumps(scores[name]), flush=True)
    reverse = all(scores[n]['timesfm + xreg']['mean'] <= (1 - MARGIN) * scores[n]['xreg + timesfm']['mean'] and
                  scores[n]['timesfm + xreg']['p95'] <= scores[n]['xreg + timesfm']['p95'] for n in SCENARIOS)
    decision = 'timesfm + xreg' if reverse else 'xreg + timesfm'
    print('DECISION:', decision, '|', RULE)
    return dict(scores=scores, rule=RULE, decision=decision, xreg_args=xreg_args, timesfm_settings=s)


def committed():
    """The test must run from committed, pushed code: returns the commit, or stops."""
    g = lambda *a: subprocess.run(['git', '-C', os.path.dirname(HERE), *a], capture_output=True, text=True)
    if g('status', '--porcelain', '--', 'weather').stdout.strip():
        sys.exit('weather/ has uncommitted changes: commit and push before running the test')
    head = g('rev-parse', 'HEAD').stdout.strip()
    remote = g('ls-remote', 'origin', 'refs/heads/main').stdout.split()
    if not remote or remote[0] != head:
        sys.exit(f'HEAD {head[:7]} is not on GitHub (origin/main is {remote[0][:7] if remote else "unknown"}): push first')
    return head


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    head = committed() if which == 'all' else 'not checked (pipeline only)'
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    w_pop, w_pv = weights()
    result = dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), commit=head,
                  code={f: sha(os.path.join(HERE, f)) for f in ('national.py', 'synth_test.py')},
                  weights=sha(WEIGHTS) if os.path.exists(WEIGHTS) else None, seeds=SEEDS, origins=ORIGINS,
                  scenarios={k: {a: str(b) for a, b in v.items()} for k, v in SCENARIOS.items()}, forecast_error=ERR_SD)
    result['pipeline'] = part_pipeline(w_pop, w_pv)
    if not result['pipeline']['pass']: sys.exit('pipeline checks failed: no mode test')
    if which == 'all': result['timesfm_mode'] = part_timesfm(w_pop, w_pv)
    result['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    json.dump(result, open(os.path.join(OUT, 'result.json' if which == 'all' else 'pipeline.json'), 'w'), indent=1)
    print('written to', OUT)


if __name__ == '__main__':
    main()
