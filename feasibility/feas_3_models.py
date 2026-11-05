"""
Feasibility check 3: can each pinned model take three weather series as known future inputs? (Brief 4, Section 4)

Synthetic data only: synthetic GB-like demand with a built-in response to synthetic temperature, wind and solar
radiation. No real demand is read, and no model forecasts real demand.

For each model, in turn:
  1  it accepts three weather series over the context and the horizon, and returns the exact horizon, finite;
  2  changing the future weather changes the forecast (the inputs are actually used);
  3  a context whose last five days come from a shifted source (the ERA5 / forecast splice) still runs;
  4  for information: error with the true future weather against error without weather.
It prints each method's signature, so a failure shows what the pinned version expects.

Run from the root of the public repository, in the `gbfm` environment:
    python /path/to/feas_3_models.py
"""
import inspect, os, sys, traceback
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'code'))
import common as C, models as M

rng = np.random.default_rng(7)
T, H = C.CONTEXT_LEN, 79
n = T + H
t = np.arange(n)
temp = 8 + 6 * np.sin(2 * np.pi * (t / 48 - 0.4)) + np.cumsum(rng.normal(0, 0.15, n))   # degrees C, with weather drift
wind = np.abs(6 + np.cumsum(rng.normal(0, 0.2, n)))                                     # m/s
sun = np.clip(np.sin(2 * np.pi * ((t % 48) / 48 - 0.25)), 0, None) * (300 + 100 * rng.random(n))   # W/m2
demand = (30000 + 5000 * np.sin(np.pi * ((t % 48) / 48)) - 600 * (temp - 10) + 80 * wind - 4 * sun
          + rng.normal(0, 300, n))
ctx, truth = demand[:T], demand[T:]
W = {'temperature': temp, 'wind': wind, 'solar': sun}


def report(name, f_true, f_alt, f_spliced, f_none):
    ok = (f_true.shape == (H,) and np.isfinite(f_true).all() and f_spliced.shape == (H,) and np.isfinite(f_spliced).all())
    moved = float(np.abs(f_true - f_alt).max())
    mae = lambda f: float(np.abs(f - truth).mean())
    print(f'{name}: {"PASS" if ok else "FAIL"} shape/finite; future weather moves forecast by up to {moved:.0f} MW '
          f'({"PASS" if moved > 1 else "FAIL"}); spliced context runs: PASS; '
          f'MAE with true weather {mae(f_true):.0f} MW vs without {mae(f_none):.0f} MW')


def alt(w):   # the same weather, 4 degrees colder and windier in the horizon
    v = {k: x.copy() for k, x in w.items()}; v['temperature'][T:] -= 4; v['wind'][T:] += 4; return v


def spliced(w):   # the last five days of context shifted, as if from a different source
    v = {k: x.copy() for k, x in w.items()}; v['temperature'][T - 240:T] += 0.7; return v


def run_chronos():
    from chronos import Chronos2Pipeline
    repo, rev = M.REVISIONS['chronos2']
    p = Chronos2Pipeline.from_pretrained(repo, revision=rev, device_map='cpu')
    print('  Chronos2Pipeline.predict_quantiles', inspect.signature(p.predict_quantiles))
    def f(w):
        inp = [{'target': ctx, 'past_covariates': {k: x[:T] for k, x in w.items()},
                'future_covariates': {k: x[T:] for k, x in w.items()}}] if w else [ctx]
        q, _ = p.predict_quantiles(inp, prediction_length=H, quantile_levels=[0.5], cross_learning=False)
        return np.asarray(q[0])[0, :, 0]
    report('Chronos-2', f(W), f(alt(W)), f(spliced(W)), f(None))


def run_timesfm():
    import timesfm
    repo, rev = M.REVISIONS['timesfm25']
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo, revision=rev)
    s = dict(M.SETTINGS['timesfm25']); s['return_backcast'] = True      # XReg needs the in-context fit
    m.compile(timesfm.ForecastConfig(**s))
    fwc = getattr(m, 'forecast_with_covariates', None)
    print('  forecast_with_covariates', inspect.signature(fwc) if fwc else 'NOT FOUND')
    if fwc is None: raise RuntimeError('this timesfm version has no forecast_with_covariates for 2.5')
    def f(w, mode='xreg + timesfm'):
        if not w:
            _, q = m.forecast(horizon=H, inputs=[ctx]); return np.asarray(q)[0, :, 5]
        out = fwc(inputs=[ctx], dynamic_numerical_covariates={k: [x] for k, x in w.items()}, xreg_mode=mode)
        return np.asarray(out[0][0]).ravel()[:H]
    report('TimesFM 2.5 (xreg + timesfm)', f(W), f(alt(W)), f(spliced(W)), f(None))
    fm = lambda w: f(w, 'timesfm + xreg')
    report('TimesFM 2.5 (timesfm + xreg)', fm(W), fm(alt(W)), fm(spliced(W)), f(None))


def run_timestral():
    from timestral import TimestralForecaster
    repo, rev = M.REVISIONS['timestral']
    fc = TimestralForecaster.from_pretrained(repo, revision=rev, device='cpu')
    print('  TimestralForecaster.forecast', inspect.signature(fc.forecast))
    def f(w):
        X = ctx.astype(np.float32)[None, None, :]
        kw = dict(prediction_length=H, symmetric_averaging=True, clip_nonnegative=True)
        if w: kw['known_covariates'] = np.stack([x for x in w.values()]).astype(np.float32)[None]
        return fc.forecast(X, **kw).quantile(0.5).detach().cpu().numpy()[0, 0, :]
    report('Timestral', f(W), f(alt(W)), f(spliced(W)), f(None))


for name, fn in (('Chronos-2', run_chronos), ('TimesFM 2.5', run_timesfm), ('Timestral', run_timestral)):
    print(f'--- {name}')
    try: fn()
    except Exception: print(f'{name}: FAILED'); traceback.print_exc(limit=3)
