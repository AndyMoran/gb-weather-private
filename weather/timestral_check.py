"""
A mechanical check of how Timestral handled E5's calendar inputs. It reads no outturn and scores nothing: it compares
Timestral's two internal passes with each other, not with demand.

Why. Timestral's default "symmetric averaging" forecasts the context and its negation, with every covariate row
negated, and averages the reflected pair. Its scaler treats a known-future row whose values are all 0 or 1 as an
indicator and leaves it unscaled; a negated indicator (0 or -1) fails that test and is instead normalised and squashed
like a continuous input. If so, E5's calendar rows reached the two passes in different forms, and the average mixed
them. Weather rows are continuous in both passes, so their reflection is exact.

What it does, on 6 backtest days spread across the period (the same days as E5's reproduction check):
  1. prints the checkpoint's patch size and decode block, to show whether a day-ahead horizon is decoded in one block;
  2. for weather only, and for calendar + weather: the main pass's median and the mirrored pass's reflected median,
     and the largest gap between them. A large gap with the calendar and a small one without points to the asymmetry.
  3. the scaler's indicator test applied to a calendar row and to its negation.

Run in `gbwx`:   JAX_PLATFORMS=cpu python weather/timestral_check.py
"""
import os, sys
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_weather as R
import models_weather as MW
import calendar_study as CS
from timestral.scaler import instance_norm_stats


def passes(f, ctx, cov, names, h):
    X = torch.as_tensor(np.asarray(ctx, np.float32)[:, None, :])
    K = torch.as_tensor(np.stack([np.stack([np.asarray(cov[v][k], np.float32) for v in names]) for k in range(len(ctx))]))
    inputs, mask = f._as_context(X, None)
    main = f._forecast_raw(inputs, mask, h, known_covariates=K)
    mirr = f._forecast_raw(-inputs, mask, h, known_covariates=-K)
    mi = f.median_index
    return main[..., mi].numpy()[:, 0], (-mirr.flip(-1))[..., mi].numpy()[:, 0]


def main():
    series, pubs, hol, days_inputs, meta, era5_hh, built = R.load_real(full=True)
    hol = {pd.Timestamp(h).normalize() for h in hol}
    days = list(pd.date_range(R.FIRST, R.LAST))[::160]
    rec, _ = R.build(series, hol, pubs, days_inputs, meta, R.load_rung1(), era5_hh, days)
    w = MW.get('timestral'); f = w.f
    print(f'patch size {f.patch_size}; decode block {f.per_shot_horizon_patches} patches '
          f'= {f.patch_size * f.per_shot_horizon_patches} steps; longest day-ahead horizon in the backtest '
          f'{max(R.C.horizon(D)[0] for D in rec)} steps')
    for D, r in rec.items():
        h, first = R.C.horizon(D)
        cal = CS.calendar(r['wx'].index, hol)
        for label, names, cov in (('weather only', CS.WX, r['cov']), ('calendar + weather', CS.WX + CS.CAL, {**r['cov'], **cal})):
            a, b = passes(f, [r['ctx']], {v: [cov[v]] for v in names}, names, h)   # a batch of one day
            gap = np.abs(a - b)[0, first:first + len(r['y'])]
            print(f'{D:%Y-%m-%d} {label:19s}: main vs reflected mirror, largest gap {gap.max():7.1f} MW, mean {gap.mean():6.1f} MW')
    row = torch.as_tensor(cal['dow_sat'][None].astype(np.float32)); obs = torch.ones_like(row, dtype=torch.bool)
    for name, x in (('Saturday row', row), ('its negation', -row)):
        loc, scale, is_bin = instance_norm_stats(x, obs, minimum_scale=f.config.minimum_scale)
        print(f'{name:14s}: treated as indicator {bool(is_bin)}; loc {float(loc):.3f}, scale {float(scale):.3f}')


if __name__ == '__main__':
    main()
