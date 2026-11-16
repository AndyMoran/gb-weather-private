import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
"""Minimal reproduction: TimesFM 2.5 forecast_with_covariates gives a different forecast for the same series
depending on what else is in the call."""
import numpy as np, timesfm, torch
torch.manual_seed(0)
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", revision="1d95242")
m.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True,
                                 use_continuous_quantile_head=True, return_backcast=True))
rng = np.random.default_rng(0); n, h = 512, 48
t = np.arange(n + h)
def series(phase, beta):
    x = np.sin(2 * np.pi * (t + phase) / 48) + rng.normal(0, 0.3, n + h)     # covariate
    y = 10 + beta * x + np.sin(2 * np.pi * t / 336) + rng.normal(0, 0.2, n + h)
    return y[:n], x
yA, xA = series(0, 2.0); yB, xB = series(17, -3.0)
def run(ys, xs):
    pf, _ = m.forecast_with_covariates(inputs=ys, dynamic_numerical_covariates={'x': xs},
                                       xreg_mode='xreg + timesfm')
    return np.asarray(pf[0])
alone = run([yA], [xA]); paired = run([yA, yB], [xA, xB])
print('A alone vs A batched with B: max abs difference', float(np.max(np.abs(alone - paired))))
