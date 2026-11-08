"""
The three foundation models with weather (analysis plan, Section 4), and a stand-in for pipeline tests.

Each wrapper takes a batch of equal-length demand contexts (CONTEXT_LEN half-hours), the weather over context plus
horizon for each ({'t2m', 'wind', 'solar'}, each a list of arrays of length CONTEXT_LEN + h), and one horizon h; it
returns the point forecast for every step, as a (batch, h) array. Revisions and the no-weather settings are rung 1's
(code/models.py), imported, not copied. Every weather argument is written out here; nothing relies on a default.
"""
import numpy as np
import models as M                                            # rung 1's code/models.py

VARS = ('t2m', 'wind', 'solar')                               # the fixed order wherever an order matters
SETTINGS_W = {
    'chronos2': dict(covariates='past_covariates over the context, future_covariates over the horizon'),
    'timestral': dict(covariates='known_covariates (batch, 3, context + horizon), order t2m, wind, solar'),
    'timesfm25': dict(xreg_mode='xreg + timesfm', normalize_xreg_target_per_input=True, ridge=0.0,
                      force_on_cpu=True, return_backcast=True),
}


def _check(cov, contexts, h):
    T = len(contexts[0])
    for v in VARS:
        if len(cov[v]) != len(contexts): raise ValueError(f'{v}: {len(cov[v])} series for {len(contexts)} contexts')
        for c in cov[v]:
            if len(c) != T + h: raise ValueError(f'{v}: length {len(c)}, expected {T + h}')
            if not np.isfinite(c).all(): raise ValueError(f'{v}: non-finite weather')


class StandInW:
    """Pipeline tests only, never scored: rung 1's stand-in, moved by the horizon's temperature against the same
    half-hours a week earlier (300 MW per degree), so a test can see the weather reach the forecast."""
    name = 'stand-in'
    def forecast(self, contexts, cov, h):
        _check(cov, contexts, h)
        base = M.StandIn().forecast(contexts, h); T = len(contexts[0])
        t = np.array([np.asarray(c, float) for c in cov['t2m']])
        return base - 300.0 * (t[:, T:T + h] - t[:, T - 336:T - 336 + h])


class Chronos2W:
    name = 'chronos2'
    def __init__(self):
        from chronos import Chronos2Pipeline
        repo, rev = M.REVISIONS[self.name]; s = M.SETTINGS[self.name]
        self.p = Chronos2Pipeline.from_pretrained(repo, revision=rev, device_map=s['device_map'])
    def forecast(self, contexts, cov, h):
        _check(cov, contexts, h); s = M.SETTINGS[self.name]; T = len(contexts[0])
        inputs = [{'target': np.asarray(x, float),
                   'past_covariates': {v: np.asarray(cov[v][k][:T], float) for v in VARS},
                   'future_covariates': {v: np.asarray(cov[v][k][T:], float) for v in VARS}}
                  for k, x in enumerate(contexts)]
        q, _ = self.p.predict_quantiles(inputs, prediction_length=h, quantile_levels=s['quantile_levels'],
                                        batch_size=s['batch_size'], context_length=s['context_length'],
                                        cross_learning=s['cross_learning'], limit_prediction_length=s['limit_prediction_length'])
        out = np.stack([np.asarray(t)[0, :, 0] for t in q]).astype(float)
        if out.shape[1] != h: raise ValueError(f'Chronos-2 returned {out.shape[1]} steps, expected {h}')
        return out


class TimestralW:
    name = 'timestral'
    def __init__(self):
        from timestral import TimestralForecaster
        repo, rev = M.REVISIONS[self.name]
        self.f = TimestralForecaster.from_pretrained(repo, revision=rev, device=M.SETTINGS[self.name]['device'])
    def forecast(self, contexts, cov, h):
        _check(cov, contexts, h); s = M.SETTINGS[self.name]
        X = np.asarray(contexts, np.float32)[:, None, :]
        K = np.stack([np.stack([np.asarray(cov[v][k], np.float32) for v in VARS]) for k in range(len(contexts))])
        out = self.f.forecast(X, prediction_length=h, known_covariates=K, symmetric_averaging=s['symmetric_averaging'],
                              clip_nonnegative=s['clip_nonnegative'])
        med = out.quantile(s['quantile']).detach().cpu().numpy()[:, 0, :].astype(float)
        if med.shape[1] != h: raise ValueError(f'Timestral returned {med.shape[1]} steps, expected {h}')
        return med


class TimesFM25W:
    name = 'timesfm25'
    def __init__(self, mode='xreg + timesfm'):
        import timesfm
        repo, rev = M.REVISIONS[self.name]
        self.mode = mode
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo, revision=rev)
        cfg = dict(M.SETTINGS[self.name]); cfg['return_backcast'] = SETTINGS_W[self.name]['return_backcast']
        self.m.compile(timesfm.ForecastConfig(**cfg))
    def forecast(self, contexts, cov, h):
        _check(cov, contexts, h); s = SETTINGS_W[self.name]
        out = self.m.forecast_with_covariates(
            inputs=[np.asarray(x, float) for x in contexts],
            dynamic_numerical_covariates={v: [np.asarray(c, float) for c in cov[v]] for v in VARS},
            xreg_mode=self.mode, normalize_xreg_target_per_input=s['normalize_xreg_target_per_input'],
            ridge=s['ridge'], force_on_cpu=s['force_on_cpu'])
        F = np.stack([np.asarray(out[0][k], float).ravel()[-h:] for k in range(len(contexts))])
        if F.shape[1] != h: raise ValueError(f'TimesFM 2.5 returned {F.shape[1]} steps, expected {h}')
        return F


def get(name):
    if name == 'timesfm25-other': return TimesFM25W(mode='timesfm + xreg')
    return {'stand-in': StandInW, 'chronos2': Chronos2W, 'timestral': TimestralW, 'timesfm25': TimesFM25W}[name]()
