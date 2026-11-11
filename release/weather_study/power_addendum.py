"""
Power addendum for the live window (weather study, plan Section 6.3), written after the backtest was scored and
before any live-window day was scored. Exploratory: it changes no test, threshold or rule in the plan.

The plan's power check (power.py) simulated gains of 2-10%. The backtest then found gains of 6-34%. This addendum asks
how the plan's agreement rule for the live window is likely to read if the live gains are of those sizes, so that the
April reading is disciplined in advance.

Method: power.py's, unchanged and imported. For each forecaster, the "with weather" arm is simulated from its own real
no-weather errors in one of the first study's two winters (20 November to 31 March, 2024-25 and 2025-26, in turn),
with the noise c taken from power.json, and scaled so that the true error ratio over the window is r. A sample is drawn
as a study would see it (blocks of 7 days), and its 90% percentile interval computed with the plan's bootstrap (blocks
of 7, here 1,000 draws). Each simulated interval is then read by the plan's agreement rule against the backtest's
scored 90% interval for that forecaster:
  consistent  the live ratio lies inside the backtest's interval, or the two intervals overlap;
  smaller     the live interval lies wholly above the backtest's (weather helped less on unseen days);
  larger      the live interval lies wholly below.
Also reported: the share of intervals that exclude 1 (a clear gain), and of one-sided p-values below 0.0125.

True live ratios r, for each forecaster:
  as backtest winter   the backtest's ratio on its own winter days (the live window is winter only)
  as backtest          the backtest's ratio on all days
  half the gain        1 - (1 - backtest ratio) / 2: the gain halved, as leakage might leave it
  no gain              1

Backtest figures: results.json of the scored run (backtest_attempt1, 11 November 2026), copied below.

Usage: python weather_study/power_addendum.py [reps]   (default 200)  ->  weather_study/power_addendum.json
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import power as P                                              # load, window_days, arrays, simulate, boot_idx, S

SEED = 20261111
BACKTEST = {   # name: (ratio, 90% interval, winter-days ratio), from backtest_attempt1/results.json
    'chronos2':   (0.6826, (0.6579, 0.7069), 0.6242),
    'timestral':  (0.7731, (0.7515, 0.7957), 0.7144),
    'timesfm25':  (0.9406, (0.9130, 0.9681), 0.8749),
    'regression': (0.6584, (0.6313, 0.6857), 0.5746),
}


def read(lo, hi, point, blo, bhi):
    if blo <= point <= bhi or (lo <= bhi and hi >= blo): return 'consistent'
    return 'smaller' if lo > bhi else 'larger'


def main(reps):
    f = P.load(); days = sorted(f['day'].unique())
    cs = json.load(open(os.path.join(HERE, 'power.json')))['c_values']
    rng = np.random.default_rng(SEED)
    out = dict(note='exploratory addendum to power.json; written after the backtest was scored, before any live-window day '
                    'was scored; changes no test, threshold or rule', c_values=cs, reps=reps, bootstrap_draws=P.B,
               block=P.BLOCK, backtest=BACKTEST, cells={})
    for name, (ratio, (blo, bhi), winter) in BACKTEST.items():
        wins = P.window_days(days, 'one winter'); data = [P.arrays(f, name, w) for w in wins]
        scen = {'as backtest winter': winter, 'as backtest': ratio, 'half the gain': 1 - (1 - ratio) / 2, 'no gain': 1.0}
        for c in cs:
            for sname, r in scen.items():
                tally = dict(consistent=0, smaller=0, larger=0, excludes_1=0, p_below_0125=0)
                for rep in range(reps):
                    E, _ = data[rep % len(data)]; n = len(E)
                    raw = P.simulate(E, rng, c); m0 = np.array([np.abs(e).mean() for e in E])
                    k = r * m0.mean() / np.mean([np.abs(x).mean() for x in raw])
                    m1 = np.array([np.abs(k * x).mean() for x in raw])
                    idx = P.S.block_indices(n, rng, P.BLOCK); a, b = m1[idx], m0[idx]
                    bi = P.boot_idx(n, rng); bs = a[bi].mean(1) / b[bi].mean(1)
                    lo, hi = np.quantile(bs, 0.05), np.quantile(bs, 0.95); point = a.mean() / b.mean()
                    tally[read(lo, hi, point, blo, bhi)] += 1
                    tally['excludes_1'] += hi < 1; tally['p_below_0125'] += (bs >= 1).mean() < 0.0125
                cell = {k: round(v / reps, 3) for k, v in tally.items()}; cell['true_ratio'] = round(r, 4)
                out['cells'][f'{name}|c{c}|{sname}'] = cell
                print(f"{name:10s} c={c} {sname:19s} r={r:.3f}: consistent {cell['consistent']:.2f}, smaller {cell['smaller']:.2f}, "
                      f"larger {cell['larger']:.2f}; clear gain {cell['excludes_1']:.2f}; p<0.0125 {cell['p_below_0125']:.2f}", flush=True)
    json.dump(out, open(os.path.join(HERE, 'power_addendum.json'), 'w'), indent=1)
    print('written to weather_study/power_addendum.json')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
