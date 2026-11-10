"""
Decision 13's minimum at 20 and 45 days (analysis plan, Section 6.4): the backtest inputs rebuilt with another minimum
number of days per variable, for the days that minimum can change.

The minimum matters only at the start of the backtest, where a variable's window D-70 .. D-11 holds few runs: temperature
and wind enter the runs on 1 February 2024, solar on 6 March. A day's inputs change only if a variable is adjusted under
one minimum and left unadjusted under the other. This script rebuilds 16 March to 31 May 2024 and refuses to write if the
last day it builds is still flagged under either minimum: from then on every variable has at least 45 days, so every
later day is the same as in the main build. Code shared with backtest_inputs.py (imported, not copied), so the only
difference is the minimum.

Outputs, beside the main build in ~/gb-weather-data/backtest/:
  inputs_min<N>_2024-03-16_2024-05-31.csv.gz, days_min<N>_2024-03-16_2024-05-31.jsonl   (same columns as the main build)

Run in `wx` or `gbwx`, after the main build:   python sensitivity_inputs.py
"""
import datetime as dt, gzip, json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import backtest_inputs as BI
N, S = BI.N, BI.S

MINIMA = (20, 45)
FIRST, LAST = BI.FIRST_DAY, dt.date(2024, 5, 31)


def build(min_days, runs, era5, late):
    tag = f'min{min_days}_{FIRST}_{LAST}'
    csv_path = os.path.join(BI.OUT, f'inputs_{tag}.csv.gz'); jl_path = os.path.join(BI.OUT, f'days_{tag}.jsonl')
    rows, recs, n_changed = [], [], 0
    for D in pd.date_range(FIRST, LAST, freq='D'):
        ctx, hor, h, i0 = BI.periods_for(D)
        rec = dict(target_day=f'{D:%Y-%m-%d}', min_days=min_days)
        f = N.history_factors(D.date(), era5, runs, min_days=min_days)
        f30 = N.history_factors(D.date(), era5, runs)
        rec['changed'] = bool(f['flagged'] != f30['flagged']); n_changed += rec['changed']
        try:
            lt = {x for x in late if x == D - N.DAY}
            out, rD = N.assemble(D.date(), ctx.append(hor), era5, runs, late=lt, factors=f)
            raw, _ = N.assemble(D.date(), ctx.append(hor), era5, runs, late=lt)
        except N.Dropped as e:
            rec.update(dropped=str(e)); recs.append(rec); continue
        rec.update(horizon_run=f'{rD:%Y-%m-%d}', fallback=bool(rD != D - N.DAY), history_factors=f)
        recs.append(rec)
        rows.append(pd.DataFrame({'target_day': f'{D:%Y-%m-%d}', 'period_start_utc': out.index.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                  'part': ['context'] * len(ctx) + ['horizon'] * len(hor),
                                  't2m_c': out['t2m'].to_numpy(), 'wind_ms': out['wind'].to_numpy(), 'solar_wm2': out['solar'].to_numpy(),
                                  't2m_c_unadjusted': raw['t2m'].to_numpy(), 'wind_ms_unadjusted': raw['wind'].to_numpy(),
                                  'solar_wm2_unadjusted': raw['solar'].to_numpy(), 'source': out['source'].to_numpy()}))
    last = recs[-1].get('history_factors', {})
    if last.get('flagged'): sys.exit(f'min {min_days}: {LAST} is still flagged ({last["flagged"]}); extend LAST')
    with gzip.open(csv_path, 'wt', newline='') as fc:
        pd.concat(rows).to_csv(fc, index=False, float_format='%.4f')
    with open(jl_path, 'w') as fj:
        for r in recs: fj.write(json.dumps(r) + '\n')
    BI.say(f'min {min_days}: {len(recs)} days, {n_changed} with factors other than at 30 days; {os.path.basename(csv_path)} '
           f'sha256 {S.sha256(csv_path)[:12]}')


def main():
    dates = BI.run_dates(FIRST, LAST)
    late = BI.phase_availability(dates, 1)                   # cached by the main build
    w_pop, w_pv = N.load_weights(BI.WEIGHTS)
    runs, _ = BI.national_runs(dates, w_pop, w_pv)
    e_first = pd.Timestamp(pd.Timestamp(FIRST) - N.FACTOR_WINDOW[0] * N.DAY).tz_localize('UTC')
    e_last = pd.Timestamp(pd.Timestamp(LAST) - N.SPLICE_DAYS * N.DAY).tz_localize('UTC')
    era5, _ = BI.national_era5(e_first, e_last, w_pop, w_pv)
    for m in MINIMA: build(m, runs, era5, late)


if __name__ == '__main__':
    main()
