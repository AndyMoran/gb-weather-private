"""
ERA5 for the LightGBM variants study's V5 (lgbm_variants/PLAN.md §4 and §9, step 1, frozen at tag lgbm-variants-plan in
the public repository): the training weather for a two-year window, D-731 to D-2, which for the first backtest day
(16 March 2024) reaches back to 16 March 2022. Committed before it runs.

Two steps, run in `wx`:
    python v5_era5.py download   ERA5 for March 2022 to February 2023, through era5_download.py with its first month
                                 moved back; the same request, files, log and single-instance lock. Resumable.
    python v5_era5.py build      the national half-hourly series for 15 March 2022 to 20 March 2023, built by the
                                 backtest's own functions (backtest_inputs.national_era5, national.era5_halfhourly) from
                                 the CDS files, then checked against the frozen era5_halfhourly file on the days both
                                 cover (from 15 March 2023). They must agree to the file's 4 decimals, or nothing is
                                 written. Writes the days before 15 March 2023 only, in the frozen file's format, and a
                                 build record with every input file's SHA-256.

Outputs, outside every repository (~/gb-weather-data/backtest/):
    era5_halfhourly_v5_2022-03-15_2023-03-14.csv.gz   raw ERA5 on each half-hour, as the frozen file
    era5_v5_build.json                                 code and weights hashes, the ERA5 files used, the overlap check,
                                                       and the output's SHA-256
The public repository's lgbm_variants/fetch_v5.py records the output's SHA-256 in its committed manifest.
The Copernicus key is read by cdsapi from ~/.cdsapirc only.
"""
import datetime as dt, gzip, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import era5_download as E                                    # noqa: E402
import backtest_inputs as BI                                 # noqa: E402  (national_era5, WEIGHTS, OUT)
import national as N, sources as S                           # noqa: E402

FIRST_MONTH = dt.date(2022, 3, 1)
NEW_FIRST = pd.Timestamp('2022-03-15T00:00Z')                # half-hours written: NEW_FIRST .. FROZEN_FIRST - 30 min
FROZEN_FIRST = pd.Timestamp('2023-03-15T00:00Z')             # the frozen file's first half-hour (backtest D0 - 367 days)
CHECK_LAST = pd.Timestamp('2023-03-20T23:30Z')               # the overlap checked: six days
FROZEN = os.path.join(BI.OUT, 'era5_halfhourly_2024-03-16_2026-10-31.csv.gz')
OUT_CSV = os.path.join(BI.OUT, 'era5_halfhourly_v5_2022-03-15_2023-03-14.csv.gz')
OUT_JSON = os.path.join(BI.OUT, 'era5_v5_build.json')
TOL = 5e-5 + 1e-9                                            # the frozen file is written to 4 decimals


def download():
    E.FIRST = FIRST_MONTH                                   # months() walks back to here; held months are skipped
    E.main()


def build():
    lock = json.load(open(os.path.join(BI.PUBLIC, 'weather_study', 'WEATHER_LOCK.json')))['files_sha256']
    if S.sha256(FROZEN) != lock.get('~/gb-weather-data/backtest/' + os.path.basename(FROZEN)):
        sys.exit('Refused: the frozen era5_halfhourly file differs from WEATHER_LOCK.json')
    w_pop, w_pv = N.load_weights(BI.WEIGHTS)
    e_first, e_last = NEW_FIRST - pd.Timedelta(hours=2), CHECK_LAST + pd.Timedelta(hours=2)
    BI.say(f'ERA5 national, {e_first} .. {e_last}')
    era5, rec = BI.national_era5(e_first, e_last, w_pop, w_pv)
    periods = pd.date_range(NEW_FIRST, CHECK_LAST, freq='30min')
    hh = N.era5_halfhourly(era5, periods)
    frozen = pd.read_csv(FROZEN, index_col=0); frozen.index = pd.to_datetime(frozen.index, utc=True)
    ov = hh.index[hh.index >= FROZEN_FIRST]
    a, b = hh.loc[ov, ['t2m', 'wind', 'solar']], frozen.reindex(ov)[['t2m', 'wind', 'solar']]
    if b.isna().any().any(): sys.exit('Refused: the frozen file does not cover the overlap')
    diff = (a - b).abs().max()
    check = dict(overlap=f'{ov[0]} .. {ov[-1]}', half_hours=len(ov), max_abs_diff={k: float(v) for k, v in diff.items()},
                 tolerance=TOL, passed=bool((diff <= TOL).all()))
    BI.say('overlap check', json.dumps(check))
    if not check['passed']: sys.exit('Refused: the rebuilt series differs from the frozen file on the overlap')
    new = hh[hh.index < FROZEN_FIRST]
    assert len(new) == int((FROZEN_FIRST - NEW_FIRST) / pd.Timedelta('30min')) and new.notna().all().all()
    new.index.name = 'period_start_utc'
    new.to_csv(OUT_CSV, float_format='%.4f', date_format='%Y-%m-%dT%H:%M:%SZ', compression='gzip')
    out = dict(built_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), half_hours=len(new),
               first=f'{new.index[0]}', last=f'{new.index[-1]}', overlap_check=check,
               code_sha256={f: S.sha256(os.path.join(HERE, f)) for f in
                            ('v5_era5.py', 'national.py', 'sources.py', 'backtest_inputs.py', 'era5_download.py')},
               weights_sha256=S.sha256(BI.WEIGHTS), era5_days=rec, output_sha256=S.sha256(OUT_CSV))
    json.dump(out, open(OUT_JSON, 'w'), indent=1)
    BI.say(f'written {OUT_CSV}: {len(new)} half-hours; sha256 {out["output_sha256"]}')


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else ''
    if step == 'download': download()
    elif step == 'build': build()
    else: sys.exit('usage: python v5_era5.py download | build')
