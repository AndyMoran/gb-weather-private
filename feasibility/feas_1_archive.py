"""
Feasibility check 1: ECMWF's open-data archive on AWS (Brief 4, Section 12).

  A  Coverage: for every day from 1 February 2024 to 31 October 2026, is the 00 UTC run at 0.25 deg on AWS, with
     every step the study needs (24 to 48 hours, every 3 hours)? One listing request per day; writes coverage.csv.
  B  Depth: for three sample runs (Feb 2024, Jun 2025, Jan 2026), download the steps and variables, and check each
     file's own metadata names the run it claims to be.
  C  Overlap: for runs since 31 October 2026, download from AWS exactly what the daily collector downloaded, crop it
     the same way, and check the values equal the collector's saved files.

No model is run and no demand data is read. Run from the root of the public repository (it reuses the collector's
crop function), in the `wx` environment:
    python /path/to/feas_1_archive.py [A|B|C]          (default: all three)
Outputs go to ~/gb-weather-data/feasibility/, outside every repository.
"""
import datetime as dt, os, re, sys, xml.etree.ElementTree as ET
import numpy as np, requests

BUCKET = 'https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com'
START, END = dt.date(2024, 2, 1), dt.date(2026, 10, 31)
NEED = list(range(24, 49, 3))                      # covers D in UK time from step 23/24 to 47, with interpolation
PARAMS = ['2t', '10u', '10v', 'ssrd']
OUT = os.path.expanduser('~/gb-weather-data/feasibility'); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(os.getcwd(), 'prospective', 'weather'))


def list_keys(prefix):
    keys, token = [], None
    while True:
        q = {'list-type': '2', 'prefix': prefix}
        if token: q['continuation-token'] = token
        r = requests.get(BUCKET, params=q, timeout=60); r.raise_for_status()
        root = ET.fromstring(r.content); ns = {'s': root.tag.split('}')[0].strip('{')}
        keys += [k.text for k in root.findall('s:Contents/s:Key', ns)]
        if root.findtext('s:IsTruncated', namespaces=ns) != 'true': return keys
        token = root.findtext('s:NextContinuationToken', namespaces=ns)


def part_a():
    first = list_keys(f'{START:%Y%m%d}/00z/')
    print('layout on the first day (sample keys):', first[:5] or 'NOTHING FOUND')
    rows, d = [], START
    while d <= END:
        # ECMWF added the ifs/ level to the path when AIFS joined the bucket; earlier runs sit one level up.
        for layout in ('ifs/0p25/oper/', '0p25/oper/'):
            keys = list_keys(f'{d:%Y%m%d}/00z/{layout}')
            if keys: break
        steps = {int(m.group(1)) for k in keys if (m := re.search(r'-(\d+)h-oper-fc\.grib2$', k))}
        missing = [s for s in NEED if s not in steps]
        rows.append((d.isoformat(), layout if keys else '', len(keys), ' '.join(map(str, missing))))
        if d.day == 1 or (rows[-2][1] != rows[-1][1] if len(rows) > 1 else False):
            print(d, layout if keys else 'NO LAYOUT', 'files', len(keys), 'missing steps', missing or 'none', flush=True)
        d += dt.timedelta(days=1)
    with open(os.path.join(OUT, 'coverage.csv'), 'w') as f:
        f.write('date,layout,files,missing_steps\n'); f.writelines(f'{a},{b},{c},{e}\n' for a, b, c, e in rows)
    bad = [r for r in rows if r[3] or r[2] == 0]
    print(f'A: {len(rows)} days checked; {len(bad)} with the run or a needed step missing')
    for r in bad[:30]: print('   ', r)


def get(url, headers=None, tries=6):
    """GET with a short, bounded retry on 5xx (S3 throttling); a 404 is final."""
    import time
    for k in range(tries):
        r = requests.get(url, headers=headers or {}, timeout=120)
        if r.status_code < 500: r.raise_for_status(); return r.content
        print(f'   {r.status_code} on {url.rsplit("/", 1)[-1]}; retry {k + 1} of {tries - 1}', flush=True)
        time.sleep(15 * (k + 1))
    r.raise_for_status()


def fetch(date, steps, target, params=PARAMS):
    """Download only the wanted fields of one 00 UTC run, straight from the bucket.

    ecmwf-opendata builds the post-AIFS path (ifs/0p25/oper) for every date, so it cannot reach runs before
    29 February 2024. This reads each step's .index (one JSON line per field, with its byte offset and length) and
    range-requests just the wanted fields, trying both layouts. Concatenated GRIB messages are a valid GRIB file.
    Returns the layout used."""
    import json
    for layout in ('ifs/0p25/oper/', '0p25/oper/'):
        base = f'{BUCKET}/{date:%Y%m%d}/00z/{layout}{date:%Y%m%d}000000'
        try: get(f'{base}-{steps[0]}h-oper-fc.index', tries=1)
        except requests.HTTPError: continue
        with open(target, 'wb') as out:
            for s in steps:
                idx = [json.loads(l) for l in get(f'{base}-{s}h-oper-fc.index').decode().splitlines() if l.strip()]
                want = [e for e in idx if e.get('param') in params and e.get('type', 'fc') == 'fc']
                got = {e['param'] for e in want}
                if got != set(params): raise RuntimeError(f'{date} step {s}: index lacks {set(params) - got}')
                for e in sorted(want, key=lambda e: e['_offset']):
                    a, n = int(e['_offset']), int(e['_length'])
                    out.write(get(f'{base}-{s}h-oper-fc.grib2', {'Range': f'bytes={a}-{a + n - 1}'}))
        return layout
    raise RuntimeError(f'{date}: run not found under either layout')


def part_b():
    import cfgrib
    for d in (dt.date(2024, 2, 15), dt.date(2025, 6, 15), dt.date(2026, 1, 15)):
        g = os.path.join(OUT, f'depth_{d:%Y%m%d}.grib2')
        try: layout = fetch(d, NEED, g)
        except Exception as e: print(f'B {d}: FAILED {e}'); continue
        seen = {}
        for ds in cfgrib.open_datasets(g, backend_kwargs=dict(indexpath='')):
            for v in ds.data_vars:
                a = ds[v].attrs
                seen[v] = (a.get('GRIB_dataDate'), a.get('GRIB_dataTime'), sorted(int(x) for x in
                           (ds.step.values / np.timedelta64(1, 'h')).ravel()))
        ok = all(s[0] == int(f'{d:%Y%m%d}') and s[1] == 0 and s[2] == NEED for s in seen.values()) and len(seen) == 4
        print(f'B {d}: {"PASS" if ok else "CHECK"} layout {layout}; variables {sorted(seen)}; metadata {next(iter(seen.values()))[:2]}')
        if not ok: print('   ', seen)


def part_c():
    import xarray as xr, ecmwf_daily as E
    data = os.path.join(os.getcwd(), 'prospective', 'weather', 'data')
    d = dt.date(2026, 10, 31)
    while d < dt.date.today():
        mine = os.path.join(data, f'ifs_0p25_{d:%Y%m%d}_00z_gb.nc')
        if os.path.exists(mine):
            g = os.path.join(OUT, f'overlap_{d:%Y%m%d}.grib2'); n = g.replace('.grib2', '_gb.nc')
            fetch(d, E.STEPS, g, E.PARAMS)                 # this script's downloader against the collector's files
            E.crop_to_gb(g, n)
            a, b = xr.open_dataset(mine), xr.open_dataset(n)
            same = all(np.array_equal(a[v].values, b[v].values, equal_nan=True) for v in a.data_vars)
            print(f'C {d}: {"IDENTICAL" if same else "DIFFERENT"} ({len(a.data_vars)} variables)')
        d += dt.timedelta(days=1)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'ABC'
    if 'A' in which: part_a()
    if 'B' in which: part_b()
    if 'C' in which: part_c()
