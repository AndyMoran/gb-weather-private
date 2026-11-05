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
import warnings
import numpy as np, requests
warnings.filterwarnings('ignore', category=FutureWarning)   # cfgrib's xarray merge notice

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


class Missing(Exception):
    """The object is not in the bucket (404 or 403). Only this means 'try the other layout'."""


def get(url, headers=None, tries=9):
    """GET from the bucket. 404/403 raise Missing at once. A 5xx (S3's 'slow down') or a dropped connection waits
    and retries: exponential backoff from 10 s, capped at 5 minutes, with jitter, honouring Retry-After. Every
    request is paced, because the 503s come from asking too fast."""
    import random, time
    for k in range(tries):
        time.sleep(0.5)
        try:
            r = requests.get(url, headers=headers or {}, timeout=120)
        except requests.RequestException as e:
            code, r = type(e).__name__, None
        else:
            if r.status_code in (403, 404): raise Missing(url)
            if r.status_code < 500: r.raise_for_status(); return r.content
            code = r.status_code
        if k == tries - 1: break
        wait = min(300, 10 * 2 ** k) * random.uniform(0.8, 1.2)
        if r is not None and r.headers.get('Retry-After', '').isdigit(): wait = max(wait, int(r.headers['Retry-After']))
        print(f'   {code} on {url.rsplit("/", 1)[-1]}; waiting {wait:.0f} s (retry {k + 1} of {tries - 1})', flush=True)
        time.sleep(wait)
    raise RuntimeError(f'gave up after {tries} tries: {url}')


def ranges(entries, gap=4_000_000):
    """Merge the wanted fields' byte ranges when they sit within `gap` bytes of each other: fewer requests, a few
    unwanted bytes. Returns (start, end) pairs, inclusive."""
    out = []
    for e in sorted(entries, key=lambda e: int(e['_offset'])):
        a, z = int(e['_offset']), int(e['_offset']) + int(e['_length']) - 1
        if out and a - out[-1][1] <= gap: out[-1][1] = max(out[-1][1], z)
        else: out.append([a, z])
    return out


def fetch(date, steps, target, params=PARAMS):
    """Download only the wanted fields of one 00 UTC run, straight from the bucket.

    ecmwf-opendata builds the post-AIFS path (ifs/0p25/oper) for every date, so it cannot reach runs before
    29 February 2024. This reads each step's .index (one JSON line per field, with its byte offset and length) and
    range-requests just the wanted fields, trying both layouts. Concatenated GRIB messages are a valid GRIB file.
    Returns the layout used."""
    import json
    for layout in ('ifs/0p25/oper/', '0p25/oper/'):
        base = f'{BUCKET}/{date:%Y%m%d}/00z/{layout}{date:%Y%m%d}000000'
        try: first = get(f'{base}-{steps[0]}h-oper-fc.index')
        except Missing: continue
        with open(target, 'wb') as out:
            for s in steps:
                raw = first if s == steps[0] else get(f'{base}-{s}h-oper-fc.index')
                idx = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
                want = [e for e in idx if e.get('param') in params and e.get('type', 'fc') == 'fc']
                got = {e['param'] for e in want}
                if got != set(params): raise RuntimeError(f'{date} step {s}: index lacks {set(params) - got}')
                blobs = {}
                for a, z in ranges(want):
                    blobs[a] = get(f'{base}-{s}h-oper-fc.grib2', {'Range': f'bytes={a}-{z}'})
                for e in sorted(want, key=lambda e: int(e['_offset'])):     # keep only the wanted messages
                    a = int(e['_offset']); start = max(k for k in blobs if k <= a)
                    msg = blobs[start][a - start:a - start + int(e['_length'])]
                    if msg[:4] != b'GRIB' or msg[-4:] != b'7777': raise RuntimeError(f'{date} step {s}: bad GRIB bytes')
                    out.write(msg)
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
                run = np.datetime64(ds['time'].values.ravel()[0], 'h')    # the run each field says it belongs to
                seen[v] = (str(run), sorted(int(x) for x in (ds.step.values / np.timedelta64(1, 'h')).ravel()))
        ok = all(x[0] == f'{d:%Y-%m-%d}T00' and x[1] == NEED for x in seen.values()) and len(seen) == 4
        print(f'B {d}: {"PASS" if ok else "CHECK"} layout {layout}; variables {sorted(seen)}; run in metadata {next(iter(seen.values()))[0]}')
        if not ok: print('   ', seen)


def part_c():
    import xarray as xr, ecmwf_daily as E   # noqa
    data = os.path.join(os.getcwd(), 'prospective', 'weather', 'data')
    d = dt.date(2026, 10, 31)
    while d < dt.date.today():
        mine = os.path.join(data, f'ifs_0p25_{d:%Y%m%d}_00z_gb.nc')
        if os.path.exists(mine):
            g = os.path.join(OUT, f'overlap_{d:%Y%m%d}.grib2'); n = g.replace('.grib2', '_gb.nc')
            try: fetch(d, E.STEPS, g, E.PARAMS)            # this script's downloader against the collector's files
            except Exception as e: print(f'C {d}: FAILED {e}'); d += dt.timedelta(days=1); continue
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
