"""
Feasibility check 1: ECMWF's open-data archive on AWS (Brief 4, Section 12).

  A  Coverage: for every day from 1 February 2024 to 31 October 2026, is the 00 UTC run at 0.25 deg on AWS, with
     every step the study needs (24 to 48 hours, every 3 hours)? One listing request per day; writes coverage.csv.
  B  Depth: for three sample runs (Feb 2024, Jun 2025, Jan 2026), download the steps and variables, and check each
     file's own metadata names the run it claims to be.
  C  Overlap: for runs since 31 October 2026, download from the archive exactly what the daily collector downloaded, crop it
     the same way, and check the values equal the collector's saved files.
  E  Availability: for runs whose AWS copy arrived after 09:30 UK time on D-1, the upload times on Google's and
     Azure's copies. A run was public at the issue time if any copy had every step 0-48 by then. Writes
     availability.csv.
  D  Parameters: for every day, which of the four variables the run's index lists (steps 24 and 48); finds when
     solar radiation first appears. Writes params.csv.

Downloads try ECMWF's Google, Azure and AWS copies in that order (AWS throttles with 503s), then ECMWF's portal.

No model is run and no demand data is read. Run from the root of the public repository (it reuses the collector's
crop function), in the `wx` environment:
    python /path/to/feas_1_archive.py [A|B|C|D]        (default: all four)
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
    """Every object under the prefix, as {key: LastModified (UTC datetime)}."""
    keys, token = {}, None
    while True:
        q = {'list-type': '2', 'prefix': prefix}
        if token: q['continuation-token'] = token
        r = requests.get(BUCKET, params=q, timeout=60); r.raise_for_status()
        root = ET.fromstring(r.content); ns = {'s': root.tag.split('}')[0].strip('{')}
        for c in root.findall('s:Contents', ns):
            keys[c.findtext('s:Key', namespaces=ns)] = dt.datetime.fromisoformat(
                c.findtext('s:LastModified', namespaces=ns).replace('Z', '+00:00'))
        if root.findtext('s:IsTruncated', namespaces=ns) != 'true': return keys
        token = root.findtext('s:NextContinuationToken', namespaces=ns)


ALL = list(range(0, 49, 3))      # history splice (steps 0-24) and the target day (21-48)


def issue_utc(run_date):
    """09:30 UK time on the run's date (D-1), in UTC: 09:30 UTC in winter, 08:30 UTC in summer time."""
    from zoneinfo import ZoneInfo
    return dt.datetime.combine(run_date, dt.time(9, 30), ZoneInfo('Europe/London')).astimezone(dt.timezone.utc)


def part_a():
    """Coverage of steps 0-48 for every run, and when each run's files reached the bucket. LastModified is the upload
    to this copy: the run cannot have been public here before it. A later re-upload would only make a day look late,
    never early, so the check is conservative."""
    first = list_keys(f'{START:%Y%m%d}/00z/')
    print('layout on the first day (sample keys):', list(first)[:3] or 'NOTHING FOUND')
    rows, d = [], START
    while d <= END:
        # ECMWF added the ifs/ level to the path when AIFS joined the bucket; earlier runs sit one level up.
        for layout in ('ifs/0p25/oper/', '0p25/oper/'):
            keys = list_keys(f'{d:%Y%m%d}/00z/{layout}')
            if keys: break
        got = {int(m.group(1)): t for k, t in keys.items() if (m := re.search(r'-(\d+)h-oper-fc\.grib2$', k))}
        missing = [x for x in ALL if x not in got]
        last = max((got[x] for x in ALL if x in got), default=None)
        iss = issue_utc(d)
        late = last is not None and last > iss
        rows.append((d.isoformat(), layout if keys else '', len(keys), ' '.join(map(str, missing)),
                     last.strftime('%Y-%m-%dT%H:%M') if last else '', iss.strftime('%H:%M'), late))
        if d.day == 1 or rows[-1][1] != (rows[-2][1] if len(rows) > 1 else None):
            print(d, layout if keys else 'NO LAYOUT', 'files', len(keys), 'missing steps', missing or 'none',
                  f'| steps 0-48 uploaded by {rows[-1][4]} UTC, issue {rows[-1][5]} UTC', flush=True)
        d += dt.timedelta(days=1)
    with open(os.path.join(OUT, 'coverage.csv'), 'w') as f:
        f.write('date,layout,files,missing_steps_0_48,last_upload_utc,issue_utc,uploaded_after_issue\n')
        f.writelines(','.join(map(str, r)) + '\n' for r in rows)
    bad = [r for r in rows if r[3] or r[2] == 0]
    late = [r for r in rows if r[6]]
    print(f'A: {len(rows)} days checked; {len(bad)} with the run or a step 0-48 missing; '
          f'{len(late)} uploaded after the 09:30 UK issue time')
    for r in bad[:30]: print('    missing', r)
    for r in late[:40]: print('    late', r[0], 'uploaded', r[4], 'issue', r[5])
    if rows:
        import statistics
        lead = [(issue_utc(dt.date.fromisoformat(r[0])) - dt.datetime.fromisoformat(r[4] + '+00:00')).total_seconds() / 3600
                for r in rows if r[4]]
        print(f'A: margin between upload and issue time, hours: min {min(lead):.1f}; median {statistics.median(lead):.1f}')


class Missing(Exception):
    """The object is not at this source (404 or 403)."""


class Busy(Exception):
    """This source kept refusing (5xx) after the retries; move to the next source."""


# ECMWF publishes identical files to four places (ECMWF forum, 5 May 2026). AWS has throttled with 503 'SlowDown'
# on and off since November 2025, so it goes last. ECMWF's own portal keeps only the last few days.
SOURCES = {'google': 'https://storage.googleapis.com/ecmwf-open-data',
           'azure': 'https://ai4edataeuwest.blob.core.windows.net/ecmwf',
           'aws': BUCKET,
           'ecmwf': 'https://data.ecmwf.int/forecasts'}
_sas, _cool = {}, {}                               # Azure token; source -> time until which it is skipped


def azure_token():
    """Planetary Computer issues a free, short-lived read token for the Azure copy (as ecmwf-opendata does)."""
    if 'tok' not in _sas:
        r = requests.get('https://planetarycomputer.microsoft.com/api/sas/v1/token/ai4edataeuwest/ecmwf', timeout=60)
        r.raise_for_status(); _sas['tok'] = r.json()['token']
    return _sas['tok']


def get(url, headers=None, tries=5):
    """GET one object. 404/403 raise Missing. A 5xx or dropped connection waits (10 s doubling, capped at 2 min,
    with jitter, honouring Retry-After) and retries; after `tries`, raise Busy so the caller switches source.
    Every request is paced."""
    import random, time
    if url.startswith(SOURCES['azure']): url += ('&' if '?' in url else '?') + azure_token()
    name = url.split('?')[0].rsplit('/', 1)[-1]
    for k in range(tries):
        time.sleep(0.5)
        try:
            r = requests.get(url, headers=headers or {}, timeout=120)
        except requests.RequestException as e:
            code, r = type(e).__name__, None
        else:
            if r.status_code in (403, 404): raise Missing(name)
            if r.status_code < 500: r.raise_for_status(); return r.content
            code = r.status_code
        if k == tries - 1: break
        wait = min(120, 10 * 2 ** k) * random.uniform(0.8, 1.2)
        if r is not None and r.headers.get('Retry-After', '').isdigit(): wait = max(wait, int(r.headers['Retry-After']))
        print(f'   {code} on {name}; waiting {wait:.0f} s (retry {k + 1} of {tries - 1})', flush=True)
        time.sleep(wait)
    raise Busy(name)


def ranges(entries, gap=4_000_000):
    """Merge the wanted fields' byte ranges when they sit within `gap` bytes of each other: fewer requests, a few
    unwanted bytes. Returns (start, end) pairs, inclusive."""
    out = []
    for e in sorted(entries, key=lambda e: int(e['_offset'])):
        a, z = int(e['_offset']), int(e['_offset']) + int(e['_length']) - 1
        if out and a - out[-1][1] <= gap: out[-1][1] = max(out[-1][1], z)
        else: out.append([a, z])
    return out


def locate(date, step, sources=SOURCES):
    """Find a source and layout holding this run. Returns (source, base URL, that step's index bytes)."""
    import json  # noqa
    import time
    tried = []
    for src, root in sources.items():
        if _cool.get(src, 0) > time.time(): tried.append(f'{src} cooling off'); continue
        for layout in ('ifs/0p25/oper/', '0p25/oper/'):
            base = f'{root}/{date:%Y%m%d}/00z/{layout}{date:%Y%m%d}000000'
            try: return src, base, layout, get(f'{base}-{step}h-oper-fc.index')
            except Missing: continue
            except Busy: tried.append(f'{src} busy'); _cool[src] = time.time() + 600; break
        else: tried.append(f'{src} missing')
    raise RuntimeError(f'{date}: run not found ({"; ".join(tried)})')


def parse_index(raw):
    import json
    return [json.loads(l) for l in raw.decode().splitlines() if l.strip()]


def fetch(date, steps, target, params=PARAMS):
    """Download only the wanted fields of one 00 UTC run, from the first source that has it and answers.

    ecmwf-opendata builds the post-AIFS path (ifs/0p25/oper) for every date, so it cannot reach runs before
    29 February 2024. This reads each step's .index (one JSON line per field, with its byte offset and length) and
    range-requests just the wanted fields. Concatenated GRIB messages are a valid GRIB file. A parameter absent from
    the index is reported, not fatal. Returns (source, layout, parameters missing)."""
    srcs = dict(SOURCES)
    while srcs:
        src, base, layout, first = locate(date, steps[0], srcs)
        absent = set()
        try:
            with open(target, 'wb') as out:
                for s in steps:
                    want = [e for e in parse_index(first if s == steps[0] else get(f'{base}-{s}h-oper-fc.index'))
                            if e.get('param') in params and e.get('type', 'fc') == 'fc']
                    absent |= set(params) - {e['param'] for e in want}
                    blobs = {a: get(f'{base}-{s}h-oper-fc.grib2', {'Range': f'bytes={a}-{z}'}) for a, z in ranges(want)}
                    for e in sorted(want, key=lambda e: int(e['_offset'])):      # keep only the wanted messages
                        a = int(e['_offset']); st = max(k for k in blobs if k <= a)
                        msg = blobs[st][a - st:a - st + int(e['_length'])]
                        if msg[:4] != b'GRIB' or msg[-4:] != b'7777':
                            raise RuntimeError(f'{date} step {s}: bad GRIB bytes from {src}')
                        out.write(msg)
            return src, layout, sorted(absent)
        except Busy:
            import time; _cool[src] = time.time() + 600
            print(f'   {src} kept refusing; skipping it for 10 minutes', flush=True)
            srcs = {k: v for k, v in srcs.items() if k != src and list(srcs).index(k) > list(srcs).index(src)}
    raise RuntimeError(f'{date}: every source refused')


def part_b():
    import cfgrib
    for d in (dt.date(2024, 2, 15), dt.date(2024, 3, 15), dt.date(2025, 6, 15), dt.date(2026, 1, 15)):
        g = os.path.join(OUT, f'depth_{d:%Y%m%d}.grib2')
        try: src, layout, absent = fetch(d, NEED, g)
        except Exception as e: print(f'B {d}: FAILED {e}'); continue
        seen = {}
        for ds in cfgrib.open_datasets(g, backend_kwargs=dict(indexpath='')):
            for v in ds.data_vars:
                run = np.datetime64(ds['time'].values.ravel()[0], 'h')    # the run each field says it belongs to
                seen[v] = (str(run), sorted(int(x) for x in (ds.step.values / np.timedelta64(1, 'h')).ravel()))
        ok = all(x[0] == f'{d:%Y-%m-%d}T00' and x[1] == NEED for x in seen.values()) and len(seen) == 4 - len(absent)
        print(f'B {d}: {"PASS" if ok and not absent else "CHECK"} from {src} {layout}; variables {sorted(seen)}; '
              f'absent from the run {absent or "none"}; run in metadata {next(iter(seen.values()))[0] if seen else "-"}')
        if not ok: print('   ', seen)


def part_c():
    import xarray as xr, ecmwf_daily as E   # noqa
    data = os.path.join(os.getcwd(), 'prospective', 'weather', 'data')
    d = dt.date(2026, 10, 31)
    while d < dt.date.today():
        mine = os.path.join(data, f'ifs_0p25_{d:%Y%m%d}_00z_gb.nc')
        if os.path.exists(mine):
            g = os.path.join(OUT, f'overlap_{d:%Y%m%d}.grib2'); n = g.replace('.grib2', '_gb.nc')
            try: src, _, _ = fetch(d, E.STEPS, g, E.PARAMS)  # this script's downloader against the collector's files
            except Exception as e: print(f'C {d}: FAILED {e}'); d += dt.timedelta(days=1); continue
            E.crop_to_gb(g, n)
            a, b = xr.open_dataset(mine), xr.open_dataset(n)
            same = all(np.array_equal(a[v].values, b[v].values, equal_nan=True) for v in a.data_vars)
            print(f'C {d}: {"IDENTICAL" if same else "DIFFERENT"} ({len(a.data_vars)} variables, from {src})')
        d += dt.timedelta(days=1)


def part_d():
    """Which of the four parameters each run's index lists, at steps 24 and 48: one small file each, every day.
    Finds when solar radiation (and anything else) first appears. Writes params.csv; prints only the changes."""
    rows, prev, d = [], None, START
    while d <= END:
        try:
            src, base, layout, i24 = locate(d, 24)
            i48 = get(f'{base}-48h-oper-fc.index')
            have = tuple(p for p in PARAMS if all(p in {e.get('param') for e in parse_index(i)} for i in (i24, i48)))
        except Exception as e:
            src, layout, have = '-', '-', ('ERROR ' + str(e)[:60],)
        rows.append((d.isoformat(), src, layout, ' '.join(have)))
        if have != prev: print(f'D {d}: {" ".join(have) or "none"} (from {src} {layout})', flush=True); prev = have
        d += dt.timedelta(days=1)
    with open(os.path.join(OUT, 'params.csv'), 'w') as f:
        f.write('date,source,layout,params\n'); f.writelines(f'{a},{b},{c},{e}\n' for a, b, c, e in rows)
    print(f'D: {len(rows)} days; all four on {sum(r[3] == " ".join(PARAMS) for r in rows)}')


def upload_times(src, date, layout):
    """{step: upload time (UTC)} for steps 0-48 of one run on one copy. Google: the JSON API's timeCreated.
    Azure: the container listing's Creation-Time (else Last-Modified), with Planetary Computer's read token."""
    from email.utils import parsedate_to_datetime
    prefix = f'{date:%Y%m%d}/00z/{layout}'; out = {}
    if src == 'google':
        url, tok = 'https://storage.googleapis.com/storage/v1/b/ecmwf-open-data/o', None
        while True:
            q = {'prefix': prefix, 'fields': 'items(name,timeCreated),nextPageToken'}
            if tok: q['pageToken'] = tok
            r = requests.get(url, params=q, timeout=60); r.raise_for_status(); j = r.json()
            for it in j.get('items', []):
                if (m := re.search(r'-(\d+)h-oper-fc\.grib2$', it['name'])):
                    out[int(m.group(1))] = dt.datetime.fromisoformat(it['timeCreated'].replace('Z', '+00:00'))
            tok = j.get('nextPageToken')
            if not tok: return out
    marker = ''
    while True:
        url = f"{SOURCES['azure']}?restype=container&comp=list&prefix={prefix}&{azure_token()}" + (f'&marker={marker}' if marker else '')
        r = requests.get(url, timeout=60); r.raise_for_status(); root = ET.fromstring(r.content)
        for b in root.iter('Blob'):
            name = b.findtext('Name')
            if (m := re.search(r'-(\d+)h-oper-fc\.grib2$', name or '')):
                t = b.findtext('Properties/Creation-Time') or b.findtext('Properties/Last-Modified')
                out[int(m.group(1))] = parsedate_to_datetime(t).astimezone(dt.timezone.utc)
        marker = root.findtext('NextMarker') or ''
        if not marker: return out


def part_e():
    """For each run the AWS listing put after the issue time (coverage.csv from part A), when did it reach Google's
    and Azure's copies? A run was public at the issue time if any copy had every step 0-48 by then."""
    import csv
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, 'coverage.csv'))) if r['uploaded_after_issue'] == 'True']
    print(f'E: {len(rows)} runs flagged by the AWS times; checking Google and Azure')
    res = []
    for r in rows:
        d = dt.date.fromisoformat(r['date']); iss = issue_utc(d); best = {'aws': r['last_upload_utc']}
        for src in ('google', 'azure'):
            try:
                t = upload_times(src, d, r['layout'])
                best[src] = max(t[x] for x in ALL).strftime('%Y-%m-%dT%H:%M') if all(x in t for x in ALL) else 'incomplete'
            except Exception as e:
                best[src] = f'error {type(e).__name__}'
        # a time before dissemination starts (05:45 UTC on the run's date) is impossible for a real upload: ignore it
        floor = dt.datetime.combine(d, dt.time(5, 45), dt.timezone.utc)
        times = [t for v in best.values() if v[:2] == '20' and (t := dt.datetime.fromisoformat(v + '+00:00')) >= floor]
        ok = bool(times) and min(times) <= iss
        res.append((r['date'], iss.strftime('%H:%M'), best['aws'], best['google'], best['azure'], ok))
        print(f"E {r['date']}: issue {iss:%H:%M} UTC | aws {best['aws'][11:]} | google {best['google'][11:] if best['google'][:2]=='20' else best['google']} "
              f"| azure {best['azure'][11:] if best['azure'][:2]=='20' else best['azure']} | {'public in time' if ok else 'LATE everywhere'}", flush=True)
    with open(os.path.join(OUT, 'availability.csv'), 'w') as f:
        f.write('date,issue_utc,aws_last_upload,google_last_upload,azure_last_upload,public_by_issue\n')
        f.writelines(','.join(map(str, x)) + '\n' for x in res)
    print(f'E: {sum(x[5] for x in res)} of {len(res)} were on another copy by the issue time; '
          f'{sum(not x[5] for x in res)} late on every copy: {[x[0] for x in res if not x[5]]}')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'ABCD'
    if 'A' in which: part_a()
    if 'B' in which: part_b()
    if 'C' in which: part_c()
    if 'D' in which: part_d()
    if 'E' in which: part_e()
