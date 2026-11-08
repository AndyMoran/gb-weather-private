"""
ECMWF's open-data archive (Brief 4, Section 3): fetch a 00 UTC run's fields, and find when a run became public.

ECMWF publishes the same files to Google Cloud, Azure and AWS. Each run is fetched whole from one copy: Google, or the
next copy only if Google answers that the run is missing or keeps refusing after retries (technical failure only; a
copy is never chosen by what its data contain). Each step's .index file lists every field's byte offset and length,
so only the wanted fields are downloaded. ECMWF's own client cannot reach runs before 29 February 2024, when the
folder layout changed, so both layouts are tried.

When a run became public: every copy records when each file arrived. A run counts as available at an issue time only
if every file it needs was on at least one copy before it (Brief 4, "When a run is missing or late").
"""
import datetime as dt, json, random, re, time, xml.etree.ElementTree as ET
import requests

SOURCES = {'google': 'https://storage.googleapis.com/ecmwf-open-data',
           'azure': 'https://ai4edataeuwest.blob.core.windows.net/ecmwf',
           'aws': 'https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com'}
LAYOUTS = ('ifs/0p25/oper/', '0p25/oper/')
_sas, _cool = {}, {}


class Missing(Exception): pass
class Busy(Exception): pass


def azure_token():
    if 'tok' not in _sas:
        r = requests.get('https://planetarycomputer.microsoft.com/api/sas/v1/token/ai4edataeuwest/ecmwf', timeout=60)
        r.raise_for_status(); _sas['tok'] = r.json()['token']
    return _sas['tok']


def get(url, headers=None, tries=5):
    if url.startswith(SOURCES['azure']): url += ('&' if '?' in url else '?') + azure_token()
    for k in range(tries):
        time.sleep(0.5)
        try:
            r = requests.get(url, headers=headers or {}, timeout=120)
        except requests.RequestException:
            r = None
        else:
            if r.status_code in (403, 404): raise Missing(url.split('?')[0])
            if r.status_code < 500: r.raise_for_status(); return r.content
        if k < tries - 1: time.sleep(min(120, 10 * 2 ** k) * random.uniform(0.8, 1.2))
    raise Busy(url.split('?')[0])


def _ranges(entries, gap=4_000_000):
    out = []
    for e in sorted(entries, key=lambda e: int(e['_offset'])):
        a, z = int(e['_offset']), int(e['_offset']) + int(e['_length']) - 1
        if out and a - out[-1][1] <= gap: out[-1][1] = max(out[-1][1], z)
        else: out.append([a, z])
    return out


def fetch(date, steps, params, target, optional=()):
    """Write the wanted fields of the 00 UTC run of `date` to `target` (GRIB). Returns (copy, layout, absent).
    Parameters in `optional` may be absent from the run (solar radiation before 6 March 2024); others may not."""
    for src, root in SOURCES.items():
        if _cool.get(src, 0) > time.time(): continue
        for layout in LAYOUTS:
            base = f'{root}/{date:%Y%m%d}/00z/{layout}{date:%Y%m%d}000000'
            try:
                absent = set()
                with open(target, 'wb') as out:
                    for s in steps:
                        idx = [json.loads(l) for l in get(f'{base}-{s}h-oper-fc.index').decode().splitlines() if l.strip()]
                        want = [e for e in idx if e.get('param') in params and e.get('type', 'fc') == 'fc']
                        lack = set(params) - {e['param'] for e in want}
                        if lack - set(optional):
                            raise RuntimeError(f'{date} step {s}: index lacks {lack - set(optional)}')
                        absent |= lack
                        blobs = {a: get(f'{base}-{s}h-oper-fc.grib2', {'Range': f'bytes={a}-{z}'}) for a, z in _ranges(want)}
                        for e in sorted(want, key=lambda e: int(e['_offset'])):
                            a = int(e['_offset']); st = max(k for k in blobs if k <= a)
                            msg = blobs[st][a - st:a - st + int(e['_length'])]
                            if msg[:4] != b'GRIB' or msg[-4:] != b'7777': raise RuntimeError(f'{date} step {s}: bad GRIB')
                            out.write(msg)
                return src, layout, sorted(absent)
            except Missing:
                continue                                       # not in this layout (or not on this copy)
            except Busy:
                _cool[src] = time.time() + 600; break          # this copy keeps refusing: try the next
    raise Missing(f'{date}: run not available on any copy')


def _keys_aws(prefix):
    out, token = {}, None
    while True:
        q = {'list-type': '2', 'prefix': prefix}
        if token: q['continuation-token'] = token
        r = requests.get(SOURCES['aws'], params=q, timeout=60); r.raise_for_status()
        root = ET.fromstring(r.content); ns = {'s': root.tag.split('}')[0].strip('{')}
        for c in root.findall('s:Contents', ns):
            out[c.findtext('s:Key', namespaces=ns)] = dt.datetime.fromisoformat(
                c.findtext('s:LastModified', namespaces=ns).replace('Z', '+00:00'))
        if root.findtext('s:IsTruncated', namespaces=ns) != 'true': return out
        token = root.findtext('s:NextContinuationToken', namespaces=ns)


def _keys_google(prefix):
    out, tok = {}, None
    while True:
        q = {'prefix': prefix, 'fields': 'items(name,timeCreated),nextPageToken'}
        if tok: q['pageToken'] = tok
        r = requests.get('https://storage.googleapis.com/storage/v1/b/ecmwf-open-data/o', params=q, timeout=60)
        r.raise_for_status(); j = r.json()
        for it in j.get('items', []):
            out[it['name']] = dt.datetime.fromisoformat(it['timeCreated'].replace('Z', '+00:00'))
        tok = j.get('nextPageToken')
        if not tok: return out


def _keys_azure(prefix):
    from email.utils import parsedate_to_datetime
    out, marker = {}, ''
    while True:
        url = f"{SOURCES['azure']}?restype=container&comp=list&prefix={prefix}&{azure_token()}" + (f'&marker={marker}' if marker else '')
        r = requests.get(url, timeout=60); r.raise_for_status(); root = ET.fromstring(r.content)
        for b in root.iter('Blob'):
            t = b.findtext('Properties/Creation-Time') or b.findtext('Properties/Last-Modified')
            out[b.findtext('Name')] = parsedate_to_datetime(t).astimezone(dt.timezone.utc)
        marker = root.findtext('NextMarker') or ''
        if not marker: return out


def public_by(date, steps, by=None):
    """When every step of the run of `date` was on at least one copy: (earliest complete time UTC, copy), or
    (None, reason). Times before 05:45 UTC on the run's date are impossible for a real upload and ignored."""
    floor = dt.datetime.combine(date, dt.time(5, 45), dt.timezone.utc)
    best, notes = None, []
    for src, lister in (('aws', _keys_aws), ('google', _keys_google), ('azure', _keys_azure)):
        try:
            for layout in LAYOUTS:
                keys = lister(f'{date:%Y%m%d}/00z/{layout}')
                if keys: break
            got = {int(m.group(1)): t for k, t in keys.items() if (m := re.search(r'-(\d+)h-oper-fc\.grib2$', k))}
            if all(s in got for s in steps):
                t = max(got[s] for s in steps)
                if t >= floor and (best is None or t < best[0]): best = (t, src)
                if by is not None and best is not None and best[0] <= by: return best   # proven on time: stop
            else:
                notes.append(f'{src}: steps missing')
        except Exception as e:
            notes.append(f'{src}: {type(e).__name__}')
    return best if best else (None, '; '.join(notes) or 'no copy has the run')
