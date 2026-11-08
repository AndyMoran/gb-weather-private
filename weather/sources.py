"""
Load the weather sources for one forecast (Brief 4, Section 3), recording where every piece came from.

ERA5:  the Copernicus (CDS) monthly files from era5_download.py, in ~/gb-weather-data/era5/. Any hour they do not
       cover comes from Google's copy (ARCO-ERA5), cached by UTC day in ~/gb-weather-data/era5_arco/. The two copies
       agree to rounding (feasibility check 4); Google's is used only where the CDS file is absent (technical failure,
       never by content), and every hour's copy is recorded.
Runs:  the daily collector's cropped files (prospective/weather/data/, checked against its manifest's SHA-256).
       A missing collector file is fetched from the archive (archive.py) and cropped with the collector's own
       crop function, cached in ~/gb-weather-data/runs/. A run on no copy is absent, and national.assemble applies
       the rules for missing runs.
"""
import csv, glob, hashlib, os, re, sys, zipfile
import numpy as np, pandas as pd, xarray as xr

DATA = os.path.expanduser('~/gb-weather-data')
ERA5_DIR, ARCO_DIR, RUNS_DIR = (os.path.join(DATA, d) for d in ('era5', 'era5_arco', 'runs'))
ARCO = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
ARCO_NAMES = {'2m_temperature': 't2m', '10m_u_component_of_wind': 'u10', '10m_v_component_of_wind': 'v10',
              'surface_solar_radiation_downwards': 'ssrd'}
LATS = np.round(np.arange(61, 48.99, -0.25), 2); LONS = np.round(np.arange(-9, 2.51, 0.25), 2)
H = pd.Timedelta(hours=1)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def _open_cds(path):
    """A CDS download: a zip of NetCDF files (instantaneous and accumulated variables), or one NetCDF file."""
    if not zipfile.is_zipfile(path):
        ds = xr.open_dataset(path, engine='netcdf4').load()
    else:
        d = path[:-3] + '_unzipped'
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith('.nc')]
            if not all(os.path.exists(os.path.join(d, n)) for n in names): z.extractall(d)
        ds = xr.merge([xr.open_dataset(os.path.join(d, n), engine='netcdf4').load() for n in names], compat='override')
    if 'valid_time' not in ds.dims: ds = ds.rename({'time': 'valid_time'})
    ds = ds[['t2m', 'u10', 'v10', 'ssrd']]
    return ds.drop_vars([c for c in ds.coords if c not in ('valid_time', 'latitude', 'longitude')])


def cds_files():
    """{(year, month): path}: the complete month file if there is one, else the partial covering the most days."""
    best = {}
    for p in glob.glob(os.path.join(ERA5_DIR, 'era5_*.nc')):
        m = re.fullmatch(r'era5_(\d{4})_(\d{2})(?:_to(\d{2}))?\.nc', os.path.basename(p))
        if not m: continue
        key, upto = (int(m.group(1)), int(m.group(2))), (99 if m.group(3) is None else int(m.group(3)))
        if key not in best or upto > best[key][0]: best[key] = (upto, p)
    return {k: v[1] for k, v in best.items()}


def _arco_day(day):
    """One UTC day (hours 00..23) of ERA5 over the GB box from Google's copy, cached."""
    os.makedirs(ARCO_DIR, exist_ok=True)
    path = os.path.join(ARCO_DIR, f'{day:%Y%m%d}.nc')
    if not os.path.exists(path):
        z = xr.open_zarr(ARCO, chunks=None, storage_options=dict(token='anon'))
        sub = z[list(ARCO_NAMES)].sel(time=slice(day, day + pd.Timedelta(hours=23)), latitude=slice(61, 49))
        w = sub.sel(longitude=slice(351, 359.75)); e = sub.sel(longitude=slice(0, 2.5))
        w = w.assign_coords(longitude=w.longitude - 360)
        box = xr.concat([w, e], dim='longitude').load().rename(ARCO_NAMES).rename({'time': 'valid_time'})
        if box.sizes['valid_time'] != 24: raise RuntimeError(f'ARCO has {box.sizes["valid_time"]} hours for {day:%Y-%m-%d}')
        box.to_netcdf(path + '.part'); os.replace(path + '.part', path)
    return xr.open_dataset(path).load(), path


def era5(first, last):
    """ERA5 for every hour from `first` to `last` (UTC, inclusive), as one dataset, and a record of the copy and
    file behind each UTC day."""
    want = pd.date_range(first, last, freq='1h')
    files = cds_files(); parts, record, hashes = [], {}, {}
    fhash = lambda p: hashes.setdefault(p, sha256(p))
    for (y, m) in sorted({(t.year, t.month) for t in want}):
        if (y, m) in files:
            ds = _open_cds(files[(y, m)])
            ds = ds.sel(valid_time=ds.valid_time[(ds.valid_time >= np.datetime64(want[0].tz_localize(None))) &
                                                  (ds.valid_time <= np.datetime64(want[-1].tz_localize(None)))])
            if ds.sizes['valid_time']:
                parts.append(ds)
                for d in pd.DatetimeIndex(ds.valid_time.values).normalize().unique():
                    record[f'{d:%Y-%m-%d}'] = dict(copy='cds', file=os.path.basename(files[(y, m)]),
                                                   sha256=fhash(files[(y, m)]))
    have = set(pd.DatetimeIndex(np.concatenate([p.valid_time.values for p in parts])) if parts else [])
    missing = [t for t in want.tz_localize(None) if t not in have]
    for day in sorted({t.normalize() for t in missing}):
        ds, path = _arco_day(day)
        hrs = [t for t in missing if t.normalize() == day]
        parts.append(ds.sel(valid_time=hrs))
        record[f'{day:%Y-%m-%d}'] = dict(copy='google (ARCO-ERA5)', file=os.path.basename(path), sha256=sha256(path),
                                         hours=len(hrs))
    out = xr.concat(parts, dim='valid_time').sortby('valid_time')
    got = pd.DatetimeIndex(out.valid_time.values)
    if got.has_duplicates or len(got) != len(want) or (got != want.tz_localize(None)).any():
        raise RuntimeError(f'ERA5 hours incomplete: have {len(got)} of {len(want)}')
    return out, dict(sorted(record.items()))


def collector_manifest(public_root):
    p = os.path.join(public_root, 'prospective', 'weather', 'manifest.csv')
    rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    return {r['file']: r for r in rows if r.get('run_utc') == '00' and r.get('resol') == '0p25'}


def run(date, public_root, manifest):
    """The cropped 00 UTC run of `date` (a Timestamp at midnight): (dataset, record) or (None, record)."""
    name = f'ifs_0p25_{date:%Y%m%d}_00z_gb.nc'
    mine = os.path.join(public_root, 'prospective', 'weather', 'data', name)
    if os.path.exists(mine):
        h = sha256(mine); m = manifest.get(name)
        if m is None or m['sha256'] != h:
            raise RuntimeError(f'{name}: SHA-256 {h[:12]} does not match the collector manifest '
                               f'({m["sha256"][:12] if m else "no entry"})')
        return xr.open_dataset(mine).load(), dict(copy='collector', file=name, sha256=h, downloaded_utc=m['downloaded_utc'])
    os.makedirs(RUNS_DIR, exist_ok=True)
    cached = os.path.join(RUNS_DIR, name)
    if not os.path.exists(cached):
        sys.path.insert(0, os.path.join(public_root, 'prospective', 'weather'))
        import ecmwf_daily as E, archive as A
        g = cached.replace('.nc', '.grib2')
        try:
            src, layout, absent = A.fetch(date.date(), E.STEPS, ['2t', 'ssrd', '10u', '10v'], g, optional=('ssrd',))
        except A.Missing:
            return None, dict(copy=None, file=None, note='not on the collector or any archive copy')
        E.crop_to_gb(g, cached + '.part'); os.replace(cached + '.part', cached); os.remove(g)
        open(cached + '.source', 'w').write(f'{src} {layout}' + (f' absent:{",".join(absent)}' if absent else '') + '\n')
    src = open(cached + '.source').read().strip() if os.path.exists(cached + '.source') else 'archive'
    return xr.open_dataset(cached).load(), dict(copy=f'archive ({src})', file=name, sha256=sha256(cached))
