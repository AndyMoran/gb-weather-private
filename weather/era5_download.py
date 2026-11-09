"""
Download ERA5 for the weather study (Brief 4, Section 3): hourly 2 m temperature, 10 m wind (u, v) and surface solar
radiation over the GB box, one calendar month per request, from March 2023 to the latest day ERA5 covers. (The first backtest day,
16 March 2024, trains the regression from 15 March 2023, D-367.)

Why this shape:
  - One request at a time, never in parallel, and only for days ERA5 has published. The CDS protection measures
    (October 2026) ask for sequential requests and can suspend an account for repeated invalid ones. cdsapi waits
    for each request to finish before the next is sent.
  - Newest months first. The prospective window (from 20 November 2026) needs the last year for the regression's
    training and the models' context; the backtest's older months can follow.
  - Resumable. A month already on disk and readable is skipped, so the script can be stopped and run again.
  - The latest month is partial. Its file is named for its last day, and is fetched again (complete) on a later run.

The key is read from ~/.cdsapirc by cdsapi; it never appears here. Files go to ~/gb-weather-data/era5/, outside every
repository; a CSV log records each file's time and SHA-256, for the lock. One copy runs at a time (a lock file), so
Task Scheduler can start it every hour: a start while a copy is running exits at once, and a start after a reboot
resumes. Run in `wx`:
    python era5_download.py >> ~/gb-weather-data/era5/download.log 2>&1
"""
import calendar, datetime as dt, hashlib, os, sys, time, zipfile
import requests

OUT = os.path.expanduser('~/gb-weather-data/era5'); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, 'files.csv')
FIRST = dt.date(2023, 3, 1)
AREA = [61, -9, 49, 2.5]                                 # north, west, south, east: the collector's GB box
VARS = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'surface_solar_radiation_downwards']
CATALOGUE = 'https://cds.climate.copernicus.eu/api/catalogue/v1/collections/reanalysis-era5-single-levels'


def latest_day():
    """The last full day ERA5 covers, from the catalogue: one plain GET, no download job."""
    j = requests.get(CATALOGUE, timeout=60).json()
    end = dt.datetime.fromisoformat(j['extent']['temporal']['interval'][0][1].replace('Z', '+00:00'))
    return (end - dt.timedelta(hours=23)).date()           # the extent may end at 00:00; keep whole days only


def readable(path):
    """A finished download: a zip of NetCDF files (mixed instant and accumulated variables) or one NetCDF file."""
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z: return z.testzip() is None and any(n.endswith('.nc') for n in z.namelist())
        with open(path, 'rb') as f: return f.read(4) in (b'CDF\x01', b'CDF\x02', b'\x89HDF')
    except Exception:
        return False


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def months(last):
    y, m, out = last.year, last.month, []
    while (y, m) >= (FIRST.year, FIRST.month):
        n = calendar.monthrange(y, m)[1] if (y, m) != (last.year, last.month) else last.day
        out.append((y, m, n, n == calendar.monthrange(y, m)[1]))
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return out                                              # newest first


def single_instance():
    """Hold an exclusive lock for the whole run, so an hourly scheduled start does nothing while a copy is running.
    The lock is released by the operating system if the process dies, so a restart after a reboot is never blocked."""
    import fcntl
    f = open(os.path.join(OUT, '.download.lock'), 'w')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(0)                                         # another copy is running: leave quietly
    return f                                                # keep the file open to keep the lock


def main():
    lock = single_instance()                                # noqa: F841
    import cdsapi
    c = cdsapi.Client()
    last = latest_day()
    print(f'=== start {dt.datetime.now():%Y-%m-%d %H:%M}; ERA5 covers to {last}', flush=True)
    if not os.path.exists(LOG):
        with open(LOG, 'w') as f: f.write('file,first_day,last_day,complete,downloaded_utc,seconds,bytes,sha256\n')
    for y, m, n, complete in months(last):
        name = f'era5_{y}_{m:02d}.nc' if complete else f'era5_{y}_{m:02d}_to{n:02d}.nc'
        path = os.path.join(OUT, name)
        if os.path.exists(path) and readable(path):
            continue
        for old in os.listdir(OUT):                         # an earlier partial file for this month is superseded
            if old.startswith(f'era5_{y}_{m:02d}_to') and old != name: os.replace(os.path.join(OUT, old), os.path.join(OUT, old + '.superseded'))
        t = time.time(); tmp = path + '.part'
        print(f'{dt.datetime.now():%H:%M} requesting {y}-{m:02d} days 1-{n}', flush=True)
        try:
            c.retrieve('reanalysis-era5-single-levels', {
                'product_type': 'reanalysis', 'variable': VARS, 'year': str(y), 'month': f'{m:02d}',
                'day': [f'{d:02d}' for d in range(1, n + 1)], 'time': [f'{h:02d}:00' for h in range(24)],
                'area': AREA, 'data_format': 'netcdf', 'download_format': 'unarchived'}, tmp)
        except Exception as e:
            print(f'   FAILED {y}-{m:02d}: {str(e)[:200]}; stopping (rerun to resume)', flush=True); sys.exit(1)
        if not readable(tmp):
            print(f'   {y}-{m:02d}: download not readable; stopping', flush=True); sys.exit(1)
        os.replace(tmp, path)
        with open(LOG, 'a') as f:
            f.write(f'{name},{y}-{m:02d}-01,{y}-{m:02d}-{n:02d},{complete},{dt.datetime.utcnow():%Y-%m-%dT%H:%M:%S},'
                    f'{time.time() - t:.0f},{os.path.getsize(path)},{sha256(path)}\n')
        print(f'   done in {time.time() - t:.0f} s', flush=True)
    print(f'=== end {dt.datetime.now():%Y-%m-%d %H:%M}', flush=True)


if __name__ == '__main__':
    main()
