"""
Feasibility check 2: ERA5 for Great Britain (Brief 4, Section 3).

  A  One month (January 2024) of hourly 2 m temperature, 10 m wind (u, v) and surface solar radiation over the GB box:
     confirms the request works, and times the queue, so the full download can be planned.
  B  Latency: reads the dataset's catalogue entry for the latest hour ERA5 covers (no download job). That tells how
     many recent context days usually come from the ECMWF runs' first hours instead (the splice rule).

The Copernicus key is read from ~/.cdsapirc, as cdsapi does by default; it never appears in this script.
Outputs go to ~/gb-weather-data/era5/, outside every repository. Run in any environment with cdsapi:
    python feas_2_era5.py [A|B]
"""
import datetime as dt, os, sys, time
import cdsapi

OUT = os.path.expanduser('~/gb-weather-data/era5'); os.makedirs(OUT, exist_ok=True)
AREA = [61, -9, 49, 2.5]                                 # north, west, south, east: the collector's GB box
VARS = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'surface_solar_radiation_downwards']
HOURS = [f'{h:02d}:00' for h in range(24)]
c = cdsapi.Client()


def request(days, target):
    years = sorted({d.year for d in days}); months = sorted({d.month for d in days}); dd = sorted({d.day for d in days})
    c.retrieve('reanalysis-era5-single-levels', {
        'product_type': 'reanalysis', 'variable': VARS, 'year': [str(y) for y in years],
        'month': [f'{m:02d}' for m in months], 'day': [f'{x:02d}' for x in dd], 'time': HOURS, 'area': AREA,
        'data_format': 'netcdf', 'download_format': 'unarchived'}, target)


def open_era5(f):
    """Open a CDS download. Mixing instantaneous variables (temperature, wind) with an accumulated one (solar
    radiation) makes the CDS return a zip of two NetCDF files, whatever the file is called; merge them."""
    import xarray as xr, zipfile
    if not zipfile.is_zipfile(f):
        return xr.open_dataset(f, engine='netcdf4'), ['(single NetCDF file)']
    d = f[:-3] + '_unzipped'; os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(f) as z:
        names = z.namelist(); z.extractall(d)
    parts = [xr.open_dataset(os.path.join(d, n), engine='netcdf4') for n in names if n.endswith('.nc')]
    return xr.merge(parts, compat='override'), names


def part_a():
    t = time.time(); f = os.path.join(OUT, 'era5_2024_01.nc')
    if os.path.exists(f):
        print('A: using the file already downloaded (the first run took about 3 minutes, queue included)')
    else:
        request([dt.date(2024, 1, d) for d in range(1, 32)], f)
        print(f'A: {time.time() - t:.0f} s for one month, queue included')
    ds, names = open_era5(f)
    print(f'A: files inside {names}')
    print(f'A: variables {sorted(ds.data_vars)}; dims {dict(ds.sizes)}')
    for v in ds.data_vars:
        x = ds[v]
        print(f'   {v}: units {x.attrs.get("units")}; min {float(x.min()):.4g}; max {float(x.max()):.4g}; '
              f'missing {int(x.isnull().sum())}')
    tc = 'valid_time' if 'valid_time' in ds.coords else 'time'
    print(f'   first and last times: {ds[tc].values[0]} .. {ds[tc].values[-1]}')


def part_b():
    """Latency, without download jobs. The first version asked for each of the last ten days in turn; a request for a
    day not yet published is an invalid request, which the CDS protection measures (October 2026) can penalise, and
    the jobs sat in the queue for hours. Instead read the dataset's catalogue entry: one plain GET, no key, no queue.
    Its temporal extent ends at the latest hour ERA5 (with ERA5T) covers."""
    import json, requests
    url = 'https://cds.climate.copernicus.eu/api/catalogue/v1/collections/reanalysis-era5-single-levels'
    try:
        r = requests.get(url, timeout=60); r.raise_for_status(); j = r.json()
    except Exception as e:
        print(f'B: could not read the catalogue ({e}); check the date on the dataset page instead'); return
    ext = j.get('extent', {}).get('temporal', {}).get('interval')
    print(f'B: temporal extent {ext}')
    for k in ('updated', 'update_frequency', 'published'):
        if k in j: print(f'   {k}: {j[k]}')
    try:
        end = dt.datetime.fromisoformat(str(ext[0][1]).replace('Z', '+00:00'))
        lag = dt.datetime.now(dt.timezone.utc) - end
        print(f'B: ERA5 ends {end:%Y-%m-%d %H:%M} UTC, {lag.total_seconds() / 86400:.1f} days before now')
    except Exception:
        print('B: could not parse the end date; the raw entry is above')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'AB'
    if 'A' in which: part_a()
    if 'B' in which: part_b()
