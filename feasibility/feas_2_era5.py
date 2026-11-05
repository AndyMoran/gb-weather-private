"""
Feasibility check 2: ERA5 for Great Britain (Brief 4, Section 3).

  A  One month (January 2024) of hourly 2 m temperature, 10 m wind (u, v) and surface solar radiation over the GB box:
     confirms the request works, and times the queue, so the full download can be planned.
  B  Latency: requests each of the last ten days separately, and reports which are available. That fixes how many
     recent context days must come from the ECMWF runs' first hours instead (the splice rule).

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


def part_a():
    t = time.time(); f = os.path.join(OUT, 'era5_2024_01.nc')
    request([dt.date(2024, 1, d) for d in range(1, 32)], f)
    import xarray as xr
    ds = xr.open_dataset(f)
    print(f'A: {time.time() - t:.0f} s for one month; variables {sorted(ds.data_vars)}; dims {dict(ds.sizes)}')


def part_b():
    today = dt.date.today()
    for k in range(10, 0, -1):
        d = today - dt.timedelta(days=k); f = os.path.join(OUT, f'latency_{d:%Y%m%d}.nc')
        try:
            request([d], f); print(f'B {d}: available')
        except Exception as e:
            print(f'B {d}: not available ({str(e)[:80]})')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'AB'
    if 'A' in which: part_a()
    if 'B' in which: part_b()
