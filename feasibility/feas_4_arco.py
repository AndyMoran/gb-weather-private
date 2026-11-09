"""
Feasibility check 4: Google's copy of ERA5 (ARCO-ERA5) as a fallback to the Copernicus queue.

The CDS queue for ERA5 stood at about 3,000 requests on 6 November 2026. ARCO-ERA5 is ERA5 (with the preliminary
ERA5T) in a public Zarr store on Google Cloud, readable without an account. Before relying on it, three questions:
  1  How far does it reach (its own valid_time_stop attributes), and how is it chunked? A store chunked one hour
     at a time over the whole globe makes a GB-only read download the globe.
  2  Is it the same data as the CDS? Compare one day (3 January 2024) against the January file already downloaded
     from the CDS, variable by variable.
  3  Is its solar radiation the same hourly accumulation (J/m2 per hour), or a different convention?

Reads anonymously; needs gcsfs and zarr 3:  pip install gcsfs "zarr>=3"
Run in `wx`:  python feas_4_arco.py [YYYY-MM-DD ...]   (default 2024-01-03; each day needs its month's CDS file)
Each day ends with AGREE or DIFFER against the brief's tolerances: 0.01 K, 0.01 m/s, 100 J/m2 per hour.
"""
import os, sys, time, zipfile
import numpy as np, xarray as xr

STORE = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
NAMES = {'t2m': '2m_temperature', 'u10': '10m_u_component_of_wind', 'v10': '10m_v_component_of_wind',
         'ssrd': 'surface_solar_radiation_downwards'}
TOL = {'t2m': 0.01, 'u10': 0.01, 'v10': 0.01, 'ssrd': 100.0}
DAYS = sys.argv[1:] or ['2024-01-03']
CDS = lambda day: os.path.expanduser(f'~/gb-weather-data/era5/era5_{day[:4]}_{day[5:7]}.nc')

t = time.time()
ds = xr.open_zarr(STORE, chunks=None, storage_options=dict(token='anon'))
print(f'opened in {time.time() - t:.0f} s')
for k in ('valid_time_start', 'valid_time_stop', 'valid_time_stop_era5t', 'last_updated'):
    print(f'  {k}: {ds.attrs.get(k)}')
for short, long in NAMES.items():
    v = ds[long]; enc = v.encoding
    print(f'  {long}: dims {v.dims}; shape {v.shape}; chunks {enc.get("chunks") or enc.get("preferred_chunks")}; '
          f'units {v.attrs.get("units")}')

def open_cds(f):
    if not zipfile.is_zipfile(f): return xr.open_dataset(f, engine='netcdf4')
    d = f[:-3] + '_unzipped'; os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(f) as z: names = z.namelist(); z.extractall(d)
    return xr.merge([xr.open_dataset(os.path.join(d, n), engine='netcdf4') for n in names if n.endswith('.nc')],
                    compat='override')


def compare(day):
    # The store holds one global field per chunk, so read each hour once (all longitudes) and cut GB in memory:
    # longitudes run 0..359.75 here, so take 351..359.75 and 0..2.5 and put them in -9..2.5 order.
    t = time.time(); parts = {}
    hours = ds.time.sel(time=day).values
    for long in NAMES.values():
        fields = []
        for k, h in enumerate(hours, 1):
            f = ds[long].sel(time=h, latitude=slice(61, 49)).load()
            west = f.sel(longitude=slice(351, 359.75)); east = f.sel(longitude=slice(0, 2.5))
            fields.append(xr.concat([west.assign_coords(longitude=west.longitude - 360), east], dim='longitude'))
            print(f'\r  {day} {long}: hour {k}/{len(hours)} ({time.time() - t:.0f} s)', end='', flush=True)
        parts[long] = xr.concat(fields, dim='time')
    print()
    box = xr.Dataset(parts)
    print(f'{day}: GB read from ARCO in {time.time() - t:.0f} s; shape {dict(box.sizes)}')
    cds = open_cds(CDS(day))
    tc = 'valid_time' if 'valid_time' in cds.coords else 'time'
    cds = cds.sel({tc: day}).rename({tc: 'time'})
    ok = True
    for short, long in NAMES.items():
        a = box[long].transpose('time', 'latitude', 'longitude').values
        b = cds[short].transpose('time', 'latitude', 'longitude').values
        if a.shape != b.shape:
            print(f'  {short}: shapes differ, ARCO {a.shape} vs CDS {b.shape}'); ok = False; continue
        d = np.abs(a - b); ok &= bool(np.nanmax(d) <= TOL[short])
        print(f'  {short}: max |ARCO - CDS| {np.nanmax(d):.4g} (tolerance {TOL[short]:g}); mean {np.nanmean(d):.3g}; '
              f'ratio of means {np.nanmean(a) / np.nanmean(b):.4f}')
    print(f'{day}: {"AGREE" if ok else "DIFFER"}')
    return ok


results = {day: compare(day) for day in DAYS}
print('summary:', ', '.join(f'{d} {"AGREE" if r else "DIFFER"}' for d, r in results.items()))
