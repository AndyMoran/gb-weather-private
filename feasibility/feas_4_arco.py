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
Run in `wx`:  python feas_4_arco.py
"""
import os, time, zipfile
import numpy as np, xarray as xr

STORE = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
NAMES = {'t2m': '2m_temperature', 'u10': '10m_u_component_of_wind', 'v10': '10m_v_component_of_wind',
         'ssrd': 'surface_solar_radiation_downwards'}
CDS = os.path.expanduser('~/gb-weather-data/era5/era5_2024_01.nc')
DAY = '2024-01-03'

t = time.time()
ds = xr.open_zarr(STORE, chunks=None, storage_options=dict(token='anon'))
print(f'opened in {time.time() - t:.0f} s')
for k in ('valid_time_start', 'valid_time_stop', 'valid_time_stop_era5t', 'last_updated'):
    print(f'  {k}: {ds.attrs.get(k)}')
for short, long in NAMES.items():
    v = ds[long]; enc = v.encoding
    print(f'  {long}: dims {v.dims}; shape {v.shape}; chunks {enc.get("chunks") or enc.get("preferred_chunks")}; '
          f'units {v.attrs.get("units")}')

# The GB box: longitudes run 0..359.75 here, so take 351..359.75 and 0..2.5 and put them in -9..2.5 order.
t = time.time()
sub = ds[list(NAMES.values())].sel(time=DAY, latitude=slice(61, 49))
west = sub.sel(longitude=slice(351, 359.75)); east = sub.sel(longitude=slice(0, 2.5))
west = west.assign_coords(longitude=west.longitude - 360)
box = xr.concat([west, east], dim='longitude').load()
print(f'one day over GB read in {time.time() - t:.0f} s; shape {dict(box.sizes)}')


def open_cds(f):
    if not zipfile.is_zipfile(f): return xr.open_dataset(f, engine='netcdf4')
    d = f[:-3] + '_unzipped'; os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(f) as z: names = z.namelist(); z.extractall(d)
    return xr.merge([xr.open_dataset(os.path.join(d, n), engine='netcdf4') for n in names if n.endswith('.nc')],
                    compat='override')


cds = open_cds(CDS)
tc = 'valid_time' if 'valid_time' in cds.coords else 'time'
cds = cds.sel({tc: DAY}).rename({tc: 'time'})
for short, long in NAMES.items():
    a = box[long].transpose('time', 'latitude', 'longitude').values
    b = cds[short].transpose('time', 'latitude', 'longitude').values
    if a.shape != b.shape:
        print(f'{short}: shapes differ, ARCO {a.shape} vs CDS {b.shape}'); continue
    d = np.abs(a - b)
    print(f'{short}: max |ARCO - CDS| {d.max():.4g}; mean {d.mean():.3g}; CDS mean {np.abs(b).mean():.4g}; '
          f'ratio of means {np.nanmean(a) / np.nanmean(b):.4f}')
print('Same data and the same solar convention if every max difference is ~0 and the solar ratio is 1.0000.')
