"""
Build the study's spatial weights on the 0.25 deg weather grid (Brief 4, Section 3, "From grid to Great Britain").

  w_pop  temperature and wind: share of Great Britain's residential population in each cell
         (UKCEH 1 km gridded population, 2021, British National Grid)
  w_pv   solar radiation: share of Great Britain's installed solar PV capacity in each cell
         (DESNZ renewable electricity by local authority, end-2025 capacity, spread over each authority's land area)

Rules (fixed in the brief, agreed 6 November 2026):
  - A 1 km square goes to the 0.25 deg cell containing its centre, converted from EPSG:27700 to EPSG:4326 by pyproj.
  - Each populated square is assigned to the NEAREST of the four countries (ONS full-resolution boundaries); squares
    nearest Northern Ireland are dropped. ("Inside GB only" would lose coastal squares whose centres lie offshore.)
  - Each GB local authority's capacity is spread over its land area; a cell gets the share of the area it contains.
    Northern Ireland's authorities (codes starting N) are dropped.
  - Both weights sum to one. Written once, hashed, never tuned.

Inputs, in ~/gb-weather-data/weights/ (downloaded by hand; the names can be anything with these patterns):
  population .tif            UKCEH, doi:10.5285/7beefde9-c520-4ddf-897a-0167e8918595 (the residential file, not workday)
  *ountr*  boundary file     ONS Countries boundaries, UK, full resolution (BFE or BFC), GeoJSON / GeoPackage / shapefile
  *istrict* or *LAD* file    ONS Local Authority Districts boundaries, UK, full resolution clipped to the coastline (BFC)
  *.xlsx                     DESNZ "Renewable electricity by local authority, 2014 to 2025"
Outputs, same folder: weights.nc, weights_meta.json (every input's SHA-256, the sheet and column used, totals,
checks), weights_map.png.

Run in `wx` (once:  pip install geopandas rasterio openpyxl matplotlib):
    python make_weights.py
"""
import datetime as dt, glob, hashlib, json, os, re, sys
import numpy as np

# Full-resolution boundaries (Scotland's coast as one feature) exceed GDAL's default GeoJSON feature size; lift it.
os.environ.setdefault('OGR_GEOJSON_MAX_OBJ_SIZE', '0')

D = os.path.expanduser('~/gb-weather-data/weights')
LATS = np.round(np.arange(61, 48.99, -0.25), 2)            # the weather box, north to south: 49 points
LONS = np.round(np.arange(-9, 2.51, 0.25), 2)              # west to east: 47 points
YEAR = '2025'
# DESNZ uses some authority codes that ONS has since replaced. Each entry: old code -> code in the boundary file.
# Barnsley and Sheffield took new codes in 2025 after a minor boundary change between them (Barnsley and Sheffield
# (Boundary Change) Order 2024); ONS marks E08000016 'replaced by E08000038' and lists Sheffield as E08000039.
RECODE = {'E08000016': 'E08000038', 'E08000019': 'E08000039'}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def one(patterns, what):
    hits = sorted({f for p in patterns for f in glob.glob(os.path.join(D, p))})
    hits = [h for h in hits if not h.endswith(('.aux.xml', '.ovr', '.png', '.nc', '.json'))]
    if len(hits) != 1:
        sys.exit(f'Need exactly one {what} in {D}; found {[os.path.basename(h) for h in hits] or "none"}')
    return hits[0]


def code_col(gdf, kind):
    """ONS boundary files name their code column like CTRY24CD or LAD25CD."""
    c = [k for k in gdf.columns if re.fullmatch(rf'{kind}\d\dCD', k, re.I)]
    if len(c) != 1: sys.exit(f'Cannot find the {kind}..CD column; columns are {list(gdf.columns)}')
    return c[0]


def cell_index(lat, lon):
    i = np.rint((61 - lat) / 0.25).astype(int); j = np.rint((lon + 9) / 0.25).astype(int)
    ok = (i >= 0) & (i < len(LATS)) & (j >= 0) & (j < len(LONS))
    return i, j, ok


def population(tif, countries):
    import geopandas as gpd, pandas as pd, rasterio
    from pyproj import Transformer
    with rasterio.open(tif) as r:
        if r.crs is None or r.crs.to_epsg() != 27700: sys.exit(f'Population raster CRS is {r.crs}, expected EPSG:27700')
        a = r.read(1, masked=True).filled(0).astype('float64'); a[~np.isfinite(a) | (a < 0)] = 0
        rows, cols = np.nonzero(a > 0)
        xs, ys = rasterio.transform.xy(r.transform, rows, cols, offset='center')
        res = r.res
    xs, ys, pop = np.asarray(xs), np.asarray(ys), a[rows, cols]
    pts = gpd.GeoDataFrame({'pop': pop}, geometry=gpd.points_from_xy(xs, ys), crs=27700)
    c = countries.to_crs(27700); cc = code_col(c, 'CTRY')
    # Nearest country, fast: split each country into its separate parts (mainland, each island), then
    # (1) point-in-polygon for every square (indexed, prepared shapes): a square inside a country is nearest to it;
    # (2) distance only for the few squares outside every outline (coastal squares whose centres are offshore).
    # Measuring distance from every square to whole full-resolution countries took hours.
    parts = c[[cc, 'geometry']].explode(index_parts=False).reset_index(drop=True)
    inside = gpd.sjoin(pts, parts, how='left', predicate='within')
    inside = inside[~inside.index.duplicated()]
    country = inside[cc].copy(); dist = pd.Series(0.0, index=pts.index)
    out = country.isna()
    if out.any():
        nb = gpd.sjoin_nearest(pts[out], parts, how='left', distance_col='dist_m')
        nb = nb[~nb.index.duplicated()]                       # a point equidistant from two parts: keep the first
        country[out] = nb[cc]; dist[out] = nb['dist_m']
    near = pd.DataFrame({cc: country, 'dist_m': dist})
    ni = near[cc].str.startswith('N').to_numpy()
    lon, lat = Transformer.from_crs(27700, 4326, always_xy=True).transform(xs, ys)
    i, j, ok = cell_index(np.asarray(lat), np.asarray(lon))
    keep = ok & ~ni
    w = np.zeros((len(LATS), len(LONS)))
    np.add.at(w, (i[keep], j[keep]), pop[keep])
    info = {'raster_resolution_m': list(res), 'squares_populated': int(len(pop)),
            'population_uk': float(pop.sum()), 'population_ni_dropped': float(pop[ni].sum()),
            'population_gb': float(pop[~ni].sum()), 'population_gb_outside_box': float(pop[~ni & ~ok].sum()),
            'squares_assigned_offshore_m_max': float(near.loc[~ni, 'dist_m'].max()),
            'squares_outside_every_outline': int(out.sum())}
    return w / w.sum(), info


def desnz_capacity(xlsx):
    """Find the end-of-YEAR installed-capacity sheet by local authority and its solar PV column (MW)."""
    import pandas as pd
    book = pd.ExcelFile(xlsx)
    cands = []
    for sh in book.sheet_names:
        raw = pd.read_excel(book, sheet_name=sh, header=None, nrows=15)
        text = ' '.join(str(v) for v in raw.values.ravel() if isinstance(v, str)).lower() + ' ' + sh.lower()
        if 'capacity' in text and YEAR in text and 'generation' not in sh.lower():
            cands.append(sh)
    if not cands:
        sys.exit(f'No sheet with installed capacity for {YEAR} by local authority. Sheets: {book.sheet_names}. '
                 'Fall back to the regional installed-capacity file (brief, Section 3).')
    # prefer a sheet named for capacity and the year
    cands.sort(key=lambda sh: (YEAR not in sh, 'capacity' not in sh.lower()))
    sh = cands[0]
    raw = pd.read_excel(book, sheet_name=sh, header=None)
    hdr = next((k for k in range(min(40, len(raw)))
                if any(isinstance(v, str) and v.strip().lower().startswith('local authority code') for v in raw.iloc[k])), None)
    if hdr is None: sys.exit(f'Sheet "{sh}": no "Local Authority Code" header row')
    head = [str(v).strip() if v is not None else '' for v in raw.iloc[hdr]]
    code = next(k for k, h in enumerate(head) if h.lower().startswith('local authority code'))
    pv = [k for k, h in enumerate(head) if re.match(r'(photovoltaics|solar\s*pv|solar)\b', h, re.I)]
    if len(pv) != 1: sys.exit(f'Sheet "{sh}": cannot pick one solar column from {head}')
    pv = pv[0]
    body = raw.iloc[hdr + 1:, [code, pv]].copy(); body.columns = ['code', 'mw']
    body['mw'] = pd.to_numeric(body['mw'], errors='coerce').fillna(0.0)
    coded = body['code'].astype(str).str.strip().str.fullmatch(r'[ENSW]\d{8}')
    # rows without an authority code: totals, "Unallocated", notes. Report what they hold; never use them.
    uncoded = body[~coded & (body['mw'] > 0)]
    body = body[coded].assign(code=lambda b: b['code'].astype(str).str.strip())
    title = ' '.join(str(v) for v in raw.iloc[:hdr].values.ravel() if isinstance(v, str))[:300]
    return body, {'sheet': sh, 'title_text': title, 'header_row': int(hdr) + 1, 'code_column': head[code],
                  'solar_column': head[pv],
                  'rows_without_code': [(str(c), float(m)) for c, m in uncoded[['code', 'mw']].itertuples(index=False)][:20]}


def solar(xlsx, lads):
    import geopandas as gpd
    from shapely.geometry import box
    cap, info = desnz_capacity(xlsx)
    gb = cap[~cap['code'].str.startswith('N')].assign(code=lambda c: c['code'].replace(RECODE))
    gb = gb.groupby('code', as_index=True)['mw'].sum()
    l = lads.to_crs(27700); lc = code_col(l, 'LAD')
    l = l[~l[lc].str.startswith('N')].dissolve(by=lc)
    missing = sorted(set(gb.index) - set(l.index))
    miss_mw = float(gb[missing].sum())
    info.update({'capacity_gb_mw': float(gb.sum()), 'authorities': int(len(gb)), 'recoded': RECODE,
                 'codes_not_in_boundaries': missing, 'capacity_not_in_boundaries_mw': miss_mw})
    if missing:
        sys.exit(f'{miss_mw:.1f} MW sits in authority codes missing from the boundary file: {missing}. '
                 'Add each to RECODE from ONS code history; no capacity is dropped silently.')
    cells = [(i, j, box(lon - .125, lat - .125, lon + .125, lat + .125).segmentize(0.01))
             for i, lat in enumerate(LATS) for j, lon in enumerate(LONS)]
    g = gpd.GeoDataFrame({'i': [c[0] for c in cells], 'j': [c[1] for c in cells]},
                         geometry=[c[2] for c in cells], crs=4326).to_crs(27700)
    la = gpd.GeoDataFrame({'mw': gb.reindex(l.index).fillna(0.0).to_numpy(), 'la_area': l.geometry.area.to_numpy()},
                          geometry=l.geometry.to_numpy(), crs=27700)
    la = la[la['mw'] > 0]
    x = gpd.overlay(la, g, how='intersection', keep_geom_type=True)
    x['mw_cell'] = x['mw'] * x.geometry.area / x['la_area']
    w = np.zeros((len(LATS), len(LONS)))
    np.add.at(w, (x['i'].to_numpy(), x['j'].to_numpy()), x['mw_cell'].to_numpy())
    info['capacity_placed_mw'] = float(w.sum())
    return w / w.sum(), info


def main():
    import geopandas as gpd, xarray as xr
    tif = one(['*.tif', '*.tiff'], 'population raster (.tif)')
    ctry = one(['*ountr*.geojson', '*ountr*.gpkg', '*ountr*.shp', '*ountr*.json'], 'countries boundary file')
    lad = one(['*istrict*.geojson', '*istrict*.gpkg', '*istrict*.shp', '*LAD*.geojson', '*LAD*.gpkg', '*LAD*.shp',
               '*istrict*.json', '*LAD*.json'], 'local authority boundary file')
    xlsx = one(['*.xlsx'], 'DESNZ workbook (.xlsx)')
    print('inputs:', *(os.path.basename(p) for p in (tif, ctry, lad, xlsx)), sep='\n  ')
    w_pop, ipop = population(tif, gpd.read_file(ctry))
    w_pv, ipv = solar(xlsx, gpd.read_file(lad))

    ds = xr.Dataset({'w_pop': (('latitude', 'longitude'), w_pop), 'w_pv': (('latitude', 'longitude'), w_pv)},
                    coords={'latitude': LATS, 'longitude': LONS})
    ds['w_pop'].attrs['description'] = 'share of GB residential population (UKCEH 2021), temperature and wind'
    ds['w_pv'].attrs['description'] = f'share of GB installed solar PV capacity, end-{YEAR} (DESNZ, by local authority)'
    out = os.path.join(D, 'weights.nc'); ds.to_netcdf(out)

    def top(w, n=5):
        k = np.argsort(w.ravel())[::-1][:n]
        return [(float(LATS[a // len(LONS)]), float(LONS[a % len(LONS)]), round(float(w.ravel()[a]), 4)) for a in k]

    meta = {'built_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
            'inputs': {os.path.basename(p): sha256(p) for p in (tif, ctry, lad, xlsx)},
            'population': ipop, 'solar': ipv,
            'cells_nonzero': {'w_pop': int((w_pop > 0).sum()), 'w_pv': int((w_pv > 0).sum())},
            'top_cells': {'w_pop': top(w_pop), 'w_pv': top(w_pv)},
            'weights_nc_sha256': sha256(out)}
    json.dump(meta, open(os.path.join(D, 'weights_meta.json'), 'w'), indent=1)

    print(f"population: UK {ipop['population_uk'] / 1e6:.2f} m; Northern Ireland dropped {ipop['population_ni_dropped'] / 1e6:.2f} m; "
          f"GB {ipop['population_gb'] / 1e6:.2f} m; GB outside the box {ipop['population_gb_outside_box']:.0f}; "
          f"{ipop['squares_outside_every_outline']} squares outside every outline, the furthest GB one {ipop['squares_assigned_offshore_m_max']:.0f} m offshore")
    print(f"solar: sheet '{ipv['sheet']}', column '{ipv['solar_column']}'; GB {ipv['capacity_gb_mw']:.0f} MW in "
          f"{ipv['authorities']} authorities; placed {ipv['capacity_placed_mw']:.0f} MW; codes not in boundaries "
          f"{ipv['codes_not_in_boundaries'] or 'none'}")
    print(f"solar rows without an authority code (not used; check none is a real authority): {ipv['rows_without_code'] or 'none'}")
    print('cells with weight: population', meta['cells_nonzero']['w_pop'], '| solar', meta['cells_nonzero']['w_pv'])
    print('largest population cells (lat, lon, share):', meta['top_cells']['w_pop'])
    print('largest solar cells (lat, lon, share):', meta['top_cells']['w_pv'])
    print('weights.nc sha256', meta['weights_nc_sha256'])

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 6.5), constrained_layout=True)
    ext = [LONS[0] - .125, LONS[-1] + .125, LATS[-1] - .125, LATS[0] + .125]
    for a, w, t in ((ax[0], w_pop, 'Population share (temperature, wind)'), (ax[1], w_pv, 'Solar capacity share')):
        m = a.imshow(np.where(w > 0, w, np.nan), extent=ext, cmap='viridis', aspect=1.6)
        a.set_title(t, fontsize=10); fig.colorbar(m, ax=a, shrink=0.7)
    fig.savefig(os.path.join(D, 'weights_map.png'), dpi=110)
    print('map: weights_map.png')


if __name__ == '__main__':
    main()
