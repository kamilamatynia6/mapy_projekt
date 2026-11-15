"""Landsat Collection 1: dane, wskaźniki, GeoTIFF, XYZ i statyczna mapa.
Nie wymaga QGIS ani programów gdal_translate/gdal2tiles w PATH.
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds, array_bounds
from rasterio.warp import transform, transform_geom, transform_bounds, reproject
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
INDEX_BANDS = {'NDVI': (5, 4), 'NDBI': (6, 5), 'MNDWI': (3, 6)}
CMAPS = {'NDVI': 'RdYlGn', 'NDBI': 'PuOr_r', 'MNDWI': 'BrBG', 'DEFICYT': 'RdYlGn_r'}
NS = {'e': 'http://espa.cr.usgs.gov/v2'}


def read_metadata(scene: Path) -> dict:
    """Obsługuje dostarczony produkt C1 ESPA. Inne kolekcje wymagają adaptera."""
    files = list(scene.glob('*.xml'))
    if len(files) != 1:
        raise ValueError('Oczekiwano jednego pliku ESPA XML.')
    root = ET.parse(files[0]).getroot()
    product = root.findtext('e:global_metadata/e:product_id', namespaces=NS)
    if not product or '_01_' not in product:
        raise ValueError('To ćwiczenie obsługuje wyłącznie Landsat Collection 1 z pixel_qa.')
    bands = {}
    for b in root.findall('e:bands/e:band', NS):
        name = b.attrib['name']
        valid = b.find('e:valid_range', NS)
        bands[name] = {
            'file': b.findtext('e:file_name', namespaces=NS),
            'scale': float(b.get('scale_factor', 1)),
            'offset': float(b.get('add_offset', 0)),
            'fill': float(b.get('fill_value', -9999)),
            'valid_min': float(valid.get('min')) if valid is not None else None,
            'valid_max': float(valid.get('max')) if valid is not None else None,
        }
    return {'product': product, 'date': root.findtext('e:global_metadata/e:acquisition_date', namespaces=NS), 'bands': bands}


def quality_mask(qa: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    """True = dobry piksel C1. Woda (bit 2) pozostaje ważna."""
    bad_bits = (1 << 0) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 10)
    bad = (qa & bad_bits) != 0
    bad |= ((qa >> 6) & 3) == 3  # wysoka pewność chmur
    bad |= ((qa >> 8) & 3) == 3  # wysoka pewność cirrus
    bad |= (saturation & (1 | sum(1 << b for b in range(2, 7)))) != 0
    return ~bad


def load_scene(data_dir: Path = ROOT / 'data') -> dict:
    """Czyta wycinek sceny. Zwraca DN, reflektancję, maski i georeferencję."""
    data_dir = Path(data_dir)
    meta = read_metadata(data_dir / 'scene')
    aoi = json.loads((data_dir / 'aoi.geojson').read_text(encoding='utf-8'))
    raw = {}
    profile = None
    for key in [*(f'sr_band{i}' for i in range(2, 7)), 'pixel_qa', 'radsat_qa']:
        with rasterio.open(data_dir / 'scene' / meta['bands'][key]['file']) as src:
            if profile is None:
                profile = src.profile.copy()
            elif (src.crs != profile['crs'] or src.transform != profile['transform'] or src.width != profile['width'] or src.height != profile['height']):
                raise ValueError(f'Niezgodna siatka rastra: {key}')
            raw[key] = src.read(1)
    geoms = [transform_geom('EPSG:4326', profile['crs'], f['geometry']) for f in aoi['features']]
    inside = geometry_mask(geoms, out_shape=raw['pixel_qa'].shape, transform=profile['transform'], invert=True)
    if not inside.any():
        raise ValueError('Granica nie przecina sceny. Sprawdź CRS i zasięg.')
    quality = quality_mask(raw['pixel_qa'], raw['radsat_qa'])
    valid = inside & quality
    for i in range(2, 7):
        b = meta['bands'][f'sr_band{i}']
        a = raw[f'sr_band{i}']
        valid &= a != b['fill']
        valid &= (a >= b['valid_min']) & (a <= b['valid_max'])
    if not valid.any():
        raise ValueError('Po maskowaniu nie pozostały ważne piksele.')
    reflectance = {}
    for i in range(2, 7):
        b = meta['bands'][f'sr_band{i}']
        a = raw[f'sr_band{i}'].astype('float32') * b['scale'] + b['offset']
        reflectance[i] = np.where(valid, a, np.nan)
    return {'metadata': meta, 'profile': profile, 'raw': raw, 'inside': inside,
            'valid': valid, 'quality': quality, 'reflectance': reflectance, 'aoi': aoi}


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Zachowuje zera i wartości ujemne. Zerowy mianownik daje NaN."""
    a, b = np.asarray(a, dtype='float32'), np.asarray(b, dtype='float32')
    out = np.full(np.broadcast_shapes(a.shape, b.shape), np.nan, dtype='float32')
    denominator = a + b
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(denominator) > 1e-6)
    np.divide(a - b, denominator, out=out, where=valid)
    return out


def calculate_indices(reflectance: dict) -> dict:
    return {name: normalized_difference(reflectance[a], reflectance[b]) for name, (a, b) in INDEX_BANDS.items()}


def stretch_rgb(reflectance: dict) -> np.ndarray:
    """RGB 4/3/2: osobne rozciągnięcie 2-98 percentyl wyłącznie do prezentacji."""
    return stretch_bands(reflectance, (4, 3, 2))


# Atlas (Kamila): kompozycje w fałszywych barwach – te same kanały w innej kolejności
COMPOSITES = {
    'PODCZERWIEN': (5, 4, 3),  # roślinność świeci na czerwono
    'MIASTO': (6, 5, 4),       # zabudowa i odkryty grunt odróżniają się od zieleni
}


def stretch_bands(reflectance: dict, bands: tuple) -> np.ndarray:
    """Dowolna kompozycja 3 kanałów z rozciągnięciem 2-98 percentyl (tylko do prezentacji)."""
    channels = []
    for b in bands:
        a = reflectance[b]
        lo, hi = np.nanpercentile(a, (2, 98))
        scaled = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        channels.append((np.nan_to_num(scaled) * 255).astype('uint8'))
    alpha = (np.isfinite(reflectance[4]) * 255).astype('uint8')
    return np.dstack([*channels, alpha])


# Atlas (Kamila): klasy pokrycia terenu wyznaczane progami wskaźników.
# Kolejność ma znaczenie: woda ma pierwszeństwo, potem gęsta i niska roślinność.
CLASSES = [
    {'id': 1, 'name': 'Woda', 'color': '#2c7fb8', 'rule': 'MNDWI > 0'},
    {'id': 2, 'name': 'Gęsta roślinność (lasy, zarośla, bujne uprawy)', 'color': '#1a7f37', 'rule': 'NDVI ≥ 0.7'},
    {'id': 3, 'name': 'Roślinność niska (łąki, pola, parki)', 'color': '#a6d96a', 'rule': '0.4 ≤ NDVI < 0.7'},
    {'id': 4, 'name': 'Zabudowa i odkryty grunt', 'color': '#c0504d', 'rule': 'NDVI < 0.4'},
]


def classify(indices: dict) -> np.ndarray:
    """Zwraca numer klasy dla każdego piksela; 0 = brak ważnych danych."""
    ndvi, mndwi = indices['NDVI'], indices['MNDWI']
    ok = np.isfinite(ndvi) & np.isfinite(indices['NDBI']) & np.isfinite(mndwi)
    cls = np.zeros(ndvi.shape, dtype='uint8')
    cls[ok] = 4
    cls[ok & (ndvi >= 0.4)] = 3
    cls[ok & (ndvi >= 0.7)] = 2
    cls[ok & (mndwi > 0)] = 1
    return cls


def class_rgba(cls: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*cls.shape, 4), dtype='uint8')
    for c in CLASSES:
        rgb = [int(c['color'][i:i+2], 16) for i in (1, 3, 5)]
        rgba[cls == c['id']] = [*rgb, 255]
    return rgba


def class_stats(cls: np.ndarray, pixel_area_m2: float) -> list:
    """Udział (%) i powierzchnia (km²) każdej klasy."""
    total = int((cls > 0).sum())
    return [{**c, 'pixels': int((cls == c['id']).sum()),
             'percent': round(100 * float((cls == c['id']).sum()) / total, 2),
             'km2': round(float((cls == c['id']).sum()) * pixel_area_m2 / 1e6, 1)} for c in CLASSES]


def green_deficit(indices: dict) -> np.ndarray:
    """Atlas (Kamila): deficyt zieleni = NDBI − NDVI.
    Dużo zabudowy i mało roślinności daje wartości bliskie 0 lub dodatnie,
    las daje wartości bliskie −1. Wskaźnik pomocniczy, bez kalibracji terenowej."""
    return (indices['NDBI'] - indices['NDVI']).astype('float32')


def grid_geojson(cls: np.ndarray, indices: dict, profile: dict, size_m=1000, min_valid=0.3) -> dict:
    """Atlas (Kamila): siatka kwadratów ok. 1 km. Każdy kwadrat dostaje udział klas
    i średnie wskaźników. Kwadraty z mniej niż 30% ważnych pikseli są pomijane."""
    t = profile['transform']
    step = int(round(size_m / abs(t.a)))            # 33 piksele po 30 m = 990 m
    height, width = cls.shape
    features = []
    for r0 in range(0, height, step):
        for c0 in range(0, width, step):
            block = cls[r0:r0+step, c0:c0+step]
            n = int((block > 0).sum())
            if n < min_valid * step * step:
                continue
            share = {key: round(100 * float((block == k).sum()) / n, 1)
                     for k, key in [(1, 'woda'), (2, 'gesta'), (3, 'niska'), (4, 'zabudowa')]}
            means = {name.lower(): round(float(np.nanmean(indices[name][r0:r0+step, c0:c0+step])), 3)
                     for name in ('NDVI', 'NDBI', 'MNDWI')}
            x0, y0 = t * (c0, r0)
            x1, y1 = t * (min(c0 + step, width), min(r0 + step, height))
            ring = transform_geom(profile['crs'], 'EPSG:4326', {'type': 'Polygon',
                                  'coordinates': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]})
            features.append({'type': 'Feature',
                             'geometry': {'type': 'Polygon', 'coordinates': [[[round(x, 5), round(y, 5)] for x, y in ring['coordinates'][0]]]},
                             'properties': {**share, 'zielen': round(share['gesta'] + share['niska'], 1),
                                            **means, 'km2': round(n * abs(t.a * t.e) / 1e6, 2)}})
    return {'type': 'FeatureCollection', 'features': features}


# Atlas (Kamila): przystanki wycieczki po atlasie. Statystyki liczymy z danych (koło 500 m),
# żeby opisy na stronie opierały się na wynikach, a nie na zgadywaniu.
TOUR = [
    {'title': 'Lasek Wolski', 'lat': 50.055, 'lon': 19.855, 'zoom': 13, 'layer': 'NDVI',
     'text': 'Największy kompleks leśny w granicach Krakowa. Gęsta, zdrowa roślinność daje tu najwyższe NDVI na całej mapie.'},
    {'title': 'Stare Miasto', 'lat': 50.0617, 'lon': 19.937, 'zoom': 13, 'layer': 'DEFICYT',
     'text': 'Zwarta zabudowa centrum: niskie NDVI i najwyższy deficyt zieleni. Planty i bulwary to tylko niewielki udział koła.'},
    {'title': 'Wisła i Zalew Bagry', 'lat': 50.0332, 'lon': 19.9883, 'zoom': 13, 'layer': 'MNDWI',
     'text': 'Woda ma dodatnie MNDWI, więc rzeka i zalew wyraźnie odcinają się od lądu. Wąskie odcinki są mniej pewne, bo piksel 30 m obejmuje też brzeg.'},
    {'title': 'Kombinat w Nowej Hucie', 'lat': 50.080, 'lon': 20.105, 'zoom': 13, 'layer': 'MIASTO',
     'text': 'Hale i place przemysłowe w kompozycji B6/B5/B4 są różowofioletowe. Pomiędzy nimi rośnie zieleń, dlatego okolica to mieszanina klas.'},
    {'title': 'Pola po żniwach – uwaga na błąd', 'lat': 50.25, 'lon': 20.12, 'zoom': 12, 'layer': 'KLASY',
     'text': 'Czerwone plamy to nie miasto, tylko pola po żniwach. Odkryta gleba ma niskie NDVI i NDBI podobne do zabudowy – to ograniczenie klasyfikacji progowej.'},
]


def tour_points(cls: np.ndarray, indices: dict, profile: dict, radius_m=500) -> list:
    t = profile['transform']
    rows, cols = np.indices(cls.shape)
    result = []
    for stop in TOUR:
        xs, ys = transform('EPSG:4326', profile['crs'], [stop['lon']], [stop['lat']])
        col, row = ~t * (xs[0], ys[0])
        r = radius_m / abs(t.a)
        r0, r1 = int(max(row - r, 0)), int(min(row + r + 1, cls.shape[0]))
        c0, c1 = int(max(col - r, 0)), int(min(col + r + 1, cls.shape[1]))
        inside = (rows[r0:r1, c0:c1] + .5 - row) ** 2 + (cols[r0:r1, c0:c1] + .5 - col) ** 2 <= r * r
        block = cls[r0:r1, c0:c1][inside]
        n = max(int((block > 0).sum()), 1)
        result.append({**stop, 'radius': radius_m,
                       'share': {str(c['id']): round(100 * float((block == c['id']).sum()) / n, 1) for c in CLASSES},
                       **{name.lower(): round(float(np.nanmean(indices[name][r0:r1, c0:c1][inside])), 2)
                          for name in ('NDVI', 'NDBI', 'MNDWI')}})
    return result


def analysis_png(cls: np.ndarray, indices: dict, profile: dict, path: Path, step_m=60) -> dict:
    """Atlas (Kamila): raster danych do analizy w przeglądarce, w siatce EPSG:4326
    (piksel ok. 60 m), żeby strona mogła przeliczać współrzędne na piksel bez bibliotek.
    Kanały PNG: R = klasa (0 = brak danych), G = NDVI, B = MNDWI; wartość = (wskaźnik + 1) * 127.5."""
    h, w = cls.shape
    west, south, east, north = transform_bounds(profile['crs'], 'EPSG:4326', *array_bounds(h, w, profile['transform']))
    lat0 = math.radians((south + north) / 2)
    width = int(math.ceil((east - west) / (step_m / (111320 * math.cos(lat0)))))
    height = int(math.ceil((north - south) / (step_m / 110540)))
    dst_transform = from_bounds(west, south, east, north, width, height)

    def warp(a, dtype, nodata):
        out = np.full((height, width), nodata, dtype=dtype)
        reproject(a.astype(dtype), out, src_transform=profile['transform'], src_crs=profile['crs'],
                  dst_transform=dst_transform, dst_crs='EPSG:4326', resampling=Resampling.nearest,
                  src_nodata=nodata, dst_nodata=nodata)
        return out

    def encode(a):
        v = warp(a, 'float32', np.nan)
        return np.where(np.isfinite(v), np.clip(np.round((v + 1) * 127.5), 0, 255), 0).astype('uint8')

    k = warp(cls, 'uint8', 0)
    rgba = np.dstack([k, encode(indices['NDVI']), encode(indices['MNDWI']), np.full_like(k, 255)])
    Image.fromarray(rgba, 'RGBA').save(path, optimize=True)
    return {'url': path.name, 'bounds': [[south, west], [north, east]], 'width': width, 'height': height}


def index_rgba(array: np.ndarray, name: str) -> np.ndarray:
    rgba = matplotlib.colormaps[CMAPS[name]](np.clip((array + 1) / 2, 0, 1), bytes=True)
    rgba[..., 3] = np.isfinite(array) * 255
    return rgba


def write_geotiff(path: Path, data: np.ndarray, profile: dict, rgba=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    if rgba:
        data = np.moveaxis(data, -1, 0)
        p.update(count=4, dtype='uint8', nodata=None, photometric='RGB')
    else:
        data = np.where(np.isfinite(data), data, -9999).astype('float32')[None]
        p.update(count=1, dtype='float32', nodata=-9999)
        p.pop('photometric', None)
    p.update(driver='GTiff', compress='deflate', tiled=True)
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(data)
        if rgba:
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
        else:
            dst.set_band_description(1, path.stem)
            dst.update_tags(meaning='normalized difference; no numeric clipping; NoData=-9999')


def xyz_tiles(rgba_tif: Path, target: Path, minzoom=9, maxzoom=12) -> int:
    """Kafelki XYZ 256 px w EPSG:3857. Oś y rośnie na południe (tms=False)."""
    if not 0 <= minzoom <= maxzoom <= 15:
        raise ValueError('W ćwiczeniu dozwolone 0 <= minzoom <= maxzoom <= 15.')
    radius = 20037508.342789244
    count = 0
    with rasterio.open(rgba_tif) as src:
        west, south, east, north = transform_bounds(src.crs, 'EPSG:3857', *src.bounds, densify_pts=21)
        for z in range(minzoom, maxzoom + 1):
            n = 2 ** z
            span = 2 * radius / n
            x0, x1 = max(0, math.floor((west + radius) / span)), min(n-1, math.floor((east + radius) / span))
            y0, y1 = max(0, math.floor((radius - north) / span)), min(n-1, math.floor((radius - south) / span))
            for x in range(x0, x1 + 1):
                folder = target / str(z) / str(x)
                folder.mkdir(parents=True, exist_ok=True)
                for y in range(y0, y1 + 1):
                    dst = np.zeros((4, 256, 256), dtype='uint8')
                    bounds = (x*span-radius, radius-(y+1)*span, (x+1)*span-radius, radius-y*span)
                    reproject(source=rasterio.band(src, [1,2,3,4]), destination=dst,
                              src_transform=src.transform, src_crs=src.crs,
                              dst_transform=from_bounds(*bounds,256,256), dst_crs='EPSG:3857',
                              resampling=Resampling.nearest, src_alpha=4, dst_alpha=4)
                    Image.fromarray(np.moveaxis(dst,0,-1)).save(folder / f'{y}.png')
                    count += 1
    return count


def make_figures(scene: dict, indices: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':12,'axes.titlesize':15,'figure.facecolor':'white'})
    extent = array_bounds(scene['profile']['height'], scene['profile']['width'], scene['profile']['transform'])
    west,south,east,north=extent
    extent_km=[west/1000,east/1000,south/1000,north/1000]
    def plot(a, title, name, cmap=None, limits=None, label=None):
        fig,ax=plt.subplots(figsize=(8.8,6.2),layout='constrained')
        im=ax.imshow(a,extent=extent_km,cmap=cmap,**({'vmin':limits[0],'vmax':limits[1]} if limits else {}))
        ax.set(title=title,xlabel='Easting [km], EPSG:32634',ylabel='Northing [km]')
        if cmap: fig.colorbar(im,ax=ax,shrink=.8,label=label or '')
        fig.savefig(out/name,dpi=155)
        plt.close(fig)
    plot(stretch_rgb(scene['reflectance']), 'Landsat 8 • RGB (4/3/2) • 2013-08-07','rgb.png')
    a=scene['raw']['sr_band5'].astype(float)
    a[a == scene['metadata']['bands']['sr_band5']['fill']]=np.nan
    plot(a,'Kanał B5 przed maskowaniem','band5_dn.png','gray',(0,6000),'DN')
    plot(scene['reflectance'][5],'B5: reflektancja po masce jakości i granicy','band5_masked.png','gray',(0,.6),'Reflektancja')
    quality=np.where(~scene['inside'],np.nan,np.where(scene['valid'],1,0))
    plot(quality,'Maska: 1 = ważny piksel, 0 = odrzucony','quality.png','RdYlGn',(0,1),'Ważność')
    for name,a in indices.items():
        plot(a,f'{name} • 2013-08-07',name.lower()+'.png',CMAPS[name],(-1,1),'Wartość wskaźnika')
    fig,axs=plt.subplots(1,3,figsize=(12.5,3.8),layout='constrained')
    for ax,(name,a) in zip(axs,indices.items()):
        ax.hist(a[np.isfinite(a)],bins=60,range=(-1,1),color='#167f83')
        ax.set(title=name,xlabel='Wartość',ylabel='Liczba pikseli')
    fig.savefig(out/'histograms.png',dpi=155); plt.close(fig)


def export_results(scene: dict, indices: dict, output_dir: Path = ROOT/'outputs', minzoom=9, maxzoom=12) -> dict:
    start=time.perf_counter()
    output_dir=Path(output_dir)
    raster_dir,figure_dir,site=output_dir/'rasters',output_dir/'figures',output_dir/'site'
    for p in [raster_dir,figure_dir,site]: p.mkdir(parents=True,exist_ok=True)
    summary={'product':scene['metadata']['product'],'date':scene['metadata']['date'],
             'shape':[scene['profile']['height'],scene['profile']['width']],
             'crs':str(scene['profile']['crs']),'inside_pixels':int(scene['inside'].sum()),
             'valid_pixels':int(scene['valid'].sum()),'indices':{},'tiles':{},
             'minzoom':minzoom,'maxzoom':maxzoom}
    for name,a in indices.items():
        finite=a[np.isfinite(a)]
        summary['indices'][name]={'count':int(finite.size),'min':float(finite.min()),'max':float(finite.max()),
            'mean':float(finite.mean()),'median':float(np.median(finite)),
            'outside_minus1_plus1':int((np.abs(finite)>1).sum())}
        write_geotiff(raster_dir/f'{name}.tif',a,scene['profile'])
    layers={'RGB':stretch_rgb(scene['reflectance']),**{name:index_rgba(a,name) for name,a in indices.items()}}
    # Atlas (Kamila): dodatkowe warstwy mapy
    layers.update({name:stretch_bands(scene['reflectance'],bands) for name,bands in COMPOSITES.items()})
    cls=classify(indices)
    tr=scene['profile']['transform']
    summary['classes']=class_stats(cls,abs(tr.a*tr.e))
    layers['KLASY']=class_rgba(cls)
    deficit=green_deficit(indices)
    finite=deficit[np.isfinite(deficit)]
    summary['deficit']={'mean':float(finite.mean()),'median':float(np.median(finite)),
                        'share_above_0':round(100*float((finite>0).mean()),2)}
    write_geotiff(raster_dir/'DEFICYT.tif',deficit,scene['profile'])
    layers['DEFICYT']=index_rgba(deficit,'DEFICYT')
    grid=grid_geojson(cls,indices,scene['profile'])
    (site/'siatka.geojson').write_text(json.dumps(grid,separators=(',',':')),encoding='utf-8')
    summary['grid_cells']=len(grid['features'])
    probe=analysis_png(cls,indices,scene['profile'],site/'analiza.png')
    for name,rgba in layers.items():
        p=raster_dir/f'{name}_display.tif'
        write_geotiff(p,rgba,scene['profile'],rgba=True)
        # Zakres zoomów jest zapisany w mapie; istniejące pliki innych zoomów są ignorowane.
        summary['tiles'][name]=xyz_tiles(p,site/'tiles'/name,minzoom,maxzoom)
    make_figures(scene,indices,figure_dir)
    shutil.copytree(Path(__file__).parent/'vendor',site/'vendor',dirs_exist_ok=True)
    (site/'aoi.geojson').write_text(json.dumps(scene['aoi']),encoding='utf-8')
    bounds=array_bounds(scene['profile']['height'],scene['profile']['width'],scene['profile']['transform'])
    w,s,e,n=transform_bounds(scene['profile']['crs'],'EPSG:4326',*bounds)
    config={'bounds':[[s,w],[n,e]],'minzoom':minzoom,'maxzoom':maxzoom,'date':summary['date'],
            'ramps':{name:['#'+''.join(f'{v:02x}' for v in matplotlib.colormaps[CMAPS[name]](float(t),bytes=True)[:3]) for t in np.linspace(0,1,9)] for name in [*indices,'DEFICYT']},
            'classes':[{k:c[k] for k in ('id','name','color','rule')} for c in CLASSES],
            'probe':probe,'tour':tour_points(cls,indices,scene['profile'])}
    template=(Path(__file__).parent/'map_template.html').read_text(encoding='utf-8')
    (site/'index.html').write_text(template.replace('__MAP_CONFIG__',json.dumps(config)),encoding='utf-8')
    summary['export_seconds']=round(time.perf_counter()-start,2)
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (site/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--minzoom',type=int,default=9)
    parser.add_argument('--maxzoom',type=int,default=12)
    args=parser.parse_args()
    scene=load_scene(args.data)
    result=export_results(scene,calculate_indices(scene['reflectance']),args.output,args.minzoom,args.maxzoom)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('Podgląd: python -m http.server 8000 --bind 127.0.0.1 --directory '+str(args.output/'site'))

if __name__=='__main__': main()
