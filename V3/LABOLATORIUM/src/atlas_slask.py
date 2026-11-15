"""Atlas środowiskowy Górnego Śląska (Kamila Matynia).

Dwie sceny Landsat 8 Collection 2 Level-2 (20.08.2015 i 12.08.2024, ścieżka 189/025):
wskaźniki NDVI/NDBI/MNDWI, temperatura powierzchni z kanału termalnego B10,
klasy pokrycia terenu, zmiany 2015 → 2024, siatka 1 km, dane do analiz w przeglądarce
i automatycznie wyszukane „problemy regionu”. Wynik: gotowa strona w outputs/site.

Najpierw pobierz dane:   python tools/pobierz_slask.py
Potem zbuduj atlas:      python src/atlas_slask.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom

from landsat_lab import normalized_difference

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'slask'
YEARS = ('2015', '2024')

# Collection 2 Level-2: współczynniki z dokumentacji USGS (inne niż w Collection 1!)
SR_SCALE, SR_OFFSET = 0.0000275, -0.2            # reflektancja = DN * skala + offset
ST_SCALE, ST_OFFSET = 0.00341802, 149.0          # temperatura [K] = DN * skala + offset
# QA_PIXEL C2: bit 0 wypełnienie, 1 rozszerzona chmura, 2 cirrus, 3 chmura, 4 cień, 5 śnieg
QA_BAD_BITS = 0b111111
# QA_RADSAT: nasycenie kanałów B2–B7 (bity 1–6)
RADSAT_BITS = 0b1111110


def load_year(year: str, data_dir: Path = DATA) -> dict:
    """Wczytuje jeden rok: reflektancję B2–B7, temperaturę powierzchni [°C] i maskę ważnych pikseli."""
    folder = Path(data_dir) / year
    raw, profile = {}, None
    for band in ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL', 'QA_RADSAT']:
        with rasterio.open(folder / f'{band}.tif') as src:
            if profile is None:
                profile = src.profile.copy()
            elif src.transform != profile['transform'] or src.shape != (profile['height'], profile['width']):
                raise ValueError(f'Niezgodna siatka rastra: {year}/{band}')
            raw[band] = src.read(1)
    catalog = json.loads((Path(data_dir) / 'sceny.json').read_text(encoding='utf-8'))
    aoi = catalog['aoi']
    ring = [[aoi['west'], aoi['south']], [aoi['east'], aoi['south']], [aoi['east'], aoi['north']],
            [aoi['west'], aoi['north']], [aoi['west'], aoi['south']]]
    aoi_geojson = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {'name': 'Obszar atlasu'},
                   'geometry': {'type': 'Polygon', 'coordinates': [ring]}}]}
    inside = geometry_mask([transform_geom('EPSG:4326', profile['crs'], aoi_geojson['features'][0]['geometry'])],
                           out_shape=raw['QA_PIXEL'].shape, transform=profile['transform'], invert=True)
    bad = ((raw['QA_PIXEL'] & QA_BAD_BITS) != 0) | ((raw['QA_RADSAT'] & RADSAT_BITS) != 0)
    valid = inside & ~bad
    for i in range(2, 8):
        valid &= raw[f'SR_B{i}'] != 0                     # 0 = brak danych
    valid &= raw['ST_B10'] != 0
    reflectance = {i: np.where(valid, raw[f'SR_B{i}'] * SR_SCALE + SR_OFFSET, np.nan).astype('float32') for i in range(2, 8)}
    lst = np.where(valid, raw['ST_B10'] * ST_SCALE + ST_OFFSET - 273.15, np.nan).astype('float32')
    return {'year': year, 'scene': catalog['scenes'][year], 'profile': profile, 'aoi': aoi_geojson,
            'inside': inside, 'valid': valid, 'reflectance': reflectance, 'lst': lst}


def indices(scene: dict) -> dict:
    r = scene['reflectance']
    return {'NDVI': normalized_difference(r[5], r[4]),
            'NDBI': normalized_difference(r[6], r[5]),
            'MNDWI': normalized_difference(r[3], r[6]),
            'TEMP': scene['lst']}


# Klasy pokrycia terenu – progi wskaźników; woda ma pierwszeństwo
CLASSES = [
    {'id': 1, 'name': 'Woda', 'color': '#2c7fb8', 'rule': 'MNDWI > 0'},
    {'id': 2, 'name': 'Gęsta roślinność (lasy, parki, bujne uprawy)', 'color': '#1a7f37', 'rule': 'NDVI ≥ 0.7'},
    {'id': 3, 'name': 'Roślinność niska (łąki, pola, trawniki)', 'color': '#a6d96a', 'rule': '0.4 ≤ NDVI < 0.7'},
    {'id': 4, 'name': 'Zabudowa, przemysł i odkryty grunt', 'color': '#c0504d', 'rule': 'NDVI < 0.4'},
]


def classify(ix: dict) -> np.ndarray:
    """Numer klasy dla każdego piksela; 0 = brak ważnych danych."""
    ndvi, mndwi = ix['NDVI'], ix['MNDWI']
    ok = np.isfinite(ndvi) & np.isfinite(mndwi)
    cls = np.zeros(ndvi.shape, dtype='uint8')
    cls[ok] = 4
    cls[ok & (ndvi >= 0.4)] = 3
    cls[ok & (ndvi >= 0.7)] = 2
    cls[ok & (mndwi > 0)] = 1
    return cls


def temperature_anomaly(lst: np.ndarray) -> np.ndarray:
    """Temperatura względem mediany całej sceny. Dwa dni mają inną pogodę,
    więc lata porównujemy anomaliami (o ile cieplej niż typowe miejsce), a nie surowymi °C."""
    return (lst - np.nanmedian(lst)).astype('float32')


# ---------------------------------------------------------------------------
# Warstwy mapy: definicje, kolory i kafelki dla obu lat
# ---------------------------------------------------------------------------
import shutil
import time
import matplotlib
matplotlib.use('Agg')
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from landsat_lab import write_geotiff, xyz_tiles

LAYERS = {
    'RGB': {'label': 'Zdjęcie – naturalne barwy', 'kind': 'composite', 'bands': (4, 3, 2),
            'description': 'Kanały B4/B3/B2, tak jak widzi oko. Ten sam kontrast w obu latach, więc kolory można porównywać.'},
    'PODCZERWIEN': {'label': 'Podczerwień (B5/B4/B3)', 'kind': 'composite', 'bands': (5, 4, 3),
                    'description': 'Fałszywe barwy: zdrowa roślinność jest czerwona, zabudowa i przemysł szaroniebieskie, woda prawie czarna.'},
    'TEMP': {'label': 'Temperatura powierzchni', 'kind': 'scalar', 'cmap': 'RdYlBu_r', 'min': 18, 'max': 44, 'unit': '°C',
             'description': 'Temperatura powierzchni z kanału termalnego B10 (nie temperatura powietrza). Czerwone plamy to miejskie wyspy ciepła.'},
    'NDVI': {'label': 'NDVI – roślinność', 'kind': 'scalar', 'cmap': 'RdYlGn', 'min': -0.1, 'max': 0.9, 'unit': '',
             'description': 'NDVI = (B5 − B4) / (B5 + B4). Im bardziej zielono, tym więcej zdrowej roślinności.'},
    'NDBI': {'label': 'NDBI – zabudowa', 'kind': 'scalar', 'cmap': 'PuOr_r', 'min': -0.5, 'max': 0.3, 'unit': '',
             'description': 'NDBI = (B6 − B5) / (B6 + B5). Dodatnie wartości wskazują zabudowę, przemysł lub odsłonięty grunt.'},
    'MNDWI': {'label': 'MNDWI – woda', 'kind': 'scalar', 'cmap': 'BrBG', 'min': -0.7, 'max': 0.5, 'unit': '',
              'description': 'MNDWI = (B3 − B6) / (B3 + B6). Dodatnie wartości wskazują wodę – jeziora po wydobyciu piasku i zbiorniki.'},
    'KLASY': {'label': 'Pokrycie terenu – klasy', 'kind': 'classes',
              'description': 'Klasyfikacja progowa: woda (MNDWI > 0), gęsta roślinność (NDVI ≥ 0.7), roślinność niska (0.4–0.7), zabudowa, przemysł i odkryty grunt (NDVI < 0.4).'},
}
CHANGE_LAYERS = {
    'ZMIANA_NDVI': {'label': 'Zmiana zieleni 2015 → 2024', 'kind': 'scalar', 'cmap': 'BrBG', 'min': -0.4, 'max': 0.4, 'unit': 'Δ NDVI',
                    'description': 'NDVI 2024 minus NDVI 2015. Brąz = ubyło zieleni, zieleń = przybyło. Na polach duże zmiany wynikają z terminu żniw i suszy w 2015 r.'},
    'ZMIANA_TEMP': {'label': 'Zmiana przegrzania 2015 → 2024', 'kind': 'scalar', 'cmap': 'RdBu_r', 'min': -8, 'max': 8, 'unit': 'Δ °C',
                    'description': 'Zmiana anomalii temperatury (miejsce względem mediany całej sceny). Czerwone: miejsce w 2024 r. jest bardziej przegrzane na tle regionu niż w 2015 r.'},
    'PRZEJSCIA': {'label': 'Przejścia klas 2015 → 2024', 'kind': 'transitions',
                  'description': 'Gdzie zieleń zamieniła się w zabudowę lub grunt (czerwone), gdzie zarosła (zielone) i gdzie pojawiła się lub zniknęła woda.'},
}
TRANSITIONS = [
    {'id': 1, 'name': 'Ubytek zieleni', 'color': '#d7301f'},
    {'id': 2, 'name': 'Przybytek zieleni', 'color': '#1a9850'},
    {'id': 3, 'name': 'Nowa woda', 'color': '#2171b5'},
    {'id': 4, 'name': 'Zanik wody', 'color': '#fdae61'},
]


def hex_rgb(color: str) -> list:
    return [int(color[i:i+2], 16) for i in (1, 3, 5)]


def ramp(cmap: str, n=9) -> list:
    return ['#' + ''.join(f'{v:02x}' for v in matplotlib.colormaps[cmap](float(t), bytes=True)[:3]) for t in np.linspace(0, 1, n)]


def scalar_rgba(a: np.ndarray, spec: dict) -> np.ndarray:
    t = np.clip((a - spec['min']) / (spec['max'] - spec['min']), 0, 1)
    rgba = matplotlib.colormaps[spec['cmap']](np.nan_to_num(t), bytes=True)
    rgba[..., 3] = np.isfinite(a) * 255
    return rgba


def categorical_rgba(codes: np.ndarray, items: list) -> np.ndarray:
    rgba = np.zeros((*codes.shape, 4), dtype='uint8')
    for item in items:
        rgba[codes == item['id']] = [*hex_rgb(item['color']), 255]
    return rgba


def composite_limits(scenes: dict) -> dict:
    """Wspólne granice kontrastu (percentyle 2–98) dla obu lat – inaczej porównanie kolorów byłoby mylące."""
    limits = {}
    for b in range(2, 8):
        values = np.concatenate([s['reflectance'][b][s['valid']][::7] for s in scenes.values()])
        limits[b] = np.percentile(values, (2, 98))
    return limits


def composite_rgba(scene: dict, bands: tuple, limits: dict) -> np.ndarray:
    channels = []
    for b in bands:
        lo, hi = limits[b]
        channels.append((np.nan_to_num(np.clip((scene['reflectance'][b] - lo) / (hi - lo), 0, 1)) * 255).astype('uint8'))
    return np.dstack([*channels, (scene['valid'] * 255).astype('uint8')])


def transitions(c15: np.ndarray, c24: np.ndarray) -> np.ndarray:
    green15, green24 = np.isin(c15, (2, 3)), np.isin(c24, (2, 3))
    out = np.zeros(c15.shape, dtype='uint8')
    out[green15 & (c24 == 4)] = 1
    out[(c15 == 4) & green24] = 2
    out[(c15 != 1) & (c15 > 0) & (c24 == 1)] = 3
    out[(c15 == 1) & (c24 != 1) & (c24 > 0)] = 4
    return out


def class_stats(cls: np.ndarray, pixel_km2: float) -> list:
    total = max(int((cls > 0).sum()), 1)
    return [{**c, 'percent': round(100 * float((cls == c['id']).sum()) / total, 2),
             'km2': round(float((cls == c['id']).sum()) * pixel_km2, 1)} for c in CLASSES]


def compress_tiles(folder: Path) -> int:
    """Zapisuje kafelki jako PNG z paletą 256 kolorów: ok. 4× mniej danych do wysłania na serwer,
    średni błąd koloru 2–4 na 255 (niewidoczny). Zwraca łączny rozmiar w bajtach."""
    from PIL import Image
    total = 0
    for path in Path(folder).rglob('*.png'):
        Image.open(path).convert('RGBA').quantize(256, method=Image.Quantize.FASTOCTREE).save(path, optimize=True)
        total += path.stat().st_size
    return total


def build_site(output_dir: Path = ROOT / 'outputs', minzoom=9, maxzoom=12) -> dict:
    start = time.perf_counter()
    output_dir = Path(output_dir)
    site, rasters = output_dir / 'site', output_dir / 'rasters'
    for p in (site, rasters):
        p.mkdir(parents=True, exist_ok=True)
    scenes = {y: load_year(y) for y in YEARS}
    ix = {y: indices(scenes[y]) for y in YEARS}
    cls = {y: classify(ix[y]) for y in YEARS}
    profile = scenes[YEARS[0]]['profile']
    pixel_km2 = abs(profile['transform'].a * profile['transform'].e) / 1e6
    limits = composite_limits(scenes)

    summary = {'title': 'Atlas środowiskowy Górnego Śląska', 'years': {}, 'tiles': {}, 'minzoom': minzoom, 'maxzoom': maxzoom}
    layers = {}
    for y in YEARS:
        s = scenes[y]
        summary['years'][y] = {'scene': s['scene'], 'valid_pixels': int(s['valid'].sum()),
                               'classes': class_stats(cls[y], pixel_km2),
                               'indices': {name: {'mean': round(float(np.nanmean(a)), 3), 'median': round(float(np.nanmedian(a)), 3),
                                                  'p2': round(float(np.nanpercentile(a, 2)), 2), 'p98': round(float(np.nanpercentile(a, 98)), 2)}
                                           for name, a in ix[y].items()}}
        for name, spec in LAYERS.items():
            if spec['kind'] == 'composite':
                layers[(y, name)] = composite_rgba(s, spec['bands'], limits)
            elif spec['kind'] == 'scalar':
                layers[(y, name)] = scalar_rgba(ix[y][name], spec)
            else:
                layers[(y, name)] = categorical_rgba(cls[y], CLASSES)
        for name in ('NDVI', 'TEMP'):
            write_geotiff(rasters / f'{name}_{y}.tif', ix[y][name], profile)

    d_ndvi = (ix['2024']['NDVI'] - ix['2015']['NDVI']).astype('float32')
    d_temp = (temperature_anomaly(ix['2024']['TEMP']) - temperature_anomaly(ix['2015']['TEMP'])).astype('float32')
    trans = transitions(cls['2015'], cls['2024'])
    write_geotiff(rasters / 'ZMIANA_NDVI.tif', d_ndvi, profile)
    layers[('zmiany', 'ZMIANA_NDVI')] = scalar_rgba(d_ndvi, CHANGE_LAYERS['ZMIANA_NDVI'])
    layers[('zmiany', 'ZMIANA_TEMP')] = scalar_rgba(d_temp, CHANGE_LAYERS['ZMIANA_TEMP'])
    layers[('zmiany', 'PRZEJSCIA')] = categorical_rgba(trans, TRANSITIONS)
    summary['transitions'] = [{**t, 'km2': round(float((trans == t['id']).sum()) * pixel_km2, 1)} for t in TRANSITIONS]

    for (folder, name), rgba in layers.items():
        tif = rasters / f'{name}_{folder}_display.tif'
        write_geotiff(tif, rgba, profile, rgba=True)
        summary['tiles'][f'{folder}/{name}'] = xyz_tiles(tif, site / 'tiles' / folder / name, minzoom, maxzoom)
    summary['tiles_mb'] = round(compress_tiles(site / 'tiles') / 1e6, 1)

    shutil.copytree(Path(__file__).parent / 'vendor', site / 'vendor', dirs_exist_ok=True)
    (site / 'aoi.geojson').write_text(json.dumps(scenes[YEARS[0]]['aoi']), encoding='utf-8')
    w, s_, e, n = transform_bounds(profile['crs'], 'EPSG:4326', *array_bounds(profile['height'], profile['width'], profile['transform']))
    layer_meta = {}
    for group, specs in (('rok', LAYERS), ('zmiany', CHANGE_LAYERS)):
        for name, spec in specs.items():
            meta = {k: spec[k] for k in ('label', 'kind', 'description')}
            meta['group'] = group
            if spec['kind'] == 'scalar':
                meta.update(min=spec['min'], max=spec['max'], unit=spec['unit'], ramp=ramp(spec['cmap']))
            layer_meta[name] = meta
    config = {'bounds': [[s_, w], [n, e]], 'minzoom': minzoom, 'maxzoom': maxzoom,
              'years': {y: scenes[y]['scene'] for y in YEARS}, 'layers': layer_meta,
              'classes': [{k: c[k] for k in ('id', 'name', 'color', 'rule')} for c in CLASSES],
              'transitions': TRANSITIONS}
    template = (Path(__file__).parent / 'map_template.html').read_text(encoding='utf-8')
    (site / 'index.html').write_text(template.replace('__MAP_CONFIG__', json.dumps(config, ensure_ascii=False)), encoding='utf-8')
    summary['export_seconds'] = round(time.perf_counter() - start, 1)
    for target in (output_dir / 'summary.json', site / 'summary.json'):
        target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    result = build_site()
    print(json.dumps({k: result[k] for k in ('transitions', 'export_seconds')}, ensure_ascii=False, indent=1))
    print('Kafelki:', sum(result['tiles'].values()))
    print('Podgląd: python -m http.server 8000 --bind 127.0.0.1 --directory outputs/site')
