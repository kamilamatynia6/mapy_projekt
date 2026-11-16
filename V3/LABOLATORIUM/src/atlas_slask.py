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


# ---------------------------------------------------------------------------
# Siatka 1 km, dane do analizy w przeglądarce i „problemy regionu”
# ---------------------------------------------------------------------------
import math
import warnings
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform
from rasterio.enums import Resampling

CELL = 33          # 33 piksele × 30 m ≈ 1 km


def pl(x: float, digits=1) -> str:
    """Liczba z polskim przecinkiem dziesiętnym."""
    return f'{x:.{digits}f}'.replace('.', ',').replace('-', '−')


def block_stats(arrays: dict, cls: dict, r0: int, c0: int, size: int = CELL, mask=None) -> dict:
    """Statystyki prostokąta (lub maski w nim) dla obu lat: udział zieleni, zabudowy, wody, NDVI, temperatura."""
    out = {}
    for y in YEARS:
        k = cls[y][r0:r0+size, c0:c0+size]
        sel = (k > 0) if mask is None else ((k > 0) & mask)
        n = int(sel.sum())
        if n == 0:
            return {}
        yy = y[2:]
        out[f'z{yy}'] = round(100 * float(np.isin(k[sel], (2, 3)).mean()), 1)
        out[f'b{yy}'] = round(100 * float((k[sel] == 4).mean()), 1)
        out[f'w{yy}'] = round(100 * float((k[sel] == 1).mean()), 1)
        out[f'n{yy}'] = round(float(np.nanmean(arrays[y]['NDVI'][r0:r0+size, c0:c0+size][sel])), 3)
        out[f't{yy}'] = round(float(np.nanmean(arrays[y]['TEMP'][r0:r0+size, c0:c0+size][sel])), 1)
        out[f'a{yy}'] = round(float(np.nanmean(arrays[y]['ANOM'][r0:r0+size, c0:c0+size][sel])), 1)
        out['n'] = n
    out['dz'] = round(out['z24'] - out['z15'], 1)
    out['dn'] = round(out['n24'] - out['n15'], 3)
    out['da'] = round(out['a24'] - out['a15'], 1)
    return out


def grid_geojson(arrays: dict, cls: dict, profile: dict) -> dict:
    t = profile['transform']
    h, w = cls[YEARS[0]].shape
    features = []
    for r0 in range(0, h, CELL):
        for c0 in range(0, w, CELL):
            if (cls[YEARS[0]][r0:r0+CELL, c0:c0+CELL] > 0).sum() < 0.3 * CELL * CELL:
                continue
            props = block_stats(arrays, cls, r0, c0)
            if not props:
                continue
            props.pop('n')
            x0, y0 = t * (c0, r0)
            x1, y1 = t * (min(c0 + CELL, w), min(r0 + CELL, h))
            ring = transform_geom(profile['crs'], 'EPSG:4326', {'type': 'Polygon',
                                  'coordinates': [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]})
            features.append({'type': 'Feature', 'properties': props, 'geometry': {'type': 'Polygon',
                             'coordinates': [[[round(x, 5), round(y, 5)] for x, y in ring['coordinates'][0]]]}})
    return {'type': 'FeatureCollection', 'features': features}


def probe_png(cls: np.ndarray, ix: dict, profile: dict, path: Path, step_m=60) -> dict:
    """Raster do analiz w przeglądarce, w siatce EPSG:4326 (piksel ok. 60 m).
    Kanały PNG: R = klasa (0 = brak danych), G = (NDVI + 1) · 127,5, B = temperatura · 4 (co 0,25 °C)."""
    from PIL import Image
    h, w = cls.shape
    west, south, east, north = transform_bounds(profile['crs'], 'EPSG:4326', *array_bounds(h, w, profile['transform']))
    width = int(math.ceil((east - west) / (step_m / (111320 * math.cos(math.radians((south + north) / 2))))))
    height = int(math.ceil((north - south) / (step_m / 110540)))
    dst = from_bounds(west, south, east, north, width, height)

    def warp(a, dtype, nodata):
        out = np.full((height, width), nodata, dtype=dtype)
        reproject(a.astype(dtype), out, src_transform=profile['transform'], src_crs=profile['crs'], dst_transform=dst,
                  dst_crs='EPSG:4326', resampling=Resampling.nearest, src_nodata=nodata, dst_nodata=nodata)
        return out

    k = warp(cls, 'uint8', 0)
    ndvi, temp = warp(ix['NDVI'], 'float32', np.nan), warp(ix['TEMP'], 'float32', np.nan)
    g = np.where(np.isfinite(ndvi), np.clip(np.round((ndvi + 1) * 127.5), 0, 255), 0).astype('uint8')
    b = np.where(np.isfinite(temp), np.clip(np.round(temp * 4), 1, 255), 0).astype('uint8')
    Image.fromarray(np.dstack([k, g, b, np.full_like(k, 255)]), 'RGBA').save(path, optimize=True)
    return {'url': path.name, 'bounds': [[south, west], [north, east]], 'width': width, 'height': height}


def to_pixel(profile: dict, lat: float, lon: float):
    xs, ys = transform('EPSG:4326', profile['crs'], [lon], [lat])
    col, row = ~profile['transform'] * (xs[0], ys[0])
    return row, col


def to_latlon(profile: dict, row: float, col: float):
    x, y = profile['transform'] * (col, row)
    lons, lats = transform(profile['crs'], 'EPSG:4326', [x], [y])
    return round(lats[0], 5), round(lons[0], 5)


def circle_stats(arrays, cls, profile, lat, lon, radius_m=500) -> dict:
    row, col = to_pixel(profile, lat, lon)
    r = radius_m / abs(profile['transform'].a)
    r0, c0 = int(row - r), int(col - r)
    size = int(2 * r) + 2
    rr, cc = np.indices((size, size))
    mask = (rr + r0 + .5 - row) ** 2 + (cc + c0 + .5 - col) ** 2 <= r * r
    return block_stats(arrays, cls, r0, c0, size, mask)


def cell_center(profile, r: int, c: int):
    return to_latlon(profile, r * CELL + CELL / 2, c * CELL + CELL / 2)


def stories(arrays: dict, cls: dict, profile: dict, grid: dict) -> list:
    """Przystanki wycieczki „Problemy regionu”. Miejsca stałe mają sprawdzone współrzędne,
    a najgorętsze miejsce, największy ubytek zieleni i nowa woda są wyszukiwane automatycznie."""
    cs = lambda lat, lon, r=500: circle_stats(arrays, cls, profile, lat, lon, r)
    h, w = cls['2024'].shape
    hh, ww = h // CELL, w // CELL

    def blocks(a):
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)       # kwadraty bez danych dają NaN
            return np.nanmean(a[:hh * CELL, :ww * CELL].reshape(hh, CELL, ww, CELL), axis=(1, 3))

    water24 = blocks((cls['2024'] == 1).astype('float32')) * 100
    water15 = blocks((cls['2015'] == 1).astype('float32')) * 100
    temp24 = blocks(arrays['2024']['TEMP'])
    dndvi = blocks(arrays['2024']['NDVI']) - blocks(arrays['2015']['NDVI'])
    out = []

    # 1. Wyspa ciepła Katowic
    kat, las = cs(50.2589, 19.0235, 1000), cs(50.205, 19.06, 1000)
    out.append({'title': 'Wyspa ciepła Katowic', 'lat': 50.2589, 'lon': 19.0235, 'zoom': 12, 'year': '2024', 'layer': 'TEMP', 'radius': 1000,
                'text': f"12 sierpnia 2024 r. powierzchnia w centrum Katowic nagrzała się średnio do {pl(kat['t24'])} °C, "
                        f"a w Lesie Murckowskim, ok. 6 km dalej, tylko do {pl(las['t24'])} °C. Różnica: {pl(kat['t24'] - las['t24'])} °C. "
                        "Asfalt, dachy i brak drzew zamieniają śródmieście w wyspę ciepła.",
                'facts': [['Centrum Katowic', f"{pl(kat['t24'])} °C"], ['Las Murckowski', f"{pl(las['t24'])} °C"],
                          ['Zieleń w centrum', f"{pl(kat['z24'], 0)}%"]]})
    # 2. Park Śląski chłodzi
    park, chorzow = cs(50.29, 18.99), cs(50.2975, 18.9545)
    out.append({'title': 'Park Śląski – zielona klimatyzacja', 'lat': 50.29, 'lon': 18.99, 'zoom': 13, 'year': '2024', 'radius': 500,
                'compare': {'mode': 'warstwy', 'left': 'PODCZERWIEN', 'right': 'TEMP'},
                'text': f"Jeden z największych parków miejskich w Europie. W jego wnętrzu powierzchnia miała {pl(park['t24'])} °C, "
                        f"a w centrum pobliskiego Chorzowa {pl(chorzow['t24'])} °C. Przesuń suwak: tam, gdzie podczerwień pokazuje "
                        "roślinność (czerwień), mapa temperatury robi się niebieska.",
                'facts': [['Park Śląski', f"{pl(park['t24'])} °C"], ['Centrum Chorzowa', f"{pl(chorzow['t24'])} °C"],
                          ['NDVI w parku', pl(park['n24'], 2)]]})
    # 3. Najgorętsze miejsce regionu (automatycznie)
    r, c = np.unravel_index(np.nanargmax(np.where(water24 < 5, temp24, np.nan)), temp24.shape)
    lat, lon = cell_center(profile, r, c)
    hot = block_stats(arrays, cls, r * CELL, c * CELL)
    desert = abs(lat - 50.36) < 0.04 and abs(lon - 19.52) < 0.06
    out.append({'title': 'Pustynia Błędowska – najgorętszy kilometr' if desert else 'Najgorętszy kilometr regionu',
                'lat': lat, 'lon': lon, 'zoom': 13, 'year': '2024', 'layer': 'TEMP', 'radius': 500,
                'compare': {'mode': 'warstwy', 'left': 'RGB', 'right': 'TEMP'},
                'text': (f"Program sam przeszukał {len(grid['features'])} kwadratów 1 km i znalazł najgorętszy: {pl(hot['t24'])} °C. "
                         + ("To Pustynia Błędowska – odsłonięty piasek nie paruje i nagrzewa się mocniej niż beton. "
                            if desert else "")
                         + ("Zieleni praktycznie tu nie ma." if hot['z24'] < 1 else f"Zieleń zajmuje tu tylko {pl(hot['z24'], 0)}% powierzchni.")),
                'facts': [['Temperatura', f"{pl(hot['t24'])} °C"], ['NDVI', pl(hot['n24'], 2)], ['Zieleń', f"{pl(hot['z24'], 0)}%"]]})
    # 4. Jeziora po wydobyciu piasku
    pog = cs(50.36, 19.20, 1500)
    out.append({'title': 'Pogorie – jeziora po kopalniach piasku', 'lat': 50.36, 'lon': 19.20, 'zoom': 12, 'year': '2024', 'layer': 'MNDWI', 'radius': 1500,
                'text': f"Jeziora Pogoria w Dąbrowie Górniczej powstały w zalanych wyrobiskach po wydobyciu piasku. "
                        f"W promieniu 1,5 km woda zajmuje {pl(pog['w24'], 0)}% powierzchni, a temperatura wynosi średnio "
                        f"{pl(pog['t24'])} °C – woda działa jak naturalny chłodnik okolicy.",
                'facts': [['Woda', f"{pl(pog['w24'], 0)}%"], ['Temperatura', f"{pl(pog['t24'])} °C"], ['Zieleń', f"{pl(pog['z24'], 0)}%"]]})
    # 5. Największy ubytek zieleni (automatycznie, bez kwadratów z wodą)
    land = (water24 < 5) & (water15 < 5)
    r, c = np.unravel_index(np.nanargmin(np.where(land, dndvi, np.nan)), dndvi.shape)
    lat, lon = cell_center(profile, r, c)
    loss = block_stats(arrays, cls, r * CELL, c * CELL)
    on_desert = abs(lat - 50.36) < 0.04 and abs(lon - 19.52) < 0.06
    out.append({'title': 'Największy ubytek zieleni', 'lat': lat, 'lon': lon, 'zoom': 13, 'year': '2024', 'radius': 500,
                'compare': {'mode': 'lata', 'layer': 'RGB'},
                'text': f"W tym kwadracie NDVI spadło z {pl(loss['n15'], 2)} do {pl(loss['n24'], 2)}, a udział zieleni "
                        f"z {pl(loss['z15'], 0)}% do {pl(loss['z24'], 0)}%. "
                        + ("Kwadrat leży na Pustyni Błędowskiej – ubytek zieleni oznacza tu powrót odkrytego piasku. " if on_desert else "")
                        + "Przesuń suwak 2015 | 2024 na zdjęciu i oceń, co się zmieniło – satelita pokazuje skutek, przyczynę trzeba sprawdzić w terenie.",
                'facts': [['Zieleń 2015', f"{pl(loss['z15'], 0)}%"], ['Zieleń 2024', f"{pl(loss['z24'], 0)}%"], ['Δ NDVI', pl(loss['dn'], 2)]]})
    # 6. Nowa woda (automatycznie)
    r, c = np.unravel_index(np.nanargmax(water24 - water15), water24.shape)
    lat, lon = cell_center(profile, r, c)
    wet = block_stats(arrays, cls, r * CELL, c * CELL)
    out.append({'title': 'Tu pojawiła się woda', 'lat': lat, 'lon': lon, 'zoom': 13, 'year': '2024', 'radius': 500,
                'compare': {'mode': 'lata', 'layer': 'MNDWI'},
                'text': f"Udział wody w tym kwadracie wzrósł z {pl(wet['w15'], 0)}% do {pl(wet['w24'], 0)}%. Na Śląsku nowe zbiorniki "
                        "powstają m.in. w zalanych wyrobiskach i zapadliskach – suwak lat pokazuje zmianę.",
                'facts': [['Woda 2015', f"{pl(wet['w15'], 0)}%"], ['Woda 2024', f"{pl(wet['w24'], 0)}%"]]})
    # 7. Susza 2015 na polach
    r, c = np.unravel_index(np.nanargmax(np.where(land, dndvi, np.nan)), dndvi.shape)
    lat, lon = cell_center(profile, r, c)
    gain = block_stats(arrays, cls, r * CELL, c * CELL)
    out.append({'title': 'Pola: susza 2015 kontra 2024', 'lat': lat, 'lon': lon, 'zoom': 12, 'year': '2024', 'radius': 500,
                'compare': {'mode': 'lata', 'layer': 'NDVI'},
                'text': f"Największy „przyrost zieleni” to pola uprawne: NDVI z {pl(gain['n15'], 2)} do {pl(gain['n24'], 2)}. "
                        "Sierpień 2015 r. przypadł na wielką suszę, a stan pól zależy też od terminu żniw – nie każda zmiana "
                        "na mapie to trwała zmiana krajobrazu.",
                'facts': [['NDVI 2015', pl(gain['n15'], 2)], ['NDVI 2024', pl(gain['n24'], 2)]]})
    return out


def correlation(grid: dict) -> dict:
    z = np.array([f['properties']['z24'] for f in grid['features']])
    t = np.array([f['properties']['t24'] for f in grid['features']])
    slope, intercept = np.polyfit(z, t, 1)
    return {'r': round(float(np.corrcoef(z, t)[0, 1]), 2), 'slope_per_10pp': round(float(slope * 10), 2),
            'intercept': round(float(intercept), 2), 'cells': int(z.size)}


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

    arrays = {y: {'NDVI': ix[y]['NDVI'], 'TEMP': ix[y]['TEMP'], 'ANOM': temperature_anomaly(ix[y]['TEMP'])} for y in YEARS}
    grid = grid_geojson(arrays, cls, profile)
    (site / 'siatka.geojson').write_text(json.dumps(grid, separators=(',', ':')), encoding='utf-8')
    summary['grid_cells'] = len(grid['features'])
    summary['correlation'] = correlation(grid)
    probes = {y: probe_png(cls[y], ix[y], profile, site / f'analiza_{y}.png') for y in YEARS}
    tour = stories(arrays, cls, profile, grid)

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
              'transitions': TRANSITIONS, 'probe': probes, 'stories': tour, 'correlation': summary['correlation']}
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
