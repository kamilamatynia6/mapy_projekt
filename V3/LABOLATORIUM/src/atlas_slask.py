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
