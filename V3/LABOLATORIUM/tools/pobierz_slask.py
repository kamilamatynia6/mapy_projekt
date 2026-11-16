"""Atlas Górnego Śląska (Kamila): pobiera wycinki scen Landsat 8 Collection 2 Level-2
z publicznego archiwum Microsoft Planetary Computer (bez logowania).

Uruchom z katalogu LABOLATORIUM (wymaga internetu, ok. 60 MB):
    python tools/pobierz_slask.py

Wynik: data/slask/<data>/<kanał>.tif oraz data/slask/sceny.json.
Pliki .tif nie trafiają do gita (.gitignore), więc każda osoba pobiera je u siebie.
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'slask'
STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2/items/'
TOKEN = 'https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2'

# Obszar: Górnośląski Okręg Przemysłowy z Zagłębiem, Pustynią Błędowską i lotniskiem w Pyrzowicach
AOI = {'west': 18.65, 'south': 50.12, 'east': 19.62, 'north': 50.48}
# Ta sama ścieżka/rząd (189/025), ten sam satelita i ta sama pora roku – porównanie jest uczciwe
SCENES = {
    '2015': 'LC08_L2SP_189025_20150820_02_T1',
    '2024': 'LC08_L2SP_189025_20240812_02_T1',
}
# Kanały: B2–B7 (reflektancja), B10 (temperatura powierzchni), maski jakości
ASSETS = {'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4', 'nir08': 'SR_B5',
          'swir16': 'SR_B6', 'swir22': 'SR_B7', 'lwir11': 'ST_B10',
          'qa_pixel': 'QA_PIXEL', 'qa_radsat': 'QA_RADSAT'}


def get_json(url, attempts=6):
    """Pobiera JSON; przy limicie zapytań (HTTP 429) czeka i próbuje ponownie."""
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == attempts - 1:
                raise
            time.sleep(10 * (i + 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = {'aoi': AOI, 'source': 'Microsoft Planetary Computer, landsat-c2-l2 (USGS)', 'scenes': {}}
    token = get_json(TOKEN)['token']          # jeden token dostępu do całej kolekcji
    for year, scene_id in SCENES.items():
        item = get_json(STAC + scene_id)
        folder = OUT / year
        folder.mkdir(exist_ok=True)
        for asset, band in ASSETS.items():
            if (folder / f'{band}.tif').exists():
                print(year, band, 'już pobrany')
                continue
            href = item['assets'][asset]['href'] + '?' + token
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
                with rasterio.open(href) as src:
                    bounds = transform_bounds('EPSG:4326', src.crs, AOI['west'], AOI['south'], AOI['east'], AOI['north'])
                    window = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
                    data = src.read(1, window=window)
                    profile = src.profile.copy()
                    profile.update(driver='GTiff', width=data.shape[1], height=data.shape[0],
                                   transform=src.window_transform(window), compress='deflate', tiled=True,
                                   blockxsize=256, blockysize=256)
            with rasterio.open(folder / f'{band}.tif', 'w', **profile) as dst:
                dst.write(data, 1)
            print(year, band, data.shape)
        props = item['properties']
        catalog['scenes'][year] = {'id': scene_id, 'date': props['datetime'][:10],
                                   'platform': props['platform'], 'cloud_cover_scene': props['eo:cloud_cover'],
                                   'sun_elevation': props.get('view:sun_elevation')}
    (OUT / 'sceny.json').write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Gotowe:', OUT)


if __name__ == '__main__':
    main()
