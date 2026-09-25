"""
Cobertura de copa de árboles sobre la red peatonal.

Fuente: High Resolution Canopy Height Maps (Meta & WRI; Tolan et al. 2024),
~1,2 m, EPSG:3857, teselas quadkey nivel 9 descargadas de
s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/.

1. `build_canopy_fraction`: raster en EPSG:32719 a `res` m con la fracción de
   superficie con copa >= `min_height_m`, suavizada con una ventana de
   `focal_px` celdas para capturar árboles que dan sombra sobre la vereda.
2. `add_edge_canopy`: fracción media de copa a lo largo de cada arista de la red.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def build_canopy_fraction(tile_fps, bounds_32719, out_fp, res: float = 3.0,
                          min_height_m: float = 3.0, focal_px: int = 3, block: int = 2048) -> Path:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import from_bounds
    from scipy.ndimage import uniform_filter

    xmin, ymin, xmax, ymax = bounds_32719
    width, height = int(np.ceil((xmax - xmin) / res)), int(np.ceil((ymax - ymin) / res))
    dst_transform = from_origin(xmin, ymax, res, res)
    frac = np.zeros((height, width), dtype=np.float32)
    covered = np.zeros((height, width), dtype=bool)

    srcs = [rasterio.open(fp) for fp in tile_fps]
    for r0 in range(0, height, block):
        for c0 in range(0, width, block):
            h, w = min(block, height - r0), min(block, width - c0)
            bx0, by1 = dst_transform * (c0, r0)
            bx1, by0 = dst_transform * (c0 + w, r0 + h)
            out = np.zeros((h, w), dtype=np.float32)
            got = np.zeros((h, w), dtype=bool)
            for src in srcs:
                sb = transform_bounds('EPSG:32719', 'EPSG:3857', bx0, by0, bx1, by1, densify_pts=21)
                if sb[2] <= src.bounds.left or sb[0] >= src.bounds.right or \
                   sb[3] <= src.bounds.bottom or sb[1] >= src.bounds.top:
                    continue
                win = from_bounds(*sb, transform=src.transform).round_offsets().round_lengths()
                win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                chm = src.read(1, window=win)
                mask = (chm >= min_height_m).astype(np.float32)
                tmp = np.full((h, w), np.nan, dtype=np.float32)
                reproject(mask, tmp, src_transform=src.window_transform(win), src_crs='EPSG:3857',
                          dst_transform=from_origin(bx0, by1, res, res), dst_crs='EPSG:32719',
                          resampling=Resampling.average, dst_nodata=np.nan)
                ok = ~np.isnan(tmp)
                out[ok] = tmp[ok]
                got |= ok
            frac[r0:r0 + h, c0:c0 + w] = out
            covered[r0:r0 + h, c0:c0 + w] = got
    for s in srcs:
        s.close()

    if focal_px > 1:
        frac = uniform_filter(frac, size=focal_px, mode='nearest')
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_fp, 'w', driver='GTiff', height=height, width=width, count=1,
                       dtype='uint8', crs='EPSG:32719', transform=dst_transform, nodata=255,
                       compress='deflate', tiled=True) as dst:
        data = np.where(covered, np.round(frac * 100), 255).astype(np.uint8)
        dst.write(data, 1)
    return out_fp


def add_edge_canopy(G, canopy_fp, step_m: float = 3.0, attr: str = 'canopy_frac') -> None:
    """Agrega a cada arista de G la fracción media de copa muestreada cada `step_m` metros."""
    import rasterio
    import shapely
    from shapely.geometry import LineString

    with rasterio.open(canopy_fp) as src:
        arr = src.read(1)
        inv = ~src.transform
    h, w = arr.shape

    edges = list(G.edges(keys=True, data=True))
    geoms = np.asarray([d['geometry'] if 'geometry' in d else
                        LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
                        for u, v, k, d in edges], dtype=object)
    mean = np.zeros(len(geoms))
    for a in range(0, len(geoms), 50000):
        g = geoms[a:a + 50000]
        n = np.maximum(2, np.ceil(shapely.length(g) / step_m).astype(int) + 1)
        owner = np.repeat(np.arange(len(g)), n)
        pos = np.concatenate([np.linspace(0, 1, k) for k in n])
        pts = shapely.line_interpolate_point(g[owner], pos, normalized=True)
        cols, rows = inv * (shapely.get_x(pts), shapely.get_y(pts))
        cols, rows = np.floor(cols).astype(int), np.floor(rows).astype(int)
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        val = np.full(len(pts), np.nan)
        raw = arr[rows[inside], cols[inside]].astype(float)
        raw[raw == 255] = np.nan
        val[inside] = raw / 100.0
        s = np.bincount(owner, weights=np.nan_to_num(val), minlength=len(g))
        c = np.bincount(owner, weights=(~np.isnan(val)).astype(float), minlength=len(g))
        mean[a:a + len(g)] = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    for (u, v, k, d), m in zip(edges, mean):
        d[attr] = float(m)
