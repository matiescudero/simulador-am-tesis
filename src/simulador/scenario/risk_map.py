"""
Agregación del resultado de un escenario a nivel de manzana.

Salida de un escenario en dos representaciones del mismo resultado:
- vector por manzana (parquet) → alineado por MANZENT, entrada para PCA
- ráster COG                    → formato de servicio para GeoServer/visor

La agregación toma el heat_load de los agentes al final del último día y lo
resume por manzana de origen.
"""
from __future__ import annotations

import os
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


def _fix_proj():
    """Redirige PROJ a los datos del paquete pip (conflicto conda/pip, ver ox env)."""
    for pkg, rel in (('rasterio', 'proj_data'), ('pyproj', 'proj_dir/share/proj')):
        spec = importlib.util.find_spec(pkg)
        if spec is not None:
            cand = Path(spec.origin).parent / rel
            if (cand / 'proj.db').exists():
                os.environ['PROJ_LIB'] = os.environ['PROJ_DATA'] = str(cand)
                return


def aggregate_to_manzana(final_agents: pd.DataFrame,
                         manz: gpd.GeoDataFrame,
                         heat_col: str = 'heat_load') -> gpd.GeoDataFrame:
    """
    Resume el heat_load de los agentes por manzana de origen y lo une a la
    geometría de manzanas.

    Devuelve un GeoDataFrame con una fila por manzana (todas las manzanas de
    `manz`, con NaN/0 donde no hubo agentes) y columnas:
        n_agents, heat_mean, heat_max, heat_p90, pct_risk_high, pct_risk_medium
    """
    g = final_agents.groupby('MANZENT')
    agg = pd.DataFrame({
        'n_agents':        g.size(),
        'heat_mean':       g[heat_col].mean(),
        'heat_max':        g[heat_col].max(),
        'heat_p90':        g[heat_col].quantile(0.90),
        'pct_risk_high':   g['risk_level'].apply(lambda x: (x == 'alto').mean()),
        'pct_risk_medium': g['risk_level'].apply(lambda x: (x == 'medio').mean()),
    }).reset_index()

    out = manz[['MANZENT', 'geometry']].merge(agg, on='MANZENT', how='left')
    out['n_agents'] = out['n_agents'].fillna(0).astype(int)
    return gpd.GeoDataFrame(out, geometry='geometry', crs=manz.crs)


def write_risk_vector(manz_risk: gpd.GeoDataFrame, path) -> None:
    """Guarda el vector de riesgo por manzana (parquet, entrada para PCA)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    manz_risk.to_parquet(path)


def rasterize_to_cog(manz_risk: gpd.GeoDataFrame,
                     column: str,
                     path,
                     resolution: float = 30.0,
                     fill: float = np.nan) -> None:
    """
    Rasteriza una columna del riesgo por manzana a un Cloud-Optimized GeoTIFF.

    resolution en las unidades del CRS (m para UTM 19S). Usa el driver COG de
    GDAL; si no está disponible, cae a GTiff con tiling (equivalente para GeoServer).
    """
    _fix_proj()
    import rasterio
    from rasterio import features
    from rasterio.transform import from_origin

    minx, miny, maxx, maxy = manz_risk.total_bounds
    width  = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)

    shapes = ((geom, val) for geom, val in zip(manz_risk.geometry, manz_risk[column])
              if geom is not None and not pd.isna(val))
    arr = features.rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=fill, dtype='float32')

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    profile = dict(height=height, width=width, count=1, dtype='float32',
                   crs=manz_risk.crs, transform=transform, nodata=fill, compress='deflate')
    try:
        with rasterio.open(path, 'w', driver='COG', **profile) as dst:
            dst.write(arr, 1)
    except Exception:
        # Fallback: GTiff con tiling (GeoServer lo lee igual)
        with rasterio.open(path, 'w', driver='GTiff', tiled=True,
                           blockxsize=256, blockysize=256, **profile) as dst:
            dst.write(arr, 1)
