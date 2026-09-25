"""
NDVI a lo largo de una ruta.

Extraído de nb5 (celda 10). Usa un índice espacial STRtree (construido una vez)
para intersectar solo las manzanas candidatas, no todas. El NDVI de la ruta es
el promedio de `ndvi_mean` de las manzanas atravesadas, ponderado por la
longitud de intersección.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString


def route_to_linestring(node_route: list, G) -> LineString:
    """Convierte una lista de nodos OSMnx en la geometría de la ruta."""
    return LineString([(G.nodes[n]['x'], G.nodes[n]['y']) for n in node_route])


def compute_route_ndvi(route_geom, manz_gdf, sindex) -> float:
    """
    NDVI ponderado por longitud de intersección con las manzanas.

    Parameters
    ----------
    route_geom : LineString | Point | None
    manz_gdf   : GeoDataFrame de manzanas con columna 'ndvi_mean'
    sindex     : shapely.strtree.STRtree construido sobre manz_gdf.geometry
    """
    if route_geom is None:
        return np.nan
    cand_idx = sindex.query(route_geom, predicate='intersects')
    if len(cand_idx) == 0:
        return np.nan
    inter = manz_gdf.iloc[cand_idx]
    if route_geom.geom_type == 'Point':
        return float(inter['ndvi_mean'].mean())
    len_int = inter.geometry.intersection(route_geom).length
    total = len_int.sum()
    if total <= 0:
        return np.nan
    return float((len_int * inter['ndvi_mean']).sum() / total)
