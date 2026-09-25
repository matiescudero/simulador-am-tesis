"""
Ruteo peatonal deduplicado.

Optimización central respecto a nb5: muchos agentes comparten el par
(origin_node, dest_node) — todos los de una misma manzana que van al mismo
destino. En vez de un Dijkstra por agente, se calcula una sola vez por par
único, agrupando además por origen para resolver one-to-many en una sola
llamada igraph (`get_shortest_paths` con lista de destinos). El resultado se
mapea de vuelta a los agentes por merge.

Esto reduce el ruteo de O(n_agentes) a O(pares únicos) y habilita el
escalamiento a Gran Santiago. La lógica de ruta/NDVI es idéntica a nb5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import IGraphIndex
from .route_ndvi import route_to_linestring, compute_route_ndvi


def compute_routes_dedup(walkers: pd.DataFrame,
                         igx: IGraphIndex,
                         G,
                         manz,
                         sindex,
                         umbral_caminata_m: float = 1200,
                         route_cache: pd.DataFrame | None = None,
                         return_pairs: bool = False):
    """
    Calcula largo y NDVI de ruta para cada agente, deduplicando por par.

    Devuelve `walkers` con columnas nuevas route_length_m, ndvi_route, x_dest,
    y_dest, filtrado a rutas <= umbral_caminata_m y sin NaN. Con `route_cache`
    (origin_node, dest_node, route_length_m, ndvi_route, shade_route; sin filtrar
    ni rellenar)
    solo se rutean los pares que no estén en el caché. Con `return_pairs=True`
    devuelve además la tabla de pares, reutilizable como caché.
    """
    pares = (walkers[['origin_node', 'dest_node']]
             .dropna().drop_duplicates())

    cols = ['origin_node', 'dest_node', 'route_length_m', 'ndvi_route', 'shade_route']
    cached = None
    if route_cache is not None and len(route_cache) and 'shade_route' in route_cache.columns:
        cached = pares.merge(route_cache[cols],
                             on=['origin_node', 'dest_node'], how='inner')
        pares = (pares.merge(cached[['origin_node', 'dest_node']], on=['origin_node', 'dest_node'],
                             how='left', indicator=True)
                      .query("_merge == 'left_only'").drop(columns='_merge'))

    filas = []
    for origin, grp in pares.groupby('origin_node'):
        src = igx.node_to_idx[origin]
        destinos = grp['dest_node'].tolist()
        tgt_idx = [igx.node_to_idx[d] for d in destinos]

        # One-to-many: un solo Dijkstra desde el origen a todos sus destinos
        epaths = igx.ig_G.get_shortest_paths(
            src, tgt_idx, weights='length', output='epath')

        for dest, epath in zip(destinos, epaths):
            if not epath:  # inalcanzable u origin == dest
                filas.append({'origin_node': origin, 'dest_node': dest,
                              'route_length_m': np.nan, 'ndvi_route': np.nan, 'shade_route': np.nan})
                continue
            es_obj = igx.ig_G.es[epath]
            lens = np.asarray(es_obj['length'], dtype=float)
            length = float(lens.sum())
            shade = float((lens * np.asarray(es_obj['canopy_frac'], dtype=float)).sum() / length) if length > 0 else 0.0
            vpath = [es_obj[0].source] + [e.target for e in es_obj]
            node_route = [igx.nodes_list[i] for i in vpath]
            geom = route_to_linestring(node_route, G) if len(node_route) > 1 else None
            ndvi = compute_route_ndvi(geom, manz, sindex)
            filas.append({'origin_node': origin, 'dest_node': dest,
                          'route_length_m': length, 'ndvi_route': ndvi, 'shade_route': shade})

    rutas = pd.DataFrame(filas, columns=cols)
    if cached is not None:
        rutas = pd.concat([cached, rutas], ignore_index=True)
    walkers = walkers.merge(rutas, on=['origin_node', 'dest_node'], how='left')

    # Fallback NDVI: si la ruta no cruza manzanas con NDVI, usar el de origen
    walkers['ndvi_route'] = walkers['ndvi_route'].fillna(walkers['ndvi_mean'])

    # Filtro de plausibilidad peatonal
    walkers = (walkers[walkers['route_length_m'] <= umbral_caminata_m]
               .dropna(subset=['route_length_m', 'ndvi_route'])
               .reset_index(drop=True))

    # Coordenadas del nodo destino (para mapas/depuración)
    walkers['x_dest'] = walkers['dest_node'].map(lambda n: G.nodes[n]['x'])
    walkers['y_dest'] = walkers['dest_node'].map(lambda n: G.nodes[n]['y'])

    if return_pairs:
        return walkers, rutas
    return walkers
