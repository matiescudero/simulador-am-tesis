"""
Población de agentes y asignación origen/destino.

Extraído de nb5 (celdas 6 y 8). Mismas reglas, con dos mejoras de cómputo que
NO cambian la lógica:
- Generación de agentes por `index.repeat` en vez de loop fila-a-fila.
- `nearest_nodes` vectorizado y K-vecinos por `scipy.cKDTree` en vez de
  distancia a todos los destinos por agente.

Determinismo: un único `numpy.random.Generator` (semilla de población) recorre
will_walk → propósito → destino. Esto reemplaza la mezcla de `np.random` global
+ `rng_global` del notebook, dejando la capa reproducible con una sola semilla.
El SALUD_BOOST queda expuesto como parámetro (pendiente rediseño conceptual).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from scipy.spatial import cKDTree

# Mapea el propósito EOD al nombre de capa de destino del modelo
PURPOSE_MAP = {
    'salud':             'salud',
    'compras_tramites':  'comercio',
    'social_recreacion': 'areas_verdes',
}


def generate_agents(manz: pd.DataFrame,
                    tabla_walk: pd.DataFrame,
                    tabla_purp: pd.DataFrame,
                    rng: np.random.Generator,
                    agentes_por_am: int = 10,
                    salud_boost: float = 2.0) -> pd.DataFrame:
    """
    Genera la población de agentes a partir de las manzanas.

    Un agente por cada `agentes_por_am` adultos mayores 60+. A cada uno se le
    asigna decisión de caminar (prob. por vulnerabilidad, tabla EOD) y, si
    camina, un propósito de viaje muestreado de la distribución EOD.
    """
    manz = manz.copy()
    manz['n_agents'] = np.where(
        manz['n_edad_60_mas'] > 0,
        np.maximum(1, np.round(manz['n_edad_60_mas'] / agentes_por_am).astype(int)),
        0,
    )

    keep = ['MANZENT', 'COD_MANZANA', 'vuln_group', 'ndvi_mean', 'ndvi_norm', 'x_rep', 'y_rep']
    agents = (manz.loc[manz.index.repeat(manz['n_agents']), keep]
                  .reset_index(drop=True))
    agents['agent_id'] = np.arange(1, len(agents) + 1)

    # Decisión de caminar (probabilidad por grupo de vulnerabilidad)
    walk_prob = dict(zip(tabla_walk['vuln_group'], tabla_walk['p_walk_pure']))
    agents['p_walk'] = agents['vuln_group'].map(walk_prob)
    agents['will_walk'] = rng.random(len(agents)) < agents['p_walk'].to_numpy()

    # Propósito del viaje (distribución EOD, con ajuste hacia salud)
    purpose_probs: dict = {}
    for vg, sub in tabla_purp.groupby('vuln_group'):
        probs = sub['prob'].to_numpy(dtype=float)
        purposes = sub['purpose_group'].tolist()
        if 'salud' in purposes:
            probs[purposes.index('salud')] *= salud_boost
        probs = probs / probs.sum()
        purpose_probs[vg] = (purposes, probs)

    def sample_purpose(vg: str) -> str:
        purposes, probs = purpose_probs[vg]
        return purposes[rng.choice(len(purposes), p=probs)]

    agents['purpose_group_eod'] = None
    m = agents['will_walk'].to_numpy()
    agents.loc[m, 'purpose_group_eod'] = [sample_purpose(vg) for vg in agents.loc[m, 'vuln_group']]
    agents['purpose_group_model'] = agents['purpose_group_eod'].map(PURPOSE_MAP)

    return agents


def build_destinations(salud: gpd.GeoDataFrame,
                       comercio: gpd.GeoDataFrame,
                       verdes: gpd.GeoDataFrame,
                       G,
                       crs_proj: str = 'EPSG:32719') -> gpd.GeoDataFrame:
    """Une las 3 capas de destino, homogeneiza a puntos y asigna nodo de red."""
    def standardize(gdf, purpose, prefix):
        out = gdf[['geometry']].copy().reset_index(drop=True)
        if not out.geom_type.eq('Point').all():
            out['geometry'] = out.geometry.representative_point()
        out['dest_id'] = [f'{prefix}_{i}' for i in range(len(out))]
        out['purpose_group_model'] = purpose
        return out[['dest_id', 'purpose_group_model', 'geometry']]

    destinos = gpd.GeoDataFrame(
        pd.concat([
            standardize(salud,    'salud',        'salud'),
            standardize(comercio, 'comercio',     'com'),
            standardize(verdes,   'areas_verdes', 'verde'),
        ], ignore_index=True),
        geometry='geometry', crs=crs_proj,
    )

    # nearest_nodes vectorizado (un solo KDTree interno)
    destinos['dest_node'] = ox.distance.nearest_nodes(
        G, destinos.geometry.x.to_numpy(), destinos.geometry.y.to_numpy()
    )
    return destinos


def assign_origins_and_dests(agents: pd.DataFrame,
                             destinos: gpd.GeoDataFrame,
                             G,
                             rng: np.random.Generator,
                             k_destinos: int = 5,
                             crs_proj: str = 'EPSG:32719') -> gpd.GeoDataFrame:
    """
    Filtra a los que caminan y les asigna nodo origen (manzana) y destino.

    Destino: de los K más cercanos del propósito correspondiente, se muestrea
    uno con peso inverso a la distancia (idéntico a nb5, pero con cKDTree).
    """
    walkers = agents[(agents['will_walk']) & (agents['purpose_group_model'].notna())].copy()
    walkers = gpd.GeoDataFrame(
        walkers,
        geometry=gpd.points_from_xy(walkers['x_rep'], walkers['y_rep']),
        crs=crs_proj,
    ).reset_index(drop=True)

    # Nodo origen vectorizado
    walkers['origin_node'] = ox.distance.nearest_nodes(
        G, walkers.geometry.x.to_numpy(), walkers.geometry.y.to_numpy()
    )

    # Nodo destino: K-vecinos por propósito vía cKDTree + muestreo ponderado
    wx = walkers.geometry.x.to_numpy()
    wy = walkers.geometry.y.to_numpy()
    dest_id   = np.empty(len(walkers), dtype=object)
    dest_node = np.empty(len(walkers), dtype=object)

    for purpose, cand in destinos.groupby('purpose_group_model'):
        mask = (walkers['purpose_group_model'] == purpose).to_numpy()
        if not mask.any() or len(cand) == 0:
            continue
        cand = cand.reset_index(drop=True)
        tree = cKDTree(np.c_[cand.geometry.x.to_numpy(), cand.geometry.y.to_numpy()])
        kk = min(k_destinos, len(cand))
        dists, idxs = tree.query(np.c_[wx[mask], wy[mask]], k=kk)
        # cKDTree devuelve 1-D cuando kk==1 → forzar 2-D
        dists = np.atleast_2d(dists) if kk == 1 else dists
        idxs  = np.atleast_2d(idxs)  if kk == 1 else idxs
        cand_ids   = cand['dest_id'].to_numpy()
        cand_nodes = cand['dest_node'].to_numpy()

        sel_id, sel_node = [], []
        for row_d, row_i in zip(dists, idxs):
            w = 1.0 / (row_d + 1.0)
            w = w / w.sum()
            j = rng.choice(len(row_i), p=w)
            sel_id.append(cand_ids[row_i[j]])
            sel_node.append(cand_nodes[row_i[j]])
        dest_id[mask]   = sel_id
        dest_node[mask] = sel_node

    walkers['dest_id']   = dest_id
    walkers['dest_node'] = dest_node
    return walkers
