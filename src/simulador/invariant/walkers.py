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
                    agentes_por_am: int = 1,
                    salud_boost: float = 1.0,
                    tabla_age: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Genera la población de agentes a partir de las manzanas.

    Un agente por cada `agentes_por_am` adultos mayores 60+. Con `tabla_age`
    (tablas por persona de eod_tables.py) a cada agente se le sortea un tramo de
    edad según p(edad | vuln_group) y luego la decisión de caminar según
    p(camina | vuln_group, tramo). Sin `tabla_age` se usa la tabla por viaje del
    nb 2, solo por vuln_group. Si camina, el propósito sale de p(propósito | vuln_group).
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

    if tabla_age is not None:
        # Tramo de edad ~ p(edad | vuln_group); caminar ~ p(camina | vuln_group, tramo)
        agents['age_band'] = None
        vg = agents['vuln_group'].to_numpy()
        for g, sub in tabla_age.groupby('vuln_group'):
            m = vg == g
            if m.any():
                p = sub['prob'].to_numpy(dtype=float)
                agents.loc[m, 'age_band'] = rng.choice(sub['age_band'].to_numpy(), size=int(m.sum()), p=p / p.sum())
        walk_prob = {(r.vuln_group, r.age_band): r.p_walk for r in tabla_walk.itertuples()}
        agents['p_walk'] = [walk_prob.get(k, np.nan) for k in zip(agents['vuln_group'], agents['age_band'])]
    else:
        walk_prob = dict(zip(tabla_walk['vuln_group'], tabla_walk['p_walk_pure']))
        agents['p_walk'] = agents['vuln_group'].map(walk_prob)
    agents['will_walk'] = rng.random(len(agents)) < agents['p_walk'].to_numpy(dtype=float)

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
                             crs_proj: str = 'EPSG:32719',
                             tabla_dist: pd.DataFrame | None = None) -> gpd.GeoDataFrame:
    """
    Filtra a los que caminan y les asigna nodo origen (manzana) y destino.

    Sin `tabla_dist`: de los K más cercanos del propósito correspondiente, se
    muestrea uno con peso inverso a la distancia (regla de nb5).
    Con `tabla_dist` (distancias observadas de viajes a pie 60+ en la EOD, ver
    eod_tables.py): cada agente sortea una distancia objetivo de la distribución
    de su propósito (y vuln_group, si la tabla la distingue) y va al destino cuya
    distancia en línea recta está más cerca de ese objetivo.
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
        if tabla_dist is not None:
            target = sample_target_distances(tabla_dist, purpose, walkers.loc[mask, 'vuln_group'].to_numpy(), rng)
            idx = nearest_to_target(tree, np.c_[wx[mask], wy[mask]], target)
            dest_id[mask] = cand['dest_id'].to_numpy()[idx]
            dest_node[mask] = cand['dest_node'].to_numpy()[idx]
            continue
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


EOD_PURPOSE = {v: k for k, v in PURPOSE_MAP.items()}   # propósito del modelo → nombre EOD


def generate_walkers_access(manz: pd.DataFrame, tabla_age: pd.DataFrame, tabla_purp: pd.DataFrame,
                            tabla_car: pd.DataFrame, walk_logit: pd.DataFrame, commerce_xy: np.ndarray,
                            rng: np.random.Generator, agentes_por_am: int = 1,
                            tabla_car_comuna: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Población → quiénes caminan ese día, según un logit por persona estimado en la EOD:
    P(camina) = f(log distancia al 10º comercio desde la manzana, auto en el hogar,
    vuln_group, tramo de edad). Tramo de edad ~ p(edad | vuln); auto ~ p(auto | vuln);
    propósito de quien camina ~ p(propósito | vuln) de los viajes a pie.
    """
    manz = manz.copy()
    manz['n_agents'] = np.where(
        manz['n_edad_60_mas'] > 0,
        np.maximum(1, np.round(manz['n_edad_60_mas'] / agentes_por_am).astype(int)), 0)
    manz['d10_com'] = cKDTree(commerce_xy).query(np.c_[manz['x_rep'], manz['y_rep']], k=10)[0][:, -1]
    keep = ['MANZENT', 'COD_MANZANA', 'vuln_group', 'ndvi_mean', 'ndvi_norm', 'x_rep', 'y_rep', 'd10_com']
    agents = manz.loc[manz.index.repeat(manz['n_agents']), keep].reset_index(drop=True)
    agents['agent_id'] = np.arange(1, len(agents) + 1)
    vg = agents['vuln_group'].to_numpy()

    agents['age_band'] = None
    for g, sub in tabla_age.groupby('vuln_group'):
        m = vg == g
        if m.any():
            p = sub['prob'].to_numpy(dtype=float)
            agents.loc[m, 'age_band'] = rng.choice(sub['age_band'].to_numpy(), size=int(m.sum()), p=p / p.sum())
    p_car = agents['vuln_group'].map(dict(zip(tabla_car['vuln_group'], tabla_car['p_car']))).fillna(0)
    if tabla_car_comuna is not None:   # tenencia de auto por comuna (EOD); por vuln_group donde no hay dato
        cut = (agents['MANZENT'].astype('int64') // 10**9).astype(int)
        p_car = cut.map(dict(zip(tabla_car_comuna['CUT'], tabla_car_comuna['p_car']))).fillna(p_car)
    agents['car'] = rng.random(len(agents)) < p_car.to_numpy()

    c = dict(zip(walk_logit['term'], walk_logit['coef']))
    ab = agents['age_band'].to_numpy()
    z = (c['intercept'] + c['ln_d10_com'] * np.log(np.maximum(agents['d10_com'].to_numpy(), 1.0))
         + c['car'] * agents['car'].to_numpy(dtype=float)
         + c['vuln_media'] * (vg == 'media') + c['vuln_alta'] * (vg == 'alta')
         + c['age_70_79'] * (ab == '70-79') + c['age_80'] * (ab == '80+'))
    agents['p_walk'] = 1.0 / (1.0 + np.exp(-z))
    agents['will_walk'] = rng.random(len(agents)) < agents['p_walk'].to_numpy()

    agents['purpose_group_eod'] = None
    m = agents['will_walk'].to_numpy()
    for g, sub in tabla_purp.groupby('vuln_group'):
        mg = m & (vg == g)
        if mg.any():
            p = sub['prob'].to_numpy(dtype=float)
            agents.loc[mg, 'purpose_group_eod'] = rng.choice(sub['purpose_group'].to_numpy(), size=int(mg.sum()), p=p / p.sum())
    agents['purpose_group_model'] = agents['purpose_group_eod'].map(PURPOSE_MAP)
    return agents


def generate_trip_makers(manz: pd.DataFrame, tabla_age: pd.DataFrame, tabla_trip: pd.DataFrame,
                         tabla_purp_trip: pd.DataFrame, tabla_car: pd.DataFrame,
                         rng: np.random.Generator, agentes_por_am: int = 1) -> pd.DataFrame:
    """
    Población → quiénes hacen un viaje (cualquier modo) a salud, compras o social ese día.

    Por agente: tramo de edad ~ p(edad | vuln); viaja ~ p(viaje | vuln, edad);
    propósito ~ p(propósito | vuln) de todos los viajes; auto en el hogar ~ p(auto | vuln).
    Si camina o no se decide después, cuando se conoce la distancia al destino.
    """
    manz = manz.copy()
    manz['n_agents'] = np.where(
        manz['n_edad_60_mas'] > 0,
        np.maximum(1, np.round(manz['n_edad_60_mas'] / agentes_por_am).astype(int)), 0)
    keep = ['MANZENT', 'COD_MANZANA', 'vuln_group', 'ndvi_mean', 'ndvi_norm', 'x_rep', 'y_rep']
    agents = manz.loc[manz.index.repeat(manz['n_agents']), keep].reset_index(drop=True)
    agents['agent_id'] = np.arange(1, len(agents) + 1)
    vg = agents['vuln_group'].to_numpy()

    agents['age_band'] = None
    for g, sub in tabla_age.groupby('vuln_group'):
        m = vg == g
        if m.any():
            p = sub['prob'].to_numpy(dtype=float)
            agents.loc[m, 'age_band'] = rng.choice(sub['age_band'].to_numpy(), size=int(m.sum()), p=p / p.sum())
    p_trip = {(r.vuln_group, r.age_band): r.p_trip for r in tabla_trip.itertuples()}
    agents['p_trip'] = [p_trip.get(k, np.nan) for k in zip(agents['vuln_group'], agents['age_band'])]
    agents['will_trip'] = rng.random(len(agents)) < agents['p_trip'].to_numpy(dtype=float)
    agents['car'] = rng.random(len(agents)) < agents['vuln_group'].map(
        dict(zip(tabla_car['vuln_group'], tabla_car['p_car']))).fillna(0).to_numpy()

    agents['purpose_group_eod'] = None
    m = agents['will_trip'].to_numpy()
    for g, sub in tabla_purp_trip.groupby('vuln_group'):
        mg = m & (vg == g)
        if mg.any():
            p = sub['prob'].to_numpy(dtype=float)
            agents.loc[mg, 'purpose_group_eod'] = rng.choice(sub['purpose_group'].to_numpy(), size=int(mg.sum()), p=p / p.sum())
    agents['purpose_group_model'] = agents['purpose_group_eod'].map(PURPOSE_MAP)
    return agents


def walk_probability(dist_m: np.ndarray, car: np.ndarray, vuln: np.ndarray, logit: pd.DataFrame) -> np.ndarray:
    """P(a pie | distancia en línea recta, auto, vuln_group) según el logit estimado en la EOD."""
    c = dict(zip(logit['term'], logit['coef']))
    z = (c['intercept'] + c['log_dist_m'] * np.log(np.maximum(dist_m, 1.0)) + c['car'] * car.astype(float)
         + c['vuln_media'] * (vuln == 'media') + c['vuln_alta'] * (vuln == 'alta'))
    return 1.0 / (1.0 + np.exp(-z))


def calibrate_logit_intercept(dist_m, car, vuln, logit: pd.DataFrame, target_walkers: float) -> pd.DataFrame:
    """Desplaza la constante del logit (bisección) para que la suma de P(a pie) sea `target_walkers`."""
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        shifted = logit.copy()
        shifted.loc[shifted['term'] == 'intercept', 'coef'] += mid
        if walk_probability(dist_m, car, vuln, shifted).sum() < target_walkers:
            lo = mid
        else:
            hi = mid
    out = logit.copy()
    out.loc[out['term'] == 'intercept', 'coef'] += (lo + hi) / 2
    print(f'[invariant] constante del logit recalibrada: +{(lo + hi) / 2:.3f}')
    return out


def assign_trips_modes_dests(agents: pd.DataFrame, destinos: gpd.GeoDataFrame, G,
                             rng: np.random.Generator, tabla_dist_trip: pd.DataFrame,
                             logit: pd.DataFrame, crs_proj: str = 'EPSG:32719',
                             k: int = 1024, target_walk_share: float | None = None,
                             n_population: int | None = None) -> gpd.GeoDataFrame:
    """
    Destino y modo para quienes viajan: distancia objetivo ~ distancias de todos los
    viajes EOD (cualquier modo) de su propósito; destino = el candidato más cercano a
    ese objetivo; camina ~ P(a pie | distancia real al destino, auto, vuln). Devuelve
    solo quienes caminan, con nodo de origen y destino.

    El modelo asigna un viaje por persona, pero el logit es por viaje y quien sale
    hace en promedio más de uno. Con `target_walk_share` (proporción de personas
    que camina, EOD) se recalibra la constante del logit para reproducirla sobre
    `n_population` personas; las pendientes de distancia y auto no cambian.
    """
    trips = agents[agents['will_trip'] & agents['purpose_group_model'].notna()].reset_index(drop=True)
    xy = np.c_[trips['x_rep'].to_numpy(), trips['y_rep'].to_numpy()]
    n = len(trips)
    d_act = np.full(n, np.nan)
    choice = np.full(n, -1)
    cand_by_purpose = {}
    for purpose, cand in destinos.groupby('purpose_group_model'):
        mask = (trips['purpose_group_model'] == purpose).to_numpy()
        if not mask.any() or len(cand) == 0:
            continue
        cand = cand.reset_index(drop=True)
        tree = cKDTree(np.c_[cand.geometry.x.to_numpy(), cand.geometry.y.to_numpy()])
        cand_by_purpose[purpose] = (cand, tree)
        rows = np.flatnonzero(mask)
        target = sample_target_distances(tabla_dist_trip, purpose, trips.loc[mask, 'vuln_group'].to_numpy(), rng)
        kk = min(k, tree.n)
        for a in range(0, len(rows), 20000):
            r = rows[a:a + 20000]
            t = target[a:a + 20000]
            dd, ii = tree.query(xy[r], k=kk)
            dd, ii = (dd[:, None], ii[:, None]) if kk == 1 else (dd, ii)
            j = np.abs(dd - t[:, None]).argmin(axis=1)
            near = t <= dd[:, -1]
            choice[r[near]] = ii[near, j[near]]
            d_act[r[near]] = dd[near, j[near]]
            d_act[r[~near]] = t[~near]          # objetivo más allá del k-ésimo: destino se resuelve si camina

    if target_walk_share is not None:
        logit = calibrate_logit_intercept(d_act, trips['car'].to_numpy(), trips['vuln_group'].to_numpy(),
                                          logit, target_walk_share * n_population)
    p = walk_probability(d_act, trips['car'].to_numpy(), trips['vuln_group'].to_numpy(), logit)
    walks = rng.random(n) < p
    trips['dist_target_m'] = d_act
    walkers = trips[walks].copy()
    ch = choice[walks]

    dest_id = np.empty(len(walkers), dtype=object)
    dest_node = np.empty(len(walkers), dtype=object)
    wxy = np.c_[walkers['x_rep'].to_numpy(), walkers['y_rep'].to_numpy()]
    for purpose, (cand, tree) in cand_by_purpose.items():
        m = (walkers['purpose_group_model'] == purpose).to_numpy()
        idx = ch[m].copy()
        pending = np.flatnonzero(idx < 0)
        tt = walkers.loc[m, 'dist_target_m'].to_numpy()
        pts = wxy[m]
        for q in pending:
            c = np.asarray(tree.query_ball_point(pts[q], r=tt[q] * 1.1))
            if c.size == 0:
                c = np.array([tree.query(pts[q])[1]])
            dd = np.hypot(*(tree.data[c] - pts[q]).T)
            idx[q] = c[np.abs(dd - tt[q]).argmin()]
        dest_id[m] = cand['dest_id'].to_numpy()[idx]
        dest_node[m] = cand['dest_node'].to_numpy()[idx]
    walkers['dest_id'] = dest_id
    walkers['dest_node'] = dest_node
    walkers['will_walk'] = True
    walkers = gpd.GeoDataFrame(walkers, geometry=gpd.points_from_xy(walkers['x_rep'], walkers['y_rep']),
                               crs=crs_proj).reset_index(drop=True)
    walkers['origin_node'] = ox.distance.nearest_nodes(G, walkers.geometry.x.to_numpy(), walkers.geometry.y.to_numpy())
    return walkers


def sample_target_distances(tabla_dist: pd.DataFrame, purpose_model: str,
                            vuln: np.ndarray, rng: np.random.Generator,
                            jitter: float = 0.10) -> np.ndarray:
    """Distancia objetivo por agente: remuestreo ponderado de las distancias EOD
    observadas, con un factor uniforme ±`jitter` para no concentrar agentes en
    los mismos valores."""
    sub = tabla_dist[tabla_dist['purpose_group'] == EOD_PURPOSE[purpose_model]]
    out = np.empty(len(vuln))
    groups = sub['vuln_group'].unique()
    for g in np.unique(vuln):
        m = vuln == g
        s = sub[sub['vuln_group'] == g] if g in groups else sub[sub['vuln_group'] == 'todas']
        if s.empty:
            s = sub
        p = s['w'].to_numpy(dtype=float)
        out[m] = rng.choice(s['dist_m'].to_numpy(), size=int(m.sum()), p=p / p.sum())
    return out * rng.uniform(1 - jitter, 1 + jitter, size=len(out))


def nearest_to_target(tree: cKDTree, xy: np.ndarray, target: np.ndarray,
                      k: int = 1024, chunk: int = 20000) -> np.ndarray:
    """Índice del candidato cuya distancia al origen es la más cercana a `target`."""
    k = min(k, tree.n)
    out = np.empty(len(xy), dtype=int)
    for a in range(0, len(xy), chunk):
        d, i = tree.query(xy[a:a + chunk], k=k)
        d, i = np.atleast_2d(d), np.atleast_2d(i)
        if k == 1:
            d, i = d.T, i.T
        t = target[a:a + chunk]
        out[a:a + chunk] = i[np.arange(len(t)), np.abs(d - t[:, None]).argmin(axis=1)]
        # Objetivo más lejos que el k-ésimo vecino: buscar en todo el radio
        far = np.flatnonzero(t > d[:, -1])
        if far.size and k < tree.n:
            for r in far:
                cand = np.asarray(tree.query_ball_point(xy[a + r], r=t[r] * 1.1))
                if cand.size:
                    dd = np.hypot(*(tree.data[cand] - xy[a + r]).T)
                    out[a + r] = cand[np.abs(dd - t[r]).argmin()]
    return out
