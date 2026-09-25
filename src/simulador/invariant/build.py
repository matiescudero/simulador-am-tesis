"""
Orquestador de la capa invariante.

Produce el artefacto reutilizable `artifacts/{zona}/walkers.parquet` a partir de
las entradas de la zona activa (config.py). Se ejecuta una vez por zona; todos
los escenarios del banco leen este parquet sin recomputar red/rutas/NDVI.

Uso:
    # desde un notebook (repo root en sys.path)
    from src.simulador.invariant.build import build_invariant
    walkers = build_invariant()

    # o desde la raíz del repo
    python -m src.simulador.invariant.build
"""
from __future__ import annotations

import time

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

from .network import load_projected_graph, build_igraph
from .walkers import (generate_agents, build_destinations, assign_origins_and_dests,
                      generate_trip_makers, assign_trips_modes_dests, generate_walkers_access)
from .routes import compute_routes_dedup
from .canopy import add_edge_canopy

# Columnas del artefacto (lo que necesita la capa de escenario / build_agents)
WALKERS_COLS = [
    'agent_id', 'MANZENT', 'COD_MANZANA', 'vuln_group', 'age_band',
    'purpose_group_model', 'ndvi_mean', 'ndvi_norm',
    'x_rep', 'y_rep', 'origin_node', 'dest_node', 'dest_id',
    'route_length_m', 'ndvi_route', 'shade_route', 'x_dest', 'y_dest', 'car', 'dist_target_m', 'd10_com',
]


def build_invariant(save: bool = True):
    """Construye la población + rutas + NDVI de ruta para la zona activa."""
    import config as cfg   # import perezoso: repo root debe estar en sys.path

    t0 = time.perf_counter()
    print(f'[invariant] zona = {cfg.ZONA}')

    # ── Entradas ──────────────────────────────────────────────────────────────
    G = load_projected_graph(cfg.FP_RED, cfg.CRS_PROJ)
    fp_canopy = getattr(cfg, 'FP_CANOPY', None)
    if fp_canopy is not None and fp_canopy.exists():
        add_edge_canopy(G, fp_canopy)
        print(f'[invariant] copa de árboles por arista: {fp_canopy.name} '
              f'({time.perf_counter()-t0:.1f}s)')
    igx = build_igraph(G)
    print(f'[invariant] grafo: {len(igx.nodes_list):,} nodos '
          f'({time.perf_counter()-t0:.1f}s)')

    manz     = gpd.read_parquet(cfg.FP_MANZ_SIM).to_crs(cfg.CRS_PROJ)
    salud    = gpd.read_file(cfg.FP_SALUD).to_crs(cfg.CRS_PROJ)
    comercio = gpd.read_file(cfg.FP_COMERCIO).to_crs(cfg.CRS_PROJ)
    verdes   = gpd.read_file(cfg.FP_VERDES_PTS).to_crs(cfg.CRS_PROJ)
    tabla_walk = pd.read_csv(cfg.FP_EOD_WALK)
    tabla_purp = pd.read_csv(cfg.FP_EOD_PURP)
    tabla_age  = pd.read_csv(cfg.FP_EOD_AGE)

    rng = np.random.default_rng(cfg.POP_SEED)
    destinos = build_destinations(salud, comercio, verdes, G, cfg.CRS_PROJ)

    if getattr(cfg, 'MODE_CHOICE', None) == 'access_logit':
        # ── Camina según accesibilidad a comercio, auto, vulnerabilidad y edad (logit EOD por persona) ─
        com = destinos[destinos['purpose_group_model'] == 'comercio']
        agents = generate_walkers_access(
            manz, tabla_age, tabla_purp, pd.read_csv(cfg.FP_EOD_CAR), pd.read_csv(cfg.FP_EOD_WALK_LOGIT),
            np.c_[com.geometry.x.to_numpy(), com.geometry.y.to_numpy()], rng, agentes_por_am=cfg.AGENTES_POR_AM,
            tabla_car_comuna=pd.read_csv(cfg.FP_EOD_CAR_COMUNA))
        print(f'[invariant] agentes: {len(agents):,} ({agents["will_walk"].mean():.1%} caminan, '
              f'{agents["car"].mean():.1%} con auto)')
        walkers = assign_origins_and_dests(
            agents, destinos, G, rng, k_destinos=cfg.K_DESTINOS, crs_proj=cfg.CRS_PROJ,
            tabla_dist=pd.read_csv(cfg.FP_EOD_DIST))
        print(f'[invariant] walkers: {len(walkers):,}')
    elif getattr(cfg, 'MODE_CHOICE', None) == 'eod_logit':
        # ── Viaje (cualquier modo) → destino → modo: camina según distancia y auto ─
        agents = generate_trip_makers(
            manz, tabla_age, pd.read_csv(cfg.FP_EOD_TRIP), pd.read_csv(cfg.FP_EOD_PURP_TRIP),
            pd.read_csv(cfg.FP_EOD_CAR), rng, agentes_por_am=cfg.AGENTES_POR_AM)
        print(f'[invariant] agentes: {len(agents):,} ({agents["will_trip"].mean():.1%} viajan, '
              f'{agents["car"].mean():.1%} con auto)')
        walkers = assign_trips_modes_dests(
            agents, destinos, G, rng, pd.read_csv(cfg.FP_EOD_DIST_TRIP),
            pd.read_csv(cfg.FP_EOD_MODE_LOGIT), crs_proj=cfg.CRS_PROJ,
            target_walk_share=getattr(cfg, 'WALK_SHARE_TARGET', None), n_population=len(agents))
        print(f'[invariant] walkers: {len(walkers):,} ({len(walkers) / len(agents):.1%} de los adultos mayores)')
    else:
        # ── Caminar decidido antes del destino (versión anterior) ───────────────────
        agents = generate_agents(
            manz, tabla_walk, tabla_purp, rng,
            agentes_por_am=cfg.AGENTES_POR_AM, salud_boost=cfg.SALUD_BOOST,
            tabla_age=tabla_age)
        print(f'[invariant] agentes: {len(agents):,} '
              f'({agents["will_walk"].mean():.1%} caminan)')
        tabla_dist = pd.read_csv(cfg.FP_EOD_DIST) if cfg.DEST_CHOICE == 'eod_distance' else None
        walkers = assign_origins_and_dests(
            agents, destinos, G, rng, k_destinos=cfg.K_DESTINOS, crs_proj=cfg.CRS_PROJ,
            tabla_dist=tabla_dist)
        print(f'[invariant] walkers: {len(walkers):,}')

    # ── Ruteo deduplicado + NDVI de ruta ────────────────────────────────────────
    t1 = time.perf_counter()
    sindex = STRtree(manz.geometry.values)
    fp_pairs = cfg.ARTIFACTS_DIR / 'route_pairs.parquet'
    cache = pd.read_parquet(fp_pairs) if fp_pairs.exists() else None
    if cache is not None:
        print(f'[invariant] caché de rutas: {len(cache):,} pares')
    walkers, pairs = compute_routes_dedup(
        walkers, igx, G, manz, sindex, umbral_caminata_m=cfg.UMBRAL_CAMINATA_M,
        route_cache=cache, return_pairs=True)
    n_pares = walkers[['origin_node', 'dest_node']].drop_duplicates().shape[0]
    print(f'[invariant] ruteo dedup: {len(walkers):,} walkers válidos, '
          f'{n_pares:,} pares únicos ({time.perf_counter()-t1:.1f}s)')

    walkers_out = walkers[[c for c in WALKERS_COLS if c in walkers.columns]].copy()

    if save:
        cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        walkers_out.to_parquet(cfg.FP_WALKERS)
        pairs.to_parquet(fp_pairs)
        print(f'[invariant] guardado: {cfg.FP_WALKERS} '
              f'(total {time.perf_counter()-t0:.1f}s)')

    return walkers_out


if __name__ == '__main__':
    build_invariant()
