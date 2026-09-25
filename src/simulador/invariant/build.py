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
from .walkers import generate_agents, build_destinations, assign_origins_and_dests
from .routes import compute_routes_dedup

# Columnas del artefacto (lo que necesita la capa de escenario / build_agents)
WALKERS_COLS = [
    'agent_id', 'MANZENT', 'COD_MANZANA', 'vuln_group',
    'purpose_group_model', 'ndvi_mean', 'ndvi_norm',
    'x_rep', 'y_rep', 'origin_node', 'dest_node', 'dest_id',
    'route_length_m', 'ndvi_route', 'x_dest', 'y_dest',
]


def build_invariant(save: bool = True):
    """Construye la población + rutas + NDVI de ruta para la zona activa."""
    import config as cfg   # import perezoso: repo root debe estar en sys.path

    t0 = time.perf_counter()
    print(f'[invariant] zona = {cfg.ZONA}')

    # ── Entradas ──────────────────────────────────────────────────────────────
    G = load_projected_graph(cfg.FP_RED, cfg.CRS_PROJ)
    igx = build_igraph(G)
    print(f'[invariant] grafo: {len(igx.nodes_list):,} nodos '
          f'({time.perf_counter()-t0:.1f}s)')

    manz     = gpd.read_parquet(cfg.FP_MANZ_SIM).to_crs(cfg.CRS_PROJ)
    salud    = gpd.read_file(cfg.FP_SALUD).to_crs(cfg.CRS_PROJ)
    comercio = gpd.read_file(cfg.FP_COMERCIO).to_crs(cfg.CRS_PROJ)
    verdes   = gpd.read_file(cfg.FP_VERDES_PTS).to_crs(cfg.CRS_PROJ)
    tabla_walk = pd.read_csv(cfg.FP_EOD_WALK)
    tabla_purp = pd.read_csv(cfg.FP_EOD_PURP)

    # ── Población de agentes ────────────────────────────────────────────────────
    rng = np.random.default_rng(cfg.POP_SEED)
    agents = generate_agents(
        manz, tabla_walk, tabla_purp, rng,
        agentes_por_am=cfg.AGENTES_POR_AM, salud_boost=cfg.SALUD_BOOST)
    print(f'[invariant] agentes: {len(agents):,} '
          f'({agents["will_walk"].mean():.1%} caminan)')

    # ── Destinos + asignación origen/destino ────────────────────────────────────
    destinos = build_destinations(salud, comercio, verdes, G, cfg.CRS_PROJ)
    walkers = assign_origins_and_dests(
        agents, destinos, G, rng, k_destinos=cfg.K_DESTINOS, crs_proj=cfg.CRS_PROJ)
    print(f'[invariant] walkers: {len(walkers):,}')

    # ── Ruteo deduplicado + NDVI de ruta ────────────────────────────────────────
    t1 = time.perf_counter()
    sindex = STRtree(manz.geometry.values)
    walkers = compute_routes_dedup(
        walkers, igx, G, manz, sindex, umbral_caminata_m=cfg.UMBRAL_CAMINATA_M)
    n_pares = walkers[['origin_node', 'dest_node']].drop_duplicates().shape[0]
    print(f'[invariant] ruteo dedup: {len(walkers):,} walkers válidos, '
          f'{n_pares:,} pares únicos ({time.perf_counter()-t1:.1f}s)')

    walkers_out = walkers[[c for c in WALKERS_COLS if c in walkers.columns]].copy()

    if save:
        cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        walkers_out.to_parquet(cfg.FP_WALKERS)
        print(f'[invariant] guardado: {cfg.FP_WALKERS} '
              f'(total {time.perf_counter()-t0:.1f}s)')

    return walkers_out


if __name__ == '__main__':
    build_invariant()
