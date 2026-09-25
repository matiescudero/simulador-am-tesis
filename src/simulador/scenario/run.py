"""
Unidad de corrida de UN escenario — el trabajo elemental del banco.

Diseñada para ser el átomo paralelizable en el cluster: un proceso lee el
artefacto invariante (walkers.parquet, solo lectura, compartido entre todos los
cores del job array), aplica un punto meteorológico (T_max, HR_min, Rs_max),
corre el ABM multi-día y escribe UN output (vector por manzana + COG).

Sin estado global, determinista por semilla → embarrassingly parallel. En SLURM,
cada tarea del array llama a esto con un índice distinto del diseño LHS.
"""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .sim_config import SimConfig
from .met_profile import make_wbgt_profiles_multiday
from .heat_load import build_agents, run_multiday
from .risk_map import aggregate_to_manzana, write_risk_vector, rasterize_to_cog


def run_one_scenario(walkers_fp,
                     manz_fp,
                     met: dict,
                     out_vector_fp=None,
                     out_cog_fp=None,
                     config: SimConfig = None,
                     n_dias: int = 3,
                     nocturnal_decay: float = 0.80,
                     seed: int = 42,
                     crs_proj: str = 'EPSG:32719',
                     cog_col: str = 'heat_mean',
                     cog_res: float = 30.0):
    """
    Corre un escenario y devuelve (manz_risk, timings).

    met : dict con 'T_max', 'HR_min', 'Rs_max'.
    timings : dict con tiempos por etapa (load, sim, aggregate_write, total).
    """
    config = config or SimConfig()
    t = {}
    t0 = time.perf_counter()

    # ── Carga de artefactos (en el cluster: lectura compartida) ─────────────────
    walkers = pd.read_parquet(walkers_fp)
    manz = gpd.read_parquet(manz_fp).to_crs(crs_proj)
    ndvi_min, ndvi_max = manz['ndvi_mean'].min(), manz['ndvi_mean'].max()
    t['load'] = time.perf_counter() - t0

    # ── Simulación (lo único que varía por escenario) ───────────────────────────
    t1 = time.perf_counter()
    agents = build_agents(walkers, ndvi_min, ndvi_max, config=config, seed=seed)
    profiles = make_wbgt_profiles_multiday(
        met['T_max'], met['HR_min'], met['Rs_max'], n_dias=n_dias, config=config)
    _, _, day_finals = run_multiday(
        agents, profiles, config=config, nocturnal_decay=nocturnal_decay, seed=seed,
        snapshot_every=None)
    t['sim'] = time.perf_counter() - t1

    # ── Agregación + escritura de la salida ─────────────────────────────────────
    # Carga al cierre del último día, antes del decay nocturno: la exposición
    # acumulada del evento, no el remanente que pasaría a un día siguiente.
    t2 = time.perf_counter()
    manz_risk = aggregate_to_manzana(day_finals[-1], manz)
    if out_vector_fp is not None:
        write_risk_vector(manz_risk, out_vector_fp)
    if out_cog_fp is not None:
        rasterize_to_cog(manz_risk, cog_col, out_cog_fp, resolution=cog_res)
    t['aggregate_write'] = time.perf_counter() - t2
    t['total'] = time.perf_counter() - t0

    return manz_risk, t


def run_scenario_indexed(index: int,
                         met: dict,
                         walkers_fp,
                         manz_fp,
                         out_dir,
                         seed_base: int = 20260000,
                         **kwargs) -> dict:
    """
    Tarea elemental del banco (la que invoca cada tarea del job array SLURM).

    Escribe vectors/risk_{index}.parquet + maps/risk_{index}.tif y devuelve un
    dict-resumen liviano (no el GeoDataFrame — evita transferir datos pesados
    entre procesos). Semilla derivada del índice → reproducibilidad total.
    """
    out_dir = Path(out_dir)
    out_v = out_dir / 'vectors' / f'risk_{index:03d}.parquet'
    out_c = out_dir / 'maps'    / f'risk_{index:03d}.tif'

    manz_risk, t = run_one_scenario(
        walkers_fp, manz_fp, met,
        out_vector_fp=out_v, out_cog_fp=out_c,
        seed=seed_base + index, **kwargs)

    ca = manz_risk[manz_risk['n_agents'] > 0]
    return {
        'index': index,
        'T_max': met['T_max'], 'HR_min': met['HR_min'], 'Rs_max': met['Rs_max'],
        't_total': t['total'], 't_sim': t['sim'],
        'heat_mean': float(ca['heat_mean'].mean()),
        'heat_p90': float(ca['heat_p90'].mean()),
    }
