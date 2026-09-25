"""
Motor ABM de carga térmica (WBGT).

Movido de src/simulation.py sin cambios de lógica. Máquina de estados por agente:
    en_casa → caminando → en_destino → volviendo → completado

Regla de acumulación (solo caminando/volviendo):
    WBGT_eff = wbgt_profile[t] − ndvi_cooling[agente]
    dHeat    = max(0, WBGT_eff − wbgt_umbral[agente]) × step_min   [°C·min]

build_agents recibe los `walkers` de la CAPA INVARIANTE (walkers.parquet) y
aplica los parámetros del escenario (ndvi_cooling, wbgt_umbral) + draws
estocásticos (salida, permanencia). Es el "split" acordado: la población y las
rutas ya vienen congeladas; acá solo se aplica config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sim_config import SimConfig, DEFAULT_CFG


def step_to_time(step: int, config: SimConfig = DEFAULT_CFG) -> str:
    """String HH:MM para un índice de paso."""
    total = config.day_start_min + step * config.step_min
    return f'{total // 60:02d}:{total % 60:02d}'


def classify_risk(heat_array: np.ndarray, config: SimConfig = DEFAULT_CFG) -> np.ndarray:
    out = np.full(len(heat_array), 'bajo', dtype=object)
    out[heat_array >= config.threshold_med]  = 'medio'
    out[heat_array >= config.threshold_high] = 'alto'
    return out


_SNAP_COLS = [
    'agent_id', 'vuln_group', 'purpose_group_model',
    'departure_step', 'dwell_steps', 'arrival_step',
    'status', 'distance_traveled', 'heat_load', 'risk_level',
]


def sample_departure_abs(purposes: np.ndarray,
                         rng: np.random.Generator,
                         config: SimConfig = DEFAULT_CFG) -> np.ndarray:
    """Hora de salida de cada agente, en minutos absolutos del día (ver SimConfig.dep_mode)."""
    n = len(purposes)
    if config.dep_mode == 'normal':
        raw = rng.normal(loc=config.dep_mean_min, scale=config.dep_std_min, size=n)
        return np.clip(raw, config.dep_early_min, config.dep_late_min).round().astype(int)
    if config.dep_mode != 'eod2012':
        raise ValueError(f'dep_mode desconocido: {config.dep_mode!r}')

    hours = np.asarray(config.dep_hours)
    pooled = np.sum([np.asarray(w, dtype=float) for w in config.dep_hour_weights.values()], axis=0)
    dep_hour = np.empty(n, dtype=int)
    assigned = np.zeros(n, dtype=bool)
    for purpose, w in config.dep_hour_weights.items():
        mask = purposes == purpose
        if mask.any():
            w = np.asarray(w, dtype=float)
            dep_hour[mask] = rng.choice(hours, size=int(mask.sum()), p=w / w.sum())
            assigned |= mask
    if (~assigned).any():
        dep_hour[~assigned] = rng.choice(hours, size=int((~assigned).sum()), p=pooled / pooled.sum())
    return dep_hour * 60 + rng.integers(0, 60, size=n)


def build_agents(walkers: pd.DataFrame,
                 ndvi_min: float,
                 ndvi_max: float,
                 config: SimConfig = DEFAULT_CFG,
                 seed: int = 42) -> pd.DataFrame:
    """
    Inicializa el estado de los agentes desde `walkers` (capa invariante).

    Columnas requeridas en walkers: agent_id, vuln_group, route_length_m,
    ndvi_route, purpose_group_model.

    Constantes por agente (dependen de config → capa escenario):
        ndvi_cooling  = ndvi_alpha × ndvi_norm(route)   [°C]
        wbgt_umbral   = umbral_base − delta(vuln)        [°C]
        departure_step, dwell_steps
    """
    agents = walkers.copy().reset_index(drop=True)
    n = len(agents)
    rng = np.random.default_rng(seed)

    dep_abs = sample_departure_abs(agents['purpose_group_model'].to_numpy(), rng, config)
    agents['departure_step'] = dep_abs - config.day_start_min

    # Velocidad de caminata por agente (según vulnerabilidad)
    agents['walk_speed_m_min'] = (
        agents['vuln_group'].map(config.walk_speed_by_vuln).fillna(config.walk_speed_m_min)
    )

    # Permanencia por agente (según propósito)
    dwell_steps = np.full(n, config.dwell_min_min, dtype=int)
    for purpose, (dmin, dmax) in config.dwell_by_purpose.items():
        mask = (agents['purpose_group_model'] == purpose).values
        if mask.any():
            dwell_steps[mask] = rng.integers(dmin, dmax + 1, size=int(mask.sum()))
    unknown = ~agents['purpose_group_model'].isin(config.dwell_by_purpose)
    if unknown.any():
        dwell_steps[unknown.values] = rng.integers(
            config.dwell_min_min, config.dwell_max_min + 1, size=int(unknown.sum()))
    agents['dwell_steps'] = dwell_steps

    # Enfriamiento NDVI: reducción aditiva de WBGT (°C) para la ruta del agente
    ndvi_n = ((agents['ndvi_route'] - ndvi_min) / (ndvi_max - ndvi_min + 1e-9)).clip(0, 1)
    agents['ndvi_cooling'] = config.ndvi_alpha * ndvi_n

    # Umbral WBGT por agente (°C)
    delta = agents['vuln_group'].map(config.vuln_delta).fillna(0.0)
    agents['wbgt_umbral'] = config.wbgt_umbral_base - delta

    # Estado dinámico
    agents['status']            = 'en_casa'
    agents['distance_traveled'] = 0.0
    agents['heat_load']         = 0.0
    agents['risk_level']        = 'bajo'
    agents['arrival_step']      = -1

    return agents


def step(agents: pd.DataFrame,
         current_step: int,
         wbgt_profile: np.ndarray,
         config: SimConfig = DEFAULT_CFG) -> pd.DataFrame:
    """Avanza todos los agentes un paso."""
    wbgt_now = float(wbgt_profile[current_step])

    # 1. Salida de casa
    departs = (agents['status'] == 'en_casa') & (agents['departure_step'] == current_step)
    agents.loc[departs, 'status'] = 'caminando'

    # 2. Caminata de ida
    walking = agents['status'] == 'caminando'
    if walking.any():
        remaining = agents.loc[walking, 'route_length_m'] - agents.loc[walking, 'distance_traveled']
        advance = np.minimum(agents.loc[walking, 'walk_speed_m_min'] * config.step_min, remaining)
        agents.loc[walking, 'distance_traveled'] += advance

        wbgt_eff = wbgt_now - agents.loc[walking, 'ndvi_cooling']
        dHeat = np.maximum(0.0, wbgt_eff - agents.loc[walking, 'wbgt_umbral']) * config.step_min
        agents.loc[walking, 'heat_load'] += dHeat

        arrived = walking & (agents['distance_traveled'] >= agents['route_length_m'])
        agents.loc[arrived, 'status']            = 'en_destino'
        agents.loc[arrived, 'arrival_step']      = current_step
        agents.loc[arrived, 'distance_traveled'] = agents.loc[arrived, 'route_length_m']

    # 3. Salir del destino tras la permanencia (sin acumulación — límite v1)
    at_dest = agents['status'] == 'en_destino'
    dwell_done = (
        at_dest &
        (agents['arrival_step'] >= 0) &
        ((current_step - agents['arrival_step']) >= agents['dwell_steps'])
    )
    agents.loc[dwell_done, 'status']            = 'volviendo'
    agents.loc[dwell_done, 'distance_traveled'] = 0.0

    # 4. Caminata de vuelta (misma ruta invertida)
    returning = agents['status'] == 'volviendo'
    if returning.any():
        remaining_ret = agents.loc[returning, 'route_length_m'] - agents.loc[returning, 'distance_traveled']
        advance_ret = np.minimum(agents.loc[returning, 'walk_speed_m_min'] * config.step_min, remaining_ret)
        agents.loc[returning, 'distance_traveled'] += advance_ret

        wbgt_eff_ret = wbgt_now - agents.loc[returning, 'ndvi_cooling']
        dHeat_ret = np.maximum(0.0, wbgt_eff_ret - agents.loc[returning, 'wbgt_umbral']) * config.step_min
        agents.loc[returning, 'heat_load'] += dHeat_ret

        home = returning & (agents['distance_traveled'] >= agents['route_length_m'])
        agents.loc[home, 'status'] = 'completado'

    agents['risk_level'] = classify_risk(agents['heat_load'].values, config)
    return agents


def run(agents: pd.DataFrame,
        wbgt_profile: np.ndarray,
        config: SimConfig = DEFAULT_CFG,
        snapshot_every: int = 15) -> tuple:
    """Corre un día completo (config.n_steps pasos). Devuelve (agents, snapshots)."""
    if len(wbgt_profile) < config.n_steps:
        raise ValueError(
            f'wbgt_profile tiene {len(wbgt_profile)} valores; '
            f'se necesitan al menos {config.n_steps}.')

    snapshots = []
    for s in range(config.n_steps):
        agents = step(agents, s, wbgt_profile, config)
        if s % snapshot_every == 0:
            snap = agents[_SNAP_COLS].copy()
            snap['step'] = s
            snap['time'] = step_to_time(s, config)
            snapshots.append(snap)
    return agents, snapshots


def run_multiday(agents: pd.DataFrame,
                 wbgt_profiles: list,
                 config: SimConfig = DEFAULT_CFG,
                 snapshot_every: int = 15,
                 nocturnal_decay: float = 0.80,
                 seed: int = 42) -> tuple:
    """
    Corre N días consecutivos con decay nocturno del heat_load.

        heat_load_dia_siguiente = heat_load_fin_dia × (1 − nocturnal_decay)

    Cada día resetea el estado de movimiento y re-sortea salida/permanencia.
    Devuelve (agents_final, all_snapshots, day_finals). day_finals captura el
    estado ANTES del decay (para análisis por día).
    """
    all_snapshots: list = []
    day_finals: list = []

    for day_idx, wbgt_profile in enumerate(wbgt_profiles):
        agents['status']            = 'en_casa'
        agents['distance_traveled'] = 0.0
        agents['arrival_step']      = -1

        rng = np.random.default_rng(seed + day_idx * 1000)
        n = len(agents)

        dep_abs = sample_departure_abs(agents['purpose_group_model'].to_numpy(), rng, config)
        agents['departure_step'] = dep_abs - config.day_start_min

        dwell_steps = np.full(n, config.dwell_min_min, dtype=int)
        for purpose, (dmin, dmax) in config.dwell_by_purpose.items():
            mask = (agents['purpose_group_model'] == purpose).values
            if mask.any():
                dwell_steps[mask] = rng.integers(dmin, dmax + 1, size=int(mask.sum()))
        unknown = ~agents['purpose_group_model'].isin(config.dwell_by_purpose)
        if unknown.any():
            dwell_steps[unknown.values] = rng.integers(
                config.dwell_min_min, config.dwell_max_min + 1, size=int(unknown.sum()))
        agents['dwell_steps'] = dwell_steps

        for s in range(config.n_steps):
            agents = step(agents, s, wbgt_profile, config)
            if s % snapshot_every == 0:
                snap = agents[_SNAP_COLS].copy()
                snap['step'] = s
                snap['time'] = step_to_time(s, config)
                snap['day']  = day_idx + 1
                all_snapshots.append(snap)

        day_finals.append(agents.copy())

        agents['heat_load'] *= (1.0 - nocturnal_decay)
        agents['risk_level'] = classify_risk(agents['heat_load'].values, config)

    return agents, all_snapshots, day_finals
