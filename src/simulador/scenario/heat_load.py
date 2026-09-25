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

    if config.physiology_by == 'age':
        if 'age_band' not in agents.columns:
            raise ValueError("physiology_by='age' requiere la columna age_band en walkers")
        phys_key, speed_map, delta_map = 'age_band', config.walk_speed_by_age, config.threshold_delta_by_age
    elif config.physiology_by == 'vuln':
        phys_key, speed_map, delta_map = 'vuln_group', config.walk_speed_by_vuln, config.vuln_delta
    else:
        raise ValueError(f'physiology_by desconocido: {config.physiology_by!r}')

    agents['walk_speed_m_min'] = agents[phys_key].map(speed_map).fillna(config.walk_speed_m_min)

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

    # Fracción de la ruta bajo copa de árbol (capa invariante); 0 si no se calculó
    agents['shade_frac'] = (agents['shade_route'].fillna(0.0).clip(0, 1)
                            if 'shade_route' in agents.columns else 0.0)

    # Enfriamiento NDVI: reducción aditiva de WBGT (°C) para la ruta del agente
    ndvi_n = ((agents['ndvi_route'] - ndvi_min) / (ndvi_max - ndvi_min + 1e-9)).clip(0, 1)
    agents['ndvi_cooling'] = config.ndvi_alpha * ndvi_n

    agents['wbgt_umbral'] = config.wbgt_umbral_base - agents[phys_key].map(delta_map).fillna(0.0)

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


_STATUS = np.array(['en_casa', 'caminando', 'en_destino', 'volviendo', 'completado'], dtype=object)
_HOME, _WALK, _DEST, _RET, _DONE = range(5)


def _run_day(agents: pd.DataFrame,
             wbgt_profile: np.ndarray,
             config: SimConfig,
             snapshot_every: int | None,
             day: int | None = None) -> list:
    """
    Un día completo sobre arreglos numpy. Misma lógica y mismo orden de
    operaciones que `step`, sin comparar strings por paso. Escribe el estado
    final en `agents` y devuelve los snapshots pedidos.

    `wbgt_profile` puede ser 1D (un solo perfil, sin sombra) o (2, n_steps)
    sol/sombra: el WBGT de cada agente es la mezcla según su `shade_frac`.
    """
    prof = np.atleast_2d(np.asarray(wbgt_profile, dtype=float))
    if prof.shape[1] < config.n_steps:
        raise ValueError(
            f'wbgt_profile tiene {prof.shape[1]} valores; '
            f'se necesitan al menos {config.n_steps}.')
    sun_prof = prof[0]
    shade_prof = prof[1] if prof.shape[0] > 1 else prof[0]
    code = {name: i for i, name in enumerate(_STATUS)}
    status  = agents['status'].map(code).to_numpy(dtype=np.int8)
    dep     = agents['departure_step'].to_numpy()
    dwell   = agents['dwell_steps'].to_numpy()
    route   = agents['route_length_m'].to_numpy(dtype=float)
    speed   = agents['walk_speed_m_min'].to_numpy(dtype=float) * config.step_min
    cooling = agents['ndvi_cooling'].to_numpy(dtype=float)
    umbral  = agents['wbgt_umbral'].to_numpy(dtype=float)
    shade   = (agents['shade_frac'].to_numpy(dtype=float) if 'shade_frac' in agents.columns
               else np.zeros(len(agents)))
    dist    = agents['distance_traveled'].to_numpy(dtype=float).copy()
    heat    = agents['heat_load'].to_numpy(dtype=float).copy()
    arrival = agents['arrival_step'].to_numpy().copy()

    def write_back():
        agents['status'] = _STATUS[status]
        agents['distance_traveled'] = dist
        agents['heat_load'] = heat
        agents['arrival_step'] = arrival
        agents['risk_level'] = classify_risk(heat, config)

    snapshots = []
    for s in range(config.n_steps):
        sun_now = float(sun_prof[s])
        shade_gap = float(shade_prof[s]) - sun_now

        status[(status == _HOME) & (dep == s)] = _WALK

        idx = np.flatnonzero(status == _WALK)
        if idx.size:
            dist[idx] += np.minimum(speed[idx], route[idx] - dist[idx])
            heat[idx] += np.maximum(0.0, (sun_now + shade[idx] * shade_gap - cooling[idx]) - umbral[idx]) * config.step_min
            arrived = idx[dist[idx] >= route[idx]]
            status[arrived] = _DEST
            arrival[arrived] = s
            dist[arrived] = route[arrived]

        done = (status == _DEST) & (arrival >= 0) & ((s - arrival) >= dwell)
        status[done] = _RET
        dist[done] = 0.0

        idx = np.flatnonzero(status == _RET)
        if idx.size:
            dist[idx] += np.minimum(speed[idx], route[idx] - dist[idx])
            heat[idx] += np.maximum(0.0, (sun_now + shade[idx] * shade_gap - cooling[idx]) - umbral[idx]) * config.step_min
            status[idx[dist[idx] >= route[idx]]] = _DONE

        if snapshot_every and s % snapshot_every == 0:
            write_back()
            snap = agents[_SNAP_COLS].copy()
            snap['step'] = s
            snap['time'] = step_to_time(s, config)
            if day is not None:
                snap['day'] = day
            snapshots.append(snap)

    write_back()
    return snapshots


def run(agents: pd.DataFrame,
        wbgt_profile: np.ndarray,
        config: SimConfig = DEFAULT_CFG,
        snapshot_every: int | None = 15) -> tuple:
    """Corre un día completo (config.n_steps pasos). Devuelve (agents, snapshots)."""
    snapshots = _run_day(agents, wbgt_profile, config, snapshot_every)
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

        all_snapshots.extend(_run_day(agents, wbgt_profile, config, snapshot_every, day=day_idx + 1))
        day_finals.append(agents.copy())

        agents['heat_load'] *= (1.0 - nocturnal_decay)
        agents['risk_level'] = classify_risk(agents['heat_load'].values, config)

    return agents, all_snapshots, day_finals
