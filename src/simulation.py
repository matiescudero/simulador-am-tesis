"""
ABM core engine — heat exposure simulation for older adults.
Peñalolén case study, PhD thesis.

State machine per agent
-----------------------
en_casa → caminando → en_destino → volviendo → completado

Heat model (v2 — WBGT)
-----------------------
- WBGT (Wet Bulb Globe Temperature) is the primary environmental driver.
- WBGT is time-varying: an array of n_steps values (one per simulation minute).
- Effective WBGT along a route = WBGT_profile[t] - ndvi_alpha * ndvi_norm
  (NDVI provides an additive cooling effect, max ndvi_alpha °C at ndvi_norm=1;
   based on Bowler et al. 2010 and Santiago LST-NDVI studies: 1–3 °C range)
- Heat accumulates only while walking (caminando / volviendo):
      dHeat = max(0, WBGT_eff - wbgt_umbral_agent) × dt   [units: °C·min]
- wbgt_umbral_agent = wbgt_umbral_base − delta(vuln_group)
  (more vulnerable agents accumulate heat at lower ambient WBGT)
- No heat accumulation at destination (v1 limitation — declared in thesis:
  subestima riesgo en visitas a áreas verdes; extensión futura)

WBGT formula (Option B — ISO 7243)
-----------------------------------
  e            = (HR/100) × 6.105 × exp(17.27 × T / (237.7 + T))   [hPa]
  WBGT_shade   = 0.567 × T + 0.393 × e + 3.94
  WBGT_outdoor = WBGT_shade + solar_k × sqrt(Rs)

All behavioural parameters live in SimConfig (dataclass).
Instantiate with defaults or override any field.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    All configurable simulation parameters in one object.
    Pass the same instance to build_agents(), step(), and run()
    to guarantee consistency across a scenario.
    """

    # ── Time window ────────────────────────────────────────────────────────
    day_start_min: int   = 6 * 60       # 06:00
    day_end_min:   int   = 20 * 60      # 20:00
    step_min:      int   = 1            # minutes per timestep

    # ── Walking speed ──────────────────────────────────────────────────────
    walk_speed_m_min: float = 62.0
    # Fallback when vuln_group not found in walk_speed_by_vuln.

    walk_speed_by_vuln: dict = field(
        default_factory=lambda: {'baja': 68.0, 'media': 60.0, 'alta': 52.0}
    )
    # Per-group walking speed (m/min).
    # Refs: Bohannon (1997) Phys Ther; Studenski et al. (2011) JAMA.
    # baja  → 68 m/min (~4.1 km/h): comfortable pace, no functional limitation.
    # media → 60 m/min (~3.6 km/h): mild limitations, cautious gait.
    # alta  → 52 m/min (~3.1 km/h): frail/limited, gait speed < 1.0 m/s.

    # ── Departure distribution (truncated normal, absolute minutes of day) ─
    dep_mean_min:  int   = 9 * 60       # 09:00 — model assumption (no EOD data)
    dep_std_min:   int   = 60           # ±1h std
    dep_early_min: int   = 7 * 60       # 07:00 lower bound
    dep_late_min:  int   = 12 * 60      # 12:00 upper bound

    # ── Dwell time at destination (uniform, minutes) ───────────────────────
    dwell_min_min: int   = 20           # fallback for unknown purposes
    dwell_max_min: int   = 60

    dwell_by_purpose: dict = field(
        default_factory=lambda: {
            'salud':        (45, 90),   # Ref tentativa: EOD 2012, visitas médicas
            'comercio':     (25, 50),   # Ref tentativa: EOD 2012, compras/trámites
            'areas_verdes': (15, 40),   # Ref tentativa: EOD 2012, recreación
        }
    )

    # ── WBGT heat model ────────────────────────────────────────────────────
    wbgt_umbral_base: float = 27.0
    # °C — baseline WBGT threshold for older adults (light activity outdoors).
    # Ref: NIOSH (2016) work limits, light work non-acclimatized: ~27–28 °C.
    # Kenney & Munce (2003): thermoregulatory capacity reduced in older adults.

    vuln_delta: dict = field(
        default_factory=lambda: {'baja': 0.0, 'media': 1.5, 'alta': 3.0}
    )
    # Threshold reduction per vulnerability group (°C).
    # Ref: Kenney & Munce (2003) J Appl Physiol — reduced heat tolerance in elderly.
    # alta → effective threshold = 24 °C; media → 25.5 °C; baja → 27 °C.

    # ── NDVI cooling ───────────────────────────────────────────────────────
    ndvi_alpha: float = 2.5
    # Max additive WBGT reduction (°C) at ndvi_norm = 1.
    # Bowler et al. 2010 (Landscape Urban Plan): urban trees reduce temp 1–3 °C.
    # Santiago-specific LST-NDVI studies (U. de Chile): ~2–3 °C. → 2.5 as midpoint.

    # ── Risk classification thresholds (°C·min above agent threshold) ─────
    threshold_med:  float = 30.0        # tentative — calibrate after pilot runs
    threshold_high: float = 60.0        # tentative — calibrate after pilot runs

    # ── WBGT solar correction (Option B) ──────────────────────────────────
    solar_k: float = 0.04
    # WBGT_outdoor = WBGT_shade + solar_k × sqrt(Rs)
    # Gives ~1.2 °C correction at Rs = 850 W/m² (consistent with lower range
    # of ISO 7243 direct-sun corrections for pedestrians).
    # Liljegren et al. 2008 model (Option C) reserved as future extension.

    @property
    def n_steps(self) -> int:
        return (self.day_end_min - self.day_start_min) // self.step_min


DEFAULT_CFG = SimConfig()

# Module-level constant kept for backward compatibility
N_STEPS = DEFAULT_CFG.n_steps


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def step_to_time(step: int, config: SimConfig = DEFAULT_CFG) -> str:
    """Return wall-clock string HH:MM for a given step index."""
    total = config.day_start_min + step * config.step_min
    return f'{total // 60:02d}:{total % 60:02d}'


def classify_risk(heat_array: np.ndarray,
                  config: SimConfig = DEFAULT_CFG) -> np.ndarray:
    out = np.full(len(heat_array), 'bajo', dtype=object)
    out[heat_array >= config.threshold_med]  = 'medio'
    out[heat_array >= config.threshold_high] = 'alto'
    return out


# ---------------------------------------------------------------------------
# WBGT computation
# ---------------------------------------------------------------------------

def wbgt_from_meteo(T:  np.ndarray,
                    HR: np.ndarray,
                    Rs: np.ndarray | None = None,
                    config: SimConfig = DEFAULT_CFG) -> np.ndarray:
    """
    Compute WBGT (°C) — Australian BoM simplified formula + optional solar
    correction (Option B, ISO 7243).

    Parameters
    ----------
    T  : air temperature (°C), scalar or array
    HR : relative humidity (%), scalar or array
    Rs : global solar radiation (W/m²), optional
    """
    T  = np.asarray(T,  dtype=float)
    HR = np.asarray(HR, dtype=float)
    e  = (HR / 100.0) * 6.105 * np.exp(17.27 * T / (237.7 + T))   # vapor pressure (hPa)
    wbgt_shade = 0.567 * T + 0.393 * e + 3.94
    if Rs is None:
        return wbgt_shade
    return wbgt_shade + config.solar_k * np.sqrt(np.maximum(np.asarray(Rs, dtype=float), 0.0))


def make_wbgt_profile(T_max:  float,
                      HR_min: float,
                      Rs_max: float,
                      config: SimConfig = DEFAULT_CFG) -> np.ndarray:
    """
    Generate a synthetic WBGT profile at 1-minute resolution for the full
    simulation day, using a sinusoidal diurnal cycle.

    Diurnal model
    -------------
    Temperature peaks at 14:00, minimum at ~02:00:
        T(h) = T_mean + T_amp × sin(2π × (h − 8) / 24)
    Relative humidity is the inverse:
        HR(h) = HR_mean − HR_amp × sin(2π × (h − 8) / 24)
    Solar radiation follows a half-sine from 07:00 to 20:00, peak at 13:00:
        Rs(h) = Rs_max × sin(π × (h − 7) / 13)

    Parameters
    ----------
    T_max  : daily maximum air temperature (°C), reached at ~14:00
    HR_min : daily minimum relative humidity (%), reached at ~14:00
    Rs_max : peak solar radiation (W/m²), reached at ~13:00

    Returns
    -------
    np.ndarray of shape (config.n_steps,) with WBGT in °C.
    """
    h_start = config.day_start_min // 60   # 6
    h_end   = config.day_end_min   // 60   # 20
    hours   = np.arange(h_start, h_end + 1, dtype=float)   # [6, 7, …, 20]  15 pts

    # Temperature (peak at 14:00)
    T_min  = T_max - 12.0
    T_mean = (T_max + T_min) / 2.0
    T_amp  = (T_max - T_min) / 2.0
    T_h    = T_mean + T_amp * np.sin(2 * np.pi * (hours - 8.0) / 24.0)

    # Relative humidity (minimum at 14:00, inverse of temperature)
    HR_max  = HR_min + 40.0
    HR_mean = (HR_max + HR_min) / 2.0
    HR_amp  = (HR_max - HR_min) / 2.0
    HR_h    = np.clip(HR_mean - HR_amp * np.sin(2 * np.pi * (hours - 8.0) / 24.0), 10.0, 100.0)

    # Solar radiation (half-sine, 07:00–20:00, peak ~13:00)
    h_rise, h_set = 7.0, 20.0
    Rs_h = np.where(
        (hours >= h_rise) & (hours <= h_set),
        Rs_max * np.sin(np.pi * (hours - h_rise) / (h_set - h_rise)),
        0.0
    )

    # WBGT for each hour
    wbgt_h = wbgt_from_meteo(T_h, HR_h, Rs_h, config)

    # Interpolate hourly → minute resolution
    hours_min = (hours - h_start) * 60          # minutes from day_start: [0, 60, …, 840]
    steps     = np.arange(config.n_steps)        # [0, 1, …, 839]
    return np.interp(steps, hours_min, wbgt_h)


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------

_SNAP_COLS = [
    'agent_id', 'vuln_group', 'purpose_group_model',
    'departure_step', 'dwell_steps', 'arrival_step',
    'status', 'distance_traveled', 'heat_load', 'risk_level',
]


def build_agents(walkers:  pd.DataFrame,
                 ndvi_min: float,
                 ndvi_max: float,
                 config:   SimConfig = DEFAULT_CFG,
                 seed:     int = 42) -> pd.DataFrame:
    """
    Initialize agent state DataFrame from pre-computed walkers.

    Required columns in walkers
    ---------------------------
    agent_id, vuln_group, route_length_m, ndvi_route, purpose_group_model

    Pre-computed per-agent constants (set here, constant for whole simulation)
    --------------------------------------------------------------------------
    ndvi_cooling  : WBGT reduction from route vegetation (°C) = ndvi_alpha × ndvi_norm
                    Uses ndvi_route (length-weighted NDVI along the full route),
                    NOT origin or destination NDVI.
    wbgt_umbral   : agent-specific WBGT threshold (°C) = umbral_base − delta(vuln)
    departure_step : step index when agent leaves home
    dwell_steps   : steps agent stays at destination before returning

    Dynamic state (reset here, updated by step())
    -----------------------------------------------
    status, distance_traveled, heat_load, risk_level, arrival_step
    """
    agents = walkers.copy().reset_index(drop=True)
    n      = len(agents)
    rng    = np.random.default_rng(seed)

    # Departure steps (truncated normal → clipped to [dep_early, dep_late])
    raw     = rng.normal(loc=config.dep_mean_min, scale=config.dep_std_min, size=n)
    dep_abs = np.clip(raw, config.dep_early_min, config.dep_late_min).round().astype(int)
    agents['departure_step'] = dep_abs - config.day_start_min

    # Walk speed per agent (m/min) — varies by vulnerability group
    # Refs: Bohannon (1997), Studenski et al. (2011)
    agents['walk_speed_m_min'] = (
        agents['vuln_group']
        .map(config.walk_speed_by_vuln)
        .fillna(config.walk_speed_m_min)
    )

    # Dwell time per agent (minutes) — varies by purpose of trip
    dwell_steps = np.full(n, config.dwell_min_min, dtype=int)
    for purpose, (dmin, dmax) in config.dwell_by_purpose.items():
        mask = (agents['purpose_group_model'] == purpose).values
        if mask.any():
            dwell_steps[mask] = rng.integers(dmin, dmax + 1, size=int(mask.sum()))
    # Fallback for unrecognised purposes
    unknown = ~agents['purpose_group_model'].isin(config.dwell_by_purpose)
    if unknown.any():
        dwell_steps[unknown.values] = rng.integers(
            config.dwell_min_min, config.dwell_max_min + 1, size=int(unknown.sum())
        )
    agents['dwell_steps'] = dwell_steps

    # NDVI cooling: additive WBGT reduction (°C) for this agent's route
    ndvi_n = ((agents['ndvi_route'] - ndvi_min) / (ndvi_max - ndvi_min + 1e-9)).clip(0, 1)
    agents['ndvi_cooling'] = config.ndvi_alpha * ndvi_n

    # Agent-specific WBGT threshold (°C)
    delta = agents['vuln_group'].map(config.vuln_delta).fillna(0.0)
    agents['wbgt_umbral'] = config.wbgt_umbral_base - delta

    # Dynamic state
    agents['status']            = 'en_casa'
    agents['distance_traveled'] = 0.0
    agents['heat_load']         = 0.0
    agents['risk_level']        = 'bajo'
    agents['arrival_step']      = -1

    return agents


# ---------------------------------------------------------------------------
# Single timestep
# ---------------------------------------------------------------------------

def step(agents:       pd.DataFrame,
         current_step: int,
         wbgt_profile: np.ndarray,
         config:       SimConfig = DEFAULT_CFG) -> pd.DataFrame:
    """
    Advance all agents by one timestep.

    Heat accumulation rule (only while caminando / volviendo):
        WBGT_eff = wbgt_profile[t] − ndvi_cooling[agent]
        dHeat    = max(0, WBGT_eff − wbgt_umbral[agent]) × step_min
        heat_load += dHeat   [°C·min]

    State transitions
    -----------------
    en_casa    → caminando   when departure_step == current_step
    caminando  → en_destino  when distance_traveled >= route_length_m
    en_destino → volviendo   when elapsed dwell time >= dwell_steps
    volviendo  → completado  when distance_traveled >= route_length_m (return)
    """
    wbgt_now = float(wbgt_profile[current_step])

    # 1. Depart from home
    departs = (agents['status'] == 'en_casa') & (agents['departure_step'] == current_step)
    agents.loc[departs, 'status'] = 'caminando'

    # 2. Outbound walk
    walking = agents['status'] == 'caminando'
    if walking.any():
        remaining = agents.loc[walking, 'route_length_m'] - agents.loc[walking, 'distance_traveled']
        advance   = np.minimum(
            agents.loc[walking, 'walk_speed_m_min'] * config.step_min, remaining
        )
        agents.loc[walking, 'distance_traveled'] += advance

        wbgt_eff = wbgt_now - agents.loc[walking, 'ndvi_cooling']
        dHeat    = np.maximum(0.0, wbgt_eff - agents.loc[walking, 'wbgt_umbral']) * config.step_min
        agents.loc[walking, 'heat_load'] += dHeat

        arrived = walking & (agents['distance_traveled'] >= agents['route_length_m'])
        agents.loc[arrived, 'status']            = 'en_destino'
        agents.loc[arrived, 'arrival_step']      = current_step
        agents.loc[arrived, 'distance_traveled'] = agents.loc[arrived, 'route_length_m']

    # 3. Leave destination after dwell time (no heat accumulation — v1 limitation)
    at_dest    = agents['status'] == 'en_destino'
    dwell_done = (
        at_dest &
        (agents['arrival_step'] >= 0) &
        ((current_step - agents['arrival_step']) >= agents['dwell_steps'])
    )
    agents.loc[dwell_done, 'status']            = 'volviendo'
    agents.loc[dwell_done, 'distance_traveled'] = 0.0

    # 4. Return walk (same route reversed)
    returning = agents['status'] == 'volviendo'
    if returning.any():
        remaining_ret = agents.loc[returning, 'route_length_m'] - agents.loc[returning, 'distance_traveled']
        advance_ret   = np.minimum(
            agents.loc[returning, 'walk_speed_m_min'] * config.step_min, remaining_ret
        )
        agents.loc[returning, 'distance_traveled'] += advance_ret

        wbgt_eff_ret = wbgt_now - agents.loc[returning, 'ndvi_cooling']
        dHeat_ret    = np.maximum(0.0, wbgt_eff_ret - agents.loc[returning, 'wbgt_umbral']) * config.step_min
        agents.loc[returning, 'heat_load'] += dHeat_ret

        home = returning & (agents['distance_traveled'] >= agents['route_length_m'])
        agents.loc[home, 'status'] = 'completado'

    agents['risk_level'] = classify_risk(agents['heat_load'].values, config)
    return agents


# ---------------------------------------------------------------------------
# Full-day run
# ---------------------------------------------------------------------------

def run(agents:         pd.DataFrame,
        wbgt_profile:   np.ndarray,
        config:         SimConfig = DEFAULT_CFG,
        snapshot_every: int = 15) -> tuple:
    """
    Run full-day simulation (config.n_steps timesteps of config.step_min minutes).

    Parameters
    ----------
    agents        : initialized DataFrame from build_agents()
    wbgt_profile  : WBGT values (°C) at each step, shape (config.n_steps,)
                    Use make_wbgt_profile() to generate synthetic profiles, or
                    supply your own array from DMC observed data.
    config        : SimConfig instance (same one used in build_agents)
    snapshot_every: save a lightweight snapshot every N steps

    Returns
    -------
    agents    : pd.DataFrame  final agent state
    snapshots : list[pd.DataFrame]  one entry every snapshot_every steps
    """
    if len(wbgt_profile) < config.n_steps:
        raise ValueError(
            f'wbgt_profile has {len(wbgt_profile)} values; '
            f'need at least {config.n_steps} (one per simulation step).'
        )

    snapshots = []
    for s in range(config.n_steps):
        agents = step(agents, s, wbgt_profile, config)

        if s % snapshot_every == 0:
            snap         = agents[_SNAP_COLS].copy()
            snap['step'] = s
            snap['time'] = step_to_time(s, config)
            snapshots.append(snap)

    return agents, snapshots
