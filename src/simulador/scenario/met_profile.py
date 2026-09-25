"""
Perfil WBGT diurno desde variables meteorológicas del escenario.

Movido de src/simulation.py. `make_wbgt_profile` genera el perfil de un día a
partir de (T_max, HR_min, Rs_max); `make_wbgt_profiles_multiday` replica el
perfil para los N días consecutivos del banco operacional (mismos valores met,
el carry-over inter-día lo maneja run_multiday con decay nocturno).
"""
from __future__ import annotations

import numpy as np

from .sim_config import SimConfig, DEFAULT_CFG


def wbgt_from_meteo(T, HR, Rs=None, config: SimConfig = DEFAULT_CFG) -> np.ndarray:
    """
    WBGT (°C) — fórmula simplificada (Australian BoM) + corrección solar
    opcional (Opción B, ISO 7243).

    T : temperatura del aire (°C) · HR : humedad relativa (%) · Rs : radiación
    solar global (W/m², opcional).
    """
    T  = np.asarray(T,  dtype=float)
    HR = np.asarray(HR, dtype=float)
    e  = (HR / 100.0) * 6.105 * np.exp(17.27 * T / (237.7 + T))   # presión vapor (hPa)
    wbgt_shade = 0.567 * T + 0.393 * e + 3.94
    if Rs is None:
        return wbgt_shade
    return wbgt_shade + config.solar_k * np.sqrt(np.maximum(np.asarray(Rs, dtype=float), 0.0))


def make_wbgt_profile(T_max: float,
                      HR_min: float,
                      Rs_max: float,
                      config: SimConfig = DEFAULT_CFG) -> np.ndarray:
    """
    Perfil WBGT a resolución de 1 minuto para el día de simulación.

    config.diurnal_mode:
      'era5'     forma horaria empírica de los días calurosos de Santiago
                 (T peak ~16:00, mínimo ~07:00, amplitud 15 °C; ver SimConfig).
      'sinusoid' ciclo sinusoidal simétrico, T peak 14:00, amplitud 12 °C.

    Returns: np.ndarray de forma (config.n_steps,) con WBGT en °C.
    """
    h_start = config.day_start_min // 60
    h_end   = config.day_end_min   // 60
    hours   = np.arange(h_start, h_end + 1, dtype=float)

    if config.diurnal_mode == 'era5':
        idx  = hours.astype(int) % 24
        T_h  = T_max - config.diurnal_T_range * (1.0 - np.asarray(config.diurnal_T_shape)[idx])
        HR_h = np.clip(HR_min + config.diurnal_RH_range * np.asarray(config.diurnal_RH_shape)[idx],
                       0.0, 100.0)
        Rs_h = Rs_max * np.asarray(config.diurnal_Rs_shape)[idx]
    elif config.diurnal_mode == 'sinusoid':
        T_min  = T_max - 12.0
        T_mean = (T_max + T_min) / 2.0
        T_amp  = (T_max - T_min) / 2.0
        T_h    = T_mean + T_amp * np.sin(2 * np.pi * (hours - 8.0) / 24.0)

        HR_max  = HR_min + 40.0
        HR_mean = (HR_max + HR_min) / 2.0
        HR_amp  = (HR_max - HR_min) / 2.0
        HR_h    = np.clip(HR_mean - HR_amp * np.sin(2 * np.pi * (hours - 8.0) / 24.0), 10.0, 100.0)

        h_rise, h_set = 7.0, 20.0
        Rs_h = np.where(
            (hours >= h_rise) & (hours <= h_set),
            Rs_max * np.sin(np.pi * (hours - h_rise) / (h_set - h_rise)),
            0.0,
        )
    else:
        raise ValueError(f'diurnal_mode desconocido: {config.diurnal_mode!r}')

    wbgt_h = wbgt_from_meteo(T_h, HR_h, Rs_h, config)

    # Interpolar horario → minuto
    hours_min = (hours - h_start) * 60
    steps     = np.arange(config.n_steps)
    return np.interp(steps, hours_min, wbgt_h)


def make_wbgt_profiles_multiday(T_max: float,
                                HR_min: float,
                                Rs_max: float,
                                n_dias: int = 3,
                                config: SimConfig = DEFAULT_CFG) -> list:
    """
    Lista de N perfiles WBGT idénticos (un escenario del banco = mismos valores
    met en N días consecutivos). El acumulado inter-día lo maneja run_multiday
    con el decay nocturno.
    """
    profile = make_wbgt_profile(T_max, HR_min, Rs_max, config)
    return [profile for _ in range(n_dias)]
