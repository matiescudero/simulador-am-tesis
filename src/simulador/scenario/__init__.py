"""
Capa de escenario.

Todo lo que SÍ depende de los parámetros del escenario (meteorología) y por lo
tanto se recalcula por corrida — barato y paralelizable:

- sim_config   : SimConfig (parámetros conductuales, fijos en el banco operacional)
- met_profile  : perfil WBGT diurno desde (T_max, HR_min, Rs_max); multi-día
- heat_load    : motor ABM (build_agents, step, run, run_multiday, clasificación)
- risk_map     : agregación de heat_load por manzana → vector (PCA) + ráster COG (visor)

La lógica del motor es idéntica a src/simulation.py (formulación WBGT aditiva de
la tesis, ndvi_alpha=2.5, sin cooling_strength). Este paquete la reorganiza en
módulos para el pipeline de banco + emulador.
"""
from .sim_config import SimConfig, DEFAULT_CFG, N_STEPS
from .met_profile import wbgt_from_meteo, make_wbgt_profile, make_wbgt_profiles_multiday
from .heat_load import (
    build_agents, step, run, run_multiday,
    classify_risk, step_to_time,
)
from .risk_map import aggregate_to_manzana, write_risk_vector, rasterize_to_cog
from .run import run_one_scenario

__all__ = [
    'SimConfig', 'DEFAULT_CFG', 'N_STEPS',
    'wbgt_from_meteo', 'make_wbgt_profile', 'make_wbgt_profiles_multiday',
    'build_agents', 'step', 'run', 'run_multiday', 'classify_risk', 'step_to_time',
    'aggregate_to_manzana', 'write_risk_vector', 'rasterize_to_cog',
    'run_one_scenario',
]
