"""
SimConfig — parámetros conductuales del ABM.

Movido verbatim de src/simulation.py. Son los parámetros FIJOS en el banco
operacional (velocidad de caminata, umbrales WBGT, cooling NDVI aditivo). Lo
que varía entre escenarios del banco es la meteorología, que NO vive acá sino
que entra como argumento a make_wbgt_profile (T_max, HR_min, Rs_max).

Formulación WBGT aditiva de la tesis: ndvi_alpha=2.5 (reducción aditiva, no
multiplicativa), sin cooling_strength (ese era del paper WSC previo).
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    """
    Parámetros de simulación en un solo objeto. Pasar la misma instancia a
    build_agents(), step() y run() garantiza consistencia dentro de un escenario.
    """

    # ── Ventana temporal ───────────────────────────────────────────────────
    day_start_min: int = 6 * 60        # 06:00
    day_end_min:   int = 22 * 60       # 22:00 (EOD: salidas hasta ~21 h + permanencia + regreso)
    step_min:      int = 1             # minutos por paso

    # ── Fisiología del agente: por tramo de edad ('age') o por vuln_group ('vuln') ─
    # 'age' asigna velocidad y umbral según la edad del agente (columna age_band de
    # la capa invariante); 'vuln' reproduce la versión anterior, que los asignaba
    # según el índice territorial de la manzana. Los valores por edad reutilizan
    # los tres niveles previos — pendiente contrastarlos con literatura por edad.
    physiology_by: str = 'age'
    walk_speed_by_age: dict = field(
        default_factory=lambda: {'60-69': 68.0, '70-79': 60.0, '80+': 52.0}
    )
    threshold_delta_by_age: dict = field(
        default_factory=lambda: {'60-69': 0.0, '70-79': 1.5, '80+': 3.0}
    )

    # ── Velocidad de caminata ──────────────────────────────────────────────
    walk_speed_m_min: float = 62.0     # fallback si el grupo no está en el dict
    walk_speed_by_vuln: dict = field(
        default_factory=lambda: {'baja': 68.0, 'media': 60.0, 'alta': 52.0}
    )
    # Refs: Bohannon (1997) Phys Ther; Studenski et al. (2011) JAMA.
    # baja 68 (~4.1 km/h), media 60 (~3.6), alta 52 (~3.1, marcha frágil <1.0 m/s).

    # ── Hora de salida ─────────────────────────────────────────────────────
    # 'eod2012': hora sorteada de la distribución empírica por propósito y minuto
    #            uniforme dentro de la hora. Viajes a pie de personas 60+, días
    #            laborales normales, EOD Santiago 2012 (SECTRA), ponderados por
    #            factor de expansión; sin viajes de regreso a casa.
    #            data/processed/eod2012_pesos_horarios_salida_60mas_caminata.csv
    # 'normal':  normal truncada (supuesto original, sin respaldo empírico).
    dep_mode: str = 'eod2012'
    dep_hours: tuple = tuple(range(6, 21))          # 06:00 … 20:59
    dep_hour_weights: dict = field(
        default_factory=lambda: {
            'salud': (0.032, 0.087, 0.176, 0.164, 0.194, 0.072, 0.061, 0.041,
                      0.031, 0.072, 0.020, 0.031, 0.010, 0.000, 0.010),
            'comercio': (0.000, 0.001, 0.009, 0.056, 0.195, 0.156, 0.137, 0.046,
                         0.018, 0.026, 0.040, 0.146, 0.122, 0.036, 0.011),
            'areas_verdes': (0.002, 0.049, 0.042, 0.030, 0.063, 0.069, 0.081, 0.078,
                             0.060, 0.120, 0.102, 0.087, 0.105, 0.075, 0.036),
        }
    )
    dep_mean_min:  int = 9 * 60        # 09:00  (solo dep_mode='normal')
    dep_std_min:   int = 60
    dep_early_min: int = 7 * 60        # 07:00
    dep_late_min:  int = 12 * 60       # 12:00

    # ── Perfil diurno de T, HR y Rs ────────────────────────────────────────
    # 'era5':     forma horaria media de los días de verano con Tmax >= 30 °C en
    #             Santiago (ERA5 vía Open-Meteo, DJF 2015–2025, 389 días), escalada:
    #             T(h) = T_max − rango_T·(1 − forma_T(h)); HR(h) = HR_min + rango_HR·forma_HR(h);
    #             Rs(h) = Rs_max·forma_Rs(h).
    #             data/processed/era5_perfil_diurno_dias_calurosos_santiago.csv
    # 'sinusoid': sinusoide simétrica con peak a las 14:00 (perfil original).
    diurnal_mode: str = 'era5'
    diurnal_T_range: float = 15.0      # °C, amplitud media Tmin(mañana)→Tmax en días calurosos
    diurnal_RH_range: float = 42.5     # puntos %, HRmax(mañana) − HRmin(tarde)
    diurnal_T_shape: tuple = (
        0.184, 0.135, 0.097, 0.064, 0.042, 0.024, 0.008, 0.000, 0.072, 0.210, 0.372, 0.555,
        0.723, 0.857, 0.946, 0.992, 1.000, 0.969, 0.890, 0.766, 0.607, 0.453, 0.339, 0.255)
    diurnal_RH_shape: tuple = (
        0.755, 0.821, 0.877, 0.922, 0.948, 0.971, 0.990, 1.000, 0.894, 0.732, 0.546, 0.352,
        0.198, 0.085, 0.025, 0.000, 0.005, 0.030, 0.091, 0.190, 0.328, 0.500, 0.633, 0.737)
    diurnal_Rs_shape: tuple = (
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.002, 0.070, 0.252, 0.465, 0.665,
        0.831, 0.946, 1.000, 0.989, 0.913, 0.779, 0.603, 0.396, 0.185, 0.028, 0.000, 0.000)

    # ── Permanencia en destino (uniforme, minutos) ─────────────────────────
    dwell_min_min: int = 20
    dwell_max_min: int = 60
    dwell_by_purpose: dict = field(
        default_factory=lambda: {
            'salud':        (45, 90),
            'comercio':     (25, 50),
            'areas_verdes': (15, 40),
        }
    )

    # ── Modelo WBGT ────────────────────────────────────────────────────────
    wbgt_umbral_base: float = 27.0
    # °C — umbral base para adulto mayor, actividad ligera exterior.
    # NIOSH (2016); Kenney & Munce (2003).
    vuln_delta: dict = field(
        default_factory=lambda: {'baja': 0.0, 'media': 1.5, 'alta': 3.0}
    )
    # Reducción de umbral por vulnerabilidad (°C). Kenney & Munce (2003).

    # ── Enfriamiento NDVI (aditivo) ────────────────────────────────────────
    # Con sombra de arbolado explícita (shade_route en walkers), el NDVI solo
    # representa el enfriamiento del aire por vegetación: 1,0 °C máximo
    # (Bowler et al. 2010, ~1 °C de temperatura del aire en parques). Sin sombra
    # explícita, la versión anterior usaba 2,5 °C para cubrir ambos efectos.
    ndvi_alpha: float = 1.0

    # ── WBGT: método y sombra ──────────────────────────────────────────────
    # 'liljegren': WBGT físico (Liljegren et al. 2008, vía pywbgt) a partir de T,
    #              HR, radiación, viento, presión y posición solar; al sol y a la
    #              sombra (radiación × shade_transmittance).
    # 'bom':       fórmula simplificada BoM + corrección solar 0,04·√Rs (anterior).
    wbgt_method: str = 'liljegren'
    shade_transmittance: float = 0.15   # fracción de radiación bajo copa de árbol
    site_lat: float = -33.45
    site_lon: float = -70.65
    ref_date: str = '2019-01-26'        # fecha de referencia para la geometría solar
    utc_offset_h: int = -3              # hora de verano de Chile
    surface_pressure_hpa: float = 949.9 # ERA5, días calurosos
    # Viento medio a 10 m por hora (m/s), días de verano con Tmax >= 30 °C, ERA5 2015–2025.
    # data/processed/era5_viento_presion_dias_calurosos_santiago.csv
    diurnal_wind_10m: tuple = (
        1.21, 0.93, 0.76, 0.77, 0.80, 0.83, 0.86, 0.91, 0.75, 0.89, 1.16, 1.55,
        2.28, 3.00, 3.67, 4.13, 4.39, 4.55, 4.58, 4.28, 3.43, 2.58, 2.05, 1.55)

    # ── Umbrales de clasificación de riesgo (°C·min sobre umbral del agente) ─
    # Solo para risk_level en snapshots/visualización. Los resultados del banco se
    # reportan en continuo (heat_load): estos umbrales no están calibrados contra
    # datos de salud y, con distancias EOD, clasifican a >85% como riesgo alto.
    threshold_med:  float = 30.0
    threshold_high: float = 60.0

    # ── Corrección solar WBGT (Opción B, ISO 7243) ─────────────────────────
    solar_k: float = 0.04
    # WBGT_outdoor = WBGT_shade + solar_k × sqrt(Rs). ~1.2 °C a Rs=850 W/m².

    @property
    def n_steps(self) -> int:
        return (self.day_end_min - self.day_start_min) // self.step_min


DEFAULT_CFG = SimConfig()
N_STEPS = DEFAULT_CFG.n_steps
