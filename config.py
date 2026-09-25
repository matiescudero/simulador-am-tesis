"""
Configuración central de la zona de estudio del simulador.

Cambiar ZONA acá redefine todos los archivos de entrada/salida del pipeline
(notebooks 0.x en adelante). Los archivos históricos de Peñalolén siguen la
misma convención de nombres (`*_penalolen.*`), por lo que ZONA = 'penalolen'
reproduce exactamente el pipeline del paper sin sobrescribir nada.

Uso en notebooks:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path('..').resolve()))
    from config import *
"""
from pathlib import Path

# ── Zona de estudio activa ────────────────────────────────────────────────────
ZONA = 'suroriente'          # 'penalolen' | 'suroriente' | 'gran_santiago'

# Comunas por zona (CUT censo 2024)
ZONAS = {
    # Paper original — no tocar
    'penalolen': [13122],

    # Piloto de escalamiento: Peñalolén + vecinas del sector suroriente
    'suroriente': [
        13110,  # La Florida
        13113,  # La Reina
        13118,  # Macul
        13120,  # Ñuñoa
        13122,  # Peñalolén
    ],

    # Objetivo final: Provincia de Santiago + Puente Alto + San Bernardo
    'gran_santiago': list(range(13101, 13133)) + [13201, 13401],
}

COMUNAS = ZONAS[ZONA]

# ── Sistemas de referencia ────────────────────────────────────────────────────
CRS_GEO  = 'EPSG:4326'   # WGS84 (GEE, OSM)
CRS_PROJ = 'EPSG:32719'  # UTM 19S (métrico, para distancias/áreas)

# ── Parámetros NDVI (Sentinel-2, composite mediano de verano) ─────────────────
NDVI_START     = '2024-10-01'
NDVI_END       = '2025-03-31'
NDVI_MAX_CLOUD = 20          # % máximo de nubes por escena
NDVI_SCALE     = 10          # resolución en metros
EE_PROJECT     = 'electric-cosine-314704'

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
DATA_RAW  = ROOT / 'data' / 'raw'
DATA_PROC = ROOT / 'data' / 'processed'
OUT_DIR   = ROOT / 'outputs'

# Entradas externas (independientes de la zona)
FP_CARTO_COMUNAL = DATA_RAW / 'Cartografía_censo2024_R13_Comunal.parquet'
FP_CARTO_ZONAL   = DATA_RAW / 'Cartografía_censo2024_R13_Zonal.parquet'

# Microdatos y cartografía censal completa (fuera del repo, R13 completa)
CENSO_DIR          = Path(r'C:\Users\mbell\Desktop\UNIVERSIDAD\DOCTORADO\TESIS\DATOS\CENSO_2024')
FP_CENSO_MANZANAS  = CENSO_DIR / 'Cartografía_censo2024_R13_Manzanas.parquet'

# Archivos dependientes de la zona (generados por los notebooks 0.x)
FP_AOI  = DATA_RAW / f'aoi_{ZONA}.gpkg'                    # nb 0
FP_NDVI = DATA_RAW / f'ndvi_{ZONA}.tif'                    # nb 0.1
FP_RED  = DATA_RAW / f'red_peatonal_{ZONA}.graphml'        # nb 0.2

# Capas de destino del ABM (nb 0.4)
FP_SALUD      = DATA_RAW / f'establecimientos_salud_{ZONA}.gpkg'          # MINSAL (no OSM)
FP_COMERCIO   = DATA_RAW / f'comercio_{ZONA}.gpkg'                        # OSM
FP_VERDES_RAW = DATA_RAW / f'areas_verdes_{ZONA}.gpkg'                    # OSM (polígonos)
FP_VERDES_PTS = DATA_PROC / 'destinations' / f'verdes_points_{ZONA}.gpkg' # puntos destino

# Archivos del pipeline principal
FP_MANZ_PAPER = DATA_PROC / f'manzanas_{ZONA}_paper.parquet'         # nb 0.3
FP_MANZ_SIM   = DATA_PROC / 'censo' / f'manzanas_{ZONA}_sim.parquet' # nb 1 + 3

# Tablas EOD por persona, agrupadas por el vuln_group censal de la manzana del hogar
# (src/simulador/invariant/eod_tables.py). Las tablas por viaje del nb 2
# (tabla_walk_prob.csv / tabla_purpose_prob.csv) quedan como referencia histórica.
FP_EOD_DB   = ROOT.parent.parent / 'DATOS' / 'EOD' / 'EOD GRAN SANTIAGO' / 'base_datos_eodStgo_2012' / 'base_datos_eodStgo_2012.accdb'
FP_EOD_WALK = DATA_PROC / 'eod' / 'tabla_walk_prob_persona.csv'
FP_EOD_PURP = DATA_PROC / 'eod' / 'tabla_purpose_prob_persona.csv'
FP_EOD_AGE  = DATA_PROC / 'eod' / 'tabla_age_prob_persona.csv'
FP_EOD_DIST = DATA_PROC / 'eod' / 'tabla_dist_walk_persona.csv'
# Cadena viaje → destino → modo (MODE_CHOICE='eod_logit')
FP_EOD_TRIP       = DATA_PROC / 'eod' / 'tabla_trip_prob_persona.csv'
FP_EOD_PURP_TRIP  = DATA_PROC / 'eod' / 'tabla_purpose_trip_persona.csv'
FP_EOD_CAR        = DATA_PROC / 'eod' / 'tabla_car_prob_persona.csv'
FP_EOD_CAR_COMUNA = DATA_PROC / 'eod' / 'tabla_car_prob_comuna.csv'
FP_EOD_DIST_TRIP  = DATA_PROC / 'eod' / 'tabla_dist_trip_persona.csv'
FP_EOD_MODE_LOGIT = DATA_PROC / 'eod' / 'tabla_modo_caminata_logit.csv'
FP_EOD_WALK_LOGIT = DATA_PROC / 'eod' / 'tabla_walk_access_logit.csv'   # MODE_CHOICE='access_logit'

# Fracción de copa de árboles (>=3 m) a 3 m, EPSG:32719 (src/simulador/invariant/canopy.py)
FP_CANOPY   = DATA_PROC / 'canopy' / f'canopy_frac_3m_{ZONA}.tif'

# ── Capa invariante al escenario (src/simulador/invariant) ────────────────────
# Artefactos que se calculan 1 vez por zona y se reutilizan en todos los escenarios.
ARTIFACTS_DIR = ROOT / 'artifacts' / ZONA
FP_WALKERS    = ARTIFACTS_DIR / 'walkers.parquet'   # población + rutas + ndvi_route
FP_GRAPH_IG   = ARTIFACTS_DIR / 'graph_ig.joblib'   # grafo igraph + índice de nodos

# Parámetros de la capa invariante (población de agentes + ruteo peatonal)
POP_SEED          = 42     # semilla de la población — invariante, ≠ semilla de escenario
UMBRAL_CAMINATA_M = 4500   # descarta rutas peatonalmente implausibles (m); ~1,5 × 3 km en línea recta
MODE_CHOICE       = 'access_logit'  # 'access_logit' (logit EOD por persona: accesibilidad, auto, vuln, edad)
                                    # | 'eod_logit' (viaje→destino→modo; no reproduce el gradiente) | None (prob. fija)
WALK_SHARE_TARGET = 0.287           # EOD: % de personas 60+ con >=1 viaje a pie (3 propósitos), día laboral
DEST_CHOICE       = 'eod_distance'  # 'eod_distance' (distancia objetivo EOD) | 'nearest_k' (K más cercanos)
K_DESTINOS        = 5      # solo DEST_CHOICE='nearest_k': K destinos más cercanos para muestreo ponderado
AGENTES_POR_AM    = 1      # 1 agente por cada N adultos mayores 60+ (1 = uno por persona)
SALUD_BOOST       = 1.0    # multiplicador de la prob. EOD de viaje a salud (1 = sin ajuste)

if __name__ == '__main__':
    print(f'Zona activa : {ZONA}')
    print(f'Comunas     : {len(COMUNAS)} -> {COMUNAS}')
    print(f'AOI         : {FP_AOI}')
    print(f'NDVI        : {FP_NDVI}')
    print(f'Red peatonal: {FP_RED}')
    print(f'Manzanas sim: {FP_MANZ_SIM}')
