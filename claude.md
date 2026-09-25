# Project Context

PhD thesis project (Matías Escudero). Simulates heat exposure in older adults (65+) during
heatwave conditions in Santiago, Chile, using a geospatial agent-based model (ABM), and is
building a **statistical emulator** on top of it so the model can serve risk maps in
milliseconds instead of re-running the simulation.

**This file describes the CURRENT state (post-refactor, July 2026).** If you see an older
copy of this file, or notebooks/code that contradict it, trust this version and the actual
repo structure over memory of earlier sessions.

---

## The big picture: two-layer architecture + emulator

The whole redesign this year exists to enable **"scenario recycling"**: build an emulator
that maps (meteorological conditions) → (risk map) without re-running the ABM per query.

```
Capa invariante (1x por zona, cara)      Capa de escenario (barata, paralelizable)      Emulador
─────────────────────────────────        ──────────────────────────────────────        ────────────
red peatonal, población de agentes   →   perfil WBGT del día + build_agents      →      banco LHS (~300 pts)
rutas dedup, NDVI por ruta               + run_multiday (heat_load acumulado)    →      PCA (mapa → ~12 pesos)
→ artifacts/{zona}/walkers.parquet       → mapa de riesgo por manzana                   Gaussian Process (met → pesos)
                                                                                          → predicción en ms + intervalo
```

- **Capa invariante**: no depende del clima. Se calcula una vez por zona (red, rutas
  Dijkstra deduplicadas por par origen-destino, NDVI a lo largo de cada ruta). Vive en
  `src/simulador/invariant/`.
- **Capa de escenario**: sí depende del clima (T_max, HR_min, Rs_max). Aplica el modelo WBGT
  sobre los `walkers` ya calculados. Vive en `src/simulador/scenario/`. Es lo que se
  paraleliza en el banco de escenarios.
- **Emulador**: LHS (diseño del banco) → correr el ABM en cada punto → PCA (comprime cada
  mapa de miles de manzanas a ~12 pesos) → Gaussian Process (aprende clima→pesos, con
  incertidumbre) → en runtime, GP + PCA inversa reconstruyen el mapa en milisegundos.
  Prototipo funcionando en `notebooks/9.emulador_forward_juguete.ipynb` (skill ~88%,
  cobertura IC95% ~89-91%, validado con leave-one-out).

---

## El modelo de calor (WBGT — NO es el del paper NDVI-multiplicativo)

**Ojo**: el paper WSC 2026 usaba una formulación NDVI multiplicativa
(`heat_load = route_length_m * effective_rate`, `effective_rate = 1 - cooling_strength*ndvi_norm`).
**Esa formulación quedó obsoleta.** Daba resultados irreales (35°C × 0.2 = 7°C no tiene
sentido físico). La tesis usa una formulación **WBGT aditiva** desde entonces:

- `WBGT_eff = WBGT_ambiente - ndvi_cooling` (NDVI resta grados, no multiplica una tasa)
- `ndvi_cooling = ndvi_alpha * ndvi_norm(ruta)`, con `ndvi_alpha = 2.5°C` máximo (Bowler et al.
  2010; estudios LST-NDVI Santiago)
- `wbgt_umbral_agente = wbgt_umbral_base(27°C, NIOSH 2016) - delta(vuln_group)`, con
  `vuln_delta = {baja: 0, media: 1.5, alta: 3.0}` (Kenney & Munce 2003)
- `dHeat = max(0, WBGT_eff - wbgt_umbral_agente) * step_min`, acumulado solo mientras camina
- WBGT desde meteorología: fórmula BoM simplificada + corrección solar (`solar_k=0.04 * sqrt(Rs)`)
- Multi-día: `heat_load` persiste entre días con `nocturnal_decay=0.80` (80% se disipa en la
  noche, 20% queda como estrés residual)
- Velocidad de caminata por vulnerabilidad: `{baja: 68, media: 60, alta: 52} m/min` (Bohannon
  1997; Studenski 2011)

Todo esto vive en `src/simulador/scenario/{sim_config,met_profile,heat_load}.py`, y es una
copia fiel y **verificada bit-idéntica** de `src/simulation.py` (el motor legacy, aún
presente, usado por los notebooks 4/5 originales — no está roto, es el punto de partida
del que se extrajo `src/simulador/`).

---

## Sistema de zonas (config.py) — el switch maestro

`config.py` en la raíz define `ZONA` y todas las rutas derivan de ahí:

```python
ZONAS = {
    'penalolen':     [13122],                                    # paper original, no tocar
    'suroriente':    [13110, 13113, 13118, 13120, 13122],         # piloto (5 comunas)
    'gran_santiago': list(range(13101, 13133)) + [13201, 13401],  # objetivo (34 comunas)
}
```

Todos los archivos se nombran `*_{zona}.*` — cambiar `ZONA='penalolen'` reproduce el paper
sin tocar nada; cada zona nueva escribe en paralelo, nunca sobrescribe. **`ZONA` actual en
el repo: `'suroriente'`**, pero Gran Santiago ya está completamente construido y sus
artefactos existen en `artifacts/gran_santiago/` y `scenarios/gran_santiago/` (generados con
overrides puntuales de ruta, no cambiando el config global — ver scripts en el historial).

Rutas clave que expone `config.py`: `FP_AOI`, `FP_NDVI`, `FP_RED`, `FP_SALUD`, `FP_COMERCIO`,
`FP_VERDES_PTS`, `FP_MANZ_SIM`, `FP_EOD_WALK/PURP` (Santiago-wide, no cambian por zona),
`FP_WALKERS` (el artefacto invariante), y parámetros `POP_SEED`, `UMBRAL_CAMINATA_M`,
`K_DESTINOS`, `AGENTES_POR_AM=10` (1 agente simulado representa a 10 adultos mayores reales
de la misma manzana — comparten vulnerabilidad/ubicación, difieren en si caminan/propósito/
horario), `SALUD_BOOST=2.0` (parche de la EOD hacia salud, marcado como pendiente de
rediseño conceptual, no lo toques sin avisar).

## Escala real medida (no proyectada)

| | Peñalolén | Suroriente (5 com.) | Gran Santiago (34 com.) |
|---|---|---|---|
| Manzanas | ~1.600 | 6.782 | 42.800 |
| Nodos red peatonal | 12.535 | 48.462 | 212.314 |
| Adultos mayores 60+ | — | ~226 mil | 1,19 millones |
| Capa invariante (1 vez) | — | 47 s | **13,5 min (812 s)** |
| Por escenario (3 días) | — | ~19 s | ~65-100 s |

La red peatonal de Gran Santiago se descarga sobre la **huella urbana** (manzanas + buffer
200m + dissolve + simplify), no sobre el polígono administrativo completo (2.270 km²) — eso
evita bajar miles de km² de cordillera sin población (huella real: 822 km², ~4× menos área,
descarga en minutos en vez de horas).

---

## Pipeline de notebooks, en orden

**Setup de zona (0.x)** — genera los inputs para una zona nueva:
- `0.setup_zona` — disuelve comunas → AOI (`aoi_{zona}.gpkg`)
- `0.1.descarga_ndvi` — Sentinel-2 vía Google Earth Engine (composite mediano verano,
  máscara nubes QA60), réplica en Python del script GEE original
- `0.2.descarga_red_peatonal` — OSMnx sobre la huella urbana → `red_peatonal_{zona}.graphml`
- `0.3.censo_a_manzanas` — filtra censo 2024 por CUT → `manzanas_{zona}_paper.parquet`
- `0.4.descarga_destinos` — comercio + áreas verdes de OSM, salud del catastro nacional
  MINSAL (NO es OSM — usar `l_910_v1_establecimientos_de_salud_*.shp` filtrado por
  `CUT_REGION`/`CUT_COMUNA`, ojo que esos campos son **string**, no int)

**Pipeline principal (versiones `_zona`, paramétricas)**:
- `1.vulnerability_index_zona` — índice de vulnerabilidad (StandardScaler, tercios/quintiles)
  → `manzanas_{zona}_sim.parquet`
- `3.ndvi_to_vector_zona` — estadísticas zonales de NDVI por manzana (rasterstats), agrega
  columnas al mismo parquet
- `2.generate_eod_patterns` — **NO se toca por zona**. Las tablas de comportamiento
  (`tabla_walk_prob.csv`, `tabla_purpose_prob.csv`) salen de la EOD Gran Santiago 2012
  completa, filtrada solo por edad 60+, sin filtro geográfico — son válidas para cualquier
  zona de la RM tal cual están.
- `4.walk_simulation` / `5.abm_run` — **notebooks legacy, ya NO se usan para escalar**. Su
  lógica fue extraída y optimizada en `src/simulador/` (ver abajo). Se mantienen como
  referencia/no se han borrado, pero no son el camino para Gran Santiago.

**Validación con datos reales**:
- `6.validacion_estadistica` — exceso de mortalidad RM 65+ (DEIS defunciones), duración de
  ola explica 96% de la varianza en exceso acumulado (R²=0,55), el pico máximo NO es
  significativo — hallazgo central que valida el enfoque multi-día del ABM
- `7.urgencias_calor` — sin señal de calor en urgencias (crisis hipertensiva incluso negativa)
  → los adultos mayores mueren en casa, no buscan atención — hallazgo importante para
  interpretar por qué la validación espacial por comuna falla (ver sección Validación abajo)
- `8.egresos_calor` — N17 (renal aguda) y E87 (electrolitos) suben consistentemente en años
  con olas (+20-50%), correlación fuerte con Tmax_media/p90 del verano (r=0,84-0,88) pero
  NO con el pico (r=0,52 n.s.) — replica el hallazgo de duración del notebook 6

**Emulador**:
- `9.emulador_forward_juguete` — LHS(40) → PCA(12 comp, 98,9% varianza) → GP por componente
  → leave-one-out. Paramétrico por `ZONA` (default `gran_santiago`). Gráficos: mapas base
  PCA, reconstrucción por N componentes, predicho-vs-real, mapa choropleth predicho.

---

## src/simulador/ — el paquete (reemplaza gradualmente a los notebooks 4/5)

```
src/simulador/
├── invariant/                     # capa invariante — CAPA CARA, 1x por zona
│   ├── network.py                 #   load_projected_graph, build_igraph (osmnx→igraph)
│   ├── walkers.py                 #   generate_agents, build_destinations,
│   │                               #   assign_origins_and_dests (K-vecinos ponderados)
│   ├── route_ndvi.py              #   NDVI por ruta vía STRtree (índice espacial)
│   ├── routes.py                  #   compute_routes_dedup — DEDUPLICA por (origin,dest),
│   │                               #   one-to-many Dijkstra por origen — la optimización clave
│   └── build.py                   #   build_invariant() orquesta todo → walkers.parquet
│
└── scenario/                      # capa de escenario — BARATA, paralelizable
    ├── sim_config.py              #   SimConfig (dataclass, parámetros WBGT)
    ├── met_profile.py             #   wbgt_from_meteo, make_wbgt_profile(s)
    ├── heat_load.py               #   build_agents, step, run, run_multiday (motor ABM)
    ├── risk_map.py                #   aggregate_to_manzana, write_risk_vector,
    │                               #   rasterize_to_cog (para GeoServer/visor)
    └── run.py                     #   run_one_scenario, run_scenario_indexed (la unidad
                                    #   que se paraleliza / que llamaría un job SLURM)
```

`igraph` es ~10-50× más rápido que NetworkX para shortest-path masivo — verificado con
benchmark. El motor de `scenario/heat_load.py` es idéntico a `src/simulation.py`, verificado
con test bit-a-bit (mismo `heat_load` final dado el mismo seed).

---

## Entorno: DOS Pythons distintos, no los confundas

- **Kernel de los notebooks / entorno real del proyecto**: conda env **`ox`**
  (`C:\Users\mbell\anaconda3\envs\ox\python.exe`). Tiene **osmnx 1.4.0** (no 2.x),
  geopandas viejo (sin `union_all()`, usar `.unary_union`), scikit-learn, igraph, rasterstats.
- **Python del sistema** (`C:\Users\mbell\AppData\Local\Programs\Python\Python311`): osmnx
  2.1.0, geopandas 1.1.3 — sirve para inspección rápida vía Bash, **NO es el entorno del
  proyecto**. Si escribes código para los notebooks, apunta a la API de osmnx 1.4
  (`ox.geometries_from_polygon`, no `ox.features.features_from_polygon`).
- **Conflicto PROJ (conda + pip)**: rasterio/geemap instalados por pip traen PROJ nuevo, pero
  conda define `PROJ_LIB` apuntando a una `proj.db` vieja → `CRSError`. Fix estándar (ya
  aplicado en los notebooks que usan rasterio/GEE):
  ```python
  import os, importlib.util
  from pathlib import Path
  for pkg, rel in (('rasterio', 'proj_data'), ('pyproj', 'proj_dir/share/proj')):
      spec = importlib.util.find_spec(pkg)
      if spec is not None:
          cand = Path(spec.origin).parent / rel
          if (cand / 'proj.db').exists():
              os.environ['PROJ_LIB'] = os.environ['PROJ_DATA'] = str(cand)
              break
  ```
- **Paralelismo con joblib + geometría = riesgo de memoria**: workers que cargan geometrías
  completas (miles de manzanas) en paralelo pueden agotar RAM
  (`numpy.core._exceptions._ArrayMemoryError`). Fix: workers "livianos" que no cargan
  geometría (solo columnas necesarias), y cargar datos pesados compartidos (ej. CSV de
  defunciones con millones de filas) **una vez fuera** del `Parallel()`, no dentro de cada
  worker. `n_jobs` conservador (2-4), no asumir que más cores siempre ayuda si hay I/O pesado
  compartido.

---

## Datos reales para validación — estructura y limitaciones (importante, ya mapeado)

- **Defunciones DEIS** (`DEFUNCIONES_FUENTE_DEIS_1990_2023...csv`): resolución **diaria**,
  columna `COD_COMUNA` (CUT numérico, comuna de **residencia**), `EDAD_TIPO==1 & EDAD_CANT>=65`
  para filtrar 65+, `DIAG1` para causa. Cubre hasta 2023 (existe un archivo separado
  2024-2026, no siempre necesario).
- **Egresos hospitalarios DEIS** (`EGRE_DATOS_ABIERTOS_{año}.csv` / `EGRESOS_2023.csv` /
  `EGR_DATOS_ABIERTO_{2024,2025}.csv` — nombres de archivo cambiaron entre años, ver
  `ARCHIVOS` dict en notebook 8): **solo agregación ANUAL, sin fecha**. `COMUNA_RESIDENCIA`
  es CUT numérico. **No tienen identificador de establecimiento** (`PERTENENCIA_ESTABLECIMIENTO_SALU`
  es solo bandera público/privado, no un hospital). Esta es la limitación que motivó pedir
  datos con fecha exacta vía Ley de Transparencia (ver abajo).
- **Atenciones de Urgencia DEIS** (`AtencionesUrgencia{2018,2019,2022,2023,2024,2025}.csv`):
  resolución **diaria** (`fecha`, formato `DD/MM/YYYY`), por `IdEstablecimiento` (formato
  `13-XXX` para RM) y `IdCausa` (ver diccionario `DICCIONARIO_ATENCIONES_DE_URGENCIA.xlsx`,
  hoja `Anexo 1`). Columna 65+ es `De_65_y_mas`. **No tiene comuna de residencia del
  paciente** — solo dónde consultó, no dónde vive. Diccionario `Anexo 2` (hoja del mismo
  xlsx) da el crosswalk `IdEstablecimiento → Nombre`, que se puede cruzar por **nombre**
  (no por código — los sistemas de códigos NO coinciden) contra el catastro nacional
  georreferenciado (`l_910_v1_establecimientos_de_salud_*.shp`, campo `NOMBRE`) — cruce
  validado: 93-100% de cobertura según el año.
- **Catastro de salud MINSAL** (`l_910_v1_establecimientos_de_salud_diciembre_2025.shp`):
  nacional, georreferenciado (lat/lon), 5.181 establecimientos. `CUT_REGION`/`CUT_COMUNA` son
  **string**, no int — comparar con `=='13'`, no `==13`. Este es el archivo fuente tanto para
  `FP_SALUD` (destinos del ABM) como para geolocalizar establecimientos SADU de urgencias.

## Estado de la validación empírica (importante — resultado nulo, bien documentado)

Se probó exhaustivamente si el riesgo simulado (o incluso el índice de vulnerabilidad
estático, sin ABM) correlaciona espacialmente con datos reales de salud, en dos niveles de
agregación:

- **Nivel comuna** (n=34, Gran Santiago): mortalidad por causas sensibles al calor (5 eventos
  reales: Feb2023, Ene2019, Dic2022, Ene2024, Feb2025, con meteorología real día a día vía
  Open-Meteo/ERA5) y egresos N17+E87 anuales — **sin correlación significativa en ningún
  caso**. Un pooled naive de los 5 eventos parecía significativo (rho=0,31, p<0,001) pero era
  artefacto de severidad entre-evento; al estandarizar dentro de evento, rho≈0,02, n.s.
- **Nivel establecimiento** (n=25, Gran Santiago): urgencias por causa cardiovascular
  específica (IAM, AVE, crisis hipertensiva, arritmia), con crosswalk geolocalizado y
  baseline correcto (mismos días calendario, año 2018 sin ola vs año evento) — tampoco
  significativo en ninguna causa individual.

**Diagnóstico**: no parece ser un bug del modelo — el gradiente de vulnerabilidad simulado es
grande y correcto (comunas ricas heat_mean~0,2, comunas vulnerables heat_mean~10, gradiente
~65×, exactamente donde debería estar geográficamente). El problema es el **"small-area
estimation problem"/falacia ecológica**: con conteos de salud pequeños por unidad espacial
(comuna o establecimiento), el ruido de Poisson domina sobre cualquier señal real. La
validación **temporal** (RM completa, notebooks 6 y 8) sí es sólida — es la validación
**espacial fina** la que no está resuelta todavía.

**Camino hacia adelante** (en progreso): pedir egresos con fecha exacta + establecimiento vía
Ley de Transparencia (agregado por establecimiento×día×grupo_edad×diagnóstico, siguiendo
consejo de Cristian García, ex-Director de Epidemiología MINSAL, contacto del usuario) — con
eso se puede repetir la validación temporal (que sí funciona) a resolución de establecimiento
en vez de forzar la espacial comunal (que no funciona con n pequeño). Pendiente coordinar
videollamada con él.

---

## Lo que falta del roadmap del emulador (6 pasos, ver progreso)

1. ✅ Capa invariante (`src/simulador/invariant/`)
2. ✅ Capa de escenario (`src/simulador/scenario/`), verificada bit-idéntica al legacy
3. 🔄 CLI de orquestación (`build-invariant`, `run-scenario`) — parcialmente hecho vía
   `run.py: run_one_scenario/run_scenario_indexed`, falta el CLI formal
4. ⬜ Banco LHS en SLURM (job arrays, cluster universidad 32 cores/64GB) — prototipo local
   con joblib ya probado y funcionando (`notebooks/9`), falta portar a SLURM
5. 🔄 PCA + emulador GP — prototipo de juguete funcionando (nb9), falta banco completo
   (~300 escenarios) y entrenamiento formal
6. ⬜ FastAPI `serve/api.py` + visor (GeoServer/MapStore) — no empezado

## Reproducibilidad / portabilidad (decisiones tomadas, para cuando se implemente)

- Formato salida escenario: **vector por manzana (parquet, para PCA)** + **COG (Cloud-Optimized
  GeoTIFF, para servir)** — dos representaciones del mismo resultado, no una sola.
- Paralelización: **SLURM job arrays** (no Dask) — trabajo estático, independiente, sin
  comunicación entre tareas, retry granular gratis.
- Entorno reproducible: **conda-lock** (environment.yml + lock file), no Docker completo —
  ya se sufrió el conflicto PROJ/GDAL una vez, conda-lock lo fija sin la complejidad de
  contenedores completos.
- Seeds: un seed por escenario derivado del índice LHS (`seed = base_seed + scenario_idx`),
  reproducibilidad total.

---

## Coding preferences (siguen vigentes)

- Verificar contra el repo real antes de asumir estructura — este archivo puede desactualizarse.
- Cambios controlados y explicados, no refactors amplios sin pedirlos.
- Preservar nombres de columnas existentes salvo razón de peso.
- Antes de editar, inspeccionar variables/columnas/estructuras reales (no asumir desde memoria).
- Los notebooks paramétricos (`*_zona`) son la versión viva; los originales sin sufijo
  (`1.vulnerability_index.ipynb`, `3.ndvi_to_vector.ipynb`) se conservan intactos como
  reproducción exacta del paper — no tocarlos.
