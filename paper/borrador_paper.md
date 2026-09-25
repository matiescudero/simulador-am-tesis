# Un algoritmo de emulación estadística para reducir el costo de simulaciones
# basadas en agentes: caso de estudio de olas de calor en adultos mayores

*Borrador — Matías Escudero, [fecha]*

> **Nota de enfoque (según indicación de la profesora, reunión 31 jul 2026):**
> El foco del paper NO es "un simulador de olas de calor". Es un **algoritmo para
> reducir el costo computacional de simulaciones basadas en agentes**, aplicado
> a un caso de estudio de exposición al calor en adultos mayores. La estructura
> sigue exactamente esa lógica: (1) modelar y validar el caso de estudio, (2)
> mostrar cómo el algoritmo propuesto (emulador PCA+GP) rinde contra alternativas
> en tiempo de ejecución y calidad.

---

## Abstract (borrador)

Las simulaciones basadas en agentes (ABM) espaciales son una herramienta clave
para estudiar fenómenos de salud pública urbana, pero su costo computacional
limita su uso en aplicaciones que requieren respuesta casi inmediata — por
ejemplo, sistemas de alerta temprana ante olas de calor. Presentamos un
algoritmo de emulación estadística (diseño Latin Hypercube + compresión por
componentes principales + Gaussian Process) que reduce el tiempo de una
consulta de ~[65-100] segundos a ~2,3 milisegundos (una aceleración de
~30.000-40.000×), manteniendo una fidelidad de 87,5% (skill) frente al modelo
original, validada mediante leave-one-out. Aplicamos el algoritmo a un caso de
estudio de exposición al calor en adultos mayores (65+) en el Gran Santiago,
Chile (34 comunas, 1,19 millones de adultos mayores), comparando el proceso
Gaussiano contra Random Forest, interpolación RBF y Polynomial Chaos Expansion.
El proceso Gaussiano obtiene la mejor precisión de las cuatro alternativas y es
el único que entrega incertidumbre puntual calibrada (92% de cobertura del
intervalo de confianza al 95%), una propiedad indispensable para sistemas de
decisión en tiempo real.

---

## 1. Introducción

- Motivación: olas de calor y mortalidad en adultos mayores — contexto real
  reciente (ej. evento de calor en París 2026, ~85% de las muertes en adultos
  mayores — usar como gancho, con cita de prensa/fuente oficial).
- El problema computacional: los ABM espaciales de exposición son caros de
  correr (en nuestro caso, [65-100] s por escenario de 3 días en Gran
  Santiago), lo que impide su uso en un sistema operacional que deba responder
  a un pronóstico meteorológico en tiempo real.
- La propuesta: un algoritmo de emulación (surrogate model) que aprende la
  relación entre condiciones meteorológicas y el mapa de riesgo resultante,
  sin volver a correr el ABM por cada consulta.
- Contribución dual:
  1. Un algoritmo general de reducción de costo para ABM espaciales de salida
     de alta dimensión (aplicable más allá de este caso de estudio).
  2. Su aplicación y validación en un caso de estudio real y actual: exposición
     al calor en adultos mayores, Gran Santiago.

---

## 2. Trabajo relacionado

**Emulación de simuladores costosos (fundamento metodológico):**
- Sacks et al. (1989), *Design and Analysis of Computer Experiments* — verificar cita exacta.
- Kennedy & O'Hagan (2001), *Bayesian calibration of computer models* — verificar cita exacta.
- Higdon et al. (2008), *Computer Model Calibration Using High-Dimensional
  Output*, JASA — base directa del pipeline PCA+GP para salida espacial de
  alta dimensión.
- Gramacy (2020), *Surrogates: Gaussian Process Modeling, Design, and
  Optimization*.

**Emulación de ABM específicamente (precedente directo):**
- *Towards real-time predictions using emulators of agent-based models*,
  Journal of Simulation (2024).
- *Combining an agent-based model with Gaussian process emulation* (variante
  Ómicron, Noruega), Epidemics (2026).
- *Gaussian process emulation of spatio-temporal outputs of a 2D inland flood
  model*, Water Research (2022).

**Comparación GP vs. otras técnicas de emulación (justifica la elección):**
- *A comparison of Gaussian processes and polynomial chaos emulators*, Phil.
  Trans. Royal Society A (2025).
- Owen et al., *A comparison of polynomial chaos and Gaussian process
  emulation for UQ in computer experiments*.
- *Emulation of CPU-demanding reactive transport models: GP vs PCE vs redes
  neuronales*.

**Extensiones / trabajo futuro citable:**
- *Gaussian Process Latent Factor Regression for Low-Data Problems with
  High-Dimensional Outputs* (2026).
- *Multivariate Gaussian Process Emulators With Nonseparable Covariance
  Structures* (coregionalización).

**Dominio de aplicación (vulnerabilidad al calor urbano — el vacío que este
trabajo llena, combinando ABM + emulador + tiempo real):**
- *Urban-Hazard Risk Analysis: Heat-Related Risks in the Elderly in Major
  Italian Cities*.
- *A Novel Urban Heat Vulnerability Analysis: ML + Remote Sensing*, Remote
  Sensing (2024).

*(Pendiente: confirmar referencias exactas de Sacks 1989 y Kennedy & O'Hagan
2001 — citadas de memoria general del campo, no verificadas por búsqueda.)*

---

## 3. Caso de estudio: modelo de exposición al calor

### 3.1 Modelo de agentes

- Población: 1 agente representa a `AGENTES_POR_AM=10` adultos mayores de la
  misma manzana censal (comparten vulnerabilidad/ubicación, difieren en
  decisión de salir/propósito/horario, muestreado desde la Encuesta
  Origen-Destino Gran Santiago 2012, filtrada a 60+).
- Arquitectura de dos capas:
  - **Capa invariante** (no depende del clima): red peatonal, rutas
    peatonales deduplicadas por par origen-destino, NDVI a lo largo de cada
    ruta. Se calcula una vez por zona urbana.
  - **Capa de escenario** (depende del clima): modelo WBGT aditivo aplicado
    sobre la capa invariante.
- Modelo de calor — WBGT aditivo (NO multiplicativo, ver nota metodológica):
  - `WBGT_eff = WBGT_ambiente - ndvi_cooling`, `ndvi_cooling = ndvi_alpha *
    ndvi_norm(ruta)`, `ndvi_alpha = 2.5°C` (Bowler et al. 2010).
  - `wbgt_umbral_agente = 27°C (NIOSH 2016) - delta(vuln_group)`, con
    `delta = {baja: 0, media: 1.5, alta: 3.0}` (Kenney & Munce 2003).
  - Acumulación multi-día con `nocturnal_decay = 0.80`.

### 3.2 Escala del caso de estudio

| | Peñalolén (piloto original) | Gran Santiago (caso de estudio final) |
|---|---|---|
| Comunas | 1 | 34 |
| Manzanas censales | ~1.600 | 42.800 |
| Nodos de red peatonal | 12.535 | 212.314 |
| Adultos mayores 65+ | — | 1,19 millones |
| Tiempo capa invariante (1 vez) | — | 13,5 min |
| Tiempo por escenario (3 días) | — | 65-100 s |

---

## 4. Validación del caso de estudio

### 4.1 Validación temporal (RM completa) — resultado fuerte

Usando datos reales de defunciones DEIS (RM, 65+, 2005-2023): el exceso de
mortalidad durante olas de calor documentadas correlaciona con la **duración**
de la ola más que con su intensidad pico. Regresión múltiple estandarizada
(exceso acumulado ~ Tmax_max + n_dias): R²=0,55; la duración explica ~96% de
la varianza relativa; el Tmax pico no es significativo (p=0,695).

El mismo patrón se replica de forma independiente en egresos hospitalarios
(N17 insuficiencia renal aguda + E87 trastorno electrolítico, RM 60+): la tasa
correlaciona fuertemente con la Tmax media/p90 del verano (r=0,84-0,88,
p<0,05), no con el pico (r=0,52, n.s.).

*Este hallazgo fue corroborado independientemente por un epidemiólogo experto
(Cristian García, ex-Director de Epidemiología, MINSAL) como consistente con
literatura documentada sobre duración de eventos de calor.*

### 4.2 Validación espacial (comuna / establecimiento) — resultado nulo, documentado con rigor

Se probó si el riesgo simulado (ABM y también el índice de vulnerabilidad
estático) correlaciona espacialmente con datos reales de salud:

- **Nivel comuna** (n=34): mortalidad por causas sensibles al calor (5 eventos
  reales con meteorología día a día vía ERA5/Open-Meteo) y egresos N17+E87
  anuales — sin correlación significativa en ningún caso, incluso corrigiendo
  un artefacto de pooling entre eventos (naive rho=0,31 p<0,001 desaparece a
  rho=0,02 n.s. al estandarizar dentro de evento).
- **Nivel establecimiento de salud** (n=25): crosswalk georreferenciado entre
  códigos de urgencias DEIS y catastro nacional MINSAL (93-100% de cobertura
  validada), catchment por cercanía, urgencias por causa cardiovascular
  específica, con baseline estacionalmente correcto — tampoco significativo.

**Diagnóstico**: no es una falla del modelo — el gradiente de vulnerabilidad
simulado es correcto y grande (comunas de alto ingreso ~0,2 vs. vulnerables
~10 en heat_mean, ~65× de rango, geográficamente donde corresponde). Es el
**problema de área pequeña / falacia ecológica**: con conteos de salud
pequeños por unidad espacial, el ruido de Poisson domina la señal real —
confirmado independientemente por el experto consultado.

**Trabajo en curso**: solicitud vía Ley de Transparencia de egresos agregados
por establecimiento×día×grupo_edad×diagnóstico (siguiendo recomendación
directa del experto), para repetir la validación a resolución más fina.

### 4.3 Validación por experto (planificada)

Siguiendo indicación de la profesora: formalizar un proceso de validación
contra experto (Cristian García) mediante un instrumento estructurado —
generar corridas del simulador para escenarios conocidos, mostrar los
resultados, y recoger su evaluación formal basada en experiencia profesional
en epidemiología del calor.

### 4.4 Métricas de evaluación del simulador (a expandir)

Métricas actualmente implementadas:
1. Número de personas afectadas (agregado, KPI poblacional).
2. Riesgo por persona (heat_load individual, variable en el tiempo).
3. Riesgo promedio por zona (agregación espacial, variable en el tiempo).

*Pendiente: revisión de literatura sobre métricas adicionales usadas en
estudios/simulaciones de impacto de olas de calor (pedido explícito de la
profesora, 31 jul 2026).*

---

## 5. El algoritmo de emulación

### 5.1 Diseño

Latin Hypercube Sampling (40 puntos, T_max∈[26,38]°C, HR_min∈[15,50]%,
Rs_max∈[700,1050] W/m²) → correr el ABM en cada punto → PCA sobre los 40 mapas
resultantes (K=12 componentes, 98,9% de varianza explicada) → un Gaussian
Process por componente (kernel `ConstantKernel × RBF(ARD) + WhiteKernel`) →
predicción vía GP + PCA inversa.

### 5.2 Benchmark: GP vs. alternativas (pedido explícito de la profesora)

Validación cruzada leave-one-out completa (PCA re-ajustada en cada fold, sin
fuga de información), sobre el banco real de 40 escenarios, Gran Santiago:

| Método | RMSE | Skill | Cobertura IC95% | Entrenar | Predecir | Incertidumbre |
|---|---|---|---|---|---|---|
| **GP** | **1,50** | **87,5%** | **92%** | 810 ms | 2,5 ms | nativa, local |
| Random Forest | 2,75 | 77,1% | 99%* (no calibrada) | 2.328 ms | 131 ms | proxy, no calibrada |
| RBF | 1,79 | 85,1% | N/A | 0,3 ms | 0,04 ms | ninguna |
| PCE (orden 2) | 1,82 | 84,8% | N/A | 88 ms | 36 ms | global, no puntual |

GP obtiene la mejor precisión de las cuatro alternativas **y** es el único con
incertidumbre puntual bien calibrada — propiedad indispensable para un sistema
operacional que debe comunicar cuánto confiar en cada predicción.

*(Nota: benchmark preliminar con n=40; se repetirá sobre el banco completo de
~300 escenarios antes de la versión final.)*

### 5.3 Tiempo de ejecución aislado

| Componente | Tiempo |
|---|---|
| GP predict (12 componentes, media + desviación) | 2,02 ms |
| PCA inversa (12 pesos → mapa de 42.800 manzanas) | 0,27 ms |
| **Total** | **2,29 ms** |
| *(referencia)* 1 corrida del ABM (Gran Santiago, 3 días) | ~65.000-100.000 ms |

**Aceleración: ~28.400×-43.700×.**

---

## 6. Resultados

*(pendiente — se completa con la corrida del banco completo y la validación
por experto)*

## 7. Discusión

*(pendiente)*

## 8. Conclusión y trabajo futuro

- Banco completo (~300 escenarios) vía SLURM job arrays en el cluster.
- GP multi-salida con coregionalización (en vez de 12 GPs independientes) —
  ver *Gaussian Process Latent Factor Regression for Low-Data Problems*
  (2026).
- Validación espacial a resolución de establecimiento con datos de fecha
  exacta (pendiente respuesta de solicitud de Transparencia).
- Validación formal contra experto.
- API + visor para consulta en tiempo real de pronósticos DMC.

---

## Checklist para la próxima reunión (15 días)

- [ ] Completar secciones 6 y 7 (Resultados, Discusión)
- [ ] Buscar métricas adicionales de impacto de olas de calor en literatura (pedido directo de la profesora)
- [ ] Confirmar citas exactas de Sacks (1989) y Kennedy & O'Hagan (2001)
- [ ] Avanzar validación por urgencias con los tips de Cristian
- [ ] Enviar solicitud de Transparencia (si aún no se envió)
- [ ] Diseñar el instrumento de validación por experto (formulario + corridas de ejemplo)
