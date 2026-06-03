# Project Context

This project is part of my PhD thesis. It models heat exposure during urban heatwave conditions in Santiago, Chile, using a geospatial agent-based simulation framework.

The current case study is Peñalolén. The model is implemented mainly in Python/Jupyter using GeoPandas, Pandas, NumPy, Shapely, NetworkX and OSMnx.

## Current model logic

The model simulates older adult agents located at census blocks. Each agent is assigned a representative origin point within its census block. Agents are connected to the nearest node of a pedestrian network derived from OpenStreetMap.

Agents are assigned plausible destinations from urban amenities such as healthcare facilities, commercial points and green areas. A shortest path is computed through the pedestrian network. Along each trajectory, heat exposure is estimated as cumulative heat load.

The core exposure idea is:

heat_load = route_length_m * effective_rate

where effective_rate decreases as route-level NDVI increases. Vegetation is treated as a mitigating factor.

## Important methodological decisions

- Do not use EOD/origin-destination survey data for now.
- Do not overcomplicate behavioral rules.
- The model should remain parsimonious and interpretable.
- The focus is trajectory-based exposure, not detailed travel behavior.
- Static exposure is based on residential NDVI.
- Dynamic exposure is based on route distance and NDVI along the trajectory.
- A key comparison is static vs dynamic exposure.
- Vulnerability is analyzed mainly as a grouping/interpretation variable, not as the central behavioral driver.

## Current notebooks

1. vulnerability_index.ipynb
   - Builds census-block vulnerability indicators.
   - Uses variables such as older adults, disability, illiteracy, overcrowding, co-resident households, walking population and tertiary education.
   - Produces vulnerability index and groups.

2. ndvi_to_vector.ipynb
   - Processes NDVI raster.
   - Computes zonal statistics at census block level.
   - Adds ndvi_mean, ndvi_median, ndvi_max, ndvi_norm.

3. walk_simulation.ipynb
   - Builds agents.
   - Loads pedestrian network.
   - Loads destinations.
   - Assigns destinations.
   - Computes shortest paths.
   - Computes route-level NDVI.
   - Computes heat load.
   - Produces results and comparison between static and dynamic exposure.

## Coding preferences

- Keep code inside notebooks for now.
- Do not convert everything into external modules unless explicitly asked.
- Prefer clean, readable notebook blocks.
- Avoid changing working logic unless necessary.
- Make small, controlled modifications.
- Explain exactly where changes should be made.
- Preserve existing column names unless there is a strong reason to rename them.
- Before editing, inspect existing variables, columns and data structures.
- Avoid broad refactors unless explicitly requested.

## Common columns

Census/block data may include:

- MANZENT
- COD_MANZANA
- geometry
- indice_vulnerabilidad
- vuln_group
- vuln_q
- ndvi_mean
- ndvi_norm
- x_rep
- y_rep
- n_edad_60_mas
- n_agents

Simulation results may include:

- agent_id
- MANZENT
- vuln_group
- ndvi_mean
- ndvi_norm
- origin_node
- dest_id
- dest_type
- dest_node
- route
- route_length_m
- travel_time_min_model
- ndvi_route
- heat_load
- risk_level

## Main outputs of interest

- Route length distribution.
- Heat load by agent.
- Route-level NDVI.
- Heat load vs route length.
- Heat load vs route NDVI.
- Static vs dynamic exposure.
- Heat load by vulnerability group.
- Spatial aggregation of exposure by census block.

## Static vs dynamic exposure

Static exposure is a residential baseline:

static_exposure = 1 - ndvi_norm

Dynamic exposure is based on simulated heat load, normalized for comparison:

dynamic_exposure_norm = minmax_norm(heat_load)

The key analytical comparison is:

delta_exp = dynamic_exposure_norm - static_exposure_norm

Negative values mean the static residential approach overestimates exposure compared with the trajectory-based model.

## What I need help with

Help me implement and modify the simulation code step by step. Prioritize correctness, clarity and methodological consistency. When suggesting code, provide replaceable blocks and specify exactly where they should go.