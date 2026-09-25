"""
Capa invariante al escenario.

Todo lo que NO depende de las variables meteorológicas del escenario y por lo
tanto se calcula una sola vez por zona urbana y se serializa a disco:

- network      : grafo peatonal (osmnx) + índice igraph para ruteo rápido
- walkers      : población de agentes (quién camina, propósito) + destinos
- routes       : rutas más cortas deduplicadas por (origin, dest)
- route_ndvi   : NDVI ponderado por longitud a lo largo de cada ruta
- build        : orquesta lo anterior y produce walkers.parquet

Salida: artifacts/{zona}/walkers.parquet  (agent_id, vuln_group,
purpose_group_model, route_length_m, ndvi_route, ...) reutilizable por todos
los escenarios del banco.
"""
from .network import load_projected_graph, build_igraph, IGraphIndex
from .walkers import generate_agents, build_destinations, assign_origins_and_dests
from .route_ndvi import route_to_linestring, compute_route_ndvi
from .routes import compute_routes_dedup

__all__ = [
    'load_projected_graph', 'build_igraph', 'IGraphIndex',
    'generate_agents', 'build_destinations', 'assign_origins_and_dests',
    'route_to_linestring', 'compute_route_ndvi', 'compute_routes_dedup',
]
