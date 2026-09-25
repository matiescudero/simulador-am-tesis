"""
Red peatonal: carga del grafo OSMnx y construcción del índice igraph.

igraph es ~1-2 órdenes de magnitud más rápido que NetworkX para shortest-path
masivo (benchmark en nb5), clave para escalar el ruteo a Gran Santiago.
"""
from __future__ import annotations
from dataclasses import dataclass

import osmnx as ox
import igraph as ig


def load_projected_graph(fp_red, crs_proj: str = 'EPSG:32719'):
    """Carga el grafo peatonal (.graphml) y lo proyecta al CRS métrico."""
    G = ox.load_graphml(fp_red)
    G = ox.project_graph(G, to_crs=crs_proj)
    return G


@dataclass
class IGraphIndex:
    """Grafo igraph + mapeo entre ids de nodo OSMnx e índices igraph."""
    ig_G: 'ig.Graph'
    nodes_list: list        # índice igraph -> id de nodo OSMnx
    node_to_idx: dict       # id de nodo OSMnx -> índice igraph

    def idx(self, osm_node) -> int:
        return self.node_to_idx[osm_node]

    def node(self, i: int):
        return self.nodes_list[i]


def build_igraph(G) -> IGraphIndex:
    """
    Construye un grafo igraph dirigido desde un MultiDiGraph de OSMnx.

    Colapsa aristas paralelas quedándose con la de menor longitud (igual que
    el ruteo de nb5). El atributo de arista 'length' guarda los metros y
    'canopy_frac' la fracción bajo copa de árbol (0 si no se calculó).
    """
    nodes_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}

    edges_dict: dict = {}
    for u, v, d in G.edges(data=True):
        k = (node_to_idx[u], node_to_idx[v])
        l = float(d.get('length', 1.0))
        if k not in edges_dict or l < edges_dict[k][0]:
            edges_dict[k] = (l, float(d.get('canopy_frac', 0.0)))

    ig_G = ig.Graph(n=len(nodes_list), edges=list(edges_dict.keys()), directed=True)
    ig_G.es['length'] = [v[0] for v in edges_dict.values()]
    ig_G.es['canopy_frac'] = [v[1] for v in edges_dict.values()]
    return IGraphIndex(ig_G=ig_G, nodes_list=nodes_list, node_to_idx=node_to_idx)
