import random
import pandas as pd
import osmnx as ox
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt


# =========================
# FUNCIONES AUXILIARES
# =========================
def nearest_node_from_point(point, graph):
    return ox.distance.nearest_nodes(graph, point.x, point.y)


def get_nearest_center_node(point, centros_gdf):
    distancias = centros_gdf.geometry.distance(point)
    idx = distancias.idxmin()
    return centros_gdf.loc[idx, "node"]


def get_age_factor(age):
    if age >= 80:
        return 1.3
    elif age >= 70:
        return 1.15
    return 1.0


def get_disease_factor(has_disease):
    return 1.25 if has_disease else 1.0


def classify_risk(heat_load):
    if heat_load >= 1200:
        return "alto"
    elif heat_load >= 700:
        return "medio"
    return "bajo"


def get_edge_attributes(graph, u, v):
    edge_data = graph.get_edge_data(u, v)
    if edge_data is None:
        return None

    first_key = list(edge_data.keys())[0]
    return edge_data[first_key]


def simulate_agent_heat(route, graph, age, has_disease):
    age_factor = get_age_factor(age)
    disease_factor = get_disease_factor(has_disease)

    heat_load = 0.0
    heat_series = []
    segment_info = []

    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]

        attrs = get_edge_attributes(graph, u, v)
        if attrs is None:
            continue

        length = attrs.get("length", 0)
        green_factor = attrs.get("green_factor", 1.0)

        increment = length * green_factor * age_factor * disease_factor
        heat_load += increment
        heat_series.append(heat_load)

        segment_info.append({
            "u": u,
            "v": v,
            "length": length,
            "green_factor": green_factor,
            "increment": increment,
            "heat_load": heat_load
        })

    return heat_load, heat_series, segment_info


def create_agent(agent_id, zona_row, graph):
    origin_node = zona_row["node"]
    destination_node = zona_row["nearest_center_node"]

    try:
        route = nx.shortest_path(graph, origin_node, destination_node, weight="length")
    except Exception:
        return None

    age = random.randint(60, 90)
    has_disease = random.random() < 0.35

    return {
        "id": agent_id,
        "origin_zone": zona_row["geocodigo"] if "geocodigo" in zona_row else agent_id,
        "age": age,
        "has_disease": has_disease,
        "route": route,
        "current_idx": 0,
        "heat_load": 0.0,
        "status": "walking",
        "risk_level": "bajo"
    }


def step_agent(agent, graph):
    if agent["status"] != "walking":
        return agent

    route = agent["route"]
    idx = agent["current_idx"]

    if idx >= len(route) - 1:
        agent["status"] = "arrived"
        return agent

    u = route[idx]
    v = route[idx + 1]

    attrs = get_edge_attributes(graph, u, v)
    if attrs is None:
        agent["status"] = "error"
        return agent

    length = attrs.get("length", 0)
    green_factor = attrs.get("green_factor", 1.0)

    age_factor = get_age_factor(agent["age"])
    disease_factor = get_disease_factor(agent["has_disease"])

    increment = length * green_factor * age_factor * disease_factor
    agent["heat_load"] += increment
    agent["current_idx"] += 1
    agent["risk_level"] = classify_risk(agent["heat_load"])

    if agent["current_idx"] >= len(route) - 1:
        agent["status"] = "arrived"

    return agent


# =========================
# 1. CARGA DE INSUMOS
# =========================
print("Cargando red peatonal...")
G = ox.load_graphml("data/raw/red_peatonal_penalolen.graphml")

print("Cargando zonas censales...")
zonas = gpd.read_file("data/raw/zonas_censales_penalolen.gpkg")

print("Cargando establecimientos de salud...")
centros = gpd.read_file("data/raw/establecimientos_salud_penalolen.gpkg")

print("Cargando áreas verdes...")
areas_verdes = gpd.read_file("data/raw/areas_verdes_penalolen.gpkg")


# =========================
# 2. ASEGURAR CRS
# =========================
G = ox.project_graph(G, to_crs="EPSG:32719")

zonas = zonas.to_crs("EPSG:32719")
centros = centros.to_crs("EPSG:32719")
areas_verdes = areas_verdes.to_crs("EPSG:32719")

print("CRS zonas:", zonas.crs)
print("CRS centros:", centros.crs)
print("CRS áreas verdes:", areas_verdes.crs)


# =========================
# 3. PREPARAR ORÍGENES Y DESTINOS
# =========================
zonas["centroid"] = zonas.geometry.centroid

print("Asignando nodo más cercano a cada zona...")
zonas["node"] = zonas["centroid"].apply(lambda p: nearest_node_from_point(p, G))

print("Asignando nodo más cercano a cada centro de salud...")
centros["node"] = centros.geometry.apply(lambda p: nearest_node_from_point(p, G))

print("Buscando centro más cercano para cada zona...")
zonas["nearest_center_node"] = zonas["centroid"].apply(
    lambda p: get_nearest_center_node(p, centros)
)


# =========================
# 4. ETIQUETAR ARISTAS SEGÚN ÁREAS VERDES
# =========================
print("Etiquetando aristas con efecto de áreas verdes...")

nodes, edges = ox.graph_to_gdfs(G)

green_union = areas_verdes.union_all()
green_buffer = gpd.GeoSeries([green_union], crs=areas_verdes.crs).buffer(30).iloc[0]

edges["has_green"] = edges.geometry.intersects(green_buffer)
edges["green_factor"] = 1.0
edges.loc[edges["has_green"], "green_factor"] = 0.7

print("Total aristas:", len(edges))
print("Aristas con efecto verde:", edges["has_green"].sum())
print("Porcentaje aristas verdes:", round(edges["has_green"].mean() * 100, 2), "%")

G = ox.graph_from_gdfs(nodes, edges)


# =========================
# 5. DEMO DE UNA RUTA
# =========================
zona_test = zonas.iloc[0]
origen = zona_test["node"]
destino = zona_test["nearest_center_node"]

print("Nodo origen:", origen)
print("Nodo destino:", destino)

print("Calculando ruta...")
ruta = nx.shortest_path(G, origen, destino, weight="length")
largo_m = nx.path_weight(G, ruta, weight="length")
print(f"Largo ruta: {largo_m:.2f} metros")

age = 78
has_disease = True

heat_final, heat_series, segment_info = simulate_agent_heat(ruta, G, age, has_disease)
risk_level = classify_risk(heat_final)

print(f"Heat load final: {heat_final:.2f}")
print(f"Pasos simulados: {len(heat_series)}")
print("Nivel de riesgo:", risk_level)

print("\nPrimeros segmentos simulados:")
for seg in segment_info[:5]:
    print(seg)


# =========================
# 6. SIMULACIÓN MULTIAGENTE
# =========================
print("Creando agentes...")

n_agents = 50
sample_zonas = zonas.sample(n=min(n_agents, len(zonas)), replace=True, random_state=42)

agents = []
agent_id = 1

for _, zona_row in sample_zonas.iterrows():
    agent = create_agent(agent_id, zona_row, G)
    if agent is not None:
        agents.append(agent)
        agent_id += 1

print(f"Agentes creados: {len(agents)}")

print("Simulando múltiples agentes...")

metrics = []
timestep = 0
max_steps = 100

while timestep < max_steps:
    active_agents = [a for a in agents if a["status"] == "walking"]

    if len(active_agents) == 0:
        break

    for agent in active_agents:
        step_agent(agent, G)

    total_agents = len(agents)
    walking_agents = sum(1 for a in agents if a["status"] == "walking")
    arrived_agents = sum(1 for a in agents if a["status"] == "arrived")
    risk_medium = sum(1 for a in agents if a["risk_level"] == "medio")
    risk_high = sum(1 for a in agents if a["risk_level"] == "alto")
    mean_heat = sum(a["heat_load"] for a in agents) / total_agents
    max_heat = max(a["heat_load"] for a in agents)

    metrics.append({
        "timestep": timestep,
        "walking_agents": walking_agents,
        "arrived_agents": arrived_agents,
        "risk_medium": risk_medium,
        "risk_high": risk_high,
        "mean_heat": mean_heat,
        "max_heat": max_heat
    })

    timestep += 1

metrics_df = pd.DataFrame(metrics)
agents_df = pd.DataFrame(agents)

print(metrics_df.head())
print(metrics_df.tail())

print("\nResumen agentes:")
print(agents_df[["id", "age", "has_disease", "heat_load", "risk_level", "status"]].head(10))


# =========================
# 7. VISUALIZACIONES
# =========================
print("Mostrando ruta...")
fig, ax = ox.plot_graph_route(
    G,
    ruta,
    route_linewidth=3,
    node_size=0,
    bgcolor="white",
    show=False,
    close=False
)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(heat_series)
plt.xlabel("Paso de la ruta")
plt.ylabel("Heat load acumulado")
plt.title("Evolución del estrés térmico acumulado")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(metrics_df["timestep"], metrics_df["risk_medium"], label="Riesgo medio")
plt.plot(metrics_df["timestep"], metrics_df["risk_high"], label="Riesgo alto")
plt.xlabel("Timestep")
plt.ylabel("Número de agentes")
plt.title("Evolución de agentes en riesgo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(metrics_df["timestep"], metrics_df["mean_heat"], label="Heat promedio")
plt.plot(metrics_df["timestep"], metrics_df["max_heat"], label="Heat máximo")
plt.xlabel("Timestep")
plt.ylabel("Heat load")
plt.title("Evolución del estrés térmico agregado")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(metrics_df["timestep"], metrics_df["walking_agents"], label="Caminando")
plt.plot(metrics_df["timestep"], metrics_df["arrived_agents"], label="Llegados")
plt.xlabel("Timestep")
plt.ylabel("Número de agentes")
plt.title("Estado de los agentes en el tiempo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()