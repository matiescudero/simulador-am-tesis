"""
simulador — motor del ABM de exposición al calor en adultos mayores.

Arquitectura de dos capas (ver project_emulador_arquitectura):
- invariant/ : red, rutas, NDVI por ruta. Se calcula 1 vez por zona urbana.
- scenario/  : heat_load WBGT dependiente de parámetros. Barato, paralelizable.

La lógica del motor WBGT vive (por ahora) en src/simulation.py y migra
incrementalmente a scenario/ en pasos posteriores.
"""
