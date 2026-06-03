CREATE TABLE output.tabla_sim AS
SELECT z.geocodigo,
	COUNT (*) FILTER (WHERE p.p09 >= 60) AS pob_total,
	COUNT (*) FILTER (WHERE p.p09 >= 60) AS pob_60omas,
	ROUND(COUNT(*) FILTER (WHERE p.p09 >= 60) * 100.0 / COUNT(*), 2) AS pct_60omas,
    ROUND(COUNT (*) FILTER (WHERE p.p15 >= 12 AND p.p15 <=14) * 100.0/ COUNT (*)FILTER (WHERE p.p09 > 18) ,2) AS pct_educ_sup,
	ROUND(COUNT (*) FILTER (WHERE p.p09 >= 60) * 0.1 , 0) AS n_agentes_sim
	
FROM personas p
JOIN hogares h ON h.hogar_ref_id = p.hogar_ref_id 
JOIN viviendas v ON h.vivienda_ref_id = v.vivienda_ref_id 
JOIN zonas z ON z.zonaloc_ref_id = v.zonaloc_ref_id
JOIN comunas c ON z.codigo_comuna = c.codigo_comuna
WHERE nom_comuna = 'PEÑALOLÉN'
GROUP BY z.geocodigo, c.nom_comuna
ORDER BY pct_60omas DESC;


-- Unir la geometría a la tabla de profesionales


CREATE TABLE output.tabla_sim_geom AS
SELECT tp.*, shp.geom
FROM output.tabla_sim AS tp
JOIN dpa.zonas_censales_rm AS shp
ON shp.geocodigo = tp.geocodigo::double precision;

CREATE TABLE output.areas_verdes_penalolen AS
WITH comuna AS (
    SELECT ST_Union(ST_MakeValid(geom)) AS geom
    FROM output.tabla_sim_geom
),
areas_verdes_validas AS (
    SELECT
        id,
        ST_MakeValid(ST_Transform(geom, 32719)) AS geom
    FROM dpa.areas_verdes_stgo
)
SELECT
    av.id,
    ST_Intersection(av.geom, c.geom) AS geom
FROM areas_verdes_validas av
CROSS JOIN comuna c
WHERE ST_Intersects(av.geom, c.geom);


