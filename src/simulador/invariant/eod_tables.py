"""
Tablas de comportamiento de la capa invariante, estimadas por PERSONA desde la
EOD Santiago 2012 (días laborales normales, 60+, ponderadas por factor de
expansión) y agrupadas por el MISMO índice de vulnerabilidad del censo que usa
el modelo: cada hogar encuestado se ubica en su manzana censal.

Salidas (data/processed/eod/):
    tabla_age_prob_persona.csv      p(tramo de edad | vuln_group)
    tabla_walk_prob_persona.csv     p(≥1 viaje a pie de ida a salud/compras/social | vuln_group, tramo)
    tabla_purpose_prob_persona.csv  p(propósito | vuln_group), viajes a pie de ida

Requiere el driver ODBC de Access (Windows). Las coordenadas de hogar de la EOD
2012 están en PSAD56 / UTM 19S (EPSG:24879).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

AGE_BANDS = ['60-69', '70-79', '80+']
PURPOSES = {
    'de salud': 'salud',
    'de compras': 'compras_tramites', 'trámites': 'compras_tramites', 'buscar o dejar algo': 'compras_tramites',
    'recreación': 'social_recreacion', 'visitar a alguien': 'social_recreacion',
    'buscar o dejar a alguien': 'social_recreacion', 'comer o tomar algo': 'social_recreacion',
}


def age_band(age) -> pd.Series:
    return pd.cut(age, [60, 70, 80, 200], right=False, labels=AGE_BANDS).astype(str)


def build_eod_tables(db_path, manz_fp, out_dir, max_dist_m: float = 50.0) -> dict:
    import pyodbc
    cn = pyodbc.connect(r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' + str(db_path) + ';')
    per = pd.read_sql("""
        SELECT P.Hogar, P.Persona, (2012 - P.AnoNac) AS edad, P.Factor_LaboralNormal AS w,
               H.DirCoordX AS x, H.DirCoordY AS y, H.NumVeh AS nveh
        FROM [Persona] AS P LEFT JOIN [Hogar] AS H ON P.Hogar = H.Hogar
        WHERE (2012 - P.AnoNac) >= 60""", cn)
    trips = pd.read_sql("""
        SELECT V.Hogar, V.Persona, MA.Modo AS modo, P2.Proposito AS prop
        FROM ([Viaje] AS V
          LEFT JOIN [Modo Agregado] AS MA ON V.ModoAgregado = MA.ID)
          LEFT JOIN [Proposito] AS P2 ON V.Proposito = P2.Id""", cn)
    per = per[per['w'].notna() & per['x'].notna()].copy()

    trips['purpose'] = trips['prop'].astype(str).str.strip().str.lower().map(PURPOSES)
    walk3 = trips[trips['modo'].astype(str).str.strip().eq('Caminata') & trips['purpose'].notna()]

    manz = gpd.read_parquet(manz_fp)[['MANZENT', 'vuln_group', 'geometry']].to_crs('EPSG:32719')
    pts = gpd.GeoDataFrame(per, geometry=gpd.points_from_xy(per['x'], per['y']), crs='EPSG:24879').to_crs('EPSG:32719')
    j = gpd.sjoin_nearest(pts, manz, how='inner', max_distance=max_dist_m, distance_col='dist_m')
    j = j[~j.index.duplicated()]
    per = pd.DataFrame(j.drop(columns='geometry'))
    per['age_band'] = age_band(per['edad'])

    n_walk = walk3.groupby(['Hogar', 'Persona']).size().rename('n_walk3')
    per = per.merge(n_walk, left_on=['Hogar', 'Persona'], right_index=True, how='left').fillna({'n_walk3': 0})
    per['walks'] = per['n_walk3'] > 0

    age = (per.groupby(['vuln_group', 'age_band'])['w'].sum()
              .rename('n_expand').reset_index())
    age['prob'] = age['n_expand'] / age.groupby('vuln_group')['n_expand'].transform('sum')

    walk = per.groupby(['vuln_group', 'age_band']).apply(
        lambda d: pd.Series({'n_sample': len(d), 'n_expand_total': d['w'].sum(),
                             'n_expand_walk': d.loc[d['walks'], 'w'].sum()})).reset_index()
    walk['p_walk'] = walk['n_expand_walk'] / walk['n_expand_total']

    wt = walk3.merge(per[['Hogar', 'Persona', 'vuln_group', 'w']], on=['Hogar', 'Persona'], how='inner')
    purp = wt.groupby(['vuln_group', 'purpose'])['w'].sum().rename('n_expand').reset_index()
    purp = purp.rename(columns={'purpose': 'purpose_group'})
    purp['prob'] = purp['n_expand'] / purp.groupby('vuln_group')['n_expand'].transform('sum')

    dist = walking_distance_table(cn, per, split_by_vuln=('compras_tramites',))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = trip_and_mode_tables(cn, per, out_dir)
    access = access_walk_logit(per, Path(manz_fp).parents[2] / 'raw' / 'comercio_gran_santiago.gpkg', out_dir)
    car_comuna = car_by_comuna(cn, manz_fp, out_dir)
    dist.to_csv(out_dir / 'tabla_dist_walk_persona.csv', index=False)
    age.to_csv(out_dir / 'tabla_age_prob_persona.csv', index=False)
    walk.to_csv(out_dir / 'tabla_walk_prob_persona.csv', index=False)
    purp.to_csv(out_dir / 'tabla_purpose_prob_persona.csv', index=False)
    summary = {
        'persons_60plus_in_area': len(per),
        'persons_expanded': float(per['w'].sum()),
        'p_walk_overall': float(np.average(per['walks'], weights=per['w'])),
    }
    summary.update(mode['summary'])
    return {'age': age, 'walk': walk, 'purpose': purp, 'dist': dist, 'mode': mode, 'access': access, 'car_comuna': car_comuna, 'summary': summary}


def car_by_comuna(cn, manz_fp, out_dir: Path) -> pd.DataFrame:
    """p(hogar con ≥1 vehículo) de las personas 60+, por comuna de residencia (EOD, ponderado)."""
    import unicodedata
    norm = lambda s: unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().upper().strip()
    p = pd.read_sql("""SELECT P.Factor_LaboralNormal AS w, H.Comuna AS comuna, H.NumVeh AS nveh
        FROM [Persona] AS P LEFT JOIN [Hogar] AS H ON P.Hogar = H.Hogar
        WHERE (2012 - P.AnoNac) >= 60""", cn)
    p = p[p['w'].notna()]
    p['car'] = p['nveh'].fillna(0) > 0
    t = p.groupby('comuna').apply(lambda d: pd.Series({'n_sample': len(d),
                                                        'p_car': np.average(d['car'], weights=d['w'])})).reset_index()
    t['key'] = t['comuna'].map(norm)
    cut = pd.read_parquet(Path(manz_fp).parents[2] / 'raw' / 'Cartografía_censo2024_R13_Comunal.parquet',
                          columns=['CUT', 'COMUNA'])
    cut['key'] = cut['COMUNA'].map(norm)
    t = t.merge(cut[['CUT', 'key']], on='key', how='inner')[['CUT', 'comuna', 'n_sample', 'p_car']]
    t.to_csv(out_dir / 'tabla_car_prob_comuna.csv', index=False)
    return t


def access_walk_logit(per: pd.DataFrame, commerce_fp, out_dir: Path) -> pd.DataFrame:
    """
    Logit por persona: P(≥1 viaje a pie a salud/compras/social en un día laboral) según
    la accesibilidad a comercio desde el hogar (log de la distancia al 10º comercio más
    cercano de la capa del modelo), auto en el hogar, vuln_group y tramo de edad.
    `per` son las personas 60+ ya ubicadas en su manzana (con 'walks', 'nveh', 'x', 'y').
    """
    import statsmodels.api as sm
    from scipy.spatial import cKDTree

    com = gpd.read_file(commerce_fp).to_crs('EPSG:32719')
    g = com.geometry if com.geom_type.eq('Point').all() else com.geometry.representative_point()
    home = gpd.GeoSeries(gpd.points_from_xy(per['x'], per['y']), crs='EPSG:24879').to_crs('EPSG:32719')
    d10 = cKDTree(np.c_[g.x, g.y]).query(np.c_[home.x, home.y], k=10)[0][:, -1]
    X = pd.DataFrame({'ln_d10_com': np.log(np.maximum(d10, 1.0)),
                      'car': (per['nveh'].fillna(0).to_numpy() > 0).astype(float),
                      'vuln_media': (per['vuln_group'] == 'media').to_numpy(dtype=float),
                      'vuln_alta': (per['vuln_group'] == 'alta').to_numpy(dtype=float),
                      'age_70_79': (per['age_band'] == '70-79').to_numpy(dtype=float),
                      'age_80': (per['age_band'] == '80+').to_numpy(dtype=float)})
    m = sm.GLM(per['walks'].astype(float).to_numpy(), sm.add_constant(X), family=sm.families.Binomial(),
               freq_weights=(per['w'] / per['w'].mean()).to_numpy()).fit()
    coef = pd.DataFrame({'term': ['intercept'] + list(X.columns), 'coef': m.params.to_numpy(),
                         'se': m.bse.to_numpy(), 'p': m.pvalues.to_numpy()})
    coef.to_csv(out_dir / 'tabla_walk_access_logit.csv', index=False)
    return coef


def trip_and_mode_tables(cn, per: pd.DataFrame, out_dir: Path, max_dist_m: float = 20000.0) -> dict:
    """
    Tablas para la cadena viaje → destino → modo (todas las modalidades, 60+,
    viajes de ida a salud / compras-trámites / social-recreación):

        tabla_trip_prob_persona.csv     p(≥1 viaje | vuln_group, tramo de edad)
        tabla_purpose_trip_persona.csv  p(propósito | vuln_group)
        tabla_dist_trip_persona.csv     distancias observadas en línea recta, con factor
        tabla_car_prob_persona.csv      p(hogar con ≥1 vehículo | vuln_group)
        tabla_modo_caminata_logit.csv   logit P(a pie | log distancia, auto, vuln_group)
    """
    from sklearn.linear_model import LogisticRegression

    t = pd.read_sql("""
        SELECT V.Hogar, V.Persona, V.OrigenCoordX AS ox, V.OrigenCoordY AS oy,
               V.DestinoCoordX AS dx, V.DestinoCoordY AS dy, V.FactorLaboralNormal AS wt,
               MA.Modo AS modo, P2.Proposito AS prop
        FROM ([Viaje] AS V
          LEFT JOIN [Modo Agregado] AS MA ON V.ModoAgregado = MA.ID)
          LEFT JOIN [Proposito] AS P2 ON V.Proposito = P2.Id
        WHERE V.FactorLaboralNormal IS NOT NULL""", cn)
    t['purpose_group'] = t['prop'].astype(str).str.strip().str.lower().map(PURPOSES)
    t = t[t['purpose_group'].notna()].merge(
        per[['Hogar', 'Persona', 'vuln_group', 'age_band', 'nveh']], on=['Hogar', 'Persona'], how='inner')
    t['walk'] = t['modo'].astype(str).str.strip().eq('Caminata')
    t['car'] = (t['nveh'].fillna(0) > 0).astype(float)
    t['dist_m'] = np.hypot(t['ox'] - t['dx'], t['oy'] - t['dy'])

    n_trip = t.groupby(['Hogar', 'Persona']).size().rename('n_trip')
    p = per.merge(n_trip, left_on=['Hogar', 'Persona'], right_index=True, how='left').fillna({'n_trip': 0})
    p['trips'] = p['n_trip'] > 0
    p['car'] = (p['nveh'].fillna(0) > 0)
    trip = p.groupby(['vuln_group', 'age_band']).apply(
        lambda d: pd.Series({'n_sample': len(d), 'p_trip': np.average(d['trips'], weights=d['w'])})).reset_index()
    car = p.groupby('vuln_group').apply(lambda d: np.average(d['car'], weights=d['w'])).rename('p_car').reset_index()

    purp = t.groupby(['vuln_group', 'purpose_group'])['wt'].sum().rename('n_expand').reset_index()
    purp['prob'] = purp['n_expand'] / purp.groupby('vuln_group')['n_expand'].transform('sum')

    ok = t[t[['ox', 'oy', 'dx', 'dy']].notna().all(axis=1) & (t['dist_m'] > 0) & (t['dist_m'] <= max_dist_m)].copy()
    dist = ok[['purpose_group', 'vuln_group', 'dist_m', 'wt']].rename(columns={'wt': 'w'}).copy()
    dist['vuln_group'] = np.where(dist['purpose_group'] == 'compras_tramites', dist['vuln_group'], 'todas')

    X = np.c_[np.log(ok['dist_m']), ok['car'], (ok['vuln_group'] == 'media'), (ok['vuln_group'] == 'alta')].astype(float)
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(X, ok['walk'], sample_weight=ok['wt'])
    coef = pd.DataFrame({'term': ['intercept', 'log_dist_m', 'car', 'vuln_media', 'vuln_alta'],
                         'coef': np.r_[lr.intercept_, lr.coef_[0]]})

    trip.to_csv(out_dir / 'tabla_trip_prob_persona.csv', index=False)
    purp.to_csv(out_dir / 'tabla_purpose_trip_persona.csv', index=False)
    dist.to_csv(out_dir / 'tabla_dist_trip_persona.csv', index=False)
    car.to_csv(out_dir / 'tabla_car_prob_persona.csv', index=False)
    coef.to_csv(out_dir / 'tabla_modo_caminata_logit.csv', index=False)
    return {'trip': trip, 'purpose_trip': purp, 'car': car, 'logit': coef,
            'summary': {'p_trip_overall': float(np.average(p['trips'], weights=p['w'])),
                        'n_trips_modelo_modo': int(len(ok))}}


def walking_distance_table(cn, per: pd.DataFrame, split_by_vuln=('compras_tramites',),
                           max_dist_m: float = 3000.0) -> pd.DataFrame:
    """
    Distancias observadas (línea recta origen→destino) de los viajes a pie de ida de
    personas 60+ a los tres propósitos del modelo, con su factor de expansión.
    `per` son las personas ya ubicadas en su manzana (vuln_group). Los propósitos en
    `split_by_vuln` guardan su vuln_group; el resto se agrupa como 'todas' porque su
    muestra por grupo es chica.
    """
    t = pd.read_sql("""
        SELECT V.Hogar, V.Persona, V.OrigenCoordX AS ox, V.OrigenCoordY AS oy,
               V.DestinoCoordX AS dx, V.DestinoCoordY AS dy, V.FactorLaboralNormal AS w,
               MA.Modo AS modo, P2.Proposito AS prop
        FROM ([Viaje] AS V
          LEFT JOIN [Modo Agregado] AS MA ON V.ModoAgregado = MA.ID)
          LEFT JOIN [Proposito] AS P2 ON V.Proposito = P2.Id
        WHERE V.FactorLaboralNormal IS NOT NULL""", cn)
    t['purpose_group'] = t['prop'].astype(str).str.strip().str.lower().map(PURPOSES)
    t = t[t['modo'].astype(str).str.strip().eq('Caminata') & t['purpose_group'].notna()
          & t[['ox', 'oy', 'dx', 'dy']].notna().all(axis=1)]
    t = t.merge(per[['Hogar', 'Persona', 'vuln_group']], on=['Hogar', 'Persona'], how='inner')
    # PSAD56 / UTM 19S: la distancia euclidiana en metros no depende del datum
    t['dist_m'] = np.hypot(t['ox'] - t['dx'], t['oy'] - t['dy'])
    t = t[(t['dist_m'] > 0) & (t['dist_m'] <= max_dist_m)].copy()
    t['vuln_group'] = np.where(t['purpose_group'].isin(split_by_vuln), t['vuln_group'], 'todas')
    return t[['purpose_group', 'vuln_group', 'dist_m', 'w']].reset_index(drop=True)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import config as cfg
    res = build_eod_tables(cfg.FP_EOD_DB, cfg.DATA_PROC / 'censo' / 'manzanas_gran_santiago_sim.parquet',
                           cfg.DATA_PROC / 'eod')
    for k in ('age', 'walk', 'purpose'):
        print(res[k].round(3).to_string(), '\n')
    for k in ('trip', 'purpose_trip', 'car', 'logit'):
        print(res['mode'][k].round(3).to_string(), '\n')
    print(res['access'].round(3).to_string(), '\n')
    print(res['summary'])
