"""
Drunk Drivers Classification - Kaggle Competition 2025-2026
============================================================
Created By: Diego Martinez Silva & Carlos García-Mauriño García-Mauriño
As:"ExtremoDuro"

"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

import lightgbm as lgb
import xgboost as xgb   
from catboost import CatBoostClassifier, Pool

print("Librerias cargadas correctamente.")

# =============================================================================
# CARGA Y PREPROCESAMIENTO
# =============================================================================
x_train = pd.read_csv('data/x_train.csv', low_memory=False)
y_train = pd.read_csv('data/y_train.csv')
x_test  = pd.read_csv('data/x_test.csv', low_memory=False)

y_raw = y_train['DRUNK_DR'].clip(upper=3).astype(int)

print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")
print(f"Target: {y_raw.value_counts().sort_index().to_dict()}")


def preprocess(df):
    """
    Limpieza general del dataset: 
    """
    df = df.copy()
    name_cols = [c for c in df.columns if c.endswith('NAME')]
    # Eliminamos columnas descriptivas terminadas en NAME porque suelen duplicar información categórica ya presente pero en texto menos útil para modelos
    df.drop(columns=name_cols, inplace=True)
    # Quitamos identificadores únicos o administrativos como ST_CASE porque pueden introducir ruido o sobreajuste sin aportar patrones reales
    drop_ids = ['ST_CASE', 'TWAY_ID', 'TWAY_ID2', 'YEAR']
    df.drop(columns=[c for c in drop_ids if c in df.columns], inplace=True)
    # Transformamos la variable RAIL en binaria para simplificar si existe o no presencia ferroviaria en el accidente
    if 'RAIL' in df.columns:
        df['RAIL'] = (df['RAIL'] != '0000000').astype(int)
    # Generamos indicador de missing en MILEPT porque la ausencia de dato también puede tener valor predictivo
    if 'MILEPT' in df.columns:
        df['MILEPT_MISSING'] = (df['MILEPT'] >= 99999).astype(int)
    # Convertimos ciertos códigos especiales en flags explícitos de ausencia o desconocimiento
    if 'WEATHER' in df.columns:
        df['WEATHER_MISSING'] = df['WEATHER'].isin([98, 99]).astype(int)
    if 'RD_OWNER' in df.columns:
        df['RD_OWNER_MISSING'] = df['RD_OWNER'].isin([98, 99]).astype(int)
    time_cols = ['HOUR', 'MINUTE', 'NOT_HOUR', 'NOT_MIN',
                 'ARR_HOUR', 'ARR_MIN', 'HOSP_HR', 'HOSP_MN']
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].replace([99, 88, 98], np.nan)
    # Corregimos o eliminamos coordenadas imposibles o erróneas basándonos en límites geográficos razonables
    if 'LATITUDE' in df.columns:
        df.loc[df['LATITUDE'] > 80, 'LATITUDE'] = np.nan
    if 'LONGITUD' in df.columns:
        df.loc[df['LONGITUD'] > 0, 'LONGITUD'] = np.nan
    special_map = {
        'FUNC_SYS': [99, 98], 'RD_OWNER': [99, 98],
        'HARM_EV': [99, 98],   'MAN_COLL': [99, 98],
        'RELJCT2': [99, 98],   'TYP_INT': [99, 98],
        'REL_ROAD': [99, 98],  'WEATHER': [99, 98],
        'LGT_COND': [9],       'SP_JUR': [9],
        'ROUTE': [9],
    }
    for col, vals in special_map.items():
        if col in df.columns:
            df[col] = df[col].replace(vals, np.nan)
    # Ajustamos de nuevo el MILEPT eliminando valores reservados de no información
    if 'MILEPT' in df.columns:
        df.loc[df['MILEPT'] >= 99999, 'MILEPT'] = np.nan
    return df


def engineer_features(df):
    df = df.copy()
    # El seno y coseno como vimos en clase, se hace para hacer una transformación cíclica a la hora porque las 23:00 y las 00:00 están cerca temporalmente aunque numéricamente no lo parezcan
    df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
    df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
    # Hacemos lo mismo con los dias de la semana ya que el domingo esta cerca del lunes como lo anterior
    df['DAY_WEEK_SIN'] = np.sin(2 * np.pi * df['DAY_WEEK'] / 7)
    df['DAY_WEEK_COS'] = np.cos(2 * np.pi * df['DAY_WEEK'] / 7)
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
    # Agregamos esto ya que por logica hemos deducido que los dias del fin de semana es probable que aumente la probabilidad de riesgos asociados al alcohol
    df['IS_WEEKEND'] = df['DAY_WEEK'].isin([1, 7]).astype(int)
    #Lo mismo pero para la noche
    df['IS_NIGHT'] = ((df['HOUR'] >= 20) | (df['HOUR'] < 6)).astype(float)
    df.loc[df['HOUR'].isna(), 'IS_NIGHT'] = np.nan
    # Y hacemos la combinacion de ambas porque es lo que mas probabilidades que tiene
    df['NIGHT_WEEKEND'] = df['IS_NIGHT'] * df['IS_WEEKEND']
    #creamos interacción simple entre hora y día para detectar combinaciones temporales
    df['HOUR_X_DAYWEEK'] = df['HOUR'] * df['DAY_WEEK']
    # Agrupamos horas en franjas más generales para reducir ruido puntual
    bins = [-1, 2, 5, 9, 14, 19, 22, 24]
    df['HOUR_BIN'] = pd.cut(df['HOUR'], bins=bins, labels=list(range(7))).astype(float)
    df['BAR_CLOSE_HOURS'] = df['HOUR'].isin([0, 1, 2]).astype(float)
    df.loc[df['HOUR'].isna(), 'BAR_CLOSE_HOURS'] = np.nan
    # Ampliamos el concepto a madrugada extendida para capturar patrones similares.
    df['LATE_NIGHT'] = df['HOUR'].isin([0, 1, 2, 3, 23]).astype(float)
    df.loc[df['HOUR'].isna(), 'LATE_NIGHT'] = np.nan
    # Transformamos toda la información temporal en diferencias de minutos entre accidente, aviso, llegada y hospital para extraer patrones de respuesta y gravedad que podrían correlacionar con accidentes relacionados con alcohol, ya que no solo importa cuándo ocurre el siniestro sino también cómo evoluciona después
    acc_mins = df['HOUR'] * 60 + df['MINUTE']
    not_mins = df['NOT_HOUR'] * 60 + df['NOT_MIN']
    arr_mins = df['ARR_HOUR'] * 60 + df['ARR_MIN']
    hosp_mins = df['HOSP_HR'] * 60 + df['HOSP_MN']
    df['DELTA_NOT'] = not_mins - acc_mins
    df.loc[df['DELTA_NOT'] < 0, 'DELTA_NOT'] += 1440
    df['DELTA_ARR'] = arr_mins - not_mins
    df.loc[df['DELTA_ARR'] < 0, 'DELTA_ARR'] += 1440
    df['DELTA_HOSP'] = hosp_mins - acc_mins
    df.loc[df['DELTA_HOSP'] < 0, 'DELTA_HOSP'] += 1440
    # Generamos variables estructurales del accidente centradas en tipo de colisión, número de vehículos, peatones, severidad e iluminación porque investigaciones sobre seguridad vial muestran que conducción ebria suele asociarse más con accidentes nocturnos, de vehículo único, salida de vía o impactos contra objetos fijos
    df['HAS_NOT_TIME'] = df['NOT_HOUR'].notna().astype(int)
    df['HAS_ARR_TIME'] = df['ARR_HOUR'].notna().astype(int)
    df['HAS_HOSP_TIME'] = df['HOSP_HR'].notna().astype(int)
    df['SINGLE_VEHICLE'] = (df['VE_TOTAL'] == 1).astype(int)
    df['MULTI_FATALS'] = (df['FATALS'] > 1).astype(int)
    df['HAS_PEDS'] = (df['PEDS'] > 0).astype(int)
    df['RATIO_PERMVIT'] = df['PERMVIT'] / (df['PERSONS'] + 1)
    df['NO_COLLISION'] = (df['MAN_COLL'] == 0).astype(float)
    df.loc[df['MAN_COLL'].isna(), 'NO_COLLISION'] = np.nan
    df['IS_PED_CRASH'] = df['HARM_EV'].isin([8, 9]).astype(float)
    df.loc[df['HARM_EV'].isna(), 'IS_PED_CRASH'] = np.nan
    df['IS_FIXED_OBJECT'] = df['HARM_EV'].isin(list(range(24, 54))).astype(float)
    df.loc[df['HARM_EV'].isna(), 'IS_FIXED_OBJECT'] = np.nan
    df['IS_ROLLOVER'] = (df['HARM_EV'] == 1).astype(float)
    df.loc[df['HARM_EV'].isna(), 'IS_ROLLOVER'] = np.nan
    df['HAS_PERNOTMVIT'] = (df['PERNOTMVIT'] > 0).astype(int)
    df['PED_CRASH'] = ((df['PEDS'] > 0) & (df['PERNOTMVIT'] > 0)).astype(int)
    df['VE_DIFF'] = df['VE_TOTAL'] - df['VE_FORMS']
    df['IS_DARK'] = df['LGT_COND'].isin([2, 3]).astype(float)
    df.loc[df['LGT_COND'].isna(), 'IS_DARK'] = np.nan
    # Creamos interacciones complejas entre noche, fin de semana, cierre de bares, oscuridad y tipo de accidente porque muchas veces el verdadero valor predictivo no está en una sola variable sino en la combinación de múltiples factores de riesgo simultáneos
    night_f = df['IS_NIGHT'].fillna(0)
    dark_f = df['IS_DARK'].fillna(0)
    df['SINGLE_NIGHT'] = df['SINGLE_VEHICLE'] * night_f
    df['SINGLE_DARK'] = df['SINGLE_VEHICLE'] * dark_f
    df['WKEND_NIGHT_SV'] = df['IS_WEEKEND'] * night_f * df['SINGLE_VEHICLE']
    df['WKEND_BAR_CLOSE'] = df['IS_WEEKEND'] * df['BAR_CLOSE_HOURS'].fillna(0)
    df['NIGHT_FIXEDOBJ'] = night_f * df['IS_FIXED_OBJECT'].fillna(0)
    # Añadimos contexto geográfico y vial como ruralidad, intersecciones, zonas de obras o carreteras locales, ya que el entorno del accidente también puede modificar significativamente la probabilidad de conducción bajo efectos del alcohol
    df['IS_RURAL'] = (df['RUR_URB'] == 1).astype(int)
    df['IS_INTERSECTION'] = df['RELJCT2'].isin([2, 3, 4, 5, 6, 7, 8]).astype(float)
    df.loc[df['RELJCT2'].isna(), 'IS_INTERSECTION'] = np.nan
    df['IN_WORK_ZONE'] = (df['WRK_ZONE'] > 0).astype(int)
    df['OFF_ROADWAY'] = df['REL_ROAD'].isin([4, 5, 8]).astype(float)
    df.loc[df['REL_ROAD'].isna(), 'OFF_ROADWAY'] = np.nan
    df['IS_LOCAL_ROAD'] = df['FUNC_SYS'].isin([6, 7]).astype(float)
    df.loc[df['FUNC_SYS'].isna(), 'IS_LOCAL_ROAD'] = np.nan
    df['NIGHT_OFFROAD_SV'] = night_f * df['OFF_ROADWAY'].fillna(0) * df['SINGLE_VEHICLE']
    df['MULTI_VEHICLE'] = (df['VE_TOTAL'] >= 2).astype(int)
    df['THREE_PLUS_VEH'] = (df['VE_TOTAL'] >= 3).astype(int)
    df['MANY_PERSONS'] = (df['PERSONS'] >= 3).astype(int)
    df['NIGHT_MULTI_VEH'] = night_f * df['MULTI_VEHICLE']
    df['WKEND_NIGHT_MULTI'] = df['IS_WEEKEND'] * night_f * df['MULTI_VEHICLE']
    # Si existen coordenadas, agrupamos ubicaciones en celdas geográficas aproximadas para detectar patrones espaciales regionales sin sobreajustar a coordenadas exactas, algo aprendido de técnicas comunes en Kaggle para datos tabulares geolocalizados
    if 'LATITUDE' in df.columns and 'LONGITUD' in df.columns:
        df['LAT_BIN'] = (df['LATITUDE'] * 2).round() / 2
        df['LON_BIN'] = (df['LONGITUD'] * 2).round() / 2
        df['GEO_CELL'] = df['LAT_BIN'] * 1000 + df['LON_BIN']
    return df

# Aplicamos todo el pipeline de limpieza y feature engineering tanto a entrenamiento como a test para garantizar que ambos conjuntos compartan exactamente la misma estructura de variables
x_train_feat = engineer_features(preprocess(x_train))
x_test_feat = engineer_features(preprocess(x_test))

# Definimos que variables deben tratarse como categóricas, algo especialmente importante en CatBoost porque este modelo aprovecha mejor dichas variables que una codificación numérica tradicional
CAT_COLS = ['STATE', 'COUNTY', 'CITY', 'HARM_EV', 'FUNC_SYS', 'ROUTE',
            'LGT_COND', 'REL_ROAD', 'MAN_COLL', 'RD_OWNER', 'RELJCT2',
            'TYP_INT', 'WEATHER', 'RUR_URB', 'SP_JUR']

# Eliminamos posibles columnas de texto no utilizables y construimos la lista final de variables que realmente alimentarán a los modelos predictivos
feature_cols = x_train_feat.columns.tolist()
obj_cols = x_train_feat.select_dtypes(include=['object', 'string']).columns.tolist()
feature_cols = [c for c in feature_cols if c not in obj_cols]
# Guardamos la posición exacta de variables categóricas para pasársela correctamente a CatBoost durante el entrenamiento.
cat_idx = [feature_cols.index(c) for c in CAT_COLS if c in feature_cols]

print(f"Features: {len(feature_cols)}, Categoricas: {len(cat_idx)}")

ve_total_train = x_train_feat['VE_TOTAL'].values
ve_total_test = x_test_feat['VE_TOTAL'].values

# =============================================================================
# Fase de entrenamiento del modelo binario con CatBoost
# =============================================================================
# Establecemos múltiples seeds y validación cruzada para reducir dependencia de una sola partición aleatoria y conseguir estimaciones más robustas, algo especialmente importante en datasets desbalanceados
N_FOLDS = 5
N_SEEDS = 5
seeds = [42, 123, 2025, 7, 999]


def train_binary_catboost(y_binary, label, cb_params_override=None):
    # Esta función entrena múltiples modelos CatBoost con distintas semillas y folds para generar probabilidades más estables, reduciendo varianza y aprovechando el buen rendimiento de CatBoost en datos tabulares con variables categóricas complejas
    oof_prob = np.zeros(len(y_binary))
    test_prob = np.zeros(len(x_test_feat))

    cb_params = dict(
        iterations=3000, learning_rate=0.03, depth=6,
        auto_class_weights='Balanced', l2_leaf_reg=5.0,
        verbose=0, early_stopping_rounds=200,
        eval_metric='BalancedAccuracy',
        min_data_in_leaf=20, bootstrap_type='Bernoulli',
        subsample=0.8, colsample_bylevel=0.8,
    )
    if cb_params_override:
        cb_params.update(cb_params_override)

    # Repetimos entrenamiento con múltiples seeds para suavizar fluctuaciones estadísticas y evitar dependencia excesiva de una sola inicialización
    for seed in seeds:
        cb_params['random_seed'] = seed
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        # Cada fold genera predicciones OOF (out-of-fold), fundamentales para evaluar de forma honesta sin contaminar entrenamiento
        for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_feat, y_binary)):
            # Preparamos datos y rellenamos categóricas faltantes con valores seguros porque CatBoost necesita estructura consistente
            tr_df = x_train_feat.iloc[train_idx][feature_cols].copy()
            val_df = x_train_feat.iloc[val_idx][feature_cols].copy()
            tst_df = x_test_feat[feature_cols].copy()
            for c in CAT_COLS:
                if c in tr_df.columns:
                    tr_df[c] = tr_df[c].fillna(-1).astype(int)
                    val_df[c] = val_df[c].fillna(-1).astype(int)
                    tst_df[c] = tst_df[c].fillna(-1).astype(int)

            # Pool es la estructura optimizada propia de CatBoost para trabajar mejor con variables categóricas
            train_pool = Pool(tr_df, y_binary.iloc[train_idx], cat_features=cat_idx)
            val_pool = Pool(val_df, y_binary.iloc[val_idx], cat_features=cat_idx)
            test_pool = Pool(tst_df, cat_features=cat_idx)

            model = CatBoostClassifier(**cb_params)
            model.fit(train_pool, eval_set=val_pool, verbose=0)

            oof_prob[val_idx] += model.predict_proba(val_pool)[:, 1] / N_SEEDS
            test_prob += model.predict_proba(test_pool)[:, 1] / (N_FOLDS * N_SEEDS)

    # Evaluamos balanced accuracy porque penaliza mejor errores en clases minoritarias
    ba = balanced_accuracy_score(y_binary, (oof_prob > 0.5).astype(int))
    print(f"  {label} CB CV (BA@0.5): {ba:.5f}")
    return oof_prob, test_prob


def train_binary_lgb(y_binary, label):
    # Implementamos LightGBM como segundo modelo complementario porque aunque CatBoost suele rendir excelente, combinar distintos algoritmos de boosting puede capturar patrones diferentes y mejorar generalización mediante ensemble
    oof_prob = np.zeros(len(y_binary))
    test_prob = np.zeros(len(x_test_feat))

    # Calculamos pesos manuales para balancear clases, compensando el enorme desbalanceo y evitando que el modelo favorezca casi exclusivamente la clase mayoritaria
    n_pos = y_binary.sum()
    n_neg = len(y_binary) - n_pos
    w_map = {0: len(y_binary) / (2 * n_neg), 1: len(y_binary) / (2 * n_pos)}

    # Configuramos hiperparámetros relativamente conservadores para evitar sobreajuste mientras mantenemos capacidad predictiva suficiente
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.3,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
    }

    # Convertimos a arrays numéricos optimizados para LightGBM, mejorando eficiencia de entrenamiento
    X_train_np = x_train_feat[feature_cols].values.astype(np.float32)
    X_test_np = x_test_feat[feature_cols].values.astype(np.float32)

    # Igual que en CatBoost, repetimos múltiples seeds y folds para estabilizar resultados y reducir varianza
    for seed in seeds:
        lgb_params['random_state'] = seed
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_binary)):
            # Aplicamos pesos a cada muestra para mantener equilibrio real entre clases.
            w_tr = np.array([w_map[int(c)] for c in y_binary.iloc[train_idx]])
            dtrain = lgb.Dataset(X_train_np[train_idx], label=y_binary.iloc[train_idx],
                                 weight=w_tr, feature_name=feature_cols)
            dval = lgb.Dataset(X_train_np[val_idx], label=y_binary.iloc[val_idx],
                               reference=dtrain, feature_name=feature_cols)

            # Entrenamos con early stopping para detener aprendizaje cuando deja de mejorar
            model = lgb.train(
                lgb_params, dtrain, num_boost_round=3000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
            )

            # Acumulamos probabilidades promedio para ensemble posterior.
            oof_prob[val_idx] += model.predict(X_train_np[val_idx]) / N_SEEDS
            test_prob += model.predict(X_test_np) / (N_FOLDS * N_SEEDS)

    ba = balanced_accuracy_score(y_binary, (oof_prob > 0.5).astype(int))
    print(f"  {label} LGB CV (BA@0.5): {ba:.5f}")
    return oof_prob, test_prob


# =============================================================================
# Iniciamos el primer modelo ordinal para predecir si existe al menos un conductor ebrio, siendo este el problema más equilibrado y estadísticamente sólido dentro del dataset
# =============================================================================
print("\n" + "="*60)
print("Modelo 1: P(DRUNK_DR >= 1)")
print("="*60)

# Convertimos el problema multicategoría en binario acumulativo, estrategia ordinal que suele funcionar mejor que clasificación directa en clases ordenadas
y1 = (y_raw >= 1).astype(int)
print(f"  Positivos: {y1.sum()}, Negativos: {(1-y1).sum()}")

oof_p1_cb, test_p1_cb = train_binary_catboost(y1, "Model 1")
oof_p1_lgb, test_p1_lgb = train_binary_lgb(y1, "Model 1")

# Buscamos automáticamente la mejor combinación ponderada entre CatBoost y LightGBM para maximizar balanced accuracy, creando así un ensemble más potente que cualquiera de los modelos por separado
best_w1 = 1.0
best_ba1 = 0
for w in np.arange(0.0, 1.05, 0.05):
    blend = w * oof_p1_cb + (1-w) * oof_p1_lgb
    ba = balanced_accuracy_score(y1, (blend > 0.5).astype(int))
    if ba > best_ba1:
        best_ba1 = ba
        best_w1 = w
# Guardamos combinación óptima final.
print(f"  Model 1 ensemble: CB={best_w1:.2f}, LGB={1-best_w1:.2f}, BA={best_ba1:.5f}")
oof_p1 = best_w1 * oof_p1_cb + (1-best_w1) * oof_p1_lgb
test_p1 = best_w1 * test_p1_cb + (1-best_w1) * test_p1_lgb

# =============================================================================
# Repetimos el proceso para clase >=2, donde el problema se vuelve muchísimo más extremo debido a la escasez de positivos, obligando a ajustar ligeramente regularización para mejorar sensibilidad
# =============================================================================

print("\n" + "="*60)
print("Modelo 2: P(DRUNK_DR >= 2)")
print("="*60)

y2 = (y_raw >= 2).astype(int)
print(f"  Positivos: {y2.sum()}, Negativos: {(1-y2).sum()}")

# Reducimos algo la regularización para permitir capturar mejor señales débiles en la clase minoritaria
oof_p2_cb, test_p2_cb = train_binary_catboost(
    y2,
    "Model 2",
    cb_params_override={
        'l2_leaf_reg': 3.0,
        'min_data_in_leaf': 10
    }
)

oof_p2_lgb, test_p2_lgb = train_binary_lgb(y2, "Model 2")

# Volvemos a optimizar los pesos del ensemble
best_w2 = 1.0
best_ba2 = 0
for w in np.arange(0.0, 1.05, 0.05):
    blend = w * oof_p2_cb + (1-w) * oof_p2_lgb
    ba = balanced_accuracy_score(y2, (blend > 0.5).astype(int))
    if ba > best_ba2:
        best_ba2 = ba
        best_w2 = w
print(f"  Model 2 ensemble: CB={best_w2:.2f}, LGB={1-best_w2:.2f}, BA={best_ba2:.5f}")
oof_p2 = best_w2 * oof_p2_cb + (1-best_w2) * oof_p2_lgb
test_p2 = best_w2 * test_p2_cb + (1-best_w2) * test_p2_lgb

# =============================================================================
# Para clase >=3 apenas existen muestras suficientes para entrenar de forma fiable, por lo que recurrimos a una heurística controlada basada en probabilidades previas y restricciones físicas observadas en datos reales.
# =============================================================================
print("\n" + "="*60)
print("Modelo 3: P(DRUNK_DR >= 3) - heuristico")
print("="*60)

y3 = (y_raw >= 3).astype(int)
print(f"  Positivos: {y3.sum()} (demasiado pocos para modelo, usando heuristico)")

# Los 5 casos clase 3 tienen VE_TOTAL >= 3 (100%).
# Heuristico: P(class 3) = P(>=2) * indicador VE_TOTAL>=3 * factor
# Factor mas bajo = mas conservador en predecir clase 3
# Lo que significa basicamente que hemos observado que todos los casos extremos implicaban al menos 3 vehículos, así que utilizamos esa restricción como filtro físico, multiplicando además por factor conservador para evitar sobrepredicción

oof_p3 = oof_p2 * (ve_total_train >= 3).astype(float) * 0.5
test_p3 = test_p2 * (ve_total_test >= 3).astype(float) * 0.5

# Guardamos probabilidades intermedias para futuras pruebas rápidas sin necesidad de reentrenar todo el pipeline, algo muy útil durante optimización iterativa
np.savez('ordinal_probs.npz',
         oof_p1=oof_p1, oof_p2=oof_p2, oof_p3=oof_p3,
         test_p1=test_p1, test_p2=test_p2, test_p3=test_p3,
         ve_total_train=ve_total_train, ve_total_test=ve_total_test,
         y_raw=y_raw.values)
print("Probabilidades guardadas en ordinal_probs.npz")

# =============================================================================
# Pasamos ahora de probabilidades binarias independientes a una clasificacion ordinal completa, combinando los tres modelos bajo restricciones logicas reales para reconstruir la clase final de forma mas coherente
# =============================================================================
print("\n" + "="*60)
print("Optimizando thresholds ordinales...")
print("="*60)


def ordinal_predict(p1, p2, p3, t1, t2, t3, ve_total):
    # Esta funcion convierte probabilidades acumulativas en clases finales, aplicando thresholds independientes y limitaciones fisicas para evitar predicciones imposibles como multiples conductores ebrios en accidentes con pocos vehiculos
    n = len(p1)
    preds = np.zeros(n, dtype=int)
    # Si supera el threshold base pasa al menos a clase 1
    preds[p1 > t1] = 1
    mask2 = (p2 > t2) & (ve_total >= 2)
    # Solo permitimos clase 2 si la probabilidad lo justifica y existen al menos 2 vehiculos
    preds[mask2] = 2
    # Clase 3 aun mas restrictiva tanto en probabilidad como en estructura fisica
    mask3 = (p3 > t3) & (ve_total >= 3)
    preds[mask3] = 3
    return preds


# Realizamos grid search exhaustivo sobre thresholds porque el corte estandar 0.5 rara vez es optimo en datasets muy desbalanceados y ordinales
best_score = 0
best_thresholds = (0.5, 0.5, 0.5)

for t1 in np.arange(0.25, 0.75, 0.01):
    for t2 in np.arange(0.005, 0.60, 0.005):
        for t3 in np.arange(0.001, 0.40, 0.005):
            preds = ordinal_predict(oof_p1, oof_p2, oof_p3, t1, t2, t3, ve_total_train)
            score = balanced_accuracy_score(y_raw, preds)
            if score > best_score:
            # Conservamos la mejor combinacion global encontrada
                best_score = score
                best_thresholds = (t1, t2, t3)

t1, t2, t3 = best_thresholds
print(f"Thresholds: t1={t1:.3f}, t2={t2:.3f}, t3={t3:.3f}")
print(f"CV Balanced Accuracy: {best_score:.5f}")

# Evaluamos recall individual por clase para entender donde rinde mejor o peor el sistema, algo importante ya que balanced accuracy resume pero puede ocultar debilidades especificas
oof_final = ordinal_predict(oof_p1, oof_p2, oof_p3, t1, t2, t3, ve_total_train)
print("\nRecall por clase (OOF):")
for cls in range(4):
    mask = (y_raw == cls)
    if mask.sum() > 0:
        recall = (oof_final[mask] == cls).mean()
        print(f"  Clase {cls}: {recall:.4f} ({mask.sum()} muestras)")

# Revisamos tambien distribucion final para asegurarnos de que el modelo no colapsa hacia una sola clase
oof_dist = pd.Series(oof_final).value_counts().sort_index().to_dict()
print(f"\nDistribucion OOF: {oof_dist}")

# Hacemos una validacion adicional de robustez optimizando thresholds en subsets distintos para comprobar estabilidad y reducir riesgo de sobreajuste al set completo
print("\nCross-validated thresholds:")
cv_thresholds = []
skf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (opt_idx, eval_idx) in enumerate(skf_cv.split(x_train_feat, y_raw)):
    best_cv = 0
    best_t = (0.5, 0.5, 0.5)
    # Busqueda algo mas gruesa para reducir coste computacional    
    for t1_ in np.arange(0.30, 0.65, 0.02):
        for t2_ in np.arange(0.01, 0.55, 0.02):
            for t3_ in np.arange(0.005, 0.35, 0.01):
                p = ordinal_predict(oof_p1[opt_idx], oof_p2[opt_idx], oof_p3[opt_idx],
                                    t1_, t2_, t3_, ve_total_train[opt_idx])
                s = balanced_accuracy_score(y_raw.iloc[opt_idx], p)
                if s > best_cv:
                    best_cv = s
                    best_t = (t1_, t2_, t3_)
    cv_thresholds.append(best_t)
    # Probamos thresholds encontrados sobre fold externo no visto
    p_eval = ordinal_predict(oof_p1[eval_idx], oof_p2[eval_idx], oof_p3[eval_idx],
                             best_t[0], best_t[1], best_t[2], ve_total_train[eval_idx])
    s_eval = balanced_accuracy_score(y_raw.iloc[eval_idx], p_eval)
    print(f"  Fold {fold}: t=({best_t[0]:.2f}, {best_t[1]:.2f}, {best_t[2]:.3f}) -> eval BA={s_eval:.5f}")

# Usamos la mediana de thresholds como opcion mas robusta frente a fluctuaciones entre folds
cv_arr = np.array(cv_thresholds)
t1_med, t2_med, t3_med = np.median(cv_arr, axis=0)
print(f"\nMedian CV thresholds: t1={t1_med:.3f}, t2={t2_med:.3f}, t3={t3_med:.3f}")

# Evaluamos esta variante robusta
oof_med = ordinal_predict(oof_p1, oof_p2, oof_p3, t1_med, t2_med, t3_med, ve_total_train)
score_med = balanced_accuracy_score(y_raw, oof_med)
print(f"Median thresholds OOF BA: {score_med:.5f}")

# =============================================================================
# Finalmente generamos predicciones reales sobre test y creamos archivos de submission listos para Kaggle
# =============================================================================
print("\n" + "="*60)
print("Generando submissions...")
print("="*60)

# Submission principal usando thresholds optimizados globalmente
predictions = ordinal_predict(test_p1, test_p2, test_p3, t1, t2, t3, ve_total_test)
sub = pd.DataFrame({'Id': range(1, len(predictions) + 1), 'Label': predictions})
sub.to_csv('submission.csv', index=False)

# Mostramos distribucion de clases predicha para detectar posibles desviaciones anormales
print(f"\nsubmission.csv (t1={t1:.3f}, t2={t2:.3f}, t3={t3:.3f}):")
for cls in range(4):
    n = (predictions == cls).sum()
    print(f"  Clase {cls}: {n} ({n/len(predictions)*100:.1f}%)")

# Generamos tambien version mas robusta basada en thresholds medianos por seguridad competitiva
pred_med = ordinal_predict(test_p1, test_p2, test_p3, t1_med, t2_med, t3_med, ve_total_test)
sub_med = pd.DataFrame({'Id': range(1, len(pred_med) + 1), 'Label': pred_med})
sub_med.to_csv('submission_median.csv', index=False)

print(f"\nsubmission_median.csv (t1={t1_med:.3f}, t2={t2_med:.3f}, t3={t3_med:.3f}):")
for cls in range(4):
    n = (pred_med == cls).sum()
    print(f"  Clase {cls}: {n} ({n/len(pred_med)*100:.1f}%)")

# Verificacion final para asegurarnos de que el numero de filas coincide exactamente con el esperado por competicion
assert len(sub) == len(x_test), "ERROR: filas no coinciden"
print(f"\nVerificacion OK: {len(sub)} predicciones")

# =============================================================================
# RESUMEN
# =============================================================================
print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print(f"Model 1 (>=1): {balanced_accuracy_score(y1, (oof_p1>0.5).astype(int)):.5f}")
print(f"Model 2 (>=2): {balanced_accuracy_score(y2, (oof_p2>0.5).astype(int)):.5f}")
print(f"Model 3 (>=3): heuristico")
print(f"Ordinal 4-class CV: {best_score:.5f}")
print(f"Best thresholds: t1={t1:.3f}, t2={t2:.3f}, t3={t3:.3f}")
print(f"Median thresholds CV: {score_med:.5f}")
print(f"Median thresholds: t1={t1_med:.3f}, t2={t2_med:.3f}, t3={t3_med:.3f}")
