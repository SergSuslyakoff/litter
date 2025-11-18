pip install h3

# =============================================
# ГЕОФИЧИ С H3 - ПОЛНЫЙ ПАЙПЛАЙН С ПОЯСНЕНИЯМИ
# =============================================

import h3
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt

"""
H3 - это шестиугольная иерархическая система геопространственного индексирования.
Основные концепции:
- H3 индекс: уникальный идентификатор шестиугольника
- Resolution (разрешение): от 0 (крупные шестиугольники) до 15 (мелкие)
- Каждый шестиугольник имеет соседей, центр, границы
"""

# ЗАГРУЗКА ДАННЫХ С ГЕОКООРДИНАТАМИ
# Предположим, у нас есть данные с широтой и долготой
data = pd.DataFrame({
    'id': range(100),
    'latitude': np.random.uniform(55.0, 56.0, 100),  # пример: Москва
    'longitude': np.random.uniform(37.0, 38.0, 100),
    'target': np.random.randint(0, 2, 100)  # целевая переменная
})

print("Первые 5 строк данных:")
print(data.head())

"""
РАЗРЕШЕНИЯ H3 (res):
- res 0: ~110 км в поперечнике
- res 1: ~50 км
- res 2: ~20 км
- res 3: ~10 км
- res 4: ~5 км
- res 5: ~2 км
- res 6: ~1 км
- res 7: ~500 м
- res 8: ~200 м
- res 9: ~100 м
- res 10: ~50 м
- res 11: ~20 м
- res 12: ~10 м
- res 13: ~5 м
- res 14: ~2 м
- res 15: ~1 м

Для городских данных обычно используют res 8-10
"""

# ОСНОВНАЯ ФУНКЦИЯ: ПРЕОБРАЗОВАНИЕ КООРДИНАТ В H3 ИНДЕКС
def add_h3_features(df, lat_col='latitude', lon_col='longitude', resolutions=[7, 8, 9]):
    """
    Добавляет H3 индексы разных разрешений в DataFrame
    
    Parameters:
    - df: исходный DataFrame
    - lat_col: название колонки с широтой
    - lon_col: название колонки с долготой  
    - resolutions: список разрешений H3 для создания
    """
    df = df.copy()
    
    for res in resolutions:
        # Преобразование координат в H3 индекс
        df[f'h3_res_{res}'] = df.apply(
            lambda row: h3.latlng_to_cell(
                row[lat_col], 
                row[lon_col], 
                res
            ), 
            axis=1
        )
        
        # Также можно получить координаты центра шестиугольника
        df[f'h3_center_lat_{res}'] = df[f'h3_res_{res}'].apply(
            lambda h: h3.cell_to_latlng(h)[0] if pd.notna(h) else None
        )
        df[f'h3_center_lon_{res}'] = df[f'h3_res_{res}'].apply(
            lambda h: h3.cell_to_latlng(h)[1] if pd.notna(h) else None
        )
    
    return df

# ДОБАВЛЕНИЕ ОСНОВНЫХ H3 ИНДЕКСОВ
print("Добавляем H3 индексы...")
data_with_h3 = add_h3_features(data, resolutions=[7, 8, 9])
print("Данные с H3 индексами:")
print(data_with_h3[['latitude', 'longitude', 'h3_res_8']].head())

# ФУНКЦИЯ ДЛЯ РАСЧЕТА РАССТОЯНИЯ МЕЖДУ ТОЧКАМИ
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Вычисляет расстояние в км между двумя точками на Земле
    используя формулу Haversine
    """
    R = 6371  # радиус Земли в км
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

# ФУНКЦИЯ ДЛЯ СОЗДАНИЯ РАСШИРЕННЫХ ГЕОФИЧЕЙ
def create_advanced_geo_features(df, lat_col='latitude', lon_col='longitude', h3_col='h3_res_8'):
    """
    Создает расширенные геопространственные признаки
    """
    df = df.copy()
    
    # 1. ПЛОТНОСТЬ ТОЧЕК В ШЕСТИУГОЛЬНИКЕ
    h3_counts = df[h3_col].value_counts().to_dict()
    df['points_in_h3_cell'] = df[h3_col].map(h3_counts)
    
    # 2. РАССТОЯНИЕ ДО ЦЕНТРА ШЕСТИУГОЛЬНИКА
    df['distance_to_h3_center'] = df.apply(
        lambda row: haversine_distance(
            row[lat_col], row[lon_col],
            h3.cell_to_latlng(row[h3_col])[0],
            h3.cell_to_latlng(row[h3_col])[1]
        ), axis=1
    )
    
    # 3. СОСЕДНИЕ ШЕСТИУГОЛЬНИКИ (количество соседей)
    df['h3_neighbors_count'] = df[h3_col].apply(
        lambda x: len(h3.grid_disk(x, 1)) if pd.notna(x) else 0
    )
    
    # 4. ПЛОЩАДЬ ШЕСТИУГОЛЬНИКА (в кв. км)
    df['h3_cell_area_km2'] = df[h3_col].apply(
        lambda x: h3.cell_area(x, unit='km^2') if pd.notna(x) else 0
    )
    
    # 5. РАССТОЯНИЕ ДО ЦЕНТРА ГОРОДА (пример для Москвы)
    MOSCOW_CENTER = (55.7558, 37.6173)
    df['distance_to_city_center'] = df.apply(
        lambda row: haversine_distance(
            row[lat_col], row[lon_col],
            MOSCOW_CENTER[0], MOSCOW_CENTER[1]
        ), axis=1
    )
    
    # 6. ГЕОГРАФИЧЕСКИЕ КЛАСТЕРЫ (группировка по крупным шестиугольникам)
    df['h3_cluster_res5'] = df.apply(
        lambda row: h3.latlng_to_cell(row[lat_col], row[lon_col], 5), 
        axis=1
    )
    
    return df

# СОЗДАНИЕ РАСШИРЕННЫХ ГЕОФИЧЕЙ
print("Создаем расширенные геофичи...")
data_with_geo_features = create_advanced_geo_features(data_with_h3)
print("Расширенные геофичи:")
print(data_with_geo_features[['points_in_h3_cell', 'distance_to_h3_center', 'h3_neighbors_count']].head())

# ФУНКЦИЯ ДЛЯ АГРЕГАЦИОННЫХ ПРИЗНАКОВ
def create_aggregated_geo_features(df, group_col='h3_res_8', target_col='target'):
    """
    Создает агрегированные признаки по H3 ячейкам
    """
    # Агрегация по H3 ячейкам
    h3_aggregations = df.groupby(group_col).agg({
        target_col: ['mean', 'sum', 'count', 'std'],
        'latitude': 'count',  # количество точек в ячейке
        'distance_to_city_center': 'mean'
    }).reset_index()
    
    # Упрощение названий колонок
    h3_aggregations.columns = [
        f'{group_col}_{col[0]}_{col[1]}' if col[1] != '' else group_col 
        for col in h3_aggregations.columns
    ]
    
    # Переименовываем основные колонки
    h3_aggregations = h3_aggregations.rename(columns={
        f'{group_col}_target_mean': 'h3_target_mean',
        f'{group_col}_target_sum': 'h3_target_sum', 
        f'{group_col}_target_count': 'h3_point_count',
        f'{group_col}_target_std': 'h3_target_std',
        f'{group_col}_latitude_count': 'h3_density',
        f'{group_col}_distance_to_city_center_mean': 'h3_avg_distance_to_center'
    })
    
    # Объединяем с исходными данными
    result_df = df.merge(h3_aggregations, on=group_col, how='left')
    
    return result_df

# СОЗДАНИЕ АГРЕГАЦИОННЫХ ПРИЗНАКОВ
print("Создаем агрегированные геофичи...")
final_data = create_aggregated_geo_features(data_with_geo_features)
print("Агрегированные геофичи:")
print(final_data[['h3_res_8', 'h3_target_mean', 'h3_density']].head())

# ВИЗУАЛИЗАЦИЯ H3 СЕТКИ (опционально)
def visualize_h3_grid(df, h3_col='h3_res_8', sample_size=50):
    """
    Визуализирует H3 сетку на карте
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        
        # Берем sample для визуализации
        sample_df = df.sample(min(sample_size, len(df)))
        
        # Создаем геометрию для H3 шестиугольников
        geometries = []
        for h3_index in sample_df[h3_col].unique():
            # Получаем границы шестиугольника
            boundary = h3.cell_to_boundary(h3_index)
            polygon = Polygon(boundary)
            geometries.append(polygon)
        
        # Создаем GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'h3_index': sample_df[h3_col].unique(),
            'geometry': geometries
        })
        
        # Визуализируем
        fig, ax = plt.subplots(figsize=(12, 8))
        gdf.plot(ax=ax, alpha=0.5, edgecolor='black')
        
        # Добавляем исходные точки
        plt.scatter(sample_df['longitude'], sample_df['latitude'], 
                   c='red', s=10, alpha=0.7, label='Исходные точки')
        
        plt.title('H3 сетка и исходные точки')
        plt.xlabel('Долгота')
        plt.ylabel('Широта')
        plt.legend()
        plt.show()
        
    except ImportError:
        print("Для визуализации установите geopandas: pip install geopandas")

# ВИЗУАЛИЗАЦИЯ (раскомментируйте если нужно)
# visualize_h3_grid(final_data)

# ФУНКЦИЯ ДЛЯ ПОИСКА БЛИЖАЙШИХ ОБЪЕКТОВ
def find_nearest_pois(df, pois_df, lat_col='latitude', lon_col='longitude', k=3):
    """
    Находит k ближайших POI (Points of Interest) для каждой точки
    """
    from sklearn.neighbors import BallTree
    import numpy as np
    
    # Конвертируем координаты в радианы для BallTree
    df_rad = np.deg2rad(df[[lat_col, lon_col]].values)
    pois_rad = np.deg2rad(pois_df[[lat_col, lon_col]].values)
    
    # Создаем BallTree для быстрого поиска соседей
    tree = BallTree(pois_rad, metric='haversine')
    
    # Находим k ближайших POI
    distances, indices = tree.query(df_rad, k=k)
    
    # Конвертируем расстояния из радиан в километры
    earth_radius_km = 6371
    distances_km = distances * earth_radius_km
    
    # Добавляем результаты в DataFrame
    for i in range(k):
        df[f'distance_to_poi_{i+1}'] = distances_km[:, i]
        df[f'nearest_poi_type_{i+1}'] = pois_df.iloc[indices[:, i]]['poi_type'].values
    
    return df

# ПРИМЕР ИСПОЛЬЗОВАНИЯ С POI (Points of Interest)
# Создаем пример данных POI
pois_data = pd.DataFrame({
    'poi_id': range(10),
    'poi_type': np.random.choice(['metro', 'shop', 'park', 'school'], 10),
    'latitude': np.random.uniform(55.0, 56.0, 10),
    'longitude': np.random.uniform(37.0, 38.0, 10)
})

print("Пример POI данных:")
print(pois_data.head())

# Добавляем информацию о ближайших POI (раскомментируйте если нужно)
# final_data_with_poi = find_nearest_pois(final_data, pois_data, k=2)
# print("Данные с информацией о POI:")
# print(final_data_with_poi[['distance_to_poi_1', 'nearest_poi_type_1']].head())

# ФИНАЛЬНЫЙ ПОДГОТОВКА ДАННЫХ ДЛЯ ML
def prepare_for_ml(df):
    """
    Подготавливает финальный DataFrame для машинного обучения
    """
    ml_df = df.copy()
    
    # Кодируем категориальные H3 индексы
    for col in ml_df.columns:
        if col.startswith('h3_res_'):
            # Преобразуем H3 индекс в числовой формат
            ml_df[f'{col}_numeric'] = ml_df[col].apply(
                lambda x: int(x, 16) if pd.notna(x) else 0
            )
    
    # Удаляем исходные текстовые H3 индексы
    h3_text_cols = [col for col in ml_df.columns if col.startswith('h3_res_') and not col.endswith('_numeric')]
    ml_df = ml_df.drop(columns=h3_text_cols)
    
    # Заполняем пропуски
    ml_df = ml_df.fillna(0)
    
    return ml_df

# ПОДГОТОВКА ДАННЫХ ДЛЯ ML
final_ml_data = prepare_for_ml(final_data)
print("Финальные данные для ML:")
print(final_ml_data.info())
print("\nКолонки с геофичами:")
geo_columns = [col for col in final_ml_data.columns if any(x in col for x in ['h3', 'distance', 'poi'])]
print(geo_columns)

# ПРИМЕР ИСПОЛЬЗОВАНИЯ С AUTOML (продолжение из предыдущих примеров)
from autogluon.tabular import TabularPredictor

# Если хотите использовать с AutoGluon:
# predictor = TabularPredictor(label='target').fit(
#     train_data=final_ml_data.drop('target', axis=1),
#     time_limit=300
# )

print("\n" + "="*50)
print("СОЗДАНО ГЕОФИЧЕЙ:")
print(f"- Базовые H3 индексы: {len([col for col in final_ml_data.columns if 'h3_res_' in col])}")
print(f"- Расширенные геофичи: {len([col for col in final_ml_data.columns if 'distance' in col or 'density' in col])}")
print(f"- Агрегированные фичи: {len([col for col in final_ml_data.columns if 'h3_target' in col or 'h3_avg' in col])}")
print("="*50)
