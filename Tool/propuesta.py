import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Cargar dataset
data = pd.read_csv("netflix_titles.csv")

# 1. Regresión Lineal para Predecir la Duración
# Hipótesis: La duración de una película/serie podría relacionarse con su género, director, clasificación por edades y año de estreno.

# Definir variables
X_duration = data[['listed_in', 'director', 'rating', 'release_year']]
y_duration = data['duration'].str.extract('(\d+)').astype(float)  # Extraer duración numérica

# Codificar variables categóricas
preprocessor_duration = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['listed_in', 'director', 'rating'])
    ],
    remainder='passthrough'
)

# Dividir datos
X_train_dur, X_test_dur, y_train_dur, y_test_dur = train_test_split(X_duration, y_duration, test_size=0.2, random_state=42)

# Aplicar preprocesamiento
X_train_dur_processed = preprocessor_duration.fit_transform(X_train_dur)
X_test_dur_processed = preprocessor_duration.transform(X_test_dur)

# Entrenar modelo
model_duration = LinearRegression()
model_duration.fit(X_train_dur_processed, y_train_dur)

# Evaluar modelo
y_pred_dur = model_duration.predict(X_test_dur_processed)
print(f"Predicción de Duración - R²: {r2_score(y_test_dur, y_pred_dur):.2f}")
print(f"Predicción de Duración - MAE: {mean_absolute_error(y_test_dur, y_pred_dur):.2f} minutos")

# Evaluación y optimización
# La regresión lineal básica muestra una relación débil entre las variables independientes y la duración (R² bajo).
# Propuesta: Aplicar Ridge y Lasso para mejorar la predicción.

ridge_duration = Ridge(alpha=1.0)
ridge_duration.fit(X_train_dur_processed, y_train_dur)
y_pred_ridge_dur = ridge_duration.predict(X_test_dur_processed)
print(f"Predicción de Duración (Ridge) - R²: {r2_score(y_test_dur, y_pred_ridge_dur):.2f}")
print(f"Predicción de Duración (Ridge) - MAE: {mean_absolute_error(y_test_dur, y_pred_ridge_dur):.2f} minutos")

lasso_duration = Lasso(alpha=0.1)
lasso_duration.fit(X_train_dur_processed, y_train_dur)
y_pred_lasso_dur = lasso_duration.predict(X_test_dur_processed)
print(f"Predicción de Duración (Lasso) - R²: {r2_score(y_test_dur, y_pred_lasso_dur):.2f}")
print(f"Predicción de Duración (Lasso) - MAE: {mean_absolute_error(y_test_dur, y_pred_lasso_dur):.2f} minutos")



# 2. Regresión Lineal para Predecir el Año de Estreno
# Hipótesis: El año de estreno podría correlacionarse con cambios en la clasificación por edades o géneros.

# Definir variables
X_year = data[['listed_in', 'rating', 'duration', 'director']]
y_year = data['release_year']

# Codificar variables categóricas
preprocessor_year = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['listed_in', 'rating', 'director'])
    ],
    remainder='passthrough'
)

# Dividir datos
X_train_year, X_test_year, y_train_year, y_test_year = train_test_split(X_year, y_year, test_size=0.2, random_state=42)

# Aplicar preprocesamiento
X_train_year_processed = preprocessor_year.fit_transform(X_train_year)
X_test_year_processed = preprocessor_year.transform(X_test_year)

# Entrenar modelo
model_year = LinearRegression()
model_year.fit(X_train_year_processed, y_train_year)

# Evaluar modelo
y_pred_year = model_year.predict(X_test_year_processed)
print(f"Predicción de Año de Estreno - R²: {r2_score(y_test_year, y_pred_year):.2f}")
print(f"Predicción de Año de Estreno - MAE: {mean_absolute_error(y_test_year, y_pred_year):.2f} años")

# Evaluación y optimización
# La regresión lineal básica muestra una relación débil entre las variables independientes y el año de estreno (R² bajo).
# Propuesta: Aplicar Ridge y Lasso para mejorar la predicción.

ridge_year = Ridge(alpha=1.0)
ridge_year.fit(X_train_year_processed, y_train_year)
y_pred_ridge_year = ridge_year.predict(X_test_year_processed)
print(f"Predicción de Año de Estreno (Ridge) - R²: {r2_score(y_test_year, y_pred_ridge_year):.2f}")
print(f"Predicción de Año de Estreno (Ridge) - MAE: {mean_absolute_error(y_test_year, y_pred_ridge_year):.2f} años")

lasso_year = Lasso(alpha=0.1)
lasso_year.fit(X_train_year_processed, y_train_year)
y_pred_lasso_year = lasso_year.predict(X_test_year_processed)
print(f"Predicción de Año de Estreno (Lasso) - R²: {r2_score(y_test_year, y_pred_lasso_year):.2f}")
print(f"Predicción de Año de Estreno (Lasso) - MAE: {mean_absolute_error(y_test_year, y_pred_lasso_year):.2f} años")

# 3. Regresión Lineal para Predecir la Popularidad
# Hipótesis: La popularidad podría depender del género, rating de edad, actores y mes de estreno.

# Suponiendo que tenemos una columna 'popularidad'
data['month_added'] = pd.to_datetime(data['date_added']).dt.month
X_popularity = data[['listed_in', 'rating', 'cast', 'month_added']]
y_popularity = data['popularidad']  # Suponiendo que esta columna existe

# Codificar mes como variable cíclica
data['month_sin'] = np.sin(2 * np.pi * data['month_added'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month_added'] / 12)
X_popularity = data[['listed_in', 'rating', 'cast', 'month_sin', 'month_cos']]

# Codificar variables categóricas
preprocessor_popularity = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['listed_in', 'rating', 'cast'])
    ],
    remainder='passthrough'
)

# Dividir datos
X_train_pop, X_test_pop, y_train_pop, y_test_pop = train_test_split(X_popularity, y_popularity, test_size=0.2, random_state=42)

# Aplicar preprocesamiento
X_train_pop_processed = preprocessor_popularity.fit_transform(X_train_pop)
X_test_pop_processed = preprocessor_popularity.transform(X_test_pop)

# Entrenar modelo
model_popularity = LinearRegression()
model_popularity.fit(X_train_pop_processed, y_train_pop)

# Evaluar modelo
y_pred_pop = model_popularity.predict(X_test_pop_processed)
print(f"Predicción de Popularidad - R²: {r2_score(y_test_pop, y_pred_pop):.2f}")
print(f"Predicción de Popularidad - MAE: {mean_absolute_error(y_test_pop, y_pred_pop):.2f}")

# Evaluación y optimización
# La regresión lineal básica muestra una relación débil entre las variables independientes y la popularidad (R² bajo).
# Propuesta: Aplicar Ridge y Lasso para mejorar la predicción.

ridge_popularity = Ridge(alpha=1.0)
ridge_popularity.fit(X_train_pop_processed, y_train_pop)
y_pred_ridge_pop = ridge_popularity.predict(X_test_pop_processed)
print(f"Predicción de Popularidad (Ridge) - R²: {r2_score(y_test_pop, y_pred_ridge_pop):.2f}")
print(f"Predicción de Popularidad (Ridge) - MAE: {mean_absolute_error(y_test_pop, y_pred_ridge_pop):.2f}")

lasso_popularity = Lasso(alpha=0.1)
lasso_popularity.fit(X_train_pop_processed, y_train_pop)
y_pred_lasso_pop = lasso_popularity.predict(X_test_pop_processed)
print(f"Predicción de Popularidad (Lasso) - R²: {r2_score(y_test_pop, y_pred_lasso_pop):.2f}")
print(f"Predicción de Popularidad (Lasso) - MAE: {mean_absolute_error(y_test_pop, y_pred_lasso_pop):.2f}")

# Interpretación y Objetivo:
# 1. Predicción de Duración: El objetivo es entender cómo el género, director, clasificación por edades y año de estreno influyen en la duración de películas/series. Las métricas R² y MAE ayudan a evaluar el rendimiento del modelo.
# 2. Predicción de Año de Estreno: Este modelo busca explorar la relación entre el año de estreno y otras características como el género y la clasificación por edades. Las métricas R² y MAE proporcionan información sobre la precisión del modelo.
# 3. Predicción de Popularidad: El objetivo es predecir la popularidad de películas/series basándose en el género, clasificación por edades, actores y mes de estreno. Las métricas R² y MAE indican la efectividad del modelo.

# Nota: Asegúrese de que la columna 'popularidad' exista en el conjunto de datos para el tercer modelo.
