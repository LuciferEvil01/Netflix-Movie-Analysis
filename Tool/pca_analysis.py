import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Función para convertir duración a minutos
def convert_duration(duration):
    if pd.isna(duration):
        return 0
    if 'min' in duration:
        return int(duration.replace(' min', ''))
    elif 'Season' in duration:
        return int(duration.split(' ')[0]) * 60  # Asumiendo 1 temporada = 60 minutos
    return 0

# Preparación de los datos
def prepare_data_for_pca(df, content_type):
    df['duration_minutes'] = df['duration'].apply(convert_duration)
    df['year_added'] = pd.to_datetime(df['date_added'], errors='coerce').dt.year
    df_filtered = df[df['type'] == content_type]
    numeric_columns = ['duration_minutes', 'release_year', 'year_added']
    data = df_filtered[numeric_columns].dropna()
    return data

# Realización del PCA con selección de componentes
def perform_pca(data, n_components=None):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_scaled)
    
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
    loadings = pd.DataFrame(pca.components_, columns=data.columns, index=[f'PC{i+1}' for i in range(principal_components.shape[1])])
    
    return pca, pca_df, loadings

# Graficar la varianza explicada
def plot_pca_variance(pca, content_type):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.6, align='center', color='skyblue', label='Varianza Explicada por Componente')
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', color='orange', label='Varianza Explicada Acumulativa')
    plt.xlabel('Componentes Principales')
    plt.ylabel('Varianza Explicada')
    plt.title(f'Varianza Explicada por Cada Componente Principal ({content_type})')
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
def plot_pca_3d(pca_df, content_type):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c='skyblue', s=50)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(f'PCA 3D Visualization for {content_type}')
    
    plt.show()

# Determinar el número de componentes para explicar al menos el 90% de la varianza
def determine_n_components(pca, threshold=0.90):
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    return n_components

# Análisis de PCA para Películas y Series de TV
data_movies = prepare_data_for_pca(df, 'Movie')
pca_movies, pca_df_movies, loadings_movies = perform_pca(data_movies)

data_tv_shows = prepare_data_for_pca(df, 'TV Show')
pca_tv_shows, pca_df_tv_shows, loadings_tv_shows = perform_pca(data_tv_shows)

# Determinar el número de componentes para películas y series de TV
n_components_movies = determine_n_components(pca_movies)
n_components_tv_shows = determine_n_components(pca_tv_shows)

# Realizar PCA con el número óptimo de componentes
pca_movies, pca_df_movies, loadings_movies = perform_pca(data_movies, n_components=n_components_movies)
pca_tv_shows, pca_df_tv_shows, loadings_tv_shows = perform_pca(data_tv_shows, n_components=n_components_tv_shows)

# Mostrar los resultados del PCA para películas
print(f"Componentes Principales para Películas (n={n_components_movies})")
display(pca_df_movies.head())
print("Cargas de los Componentes Principales para Películas")
display(loadings_movies)

# Mostrar los resultados del PCA para series de TV
print(f"Componentes Principales para Series de TV (n={n_components_tv_shows})")
display(pca_df_tv_shows.head())
print("Cargas de los Componentes Principales para Series de TV")
display(loadings_tv_shows)

# Graficar la varianza explicada para películas
plot_pca_variance(pca_movies, 'Películas')

# Graficar la varianza explicada para series de TV
plot_pca_variance(pca_tv_shows, 'Series de TV')


plot_pca_3d(pca_df_movies, 'Películas')
plot_pca_3d(pca_df_tv_shows, 'Series de TV')