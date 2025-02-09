# statistics.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Función para convertir la duración
def convert_duration(duration):
    if pd.isna(duration):
        return 0
    if 'min' in duration:
        return int(duration.replace(' min', ''))
    elif 'Season' in duration:
        return int(duration.split(' ')[0])  # Asumimos 1 temporada
    return 0

# Función para calcular estadísticas descriptivas
def calculate_statistics(df):
    df['duration_minutes'] = df['duration'].apply(convert_duration)
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].apply(lambda x: x.year if pd.notna(x) else None)

    movie_df = df[df['type'] == 'Movie']
    tv_show_df = df[df['type'] == 'TV Show']

    stats = {
        "Duración de Películas": {
            "Duración Media (minutos)": round(movie_df['duration_minutes'].mean(), 3),
            "Duración Mediana (minutos)": round(movie_df['duration_minutes'].median(), 3),
            "Moda de la Duración (minutos)": movie_df['duration_minutes'].mode()[0],
            "Varianza (minutos^2)": round(movie_df['duration_minutes'].var(), 3),
            "Desviación Estándar (minutos)": round(movie_df['duration_minutes'].std(), 3),
            "Percentil 25 (minutos)": round(movie_df['duration_minutes'].quantile(0.25), 3),
            "Percentil 50 (mediana) (minutos)": round(movie_df['duration_minutes'].quantile(0.50), 3),
            "Percentil 75 (minutos)": round(movie_df['duration_minutes'].quantile(0.75), 3)
        },
        "Duración de Series de TV": {
            "Duración Media (temporadas)": round(tv_show_df['duration_minutes'].mean(), 3),
            "Duración Mediana (temporadas)": round(tv_show_df['duration_minutes'].median(), 3),
            "Moda de la Duración (temporadas)": tv_show_df['duration_minutes'].mode()[0],
            "Varianza (temporadas^2)": round(tv_show_df['duration_minutes'].var(), 3),
            "Desviación Estándar (temporadas)": round(tv_show_df['duration_minutes'].std(), 3),
            "Percentil 25 (temporadas)": round(tv_show_df['duration_minutes'].quantile(0.25), 3),
            "Percentil 50 (mediana) (temporadas)": round(tv_show_df['duration_minutes'].quantile(0.50), 3),
            "Percentil 75 (temporadas)": round(tv_show_df['duration_minutes'].quantile(0.75), 3)
        },
        "Año de Estreno": {
            "Mediana": round(df['release_year'].median()),
            "Moda": df['release_year'].mode()[0],
            "Varianza": round(df['release_year'].var()),
            "Desviación Estándar": round(df['release_year'].std()),
            "Percentil 25": round(df['release_year'].quantile(0.25)),
            "Percentil 50 (mediana)": round(df['release_year'].quantile(0.50)),
            "Percentil 75": round(df['release_year'].quantile(0.75))
        },
        "Año de Adición a Netflix": {
            "Mediana": round(df['year_added'].median()),
            "Moda": df['year_added'].mode()[0],
            "Varianza": round(df['year_added'].var()),
            "Desviación Estándar": round(df['year_added'].std()),
            "Percentil 25": round(df['year_added'].quantile(0.25)),
            "Percentil 50 (mediana)": round(df['year_added'].quantile(0.50)),
            "Percentil 75": round(df['year_added'].quantile(0.75))
        }
    }

    return stats

# Métodos para calcular directores con más contenido
def top_movie_directors(df, n=10):
    movie_director_counts = df[df['type'] == 'Movie']['director'].value_counts().nlargest(n)
    return movie_director_counts

def top_tv_show_directors(df, n=10):
    tv_show_director_counts = df[df['type'] == 'TV Show']['director'].value_counts().nlargest(n)
    return tv_show_director_counts

# Métodos para calcular países con más contenido
def top_movie_countries(df, n=10):
    movie_country_counts = df[df['type'] == 'Movie']['country'].value_counts().nlargest(n)
    return movie_country_counts

def top_tv_show_countries(df, n=10):
    tv_show_country_counts = df[df['type'] == 'TV Show']['country'].value_counts().nlargest(n)
    return tv_show_country_counts

# Métodos para graficar directores con más contenido
def plot_top_movie_directors(movie_director_counts):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=movie_director_counts.values, y=movie_director_counts.index, palette='viridis', hue=movie_director_counts.index, dodge=False, legend=False)
    plt.title('Top Directores de Películas con Más Contenidos')
    plt.xlabel('Cantidad de Contenidos')
    plt.ylabel('Director')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

def plot_top_tv_show_directors(tv_show_director_counts):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=tv_show_director_counts.values, y=tv_show_director_counts.index, palette='viridis', hue=tv_show_director_counts.index, dodge=False, legend=False)
    plt.title('Top Directores de Series de TV con Más Contenidos')
    plt.xlabel('Cantidad de Contenidos')
    plt.ylabel('Director')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

# Métodos para graficar países con más contenido
def plot_top_movie_countries(movie_country_counts):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=movie_country_counts.values, y=movie_country_counts.index, palette='viridis', hue=movie_country_counts.index, dodge=False, legend=False)
    plt.title('Top Países de Películas con Más Contenidos')
    plt.xlabel('Cantidad de Contenidos')
    plt.ylabel('País')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

def plot_top_tv_show_countries(tv_show_country_counts):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=tv_show_country_counts.values, y=tv_show_country_counts.index, palette='viridis', hue=tv_show_country_counts.index, dodge=False, legend=False)
    plt.title('Top Países de Series de TV con Más Contenidos')
    plt.xlabel('Cantidad de Contenidos')
    plt.ylabel('País')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()
    
    
def calculate_category_counts_by_type(df, content_type):
    # Filtrar por tipo de contenido
    filtered_df = df[df['type'] == content_type]

    # Crear una lista vacía para almacenar las categorías
    all_categories = []

    # Recorrer cada fila en el DataFrame
    for categories in filtered_df['listed_in'].dropna():
        # Separar las categorías por coma y eliminar espacios en blanco
        category_list = [category.strip() for category in categories.split(',')]
        # Añadir las categorías a la lista
        all_categories.extend(category_list)

    # Contar la frecuencia de cada categoría
    category_counts = pd.Series(all_categories).value_counts()

    return category_counts

def plot_top_categories_by_type(category_counts, content_type, n=10):
    # Seleccionar las n categorías más comunes
    top_categories = category_counts.nlargest(n)

    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_categories.values, y=top_categories.index, hue=top_categories.index, palette='viridis', dodge=False, legend=False)
    plt.title(f'Top Categorías Más Comunes en {content_type}')
    plt.xlabel('Cantidad de Contenidos')
    plt.ylabel('Categoría')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
