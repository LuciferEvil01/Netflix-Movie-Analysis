import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Hipotesis as hipotesis

# Cargar el dataset
df = pd.read_csv('.\netflix_titles.csv')

# Exploración inicial del dataset
print(df.head())

# Gráfico de distribución de tipos de contenido (Películas vs Series)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='type')
plt.title('Distribución de tipos de contenido en Netflix')
plt.xlabel('Tipo')
plt.ylabel('Conteo')
plt.show()

# Gráfico de países con más contenido en Netflix
top_countries = df['country'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.index, y=top_countries.values)
plt.title('Top 10 países con más contenido en Netflix')
plt.xlabel('País')
plt.ylabel('Número de contenidos')
plt.xticks(rotation=45)
plt.show()

# Gráfico de la evolución del número de títulos por año
df['release_year'] = pd.to_datetime(df['release_year'], format='%Y')
plt.figure(figsize=(10, 6))
df['release_year'].value_counts().sort_index().plot()
plt.title('Evolución del número de títulos por año')
plt.xlabel('Año')
plt.ylabel('Número de títulos')
plt.show()

# Analizar duración de películas y número de temporadas de series
df['duration'] = df['duration'].str.extract('(\d+)').astype(float)
movies_duration = df[df['type'] == 'Movie']['duration']
shows_seasons = df[df['type'] == 'TV Show']['duration']

# Visualización de la duración de películas
plt.figure(figsize=(10, 6))
sns.histplot(movies_duration, kde=True, color='blue', label='Movies')
plt.title('Distribución de la duración de películas')
plt.xlabel('Duración (minutos)')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

# Visualización del número de temporadas de series
plt.figure(figsize=(10, 6))
sns.histplot(shows_seasons, kde=True, color='green', label='TV Shows')
plt.title('Distribución del número de temporadas de series')
plt.xlabel('Número de temporadas')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()
