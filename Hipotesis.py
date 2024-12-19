import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('ruta_a_tu_archivo.csv')

# Hipótesis 1: Calificaciones promedio por país
plt.figure(figsize=(10, 6))
average_ratings_by_country = df.groupby('country')['rating'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=average_ratings_by_country.values, y=average_ratings_by_country.index)
plt.title('Calificaciones promedio por país')
plt.xlabel('Calificación promedio')
plt.ylabel('País')
plt.show()

# Hipótesis 2: Evolución del número de títulos agregados
df['date_added'] = pd.to_datetime(df['date_added'])
titles_added_by_year = df['date_added'].dt.year.value_counts().sort_index()
plt.figure(figsize=(10, 6))
titles_added_by_year.plot()
plt.title('Evolución del número de títulos añadidos por año')
plt.xlabel('Año')
plt.ylabel('Número de títulos')
plt.show()

# Hipótesis 3: Distribución de géneros
genres = df['listed_in'].str.split(',').explode()
plt.figure(figsize=(10, 6))
genres.value_counts().head(10).plot(kind='bar')
plt.title('Distribución de géneros en Netflix')
plt.xlabel('Género')
plt.ylabel('Número de títulos')
plt.show()

# Hipótesis 4: Duración promedio por género
df['duration'] = df['duration'].str.extract('(\d+)').astype(float)
average_duration_by_genre = df.groupby('listed_in')['duration'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=average_duration_by_genre.values, y=average_duration_by_genre.index)
plt.title('Duración promedio por género')
plt.xlabel('Duración promedio (minutos)')
plt.ylabel('Género')
plt.show()

# Hipótesis 5: Actores recurrentes
actors = df['cast'].str.split(',').explode()
plt.figure(figsize=(10, 6))
actors.value_counts().head(10).plot(kind='bar')
plt.title('Actores más recurrentes en Netflix')
plt.xlabel('Actor')
plt.ylabel('Número de títulos')
plt.show()
