import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def ks_test(data):
    stat, p_value = stats.kstest(data, 'norm')
    return stat, p_value

def perform_ks_test(df):
    df['duration_minutes'] = df['duration'].apply(convert_duration)
    
    movie_df = df[df['type'] == 'Movie']['duration_minutes']
    tv_show_df = df[df['type'] == 'TV Show']['duration_minutes']
    
    stat_movie, p_value_movie = ks_test(movie_df)
    stat_tv_show, p_value_tv_show = ks_test(tv_show_df)
    
    results = pd.DataFrame({
        'Tipo': ['Películas', 'Series de TV'],
        'Estadístico': [stat_movie, stat_tv_show],
        'p-valor': [p_value_movie, p_value_tv_show]
    })
    
    return results

def plot_ks_test_results(results):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Crear una tabla visual de los resultados del Test de Kolmogorov-Smirnov
    table = ax.table(cellText=results.values,
                     colLabels=results.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    plt.title('Resultados del Test de Kolmogorov-Smirnov', fontsize=15, fontweight='bold')
    plt.show()

def enhanced_ks_test_plot(df):
    df['duration_minutes'] = df['duration'].apply(convert_duration)
    
    movie_df = df[df['type'] == 'Movie']['duration_minutes']
    tv_show_df = df[df['type'] == 'TV Show']['duration_minutes']
    
    stat_movie, p_value_movie = ks_test(movie_df)
    stat_tv_show, p_value_tv_show = ks_test(tv_show_df)
    
    # Data for visualization
    data = {
        'Categoría': ['Películas', 'Series de TV'],
        'Estadístico': [stat_movie, stat_tv_show],
        'p-valor': [p_value_movie, p_value_tv_show]
    }
    
    results_df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Create a seaborn barplot for p-values
    sns.barplot(x='Categoría', y='p-valor', data=results_df, hue='Categoría', palette='viridis', ax=ax, dodge=False)
    
    # Check if legend exists and remove it
    if ax.legend_:
        ax.legend_.remove()
        
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2e', padding=5, fontsize=12)
    
    ax.set_title('Resultados del Test de Kolmogorov-Smirnov', fontsize=18, fontweight='bold')
    ax.set_xlabel('Categoría', fontsize=14)
    ax.set_ylabel('p-valor', fontsize=14)
    
    plt.show()

# Realizar el test y mostrar resultados
ks_results = perform_ks_test(df)
plot_ks_test_results(ks_results)

# Graficar resultados mejorados
enhanced_ks_test_plot(df)
