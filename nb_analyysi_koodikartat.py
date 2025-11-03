# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Theme Maps
# 
# This notebook creates maps from observations and their themes.

# %% 
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
import os

# %%
# Load and merge datasets
try:
    themes = pd.read_csv('data/bird-metadata/themes.csv', index_col='rec_id')
    recs = pd.read_csv('data/bird-metadata/recs_since_June25.csv')
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure the data files are in the 'data/bird-metadata/' directory.")
    exit()

# Merge recording metadata with themes
merged_data = pd.merge(themes, recs, left_index=True, right_on='rec_id')

# %%
# Create GeoDataFrame
geometry = [Point(xy) for xy in zip(merged_data['lon'], merged_data['lat'])]
gdf = gpd.GeoDataFrame(merged_data, geometry=geometry, crs="EPSG:4326")

# %% [markdown]
# ## Visualization Experiments
# 
# Trying out five different ways to visualize the data on a map.

# %% 
# --- Visualization 1: Scatter Plot ---
def plot_scatter(gdf, theme_name, output_filename):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world[world.name == 'Finland'].plot(color='white', edgecolor='black', ax=ax)
    gdf.plot(ax=ax, column=theme_name, cmap='viridis', markersize=10, alpha=0.6, legend=True)
    plt.title(f'Theme: {theme_name}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(output_filename)
    plt.close(fig)

# %% 
# --- Visualization 2: Hexbin Plot ---
def plot_hexbin(gdf, theme_name, output_filename):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world[world.name == 'Finland'].plot(color='white', edgecolor='black', ax=ax)
    
    gdf = gdf[gdf.geometry.is_valid]
    if not gdf.empty:
        hb = ax.hexbin(gdf.geometry.x, gdf.geometry.y, C=gdf[theme_name], gridsize=50, cmap='inferno', alpha=0.7)
        fig.colorbar(hb, ax=ax, label=f'{theme_name} Probability')
    
    plt.title(f'Theme: {theme_name} (Hexbin)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(output_filename)
    plt.close(fig)

# %%
# --- Visualization 3: Kernel Density Estimation (KDE) ---
def plot_kde(gdf, theme_name, output_filename):
    import seaborn as sns
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world[world.name == 'Finland'].plot(color='white', edgecolor='black', ax=ax)
    
    gdf = gdf[gdf.geometry.is_valid]
    if not gdf.empty:
        sns.kdeplot(x=gdf.geometry.x, y=gdf.geometry.y, weights=gdf[theme_name], cmap="Reds", shade=True, alpha=0.5, ax=ax)
    
    plt.title(f'Theme: {theme_name} (KDE)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(output_filename)
    plt.close(fig)

# %%
# --- Visualization 4: Bubble Plot ---
def plot_bubble(gdf, theme_name, output_filename):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world[world.name == 'Finland'].plot(color='white', edgecolor='black', ax=ax)
    
    sizes = gdf[theme_name] * 100
    gdf.plot(ax=ax, column=theme_name, cmap='coolwarm', markersize=sizes, alpha=0.7, legend=True)
    
    plt.title(f'Theme: {theme_name} (Bubble Plot)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(output_filename)
    plt.close(fig)

# %% 
# --- Visualization 5: Stamen Toner Base Map ---
def plot_stamen(gdf, theme_name, output_filename):
    import contextily as cx
    
    gdf_wm = gdf.to_crs(epsg=3857)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    gdf_wm = gdf_wm[gdf_wm.geometry.is_valid]
    if not gdf_wm.empty:
        gdf_wm.plot(ax=ax, column=theme_name, cmap='gist_heat', markersize=15, alpha=0.7, legend=True)
        cx.add_basemap(ax, crs=gdf_wm.crs, source=cx.providers.Stamen.TonerLite)
    
    plt.title(f'Theme: {theme_name} (Stamen Toner)')
    ax.set_axis_off()
    plt.savefig(output_filename)
    plt.close(fig)

# %% 
if __name__ == '__main__':
    theme_names = themes.columns

    # Create a directory for the maps
    if not os.path.exists('maps'):
        os.makedirs('maps')

    for theme in theme_names:
        print(f"Generating maps for theme: {theme}...")
        theme_dir = os.path.join('maps', theme)
        if not os.path.exists(theme_dir):
            os.makedirs(theme_dir)

        plot_scatter(gdf, theme, os.path.join(theme_dir, '01_scatter.png'))
        plot_hexbin(gdf, theme, os.path.join(theme_dir, '02_hexbin.png'))
        plot_kde(gdf, theme, os.path.join(theme_dir, '03_kde.png'))
        plot_bubble(gdf, theme, os.path.join(theme_dir, '04_bubble.png'))
        plot_stamen(gdf, theme, os.path.join(theme_dir, '05_stamen.png'))
    
    print("Maps generated in the 'maps' directory.")