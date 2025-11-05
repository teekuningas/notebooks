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
# This notebook creates maps from observations and their themes, focusing on creating visually appealing, presentation-ready outputs.

# %%
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import os
import colorsys
from io import StringIO

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point, Polygon

def to_pastel_color(h, s=0.5, l=0.85):
    # Convert HSL to RGB
    rgb = colorsys.hls_to_rgb(h, l, s)
    r, g, b = [int(x * 255) for x in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'

def generate_greenish_pastel_colors(n=5):
    base_hue = 0.33  # Green
    single_pastel_color = to_pastel_color(base_hue, s=0.5, l=0.85)
    return [single_pastel_color] * n


# %%
def load_data():
    """
    Loads, merges, and prepares the bird recording data.

    Reads the themes and recording metadata, merges them, and returns a
    GeoDataFrame with point geometries for each recording.
    """
    try:
        themes = pd.read_csv('koodidata/bird-themes/2025-11-04/themes_1x400.csv', index_col=0)
        recs = pd.read_csv('koodidata/bird-metadata/recs_since_June25.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please make sure the data files are in the 'data/bird-metadata/' directory.")
        return None, None, None

    merged_data = pd.merge(themes, recs, left_index=True, right_on='rec_id')
    geometry = [Point(xy) for xy in zip(merged_data['lon'], merged_data['lat'])]
    gdf = gpd.GeoDataFrame(merged_data, geometry=geometry, crs="EPSG:4326")

    url = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson"
    response = requests.get(url)

    world = gpd.read_file(StringIO(response.text))

    finland_shape = world[world.name.isin(['Finland', 'Aland'])]

    return gdf, finland_shape, list(themes.columns)

def plot_theme_map(gdf, finland_shape, maakuntarajat, theme_name, output_filename, grid_size=20, projection='EPSG:3067'):
    """
    Generates a grid-based map for a given theme, coloring cells by the ratio of presence.
    """
    if not pd.api.types.is_numeric_dtype(gdf[theme_name]):
        print(f"Skipping non-numeric theme '{theme_name}'")
        return

    # 1. Reproject all data
    gdf = gdf.to_crs(projection)
    finland_shape = finland_shape.to_crs(projection)
    maakuntarajat = maakuntarajat.to_crs(projection)

    # Clip maakuntarajat to finland_shape to exclude water areas
    maakuntarajat_clipped = gpd.overlay(maakuntarajat, finland_shape, how='intersection')

    # Generate pastel colors for maakuntarajat_clipped
    num_regions = len(maakuntarajat_clipped)
    pastel_colors = generate_greenish_pastel_colors(num_regions)
    maakuntarajat_clipped['color'] = pastel_colors

    # 2. Create a grid in the new projection
    xmin, ymin, xmax, ymax = finland_shape.total_bounds
    cell_size = (xmax - xmin) / grid_size
    grid_cells = []
    for x in np.arange(xmin, xmax, cell_size):
        for y in np.arange(ymin, ymax, cell_size):
            grid_cells.append(Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)]))
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=projection)

    # 3. Spatially join data to the grid
    joined = gpd.sjoin(gdf, grid, how='inner', predicate='within')

    # 4. Calculate presence ratio for each grid cell
    grouped = joined.groupby('index_right')[theme_name].agg(
        present_count=lambda x: (x > 0).sum(),
        total_count='count'
    ).reset_index()
    grouped['ratio'] = grouped['present_count'] / grouped['total_count']

    # 5. Merge ratio back to the grid
    grid = grid.merge(grouped, left_index=True, right_on='index_right', how='left')
    grid['ratio'] = grid['ratio'].fillna(-1) # Use -1 for no data

    # 6. Plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor('white')
    ax.set_aspect('equal')

    # Define a pastel colormap (blue -> off-white -> red)
    pastel_cmap = LinearSegmentedColormap.from_list(
        'pastel_map', 
        [(0, '#6495ED'), (0.5, '#ffffef'), (1, '#FA8072')],
        N=256
    )

    # Base layer: Finland shape
    finland_shape.plot(ax=ax, color='#e0e0e0', edgecolor='#cccccc', linewidth=0.5)

    # Overlay regional borders
    maakuntarajat_clipped.plot(ax=ax, color=maakuntarajat_clipped['color'], edgecolor='#B0B0B0', linewidth=0.7, alpha=0.6)

    # Plot the entire grid with a light background and outlines
    grid.plot(ax=ax, color='none', edgecolor='white', linewidth=0.5)

    # Plot grid cells with data on top
    grid_with_data = grid[grid['ratio'] != -1]
    grid_with_data.plot(column='ratio', cmap=pastel_cmap, linewidth=0.5, ax=ax, edgecolor='white', vmin=0, vmax=1)

    ax.set_axis_off()
    plt.title(f"Teeman '{theme_name}' yleisyys", fontsize=16, pad=10)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=pastel_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = [] # Fake up the array of the scalar mappable
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Osuus')

    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    name = name.lower()
    name = name.replace('ä', 'a').replace('ö', 'o').replace('å', 'a')
    name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()
    name = name.replace(' ', '_')
    return name

def main():
    """Main function to generate maps for all themes."""
    gdf, finland_shape, theme_names = load_data()
    if gdf is None:
        return

    # Load regional borders
    url = "https://raw.githubusercontent.com/samilaine/hallinnollisetrajat/refs/heads/main/maakuntarajat.json"
    response = requests.get(url)
    maakuntarajat = gpd.read_file(StringIO(response.text))

    output_dir = 'output/maps'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for theme in theme_names:
        safe_theme_name = sanitize_filename(theme)
        print(f"Generating map for theme: {theme}...")
        theme_dir = os.path.join(output_dir, safe_theme_name)
        if not os.path.exists(theme_dir):
            os.makedirs(theme_dir)

        plot_theme_map(gdf, finland_shape, maakuntarajat, theme, os.path.join(theme_dir, f'{safe_theme_name}_grid.png'))

    print(f"\nMaps generated in the '{output_dir}' directory.")

if __name__ == '__main__':
    main()
