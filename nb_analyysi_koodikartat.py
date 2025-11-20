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
# # Code Maps
#
# This notebook creates maps from observations and their codes, focusing on creating visually appealing, presentation-ready outputs.

# %%
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import os
import colorsys
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- Configuration ---
codes_path = 'output/analyysi_koodit/91a63520/koodit_16x452.csv'
metadata_path = '/home/user/bird-metadata/recs_since_June25.csv'
# ---------------------

def to_pastel_color(h, s=0.5, l=0.85):
    # Convert HSL to RGB
    rgb = colorsys.hls_to_rgb(h, l, s)
    r, g, b = [int(x * 255) for x in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'

def generate_greenish_pastel_colors(n=5):
    base_hue = 0.33  # Green
    single_pastel_color = to_pastel_color(base_hue, s=0.5, l=0.85)
    return [single_pastel_color] * n

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    name = name.lower()
    name = name.replace('ä', 'a').replace('ö', 'o').replace('å', 'a')
    name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()
    name = name.replace(' ', '_')
    return name

def plot_code_map(gdf, finland_shape, maakuntarajat, code_name, output_filename, grid_size=20, projection='EPSG:3067'):
    """
    Generates a grid-based map for a given code, coloring cells by the ratio of presence.
    """
    if not pd.api.types.is_numeric_dtype(gdf[code_name]):
        print(f"Skipping non-numeric code '{code_name}'")
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
    grouped = joined.groupby('index_right')[code_name].agg(
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
    plt.title(f"Koodin '{code_name}' yleisyys", fontsize=16, pad=10)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=pastel_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = [] # Fake up the array of the scalar mappable
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Osuus')

    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()


# --- Data Loading ---
from uuid import uuid4

try:
    codes = pd.read_csv(codes_path, index_col=0)
    recs = pd.read_csv(metadata_path)
    code_names = list(codes.columns)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please make sure the data files are in the correct directory.")
    code_names = [] # Ensure code_names is defined

if code_names:
    merged_data = pd.merge(codes, recs, left_index=True, right_on='rec_id')
    geometry = [Point(xy) for xy in zip(merged_data['lon'], merged_data['lat'])]
    gdf = gpd.GeoDataFrame(merged_data, geometry=geometry, crs="EPSG:4326")

    world = gpd.read_file('geo/ne_50m_admin_0_countries.geojson')
    finland_shape = world[world.name.isin(['Finland', 'Aland'])]

    maakuntarajat = gpd.read_file('geo/maakuntarajat.json')

    run_id = str(uuid4())[:8]
    output_dir = f'output/analyysi_koodikartat/{run_id}'
    os.makedirs(output_dir, exist_ok=True)

    for code in code_names:
        safe_code_name = sanitize_filename(code)
        print(f"Generating map for code: {code}...")
        code_dir = os.path.join(output_dir, safe_code_name)
        os.makedirs(code_dir, exist_ok=True)

        plot_code_map(gdf, finland_shape, maakuntarajat, code, os.path.join(code_dir, f'{safe_code_name}_grid.png'))

    print(f"\nMaps generated in the '{output_dir}' directory.")

# %%
