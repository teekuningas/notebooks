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
import sys

# --- Configuration ---
# Parse command-line arguments for flexibility
use_all_themes = '--all-themes' in sys.argv
custom_output = None
for i, arg in enumerate(sys.argv):
    if arg == '--output' and i + 1 < len(sys.argv):
        custom_output = sys.argv[i + 1]

# Choose which data to map:
# - 'koodit': Thematic codes (default)
# - 'paikat': Location categories
data_type = os.environ.get('MAP_TYPE', 'paikat')  # Can be overridden with MAP_TYPE env var

# Data paths
if use_all_themes:
    codes_path = './output/analyysi_koodit/7176421e/themes_98x710.csv'  # All 98 themes
    data_type = 'koodit'
else:
    codes_path = 'inputs/correlation-data/koodit_16x452.csv'

paikat_path = 'inputs/correlation-data/paikat_10x452.csv'
metadata_path = 'inputs/bird-metadata/recs_since_June25.csv'

# PDF output configuration
save_for_pdf = True  # Save maps to figures folder for PDF generation
# ---------------------

def to_pastel_color(h, s=0.5, l=0.85):
    # Convert HSL to RGB
    rgb = colorsys.hls_to_rgb(h, l, s)
    r, g, b = [int(x * 255) for x in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'

def generate_neutral_background_colors(n=5):
    """Generates a list of neutral light grey colors."""
    return ['#f0f0f0'] * n

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
    pastel_colors = generate_neutral_background_colors(num_regions)
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



    # Define a custom, light, diverging Purple-Green colormap
    pastel_prgn_cmap = LinearSegmentedColormap.from_list(
        'pastel_prgn_map',
        ['#c2a5cf', '#f7f7f7', '#88d48a'] # Final Pastel Purple -> Off-White -> Stronger Pastel Green
    )

    # Base layer: Finland shape
    finland_shape.plot(ax=ax, color='#e0e0e0', edgecolor='#cccccc', linewidth=0.5)

    # Overlay regional borders
    maakuntarajat_clipped.plot(ax=ax, color=maakuntarajat_clipped['color'], edgecolor='#B0B0B0', linewidth=0.7, alpha=0.6)

    # Plot the entire grid with a light background and outlines
    grid.plot(ax=ax, color='none', edgecolor='white', linewidth=0.5)

    # Plot grid cells with data on top
    grid_with_data = grid[grid['ratio'] != -1]

    # Use TwoSlopeNorm to center the diverging map correctly on the 0-1 ratio data
    from matplotlib.colors import TwoSlopeNorm
    div_norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

    grid_with_data.plot(column='ratio', cmap=pastel_prgn_cmap, norm=div_norm, linewidth=0.5, ax=ax, edgecolor='white')

    ax.set_axis_off()
    plt.title(f"'{code_name}' prevalence", fontsize=16, pad=10)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=pastel_prgn_cmap, norm=div_norm)
    sm._A = [] # Fake up the array of the scalar mappable
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Ratio')

    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()


# --- Data Loading ---
from uuid import uuid4

try:
    # Load data based on configuration
    if data_type == 'paikat':
        codes = pd.read_csv(paikat_path, index_col=0)
        data_label = 'paikat'
    else:
        codes = pd.read_csv(codes_path, index_col=0)
        # For 98 themes dataset, convert to binary (>= 0.5)
        if use_all_themes:
            codes = (codes >= 0.5).astype(int)
        data_label = 'koodit'
        
    # Optionally translate theme names to English
    if os.environ.get('ENABLE_TRANSLATION', '0') == '1' and data_type == 'koodit':
        from utils_translate import translate_theme_names
        codes = translate_theme_names(codes)
        print(f"✓ Translated {len(codes.columns)} theme names to English")
    
    recs = pd.read_csv(metadata_path)
    code_names = list(codes.columns)
    print(f"Loaded {len(code_names)} {data_label}: {', '.join(code_names[:5])}{'...' if len(code_names) > 5 else ''}")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please make sure the data files are in the correct directory.")
    code_names = []

if code_names:
    merged_data = pd.merge(codes, recs, left_index=True, right_on='rec_id')
    geometry = [Point(xy) for xy in zip(merged_data['lon'], merged_data['lat'])]
    gdf = gpd.GeoDataFrame(merged_data, geometry=geometry, crs="EPSG:4326")

    world = gpd.read_file('geo/ne_50m_admin_0_countries.geojson')
    finland_shape = world[world.name.isin(['Finland', 'Aland'])]

    maakuntarajat = gpd.read_file('geo/maakuntarajat.json')

    # Determine output directory
    if custom_output:
        output_dir = custom_output
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving maps to custom directory: {output_dir}")
    elif save_for_pdf:
        output_dir = 'output/figures_maps'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving maps to figures directory: {output_dir}")
    else:
        run_id = str(uuid4())[:8]
        output_dir = f'output/analyysi_koodikartat/{run_id}'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving maps to: {output_dir}")

    for i, code in enumerate(code_names, start=1):
        safe_code_name = sanitize_filename(code)
        print(f"Generating map for {data_label}: {code}...")
        
        if save_for_pdf:
            # Save directly to figures_maps with numbered prefix
            output_file = os.path.join(output_dir, f'{i:02d}_kartta_{safe_code_name}.png')
        else:
            # Save to code-specific subdirectory
            code_dir = os.path.join(output_dir, safe_code_name)
            os.makedirs(code_dir, exist_ok=True)
            output_file = os.path.join(code_dir, f'{safe_code_name}_grid.png')
        
        plot_code_map(gdf, finland_shape, maakuntarajat, code, output_file)

    print(f"\nAll maps generated successfully!")

# %%
