# ---
# jupyter:
#   jupytext:
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
# # Correlational Analysis of Interview Data and Metadata
#
# This notebook aims to explore correlations between qualitative interview data and associated metadata.
# We will:
# 1. Load interview texts from the `data/luontokokemus_simulaatio` directory.
# 2. Load corresponding metadata from `data/luontokokemus_simulaatio/metadata.csv`.
# 3. Prepare the data for further analysis.

# %%
import pandas as pd
import os
from utils import read_files, strip_webvtt_to_plain_text
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point # Though not directly used in plotting, good for consistency if adapting more
import requests
from io import StringIO
import json
from llm import generate_simple

# %% [markdown]
# ## Load Interview Data
#
# We read all files starting with "interview" from the specified folder.

# %%
# Define the folder and prefix for interview files
interview_folder = "data/luontokokemus_simulaatio"
interview_prefix = "interview" # Based on files like interview1.txt

# Read the interview files
# The read_files function returns a list of tuples (filename, content)
interview_contents = read_files(folder=interview_folder, prefix=interview_prefix)

# Clean up texts if they were in WebVTT format (though these are .txt, it's a good general step)
interview_contents = [(fname, strip_webvtt_to_plain_text(text)) for fname, text in interview_contents]

# Print a summary of loaded interviews
print(f"Loaded {len(interview_contents)} interviews:")
for fname, text in interview_contents:
    print(f"- {fname} (length: {len(text)} characters)")

# Display the first few lines of the first interview as an example
if interview_contents:
    print(f"\nExample from {interview_contents[0][0]}:\n")
    print(interview_contents[0][1][:300] + "...") # Print first 300 characters
else:
    print("\nNo interview files found.")

# %% [markdown]
# ## Load Metadata
#
# We load the metadata from the CSV file. The metadata contains information associated with each interview, such as location type, distance to city, etc.

# %%
# Define the path to the metadata file
metadata_file_path = os.path.join(interview_folder, "metadata.csv")

# Read the metadata CSV file into a pandas DataFrame
try:
    metadata_df = pd.read_csv(metadata_file_path)
    print("\nMetadata loaded successfully:")
    print(metadata_df.info())
    print("\nFirst 5 rows of metadata:")
    print(metadata_df.head())
except FileNotFoundError:
    print(f"\nError: Metadata file not found at {metadata_file_path}")
    metadata_df = pd.DataFrame() # Create an empty DataFrame if file not found

# %% [markdown]
# ## Metadata Table

# %%
# Pick the most informative columns for a quick overview
meta_cols = [
    'interview_id',
    'city_context',
    'distance_to_closest_city_km',
    'location_type_generated',
    'themes_used_for_generation'
]
meta_display = metadata_df[meta_cols].copy()

# Define a simple highlighter for urban vs. rural
def highlight_location(val):
    if val == 'urban':
        return 'background-color: rgba(255,182,193,0.3)'  # light pink
    elif val == 'rural':
        return 'background-color: rgba(144,238,144,0.3)'  # light green
    return ''

# Build the styled metadata table
styled_metadata = (
    meta_display
      .style
      .applymap(highlight_location, subset=['location_type_generated'])
      .format({
          'distance_to_closest_city_km': '{:.1f}',
          'themes_used_for_generation': lambda x: x  # leave as-is
      })
      .set_properties(**{'text-align': 'left'})
)

# Display it
styled_metadata

# %% [markdown]
# ## Visualize Interview Locations on a Map

# %%
# Helper function to get Finland's geometry (adapted from nb_simulaatio.py)
def get_finland_geometry():
    """Loads Finland's geometry from an online GeoJSON source."""
    print("Loading Finland geometry...")
    try:
        url = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        geojson_str = response.text
        world = gpd.read_file(StringIO(geojson_str))
        finland_gdf = world[world['name'] == "Finland"]
        
        if finland_gdf.empty:
            print("Warning: Could not find Finland in the GeoJSON data. Map background will be missing.")
            return None
        
        finland_geometry = finland_gdf.geometry.iloc[0]
        print("Finland geometry loaded successfully.")
        return finland_geometry
    except Exception as e:
        print(f"Error loading Finland geometry: {e}")
        return None

# City data for context on the map (from nb_simulaatio.py)
CITIES_FOR_MAP = [
    {"city": "Helsinki", "latitude": 60.1695, "longitude": 24.9354},
    {"city": "Tampere", "latitude": 61.4978, "longitude": 23.7608},
    {"city": "Turku", "latitude": 60.4518, "longitude": 22.2666},
    {"city": "Oulu", "latitude": 65.0121, "longitude": 25.4651},
    {"city": "Rovaniemi", "latitude": 66.5039, "longitude": 25.7294},
    {"city": "Kuopio", "latitude": 62.8910, "longitude": 27.6780},
    {"city": "Joensuu", "latitude": 62.6010, "longitude": 29.7639},
    {"city": "Jyväskylä", "latitude": 62.2426, "longitude": 25.7473},
]

def plot_interview_locations(metadata_df, finland_geom, cities_for_map):
    """Plots interview locations on a map of Finland."""
    if metadata_df.empty or 'latitude' not in metadata_df.columns or 'longitude' not in metadata_df.columns:
        print("Metadata DataFrame is empty or missing coordinate columns. Cannot plot locations.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 15))
    
    # Plot Finland's geometry
    if finland_geom:
        finland_gdf = gpd.GeoDataFrame([1], geometry=[finland_geom], crs="EPSG:4326")
        finland_gdf.plot(ax=ax, facecolor='lightgray', edgecolor='black', alpha=0.8)
    else:
        print("Finland geometry not available, plotting points without map background.")

    # Plot Cities for context
    city_lons = [city['longitude'] for city in cities_for_map]
    city_lats = [city['latitude'] for city in cities_for_map]
    ax.scatter(city_lons, city_lats, c='blue', marker='*', s=100, label='Major Cities', zorder=3, alpha=0.7)
    for city in cities_for_map:
        ax.text(city['longitude'] + 0.1, city['latitude'] + 0.1, city['city'], fontsize=8, color='blue', zorder=4)

    # Plot Interview Locations, colored by location_type_generated
    if 'location_type_generated' in metadata_df.columns:
        urban_points = metadata_df[metadata_df['location_type_generated'] == 'urban']
        rural_points = metadata_df[metadata_df['location_type_generated'] == 'rural']
        
        ax.scatter(urban_points['longitude'], urban_points['latitude'], 
                   c='red', marker='o', s=50, label='Urban Interviews', zorder=5, alpha=0.7, edgecolor='black')
        ax.scatter(rural_points['longitude'], rural_points['latitude'], 
                   c='green', marker='^', s=50, label='Rural Interviews', zorder=5, alpha=0.7, edgecolor='black')
        
        # Add tooltips for interview_id (optional, can be a bit cluttered with many points)
        # for idx, row in metadata_df.iterrows():
        #     ax.text(row['longitude'] + 0.05, row['latitude'] + 0.05, row['interview_id'], fontsize=7, zorder=6)

    else: # Fallback if location_type_generated is not available
        ax.scatter(metadata_df['longitude'], metadata_df['latitude'], 
                   c='purple', marker='o', s=50, label='Interview Locations', zorder=5, alpha=0.7, edgecolor='black')

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Simulated Interview Locations in Finland')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# Load Finland geometry
finland_geometry = get_finland_geometry()

# Plot the locations if metadata was loaded
if not metadata_df.empty:
    plot_interview_locations(metadata_df, finland_geometry, CITIES_FOR_MAP)
else:
    print("Skipping map plot as metadata is not available.")

# %% [markdown]
# ## Code‐Presence Tabulation

# %%
# 1) Configure codes and iterations
codes = ['Luonto', 'Rauha', 'Hiljaisuus', 'Äänet', 'Kaupunki', 'Kauneus', 'Yhteys', 'Linnut', 'rauha', 'Inspiratio', 'Kontrasti', 'vapaus']
n_iter = 5

output_format = {
    "type": "object",
    "properties": {
        "code_present": {"type": "boolean"}
    },
    "required": ["code_present"]
}

# 2) Gather LLM decisions
results = []
for fname, text in interview_contents:
    for code in codes:
        for idx in range(n_iter):
            instruction = """
            Olet laadullisen tutkimuksen avustaja. Saat tekstinäytteen sekä aineiston pohjalta rakennetun koodikirjan yksittäisen koodin. Lue teksti huolella ja päätä kuvaako koodi tekstinäytettä.
            """
            content = f"Koodi: {code} \n\n Tekstinäyte: \n\n {text}"
            res = generate_simple(instruction, content, seed=idx, output_format=output_format)
            present = json.loads(res['message']['content'])['code_present']
            results.append({
                "fname": fname,
                "code": code,
                "iter": idx,
                "result": present
            })

# 3) Pivot into percentage table
import pandas as pd

# build nested dict: fname → code → [bools]
td = {}
for r in results:
    td.setdefault(r['fname'], {}).setdefault(r['code'], []).append(r['result'])

# compute percentage of True
for fn in td:
    for c in td[fn]:
        trues = td[fn][c].count(True)
        td[fn][c] = (trues / len(td[fn][c])) * 100

df_codes = pd.DataFrame.from_dict(td, orient='index')
df_codes['total'] = df_codes.mean(axis=1)
df_codes.loc['total'] = df_codes.mean()
df_codes = df_codes.round(2)

# 4) Style and display
def color_high(val):
    return 'background-color: rgba(144,238,144,0.3)' if val >= 80 else ''

styled_codes = df_codes.style.map(color_high).format("{:.2f}")
styled_codes

# %% [markdown]
# ## T-Test of Code Frequencies by Location Type

# %%
from scipy.stats import ttest_ind

# 1) Merge code-percentages with metadata on interview_id
#    (assumes df_codes.index matches metadata_df['interview_id'])
merged = (
    df_codes
      .reset_index()                    # turn the index into a column
      .rename(columns={'index':'fname'})
      .assign(
          # pull the digits out of e.g. "interview23.txt", convert to int,
          # subtract 1 (because interview1.txt → sim_0000), then format
          interview_id=lambda df: (
              df['fname']
                .str.extract(r'(\d+)', expand=False)    # get "23"
                .astype(int)                            # to 23
                .sub(1)                                 # to 22
                .apply(lambda x: f"sim_{x:04d}")        # to "sim_0022"
          )
      )
      .merge(
          metadata_df[['interview_id', 'location_type_generated']],
          on='interview_id',
          how='inner'
      )
)

# 2) Split into Urban vs. Rural
urban = merged[merged.location_type_generated == 'urban']
rural = merged[merged.location_type_generated == 'rural']

# 3) Run Welch’s t-test for each code (exclude the 'total' column)
ttest_results = []
for code in df_codes.columns.drop('total'):
    u_vals = urban[code].dropna()
    r_vals = rural[code].dropna()
    if len(u_vals) >= 2 and len(r_vals) >= 2:
        tstat, pval = ttest_ind(u_vals, r_vals, equal_var=False)
    else:
        tstat, pval = float('nan'), float('nan')
    ttest_results.append({
        'code': code,
        'mean_urban': u_vals.mean(),
        'mean_rural': r_vals.mean(),
        't_stat': tstat,
        'p_value': pval
    })

import pandas as pd
tt_df = pd.DataFrame(ttest_results).set_index('code').round(3)

# 4) Style and display
def highlight_sig(val):
    return 'background-color: rgba(144,238,144,0.3)' if val < 0.05 else ''

styled_ttest = (
    tt_df.style
        .format({
            'mean_urban':'{:.1f}',
            'mean_rural':'{:.1f}',
            't_stat':'{:.2f}',
            'p_value':'{:.3f}'
        })
        .applymap(highlight_sig, subset=['p_value'])
)
styled_ttest

# %%
