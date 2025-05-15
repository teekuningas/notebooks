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
# ## Next Steps
#
# With both interview texts and metadata loaded, we can proceed to:
# 1. Preprocess the text data (e.g., feature extraction, sentiment analysis, topic modeling).
# 2. Merge or align text-derived features with the metadata based on `interview_id`.
# 3. Perform statistical correlational analyses.
# 4. Visualize findings.

# %%
# Placeholder for future analysis
print("\nSetup complete. Ready for correlational analysis.")


# %%
