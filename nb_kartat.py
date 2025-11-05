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
# ## Joitakin pieniä karttaesimerkkejä

# %%
import geopandas as gpd
import matplotlib.pyplot as plt

# Let's draw a very simple map of Finland

# Read the GeoJSON content into a GeoDataFrame
world = gpd.read_file('geo/ne_50m_admin_0_countries.geojson')
    
# Filter for Finland using the appropriate column
finland = world[world['name'] == "Finland"]

# Plot Finland
finland.plot()
plt.title("Map of Finland")
plt.show()

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import random
import colorsys

# Let's draw a map of Finland subdivided to counties.

def to_pastel_color(h, s=0.5, l=0.85):
    # Convert HSL to RGB
    rgb = colorsys.hls_to_rgb(h, l, s)
    r, g, b = [int(x * 255) for x in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'

def generate_pastel_colors(n=5):
    # Generate evenly spaced hues
    hues = [i / n for i in range(n)]
    pastel_colors = [to_pastel_color(h) for h in hues]
    return pastel_colors

# Load data into GeoDataFrame
counties = gpd.read_file('geo/maakuntarajat.json')

# Create random pastel colors
pastel_colors = generate_pastel_colors(len(counties))
random.shuffle(pastel_colors)
counties['color'] = [pastel_colors[i] for i in range(len(counties))]

# Plot the counties with pastel colors
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
counties.plot(ax=ax, color=counties['color'], edgecolor='black')

plt.title("Map of Finland")
plt.show()

# %%
import geopandas as gpd
import random
import colorsys
import folium

def to_pastel_color(h, s=0.5, l=0.85):
    rgb = colorsys.hls_to_rgb(h, l, s)
    r, g, b = [int(x * 255) for x in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'

def generate_pastel_colors(n=5):
    hues = [i / n for i in range(n)]  # Generate evenly spaced hues
    pastel_colors = [to_pastel_color(h) for h in hues]
    return pastel_colors

# Let's draw counties overlaid to the folium default map (openstreetmap)

# Load data into GeoDataFrame
counties = gpd.read_file('geo/maakuntarajat.json')

# Generate pastel colors and assign them to counties
pastel_colors = generate_pastel_colors(len(counties))
random.shuffle(pastel_colors)
counties['color'] = pastel_colors  # Now ensure each county has a color

# Convert GeoDataFrame to JSON, preserving selected properties, including 'color'
counties_json = counties.to_json()

# Create a folium map centered on Finland
m = folium.Map(location=[64.9667, 25.6667], zoom_start=5, height=800, width=500)

# Add counties to the map with hover effects
geojson = folium.GeoJson(
    counties_json,
    style_function=lambda feature: {
        'fillColor': feature['properties']['color'],  # Use color property
        'color': 'black',
        'weight': 0.1,
        'fillOpacity': 0.2,
    },
    highlight_function=lambda feature: {
        'weight': 1,
    },
    tooltip=folium.GeoJsonTooltip(fields=['NAMEFIN'])
)
geojson.add_to(m)

f = folium.Figure(height=800, width=500)
m.add_to(f)

# Display the map
m

# %%
import geopandas as gpd
import random
import colorsys
import folium

# Let's draw munincipalities overlaid to the folium default map (openstreetmap)

def to_pastel_color(h, s=0.5, l=0.85):
    rgb = colorsys.hls_to_rgb(h, l, s)
    r, g, b = [int(x * 255) for x in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'

def generate_pastel_colors(n=5):
    hues = [i / n for i in range(n)]  # Generate evenly spaced hues
    pastel_colors = [to_pastel_color(h) for h in hues]
    return pastel_colors

# Load data into GeoDataFrame
muns = gpd.read_file('geo/kuntarajat.json')

# Generate pastel colors and assign them to munincipalities
pastel_colors = generate_pastel_colors(len(muns))
random.shuffle(pastel_colors)
muns['color'] = pastel_colors

# Convert GeoDataFrame to JSON, preserving selected properties, including 'color'
muns_json = muns.to_json()

# Create a folium map centered on Finland
m = folium.Map(location=[64.9667, 25.6667], zoom_start=5, height=800, width=500)

# Add munincipalities to the map with hover effects
geojson = folium.GeoJson(
    muns_json,
    style_function=lambda feature: {
        'fillColor': feature['properties']['color'],  # Use color property
        'color': 'black',
        'weight': 0.1,
        'fillOpacity': 0.2,
    },
    highlight_function=lambda feature: {

        'weight': 1,
    },
    tooltip=folium.GeoJsonTooltip(fields=['NAMEFIN'])
)
geojson.add_to(m)

f = folium.Figure(height=800, width=500)
m.add_to(f)

# Display the map
m

# %%
import geopandas as gpd
import random
import colorsys
import folium

# Let's draw the map of Jyväskylä overlaid to the folium default map (openstreetmap)

def to_pastel_color(h, s=0.5, l=0.85):
    rgb = colorsys.hls_to_rgb(h, l, s)
    r, g, b = [int(x * 255) for x in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'

def generate_pastel_colors(n=5):
    hues = [i / n for i in range(n)]  # Generate evenly spaced hues
    pastel_colors = [to_pastel_color(h) for h in hues]
    return pastel_colors

# Load data into GeoDataFrame
muns = gpd.read_file('geo/kuntarajat.json')

# Filter for Jyväskylä
jyvaskyla = muns[muns['NAMEFIN'] == 'Jyväskylä'].copy()

# Generate pastel colors and assign them to the filtered municipality
pastel_colors = generate_pastel_colors(len(jyvaskyla))
random.shuffle(pastel_colors)
jyvaskyla.loc[:, 'color'] = pastel_colors

# Re-project to a projected CRS (e.g., EPSG:3857) for accurate centroid calculations
jyvaskyla_projected = jyvaskyla.to_crs(epsg=3857)

# Calculate the centroid in the projected CRS
projected_centroid = jyvaskyla_projected.geometry.centroid.iloc[0]

# Convert the centroid back to geographic CRS (EPSG:4326)
centroid_geom = gpd.GeoSeries([projected_centroid], crs=3857).to_crs(epsg=4326)

# Extract the correct geographic coordinates
coords = centroid_geom.iloc[0].coords[0]

# Create a folium map centered on Jyväskylä with the original CRS
m = folium.Map(location=[coords[1], coords[0]], zoom_start=10, height=800, width=500)

# Add Jyväskylä to the map with hover effects
geojson = folium.GeoJson(
    jyvaskyla.to_json(),
    style_function=lambda feature: {
        'fillColor': feature['properties']['color'],
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.2,
    },
    tooltip=folium.GeoJsonTooltip(fields=['NAMEFIN'])
)
geojson.add_to(m)

f = folium.Figure(height=800, width=500)
m.add_to(f)

# Display the map
m

# %%
import geopandas as gpd
import numpy as np
import folium
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib import cm

# Let's try drawing a heatmap over the map of Finland based on some point data.

# Load county borders
counties = gpd.read_file('geo/maakuntarajat.json')

# Initialize folium map
m = folium.Map(location=[64.9667, 25.6667], zoom_start=5, tiles='cartodbpositron')

# Define grid parameters
finland_lat_min, finland_lat_max = 59.5, 70.1
finland_lon_min, finland_lon_max = 19.0, 31.6
grid_res = 0.1

# Create grid
grid_lat = np.arange(finland_lat_min, finland_lat_max, grid_res)
grid_lon = np.arange(finland_lon_min, finland_lon_max, grid_res)

# City data
finland_data = [
    {"city": "Helsinki", "latitude": 60.1695, "longitude": 24.9354, "pop": 684589},
    {"city": "Tampere", "latitude": 61.4978, "longitude": 23.7608, "pop": 260358},
    {"city": "Turku", "latitude": 60.4518, "longitude": 22.2666, "pop": 206035},
    {"city": "Oulu", "latitude": 65.0121, "longitude": 25.4651, "pop": 216194},
    {"city": "Rovaniemi", "latitude": 66.5039, "longitude": 25.7294, "pop": 65738},
    {"city": "Kuopio", "latitude": 62.8910, "longitude": 27.6780, "pop": 125668},
    {"city": "Joensuu", "latitude": 62.6010, "longitude": 29.7639, "pop": 78743},
    {"city": "Jyväskylä", "latitude": 62.2426, "longitude": 25.7473, "pop": 149269},
]

# IDW interpolation function (get a weighted average value to each of the points in the grid)
def idw_interpolate(x, y, values, xi, yi, power=2):
    distances = np.sqrt((xi - x[:, None, None])**2 + (yi - y[:, None, None])**2)
    weights = 1 / (np.where(distances == 0, np.nan, distances) ** power)
    total_weights = np.nansum(weights, axis=0)
    weighted_values = np.nansum(weights * values[:, None, None], axis=0)
    return np.divide(weighted_values, total_weights, where=total_weights != 0)

# Interpolation
x = np.array([city["latitude"] for city in finland_data])
y = np.array([city["longitude"] for city in finland_data])
pop = np.array([city["pop"] for city in finland_data])
zi = idw_interpolate(x, y, pop, grid_lat[:, None], grid_lon[None, :])

# Normalize and color mapping
norm = plt.Normalize(vmin=zi.min(), vmax=zi.max())
colors = cm.viridis(norm(zi))

# Use rectangles to cover the area (not for production)
if counties is not None:
    for i, lat in enumerate(grid_lat[:-1]):
        for j, lon in enumerate(grid_lon[:-1]):
            pt = Point(lon, lat)
            if counties.contains(pt).any():
                color = colors[i, j]
                color_hex = '#{:02x}{:02x}{:02x}'.format(int(255 * color[0]), int(255 * color[1]), int(255 * color[2]))
                rectangle = folium.Rectangle(
                    bounds=[[lat, lon], [lat + grid_res, lon + grid_res]],
                    color=color_hex,
                    opacity=0.5,
                    fill=True,
                    fill_color=color_hex,
                    fill_opacity=0.5,
                    weight=0.0
                )
                m.add_child(rectangle)

# Display the map
m

# %%
