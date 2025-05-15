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

# %%
import random
import math
import geopandas as gpd
from shapely.geometry import Point
import requests
import matplotlib.pyplot as plt
from io import StringIO

from llm import generate_simple

# Number of interviews to simulate
N_INTERVIEWS = 30

# City data
CITIES = [
    {"city": "Helsinki", "latitude": 60.1695, "longitude": 24.9354, "pop": 684589},
    {"city": "Tampere", "latitude": 61.4978, "longitude": 23.7608, "pop": 260358},
    {"city": "Turku", "latitude": 60.4518, "longitude": 22.2666, "pop": 206035},
    {"city": "Oulu", "latitude": 65.0121, "longitude": 25.4651, "pop": 216194},
    {"city": "Rovaniemi", "latitude": 66.5039, "longitude": 25.7294, "pop": 65738},
    {"city": "Kuopio", "latitude": 62.8910, "longitude": 27.6780, "pop": 125668},
    {"city": "Joensuu", "latitude": 62.6010, "longitude": 29.7639, "pop": 78743},
    {"city": "Jyväskylä", "latitude": 62.2426, "longitude": 25.7473, "pop": 149269},
]

# Distance threshold in kilometers to differentiate rural/urban themes
DISTANCE_THRESHOLD_KM = 50.0

# Standard deviation for adding noise to city coordinates (in degrees)
# Adjust this to control how far from the city center points are generated
LOCATION_NOISE_STD_DEV = 0.5

# Themes
COMMON_THEMES = ["linnut", "luonto"] # Themes always included
RURAL_THEMES = ["maaseutu", "hiljaisuus"] # Added if location is rural
URBAN_THEMES = ["kaupunki", "äänet"] # Added if location is urban

# LLM Model for generation
GENERATION_MODEL = "llama3.3:70b"


# %%
def generate_location(cities, noise_std_dev, finland_geometry):
    """
    Generates a random location biased towards a city, ensuring it's within Finland borders if geometry is provided.
    Returns (latitude, longitude, chosen_city_dict).
    """
    while True:
        city = random.choice(cities)
        base_lat, base_lon = city['latitude'], city['longitude']

        # Add Gaussian noise to the coordinates
        gen_lat = base_lat + random.gauss(0, noise_std_dev)
        gen_lon = base_lon + random.gauss(0, noise_std_dev)

        # Create a Shapely Point for checking
        point = Point(gen_lon, gen_lat) # Note: Point uses (longitude, latitude) order

        # Check if point is within Finland, if geometry is available and valid
        if finland_geometry is None or finland_geometry.contains(point):
            # Valid location found, return it and the city info
            return gen_lat, gen_lon, city
        # else: Point is outside Finland (or geometry failed loading), loop again

def calculate_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculates the approximate distance between two points using the Haversine formula.
    """
    R = 6371 # Earth radius in kilometers

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    distance = R * c
    return distance

def select_themes(location_type):
    """Selects themes based on distance."""
    if location_type == 'rural':
        return COMMON_THEMES + RURAL_THEMES
    else:
        return COMMON_THEMES + URBAN_THEMES

def get_finland_geometry():
    print("Loading Finland geometry...")
    
    url = "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson"
    
    response = requests.get(url)
    response.raise_for_status() # Raise an exception for bad status codes
    
    geojson_str = response.text
    world = gpd.read_file(StringIO(geojson_str))
    finland_gdf = world[world['name'] == "Finland"]
    
    if finland_gdf.empty:
        raise ValueError("Could not find Finland in the GeoJSON data.")
    
    # Get the primary geometry object (likely a MultiPolygon)
    finland_geometry = finland_gdf.geometry.iloc[0]
    
    print("Finland geometry loaded successfully.")
        
    return finland_geometry

def draw_map_with_locations(cities, locations, finland_geometry):
    # Implement the map drawing.
    print("Drawing a map of locations.")
    fig, ax = plt.subplots(1, 1, figsize=(10, 15)) # Adjust figsize as needed
    if finland_geometry is not None:
        finland_gdf = gpd.GeoDataFrame([1], geometry=[finland_geometry], crs="EPSG:4326")
        finland_gdf.plot(ax=ax, facecolor='lightgray', edgecolor='black')
    else:
        print("Finland geometry not available, skipping map background.")

    # Plot Cities
    city_lons = [city['longitude'] for city in cities]
    city_lats = [city['latitude'] for city in cities]
    ax.scatter(city_lons, city_lats, c='blue', marker='*', s=50, label='Cities', zorder=5) # zorder puts markers on top

    # Plot Generated Locations
    loc_lons = [loc['longitude'] for loc in locations]
    loc_lats = [loc['latitude'] for loc in locations]
    ax.scatter(loc_lons, loc_lats, c='red', marker='o', s=25, label='Generated Locations', zorder=5)

    # Set aspect ratio to be equal (important for maps)
    ax.set_aspect('equal', adjustable='box')

    ax.set_title('Simulated Interview Locations in Finland')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.show()



# %%
finland_geometry = get_finland_geometry()

simulated_data = []

# First, get locations
print(f"Getting random locations.")
locations = []
for _ in range(N_INTERVIEWS):
    latitude, longitude, city = generate_location(CITIES, LOCATION_NOISE_STD_DEV, finland_geometry)
    city_lat, city_lon = city['latitude'], city['longitude']
    city_name = city['city']

    min_distance_to_any_city = float('inf')
    for c in CITIES:
        dist = calculate_distance_km(latitude, longitude, c['latitude'], c['longitude'])
        if dist < min_distance_to_any_city:
            min_distance_to_any_city = dist
    
    locations.append({
        "city_name": city_name,
        "distance": min_distance_to_any_city,
        "latitude": latitude,
        "longitude": longitude
    })

perc_rural = len([loc for loc in locations if loc['distance'] > DISTANCE_THRESHOLD_KM]) / float(len(locations))
print(f"Percentage of rurals: {perc_rural}")
draw_map_with_locations(CITIES, locations, finland_geometry)

# %%
# Then get interviews
print(f"Generating interviews for each location.")
for i, location in enumerate(locations):
    distance = location['distance']
    city_name = location['city_name']
    latitude = location['latitude']
    longitude = location['longitude']
    
    # Select themes and location type based on the minimum distance
    location_type = "rural" if distance > DISTANCE_THRESHOLD_KM else "urban"
    themes_for_interview = select_themes(location_type)

    # Prepare LLM Prompt
    instruction = f"""
Olet haastattelusimulaattori. Tehtäväsi on luoda lyhyt kuvitteellinen haastattelukatkelma (muutama kappale).
Henkilö puhuu kokemuksistaan luonnosta ja linnuista.
Sisällytä vastaukseen viittauksia annettuihin teemoihin.
Älä mainitse teemoja suoraan termeillä, vaan kuvaile niihin liittyviä asioita luontevasti osana kertomusta.
Vastaa suomeksi.
"""
    content = f"""
Teemat, joita tulee käsitellä implisiittisesti: {', '.join(themes_for_interview)}
"""

    # Generate Interview Text using LLM
    try:
        result = generate_simple(
            instruction,
            content,
            model=GENERATION_MODEL,
            seed=i # Use loop index for reproducibility if needed, but vary for diverse outputs
        )
        interview_text = result['message']['content']
    except Exception as e:
        print(f"Error generating interview {i}: {e}")
        interview_text = f"ERROR: Could not generate text (Themes: {', '.join(themes_for_interview)})"

    # Store results
    simulated_data.append({
        "interview_id": f"sim_{i:04d}",
        "latitude": latitude,
        "longitude": longitude,
        "city_context": city_name,
        "distance_to_closest_city_km": round(distance, 2),
        "location_type_generated": location_type,
        "themes_used_for_generation": themes_for_interview,
        "interview_text": interview_text.strip()
    })

    # Print progress
    print(f"Generated interview {i + 1}/{N_INTERVIEWS}")

# %%
import os
import csv

# from pprint import pprint
# pprint(simulated_data)

TARGET_DIR = "data/luontokokemus_simulaatio"

def save_interviews_to_files(data, directory):
    # Create the target directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Prepare a list to hold metadata for the CSV file
    metadata = []

    for index, item in enumerate(data):
        # Generate a filename for each interview text
        interview_id = item['interview_id']
        interview_file_path = os.path.join(directory, f"interview{index + 1}.txt")
        
        # Write the interview text to its respective file
        with open(interview_file_path, 'w', encoding='utf-8') as interview_file:
            interview_file.write(item['interview_text'])
        
        # Gather metadata for CSV file
        metadata.append({
            'interview_id': interview_id,
            'city_context': item['city_context'],
            'distance_to_closest_city_km': item['distance_to_closest_city_km'],
            'latitude': item['latitude'],
            'longitude': item['longitude'],
            'location_type_generated': item['location_type_generated'],
            'themes_used_for_generation': ', '.join(item['themes_used_for_generation'])  # join list to string
        })

    # Write the metadata to a CSV file
    metadata_file_path = os.path.join(directory, "metadata.csv")
    with open(metadata_file_path, 'w', encoding='utf-8', newline='') as metadata_file:
        fieldnames = metadata[0].keys()
        writer = csv.DictWriter(metadata_file, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(metadata)

print(f"Saving to {TARGET_DIR}")
save_interviews_to_files(simulated_data, TARGET_DIR)
print(f"Done.")


# %%
