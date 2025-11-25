#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3 python3Packages.requests
"""
Precompute CORINE Land Cover types for specific recordings.

This script:
1. Loads recording IDs from koodit_16x452.csv
2. Filters to recordings in recs_since_June25.csv with Finland coordinates
3. Generates center point + 4 offset points (±50m N/S/E/W) for each location
4. Queries CORINE Land Cover API with rate limiting
5. Saves to CSV: lon, lat, corine_code, corine_type

Output: inputs/location_lookups/corine_locations_lookup.csv

Note: This takes ~20-30 minutes due to API rate limiting (0.5s per request).
"""

import csv
import time
import requests
from pathlib import Path

# CORINE Land Cover codes (44 classes)
CORINE_CODES = {
    "111": "Continuous urban fabric",
    "112": "Discontinuous urban fabric",
    "121": "Industrial or commercial units",
    "122": "Road and rail networks and associated land",
    "123": "Port areas",
    "124": "Airports",
    "131": "Mineral extraction sites",
    "132": "Dump sites",
    "133": "Construction sites",
    "141": "Green urban areas",
    "142": "Sport and leisure facilities",
    "211": "Non-irrigated arable land",
    "212": "Permanently irrigated land",
    "213": "Rice fields",
    "221": "Vineyards",
    "222": "Fruit trees and berry plantations",
    "223": "Olive groves",
    "231": "Pastures",
    "241": "Annual crops associated with permanent crops",
    "242": "Complex cultivation patterns",
    "243": "Land principally occupied by agriculture with significant areas of natural vegetation",
    "244": "Agro-forestry areas",
    "311": "Broad-leaved forest",
    "312": "Coniferous forest",
    "313": "Mixed forest",
    "321": "Natural grasslands",
    "322": "Moors and heathland",
    "323": "Sclerophyllous vegetation",
    "324": "Transitional woodland-shrub",
    "331": "Beaches, dunes, sands",
    "332": "Bare rocks",
    "333": "Sparsely vegetated areas",
    "334": "Burnt areas",
    "335": "Glaciers and perpetual snow",
    "411": "Inland marshes",
    "412": "Peat bogs",
    "421": "Salt marshes",
    "422": "Salines",
    "423": "Intertidal flats",
    "511": "Water courses",
    "512": "Water bodies",
    "521": "Coastal lagoons",
    "522": "Estuaries",
    "523": "Sea and ocean"
}

# CORINE API endpoint
ARCGIS_BASE_URL = "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer"

# GPS offset for ±50m (approximate at Finland latitude ~60-70°N)
LAT_OFFSET = 50 / 111000  # ~0.00045° for 50m north/south
LON_OFFSET = 50 / 55000   # ~0.00091° for 50m east/west


def load_target_recordings():
    """
    Load recordings from koodit_16x452.csv that exist in recs_since_June25.csv
    and are located in Finland.
    
    Returns: list of (lon, lat) tuples
    """
    koodit_path = Path("inputs/correlation-data/koodit_16x452.csv")
    recs_path = Path("inputs/bird-metadata/recs_since_June25.csv")
    
    # Load rec_ids from koodit
    rec_ids = set()
    with open(koodit_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec_ids.add(row[''])  # First column (index)
    
    print(f"Loaded {len(rec_ids)} recording IDs from koodit_16x452.csv")
    
    # Filter to Finland locations
    locations = []
    with open(recs_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec_id = row['rec_id']
            if rec_id in rec_ids:
                try:
                    lat = float(row['lat'])
                    lon = float(row['lon'])
                    # Finland bounding box
                    if 59 <= lat <= 71 and 19 <= lon <= 32:
                        locations.append((lon, lat))
                except (ValueError, KeyError):
                    pass
    
    print(f"Found {len(locations)} recordings in Finland")
    return locations


def generate_offset_points(locations):
    """
    For each location, generate center + 4 offset points (N/S/E/W at 50m).
    
    Returns: set of unique (lon, lat) tuples
    """
    points = set()
    
    for lon, lat in locations:
        # Center point
        points.add((lon, lat))
        # North (+50m latitude)
        points.add((lon, lat + LAT_OFFSET))
        # South (-50m latitude)
        points.add((lon, lat - LAT_OFFSET))
        # East (+50m longitude)
        points.add((lon + LON_OFFSET, lat))
        # West (-50m longitude)
        points.add((lon - LON_OFFSET, lat))
    
    return points


def query_corine_api(longitude, latitude):
    """
    Query CORINE Land Cover API for given coordinates.
    
    Returns: (code, description) tuple or (None, error_message)
    """
    url = f"{ARCGIS_BASE_URL}/identify"
    
    params = {
        'geometry': f'{longitude},{latitude}',
        'geometryType': 'esriGeometryPoint',
        'sr': '4326',
        'layers': 'all:0',
        'tolerance': '1',
        'mapExtent': f'{longitude-0.1},{latitude-0.1},{longitude+0.1},{latitude+0.1}',
        'imageDisplay': '400,400,96',
        'returnGeometry': 'false',
        'f': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            result = data['results'][0]
            code = result['attributes']['Code_18']
            description = CORINE_CODES.get(code, "Unknown")
            return (code, description)
        else:
            return (None, "No data")
            
    except requests.RequestException as e:
        return (None, f"Error: {str(e)}")
    except (KeyError, ValueError) as e:
        return (None, f"Parse error: {str(e)}")


def main():
    # Configuration
    output_file = Path("inputs/location_lookups/corine_locations_lookup.csv")
    request_delay = 0.5  # seconds between requests (be nice to the API)
    
    print("=" * 70)
    print("CORINE LAND COVER LOCATION PRECOMPUTATION")
    print("=" * 70)
    print()
    
    # Step 1: Load target recordings
    print("Step 1: Loading target recordings...")
    locations = load_target_recordings()
    
    if not locations:
        print("❌ No valid recordings found!")
        return
    
    print()
    
    # Step 2: Generate offset points
    print("Step 2: Generating offset points (±50m)...")
    all_points = generate_offset_points(locations)
    print(f"  Generated {len(all_points):,} unique points")
    print(f"  (Original: {len(locations)}, with offsets: {len(all_points)})")
    print()
    
    # Estimate time
    estimated_time = len(all_points) * request_delay
    print(f"⏱️  Estimated time: ~{estimated_time / 60:.1f} minutes ({estimated_time / 3600:.1f} hours)")
    print(f"   API rate: {request_delay}s per request")
    print()
    
    # Confirm
    response = input(f"Proceed with {len(all_points):,} API queries? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    print()
    print("Step 3: Querying CORINE API...")
    print("(This will take a while, please be patient...)")
    print()
    
    # Query API for each point
    results = []
    errors = 0
    start_time = time.time()
    
    sorted_points = sorted(all_points)
    
    for i, (lon, lat) in enumerate(sorted_points, 1):
        # Progress update every 100 points
        if i % 100 == 0 or i == len(sorted_points):
            elapsed = time.time() - start_time
            points_per_sec = i / elapsed if elapsed > 0 else 0
            remaining = (len(sorted_points) - i) / points_per_sec if points_per_sec > 0 else 0
            print(f"  [{i:,}/{len(sorted_points):,}] "
                  f"Progress: {i/len(sorted_points)*100:.1f}% | "
                  f"ETA: {remaining/60:.1f} min | "
                  f"Errors: {errors}")
        
        code, description = query_corine_api(lon, lat)
        
        if code is not None:
            results.append({
                'lon': lon,
                'lat': lat,
                'corine_code': code,
                'corine_type': description
            })
        else:
            errors += 1
            # Still save the point but with error info
            results.append({
                'lon': lon,
                'lat': lat,
                'corine_code': '',
                'corine_type': description  # Contains error message
            })
        
        # Rate limiting: wait between requests
        if i < len(sorted_points):
            time.sleep(request_delay)
    
    print()
    
    # Step 4: Save results
    print("Step 4: Saving results...")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lon', 'lat', 'corine_code', 'corine_type'])
        writer.writeheader()
        writer.writerows(results)
    
    elapsed_total = time.time() - start_time
    
    print()
    print("=" * 70)
    print("✓ PRECOMPUTATION COMPLETE")
    print("=" * 70)
    print(f"Output file: {output_file}")
    print(f"Total points: {len(results):,}")
    print(f"Successful: {len(results) - errors:,}")
    print(f"Errors: {errors}")
    print(f"Time elapsed: {elapsed_total / 60:.1f} minutes")
    
    if errors > 0:
        print(f"\n⚠️  Warning: {errors} points could not be processed")
    
    # Show distribution (only successful queries)
    from collections import Counter
    successful = [r for r in results if r['corine_code']]
    type_counts = Counter(r['corine_type'] for r in successful)
    
    if successful:
        print("\nLand cover distribution:")
        for land_type, count in type_counts.most_common(10):  # Top 10
            pct = count / len(successful) * 100
            print(f"  {land_type[:40]:40s}: {count:6,} ({pct:5.1f}%)")
        
        if len(type_counts) > 10:
            print(f"  ... and {len(type_counts) - 10} more types")
    
    print()
    print("🎉 Lookup file ready! Use lookup_landcover.py to query locations.")


if __name__ == '__main__':
    main()
