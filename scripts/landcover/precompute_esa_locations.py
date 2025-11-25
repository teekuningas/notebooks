#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3 python3Packages.rasterio python3Packages.numpy
"""
Precompute ESA WorldCover land types for specific recordings.

This script:
1. Loads recording IDs from koodit_16x452.csv
2. Filters to recordings in recs_since_June25.csv with Finland coordinates
3. Generates center point + 4 offset points (±50m N/S/E/W) for each location
4. Extracts land cover from local ESA WorldCover tiles
5. Saves to CSV: lon, lat, worldcover_code, worldcover_type

Output: inputs/location_lookups/esa_locations_lookup.csv
"""

import csv
import math
from pathlib import Path
from collections import defaultdict

# ESA WorldCover classes (11 classes)
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen"
}

# GPS offset for ±50m (approximate at Finland latitude ~60-70°N)
# At 60°N: 1° lat ≈ 111km, 1° lon ≈ 55km
LAT_OFFSET = 50 / 111000  # ~0.00045° for 50m north/south
LON_OFFSET = 50 / 55000   # ~0.00091° for 50m east/west (at ~60°N)


def get_tile_for_coord(longitude, latitude):
    """Determine which 3°×3° tile contains this coordinate."""
    lat_tile = math.floor(latitude / 3) * 3
    lon_tile = math.floor(longitude / 3) * 3
    return (lat_tile, lon_tile)


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


def group_points_by_tile(points):
    """
    Group points by their ESA WorldCover tile for efficient batch processing.
    
    Returns: dict of (lat_tile, lon_tile) -> list of (lon, lat)
    """
    points_by_tile = defaultdict(list)
    
    for lon, lat in points:
        tile = get_tile_for_coord(lon, lat)
        points_by_tile[tile].append((lon, lat))
    
    return points_by_tile


def process_tile(tile_path, coords):
    """
    Process all coordinates within a single tile at once.
    
    Returns: dict of (lon, lat) -> (code, description)
    """
    import rasterio
    from rasterio.sample import sample_gen
    
    results = {}
    
    try:
        with rasterio.open(tile_path) as src:
            # Sample all coordinates at once
            for coord, val in zip(coords, sample_gen(src, coords)):
                code = int(val[0])
                description = WORLDCOVER_CLASSES.get(code, "Unknown")
                results[coord] = (code, description)
    except Exception as e:
        print(f"  ⚠️  Error processing {tile_path.name}: {e}")
        for coord in coords:
            results[coord] = (None, f"Error: {str(e)}")
    
    return results


def main():
    # Configuration
    tiles_dir = Path("inputs/worldcover_tiles")
    output_file = Path("inputs/location_lookups/esa_locations_lookup.csv")
    
    print("=" * 70)
    print("ESA WORLDCOVER LOCATION PRECOMPUTATION")
    print("=" * 70)
    print()
    
    # Check tiles directory
    if not tiles_dir.exists():
        print(f"❌ ERROR: Tiles directory not found: {tiles_dir}")
        print("Please download tiles first using download_worldcover_tiles.py")
        return
    
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
    
    # Step 3: Group by tile
    print("Step 3: Grouping points by tile...")
    points_by_tile = group_points_by_tile(all_points)
    print(f"  Points span {len(points_by_tile)} tiles")
    
    for tile, coords in sorted(points_by_tile.items()):
        lat_tile, lon_tile = tile
        tile_name = f"N{lat_tile:02d}E{lon_tile:03d}"
        print(f"    {tile_name}: {len(coords):,} points")
    print()
    
    # Step 4: Process tiles
    print("Step 4: Extracting land cover from tiles...")
    all_results = {}
    
    for i, (tile, coords) in enumerate(points_by_tile.items(), 1):
        lat_tile, lon_tile = tile
        tile_name = f"ESA_WorldCover_10m_2020_v100_N{lat_tile:02d}E{lon_tile:03d}_Map.tif"
        tile_path = tiles_dir / tile_name
        
        print(f"  [{i}/{len(points_by_tile)}] Processing {tile_name} ({len(coords):,} points)...", 
              end=" ", flush=True)
        
        if not tile_path.exists():
            print(f"❌ MISSING!")
            for coord in coords:
                all_results[coord] = (None, "Tile not found")
        else:
            tile_results = process_tile(tile_path, coords)
            all_results.update(tile_results)
            print(f"✓")
    
    print()
    
    # Step 5: Save results
    print("Step 5: Saving results...")
    
    results = []
    errors = 0
    
    for (lon, lat), (code, description) in sorted(all_results.items()):
        if code is not None:
            results.append({
                'lon': lon,
                'lat': lat,
                'esa_code': code,
                'esa_type': description
            })
        else:
            errors += 1
    
    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lon', 'lat', 'esa_code', 'esa_type'])
        writer.writeheader()
        writer.writerows(results)
    
    print()
    print("=" * 70)
    print("✓ PRECOMPUTATION COMPLETE")
    print("=" * 70)
    print(f"Output file: {output_file}")
    print(f"Total points: {len(results):,}")
    print(f"Errors: {errors}")
    
    if errors > 0:
        print(f"\n⚠️  Warning: {errors} points could not be processed")
    
    # Show distribution
    from collections import Counter
    type_counts = Counter(r['esa_type'] for r in results)
    print("\nLand cover distribution:")
    for land_type, count in type_counts.most_common():
        pct = count / len(results) * 100 if len(results) > 0 else 0
        print(f"  {land_type:30s}: {count:6,} ({pct:5.1f}%)")
    
    print()
    print("🎉 Lookup file ready! Use lookup_landcover.py to query locations.")


if __name__ == '__main__':
    main()
