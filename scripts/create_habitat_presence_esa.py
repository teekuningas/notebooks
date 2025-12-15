#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3 python3Packages.rasterio python3Packages.numpy
"""
Create ESA WorldCover habitat presence matrices (all-in-one script).

This script:
1. Loads recording locations from bird metadata
2. Generates 9-point circular sampling grid (100m radius, 200m diameter)
3. Extracts land cover from ESA WorldCover raster data (10m resolution, 11 classes)
4. Creates three CSV files with habitat presence (0/1) per recording:
   - esa_habitat_presence_original.csv (original WorldCover class names)
   - esa_habitat_presence_en.csv (simplified English)
   - esa_habitat_presence_fi.csv (simplified Finnish)

Output: inputs/bird-metadata-refined/esa_habitat_presence_*.csv

Runtime: ~30 seconds
"""

import csv
from pathlib import Path
from collections import Counter
import math

# ESA WorldCover classes (11 classes)
ESA_CLASSES = {
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

# Simplified habitat labels mapping
HABITAT_LABELS = {
    "Tree cover": ("forest", "metsä"),
    "Shrubland": ("shrubland", "pensaikko"),
    "Grassland": ("grassland", "niitty"),
    "Cropland": ("cropland", "pelto"),
    "Built-up": ("urban", "kaupunki"),
    "Bare / sparse vegetation": ("bare", "paljas_maa"),
    "Snow and ice": ("snow_ice", "lumi_jää"),
    "Permanent water bodies": ("water", "vesialue"),
    "Herbaceous wetland": ("wetland", "kosteikko"),
    "Mangroves": ("mangroves", "mangrovemetsä"),
    "Moss and lichen": ("moss_lichen", "jäkälä_sammal"),
}


def get_simplified_labels(original_type):
    """Get simplified English and Finnish labels for an ESA type."""
    if original_type in HABITAT_LABELS:
        return HABITAT_LABELS[original_type]
    # Fallback: create safe names from original
    safe_en = original_type.lower().replace(' ', '_').replace('/', '_')
    safe_fi = safe_en  # Use English as fallback
    return (safe_en, safe_fi)


def load_recording_locations():
    """Load recording locations from bird metadata."""
    # Load recording IDs from analysis dataset
    input_path = 'output/analyysi_koodit/99699c4b/themes_143x452.csv'
    rec_ids = set()
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                rec_ids.add(row[0])
    
    # Load coordinates from metadata
    centers = {}  # rec_id -> (lon, lat)
    with open('inputs/bird-metadata/recs_since_June25.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['rec_id'] in rec_ids:
                try:
                    lat = float(row['lat'])
                    lon = float(row['lon'])
                    if 59 <= lat <= 71 and 19 <= lon <= 32:  # Finland bounds
                        centers[row['rec_id']] = (lon, lat)
                except (ValueError, KeyError):
                    pass
    
    return centers


def generate_9point_circle(lon, lat):
    """Generate 9-point circular sampling grid at 100m radius."""
    # Offsets for 100m radius
    LAT_100M = 100 / 111000  # ~0.0009° for 100m north/south
    LON_100M = 100 / 55000   # ~0.0018° for 100m east/west at ~60°N
    
    # Diagonal offset (100m at 45° angle)
    DIAG_LAT = 100 * math.cos(math.radians(45)) / 111000
    DIAG_LON = 100 * math.sin(math.radians(45)) / 55000
    
    return [
        (lon, lat),                              # Center
        (lon, lat + LAT_100M),                   # N
        (lon, lat - LAT_100M),                   # S
        (lon + LON_100M, lat),                   # E
        (lon - LON_100M, lat),                   # W
        (lon + DIAG_LON, lat + DIAG_LAT),       # NE
        (lon + DIAG_LON, lat - DIAG_LAT),       # SE
        (lon - DIAG_LON, lat - DIAG_LAT),       # SW
        (lon - DIAG_LON, lat + DIAG_LAT)        # NW
    ]


def find_esa_files(data_dir):
    """Find all ESA WorldCover tile files in the directory."""
    patterns = ['*.tif', '*.tiff']
    files = []
    
    for pattern in patterns:
        files.extend(list(data_dir.glob(pattern)))
    
    return files


def extract_esa_data(esa_files, points):
    """Extract ESA WorldCover land cover types for given points."""
    import rasterio
    from rasterio.sample import sample_gen
    
    results = {}
    
    # Try each file to find which tiles contain our points
    for esa_file in esa_files:
        try:
            with rasterio.open(esa_file) as src:
                # Check which points are in this tile's bounds
                points_in_tile = []
                for lon, lat in points:
                    if (src.bounds.left <= lon <= src.bounds.right and
                        src.bounds.bottom <= lat <= src.bounds.top):
                        points_in_tile.append((lon, lat))
                
                if not points_in_tile:
                    continue
                
                # Sample these points
                for coord, val in zip(points_in_tile, sample_gen(src, points_in_tile)):
                    code = int(val[0])
                    if code in ESA_CLASSES:
                        esa_type = ESA_CLASSES[code]
                        results[coord] = esa_type
        except:
            pass
    
    return results


def main():
    print("=" * 70)
    print("ESA WORLDCOVER HABITAT PRESENCE MATRIX GENERATOR")
    print("=" * 70)
    print()
    
    # Step 1: Load recording locations
    print("Step 1: Loading recording locations...")
    centers = load_recording_locations()
    print(f"✓ Loaded {len(centers)} recording locations")
    print()
    
    # Step 2: Find ESA data files
    print("Step 2: Finding ESA WorldCover data files...")
    esa_dir = Path('inputs/esa_worldcover')
    
    if not esa_dir.exists():
        print("✗ ERROR: No ESA WorldCover directory found")
        print("  Expected: inputs/esa_worldcover/")
        return
    
    esa_files = find_esa_files(esa_dir)
    
    if not esa_files:
        print("✗ ERROR: No ESA WorldCover tile files found")
        print("  Expected: .tif or .tiff files in inputs/esa_worldcover/")
        return
    
    print(f"✓ Found {len(esa_files)} tile files")
    print()
    
    # Step 3: Generate 9-point circles and extract data
    print("Step 3: Extracting ESA WorldCover data for 9-point circles...")
    print(f"  {len(centers)} recordings × 9 points = {len(centers) * 9} samples")
    print()
    
    rec_habitats = {}  # rec_id -> set of ESA types
    
    for i, (rec_id, (lon, lat)) in enumerate(centers.items(), 1):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(centers)}...")
        
        # Generate 9-point circle
        points = generate_9point_circle(lon, lat)
        
        # Extract ESA data
        esa_data = extract_esa_data(esa_files, points)
        
        # Collect unique types
        types_present = set(esa_data.values())
        
        rec_habitats[rec_id] = types_present
    
    print(f"✓ Extracted data for {len(rec_habitats)} recordings")
    print()
    
    # Step 4: Find all unique habitat types
    all_types = set()
    for types in rec_habitats.values():
        all_types.update(types)
    
    all_types = sorted(all_types)
    print(f"Found {len(all_types)} unique habitat types")
    print()
    
    # Step 5: Create simplified label mappings
    all_types_en = [get_simplified_labels(t)[0] for t in all_types]
    all_types_fi = [get_simplified_labels(t)[1] for t in all_types]
    
    # Step 6: Create output directory
    output_dir = Path('inputs/bird-metadata-refined')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_original = output_dir / 'esa_habitat_presence_original.csv'
    output_en = output_dir / 'esa_habitat_presence_en.csv'
    output_fi = output_dir / 'esa_habitat_presence_fi.csv'
    
    print("Step 4: Creating presence/absence matrices (3 versions)...")
    print()
    
    # Prepare data rows
    data_rows = []
    for rec_id in sorted(rec_habitats.keys()):
        data_rows.append({
            'rec_id': rec_id,
            'lon': centers[rec_id][0],
            'lat': centers[rec_id][1],
            'types': rec_habitats[rec_id]
        })
    
    # 1. ORIGINAL ESA names
    with open(output_original, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + all_types
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'],
                'lat': row_data['lat']
            }
            for habitat_type in all_types:
                row[habitat_type] = 1 if habitat_type in row_data['types'] else 0
            writer.writerow(row)
    
    print(f"✓ Created: {output_original.name}")
    
    # 2. ENGLISH simplified
    with open(output_en, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + all_types_en
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'],
                'lat': row_data['lat']
            }
            for i, habitat_type in enumerate(all_types):
                row[all_types_en[i]] = 1 if habitat_type in row_data['types'] else 0
            writer.writerow(row)
    
    print(f"✓ Created: {output_en.name}")
    
    # 3. FINNISH simplified
    with open(output_fi, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + all_types_fi
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'],
                'lat': row_data['lat']
            }
            for i, habitat_type in enumerate(all_types):
                row[all_types_fi[i]] = 1 if habitat_type in row_data['types'] else 0
            writer.writerow(row)
    
    print(f"✓ Created: {output_fi.name}")
    print()
    
    # Summary
    print("=" * 70)
    print("✓ ESA WORLDCOVER HABITAT PRESENCE MATRICES COMPLETE")
    print("=" * 70)
    print()
    print("Files created:")
    print(f"  1. {output_original.name} - Original WorldCover class names")
    print(f"  2. {output_en.name} - Simplified English")
    print(f"  3. {output_fi.name} - Simplified Finnish")
    print()
    print(f"Dimensions: {len(data_rows)} recordings × {len(all_types)} habitat types")
    print()
    print("Top 5 habitat types:")
    type_counts = {}
    for types in rec_habitats.values():
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
    
    for habitat_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = count / len(rec_habitats) * 100
        en_label = get_simplified_labels(habitat_type)[0]
        fi_label = get_simplified_labels(habitat_type)[1]
        print(f"  {en_label:15s} ({fi_label:15s}): {count:3} ({pct:5.1f}%)")
    
    print()
    print(f"All files in: {output_dir}/")


if __name__ == '__main__':
    main()
