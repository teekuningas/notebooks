#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3 python3Packages.rasterio python3Packages.numpy
"""
Create ESA WorldCover habitat presence matrices.

Generates presence/absence matrices for land cover types around recording
locations using ESA WorldCover data (10m resolution, 11 classes).

Method:
1. Load target recording IDs and coordinates
2. Generate 9-point circular sampling grid (100m radius) per recording
3. Extract land cover types from ESA WorldCover raster tiles
4. Create presence matrices with habitat types as columns

Habitat types are sorted alphabetically by internal key (consistent order
across both language outputs). Column headers use display names.

Outputs: habitat_esa_english.csv, habitat_esa_finnish.csv
"""

import csv
from pathlib import Path
import math

# ESA WorldCover class codes (11 classes)
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

# Language-independent habitat keys (sorted alphabetically)
HABITAT_ORDER = [
    'bare',
    'cropland',
    'forest',
    'grassland',
    'mangroves',
    'moss_lichen',
    'shrubland',
    'snow_ice',
    'urban',
    'water',
    'wetland'
]

# Display names for outputs
DISPLAY_NAMES_ENGLISH = {
    'bare': 'Bare / sparse vegetation',
    'cropland': 'Cropland',
    'forest': 'Forest',
    'grassland': 'Grassland',
    'mangroves': 'Mangroves',
    'moss_lichen': 'Moss and lichen',
    'shrubland': 'Shrubland',
    'snow_ice': 'Snow and ice',
    'urban': 'Built-up',
    'water': 'Permanent water bodies',
    'wetland': 'Herbaceous wetland'
}

DISPLAY_NAMES_FINNISH = {
    'bare': 'Paljas maa',
    'cropland': 'Pelto',
    'forest': 'Metsä',
    'grassland': 'Niitty',
    'mangroves': 'Mangrovemetsä',
    'moss_lichen': 'Jäkälä ja sammal',
    'shrubland': 'Pensaikko',
    'snow_ice': 'Lumi ja jää',
    'urban': 'Kaupunki',
    'water': 'Vesialue',
    'wetland': 'Kosteikko'
}

# Mapping from ESA class names to internal keys
ESA_TO_KEY = {
    "Tree cover": 'forest',
    "Shrubland": 'shrubland',
    "Grassland": 'grassland',
    "Cropland": 'cropland',
    "Built-up": 'urban',
    "Bare / sparse vegetation": 'bare',
    "Snow and ice": 'snow_ice',
    "Permanent water bodies": 'water',
    "Herbaceous wetland": 'wetland',
    "Mangroves": 'mangroves',
    "Moss and lichen": 'moss_lichen'
}


def load_recording_locations():
    """Load recording locations from target dataset."""
    # Load recording IDs
    input_path = 'output/analyysi_koodit/7176421e/themes_98x710.csv'
    rec_ids = set()
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                rec_ids.add(row[0])

    # Load coordinates
    centers = {}
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
    """Find all ESA WorldCover tile files."""
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
        except (ValueError, KeyError):
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

    # Step 3: Extract ESA data for 9-point circles
    print("Step 3: Extracting ESA WorldCover data...")
    print(f"  {len(centers)} recordings × 9 points = {len(centers) * 9} samples")
    print()

    rec_habitats = {}

    for i, (rec_id, (lon, lat)) in enumerate(centers.items(), 1):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(centers)}...")

        # Generate 9-point circle
        points = generate_9point_circle(lon, lat)

        # Extract ESA data
        esa_data = extract_esa_data(esa_files, points)

        # Convert ESA class names to internal keys
        keys_present = set()
        for esa_type in esa_data.values():
            if esa_type in ESA_TO_KEY:
                keys_present.add(ESA_TO_KEY[esa_type])

        rec_habitats[rec_id] = keys_present

    print(f"✓ Extracted data for {len(rec_habitats)} recordings")
    print()

    # Step 4: Find all unique habitat types (use HABITAT_ORDER for consistency)
    all_keys_present = set()
    for keys in rec_habitats.values():
        all_keys_present.update(keys)

    # Filter HABITAT_ORDER to only include keys that are present
    habitat_keys = [k for k in HABITAT_ORDER if k in all_keys_present]
    print(f"Found {len(habitat_keys)} unique habitat types")
    print()

    # Step 5: Create output matrices
    output_dir = Path('inputs/bird-metadata-refined')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_english = output_dir / 'habitat_esa_english.csv'
    output_finnish = output_dir / 'habitat_esa_finnish.csv'

    print("Step 4: Creating habitat presence matrices...")
    print()

    # Prepare data rows
    data_rows = []
    for rec_id in sorted(rec_habitats.keys()):
        data_rows.append({
            'rec_id': rec_id,
            'lon': centers[rec_id][0],
            'lat': centers[rec_id][1],
            'keys': rec_habitats[rec_id]
        })

    # 1. English names
    with open(output_english, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + [DISPLAY_NAMES_ENGLISH[k] for k in habitat_keys]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'],
                'lat': row_data['lat']
            }
            for key in habitat_keys:
                display_name = DISPLAY_NAMES_ENGLISH[key]
                row[display_name] = 1 if key in row_data['keys'] else 0
            writer.writerow(row)

    print(f"✓ Created: {output_english.name}")

    # 2. Finnish names
    with open(output_finnish, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + [DISPLAY_NAMES_FINNISH[k] for k in habitat_keys]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'],
                'lat': row_data['lat']
            }
            for key in habitat_keys:
                display_name = DISPLAY_NAMES_FINNISH[key]
                row[display_name] = 1 if key in row_data['keys'] else 0
            writer.writerow(row)

    print(f"✓ Created: {output_finnish.name}")
    print()

    # Summary statistics
    print("=" * 70)
    print("✓ MATRICES COMPLETE")
    print("=" * 70)
    print(f"Dimensions: {len(data_rows)} recordings × {len(habitat_keys)} habitat types")
    print()

    # Count presence for each habitat type
    print("Most common habitat types:")
    key_counts = {}
    for key in habitat_keys:
        count = sum(1 for row in data_rows if key in row['keys'])
        key_counts[key] = count

    # Show top 5
    sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
    for key, count in sorted_keys[:5]:
        name_en = DISPLAY_NAMES_ENGLISH[key]
        name_fi = DISPLAY_NAMES_FINNISH[key]
        pct = count / len(data_rows) * 100
        print(f"  {name_en:30s} ({name_fi:20s}): {count:3d} ({pct:5.1f}%)")

    print()


if __name__ == '__main__':
    main()
