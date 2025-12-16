#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3 python3Packages.rasterio python3Packages.numpy
"""
Create CCI habitat presence matrices (all-in-one script).

This script:
1. Loads recording locations from bird metadata
2. Generates 9-point circular sampling grid (100m radius, 200m diameter)
3. Extracts land cover from ESA CCI raster data (300m resolution, 37 classes)
4. Creates three CSV files with habitat presence (0/1) per recording:
   - cci_habitat_presence_original.csv (original CCI class names)
   - cci_habitat_presence_en.csv (simplified English)
   - cci_habitat_presence_fi.csv (simplified Finnish)

Output: inputs/bird-metadata-refined/cci_habitat_presence_*.csv

Runtime: ~30 seconds
"""

import csv
from pathlib import Path
from collections import Counter
import math

# ESA CCI Land Cover classes (37 classes in v2.0.7)
CCI_CLASSES = {
    0: "No data",
    10: "Cropland, rainfed",
    11: "Cropland, rainfed - Herbaceous cover",
    12: "Cropland, rainfed - Tree or shrub cover",
    20: "Cropland, irrigated or post-flooding",
    30: "Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous)",
    40: "Mosaic natural vegetation (tree, shrub, herbaceous) (>50%) / cropland (<50%)",
    50: "Tree cover, broadleaved, evergreen, closed to open (>15%)",
    60: "Tree cover, broadleaved, deciduous, closed to open (>15%)",
    61: "Tree cover, broadleaved, deciduous, closed (>40%)",
    62: "Tree cover, broadleaved, deciduous, open (15-40%)",
    70: "Tree cover, needleleaved, evergreen, closed to open (>15%)",
    71: "Tree cover, needleleaved, evergreen, closed (>40%)",
    72: "Tree cover, needleleaved, evergreen, open (15-40%)",
    80: "Tree cover, needleleaved, deciduous, closed to open (>15%)",
    81: "Tree cover, needleleaved, deciduous, closed (>40%)",
    82: "Tree cover, needleleaved, deciduous, open (15-40%)",
    90: "Tree cover, mixed leaf type (broadleaved and needleleaved)",
    100: "Mosaic tree and shrub (>50%) / herbaceous cover (<50%)",
    110: "Mosaic herbaceous cover (>50%) / tree and shrub (<50%)",
    120: "Shrubland",
    121: "Shrubland - Evergreen",
    122: "Shrubland - Deciduous",
    130: "Grassland",
    140: "Lichens and mosses",
    150: "Sparse vegetation (tree, shrub, herbaceous) (<15%)",
    152: "Sparse shrub (<15%)",
    153: "Sparse herbaceous (<15%)",
    160: "Tree cover, flooded, fresh or brackish water",
    170: "Tree cover, flooded, saline water",
    180: "Shrub or herbaceous cover, flooded, fresh/saline/brackish water",
    190: "Urban areas",
    200: "Bare areas",
    201: "Consolidated bare areas",
    202: "Unconsolidated bare areas",
    210: "Water bodies",
    220: "Permanent snow and ice"
}

# Simplified habitat labels mapping
HABITAT_LABELS = {
    # CROPLAND
    "Cropland, rainfed": ("cropland", "pelto"),
    "Cropland, rainfed - Herbaceous cover": ("cropland_herbaceous", "pelto_ruohovaltainen"),
    "Cropland, rainfed - Tree or shrub cover": ("cropland_woody", "pelto_puuvartinen"),
    "Cropland, irrigated or post-flooding": ("cropland_irrigated", "pelto_kasteltu"),
    
    # MOSAIC
    "Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous)": 
        ("mosaic_cropland", "pelto_mosaiikki"),
    "Mosaic natural vegetation (tree, shrub, herbaceous) (>50%) / cropland (<50%)": 
        ("mosaic_natural", "luonnonkasvillisuus_mosaiikki"),
    "Mosaic tree and shrub (>50%) / herbaceous cover (<50%)": 
        ("mosaic_vegetation", "mosaiikkikasvillisuus"),
    "Mosaic herbaceous cover (>50%) / tree and shrub (<50%)": 
        ("mosaic_herbaceous", "ruoho_mosaiikki"),
    
    # FOREST - NEEDLELEAVED
    "Tree cover, needleleaved, evergreen, closed to open (>15%)": 
        ("needleleaved_forest", "havupuumetsä"),
    "Tree cover, needleleaved, evergreen, closed (>40%)": 
        ("needleleaved_closed", "tiheä_havupuumetsä"),
    "Tree cover, needleleaved, evergreen, open (15-40%)": 
        ("needleleaved_open", "avoin_havupuumetsä"),
    "Tree cover, needleleaved, deciduous, closed to open (>15%)": 
        ("needleleaved_deciduous", "lehtipuumainen_havupuu"),
    "Tree cover, needleleaved, deciduous, closed (>40%)": 
        ("needleleaved_deciduous_closed", "tiheä_lehtipuumainen_havupuu"),
    "Tree cover, needleleaved, deciduous, open (15-40%)": 
        ("needleleaved_deciduous_open", "avoin_lehtipuumainen_havupuu"),
    
    # FOREST - BROADLEAVED
    "Tree cover, broadleaved, evergreen, closed to open (>15%)": 
        ("broadleaved_evergreen", "ikivihreä_lehtipuumetsä"),
    "Tree cover, broadleaved, deciduous, closed to open (>15%)": 
        ("broadleaved_forest", "lehtipuumetsä"),
    "Tree cover, broadleaved, deciduous, closed (>40%)": 
        ("broadleaved_closed", "tiheä_lehtipuumetsä"),
    "Tree cover, broadleaved, deciduous, open (15-40%)": 
        ("broadleaved_open", "avoin_lehtipuumetsä"),
    
    # FOREST - MIXED & FLOODED
    "Tree cover, mixed leaf type (broadleaved and needleleaved)": 
        ("mixed_forest", "sekametsä"),
    "Tree cover, flooded, fresh or brackish water": 
        ("flooded_forest_fresh", "tulvametsä_makea"),
    "Tree cover, flooded, saline water": 
        ("flooded_forest_saline", "tulvametsä_suolainen"),
    
    # SHRUBLAND & GRASSLAND
    "Shrubland": ("shrubland", "pensaikko"),
    "Shrubland - Evergreen": ("shrubland_evergreen", "ikivihreä_pensaikko"),
    "Shrubland - Deciduous": ("shrubland_deciduous", "lehtipensaikko"),
    "Grassland": ("grassland", "niitty"),
    
    # SPARSE VEGETATION
    "Sparse vegetation (tree, shrub, herbaceous) (<15%)": 
        ("sparse_vegetation", "harvakasvillisuus"),
    "Sparse shrub (<15%)": ("sparse_shrub", "harvapensaikko"),
    "Sparse herbaceous (<15%)": ("sparse_herbaceous", "harvaruoho"),
    
    # WETLAND
    "Shrub or herbaceous cover, flooded, fresh/saline/brackish water": 
        ("wetland", "kosteikko"),
    
    # OTHER
    "Lichens and mosses": ("moss_lichen", "jäkälä_sammal"),
    "Urban areas": ("urban", "kaupunki"),
    "Bare areas": ("bare", "paljas_maa"),
    "Consolidated bare areas": ("bare_consolidated", "kiinteä_paljas_maa"),
    "Unconsolidated bare areas": ("bare_unconsolidated", "irtonainen_paljas_maa"),
    "Water bodies": ("water", "vesialue"),
    "Permanent snow and ice": ("snow_ice", "ikijää"),
}


def get_simplified_labels(original_type):
    """Get simplified English and Finnish labels for a CCI type."""
    if original_type in HABITAT_LABELS:
        return HABITAT_LABELS[original_type]
    # Fallback: create safe names from original
    safe_en = original_type.lower().replace(' ', '_').replace(',', '').replace('(', '').replace(')', '').replace('/', '_')[:50]
    safe_fi = safe_en  # Use English as fallback
    return (safe_en, safe_fi)


def load_recording_locations():
    """Load recording locations from bird metadata."""
    # Load recording IDs from analysis dataset
    input_path = 'output/analyysi_koodit/6525b5f3/themes_51x452.csv'
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


def extract_cci_data(cci_file, points):
    """Extract CCI land cover types for given points."""
    import rasterio
    
    results = {}
    with rasterio.open(cci_file) as src:
        for lon, lat in points:
            try:
                row, col = src.index(lon, lat)
                # Read specific band (CCI has multiple bands for different years)
                # Using band 24 (2015 - most recent in v2.0.7)
                value = src.read(24, window=((row, row+1), (col, col+1)))[0, 0]
                
                if value in CCI_CLASSES:
                    cci_type = CCI_CLASSES[value]
                    results[(lon, lat)] = cci_type
            except:
                pass
    
    return results


def main():
    print("=" * 70)
    print("CCI HABITAT PRESENCE MATRIX GENERATOR")
    print("=" * 70)
    print()
    
    # Step 1: Load recording locations
    print("Step 1: Loading recording locations...")
    centers = load_recording_locations()
    print(f"✓ Loaded {len(centers)} recording locations")
    print()
    
    # Step 2: Find CCI data file
    print("Step 2: Finding CCI data file...")
    cci_dir = Path('inputs/cci_landcover')
    cci_file = None
    for pattern in ['*.tif', '*.tiff', '*.nc']:
        files = list(cci_dir.glob(pattern))
        if files:
            cci_file = files[0]
            break
    
    if not cci_file:
        print("✗ ERROR: No CCI data file found in inputs/cci_landcover/")
        print("  Expected: .tif, .tiff, or .nc file")
        return
    
    print(f"✓ Found: {cci_file.name}")
    print()
    
    # Step 3: Generate 9-point circles and extract data
    print("Step 3: Extracting CCI data for 9-point circles...")
    print(f"  {len(centers)} recordings × 9 points = {len(centers) * 9} samples")
    print()
    
    rec_habitats = {}  # rec_id -> set of CCI types
    
    for i, (rec_id, (lon, lat)) in enumerate(centers.items(), 1):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(centers)}...")
        
        # Generate 9-point circle
        points = generate_9point_circle(lon, lat)
        
        # Extract CCI data
        cci_data = extract_cci_data(cci_file, points)
        
        # Collect unique types
        types_present = set(cci_data.values())
        types_present.discard("No data")  # Remove no-data
        
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
    
    output_original = output_dir / 'cci_habitat_presence_original.csv'
    output_en = output_dir / 'cci_habitat_presence_en.csv'
    output_fi = output_dir / 'cci_habitat_presence_fi.csv'
    
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
    
    # 1. ORIGINAL CCI names
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
    print("✓ CCI HABITAT PRESENCE MATRICES COMPLETE")
    print("=" * 70)
    print()
    print("Files created:")
    print(f"  1. {output_original.name} - Original CCI class names")
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
        print(f"  {en_label:25s} ({fi_label:25s}): {count:3} ({pct:5.1f}%)")
    
    print()
    print(f"All files in: {output_dir}/")


if __name__ == '__main__':
    main()
