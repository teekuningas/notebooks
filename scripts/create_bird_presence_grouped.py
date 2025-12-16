#!/usr/bin/env python3
"""
Create grouped bird presence matrices.

This script works exactly like create_bird_presence_matrix.py but groups birds
into 7 functional categories instead of individual species columns.

This script:
1. Loads IOC World Bird List taxonomy
2. Loads target recording IDs from koodit_16x452.csv
3. Filters species observations with prediction >= 50%
4. Groups birds into 7 functional categories based on Order
5. Creates two CSV files with bird group presence (0/1) per recording:
   - bird_groups_latin.csv (English group names)
   - bird_groups_finnish.csv (Finnish group names)

Functional categories:
1. Vesilinnut / waterfowl - ducks, geese, loons, grebes
2. Kahlaajat / waders - plovers, sandpipers, gulls
3. Petolinnut / raptors - hawks, falcons, owls
4. Varpuslinnut / songbirds - all passerines
5. Kanalinnut / gamebirds - grouse, pheasants
6. Tikkalinnut / woodpeckers
7. Muut / other - pigeons, cuckoos, swifts, etc.

Output: inputs/bird-metadata-refined/bird_groups_*.csv

Dimensions: ~450 recordings × 7 groups
"""

import csv
from pathlib import Path
from collections import defaultdict


def load_ioc_taxonomy():
    """Load IOC World Bird List and create lookup by scientific name."""
    taxonomy = {}
    with open('inputs/bird-metadata/ioc_masterlist.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['IOC_14.2']:
                continue
            scientific = row['IOC_14.2'].strip()
            taxonomy[scientific] = {
                'order': row['Order'].strip().upper() if row['Order'] else '',
                'family': row['Family'].strip() if row['Family'] else ''
            }
    return taxonomy


def classify_species(scientific_name, taxonomy):
    """Classify a species into a functional group based on Order."""
    if scientific_name not in taxonomy:
        return 'Muut'
    
    order = taxonomy[scientific_name]['order']
    
    # Waterfowl
    if order in ['ANSERIFORMES', 'GAVIIFORMES', 'PODICIPEDIFORMES']:
        return 'Vesilinnut'
    
    # Waders/Shorebirds
    if order == 'CHARADRIIFORMES':
        return 'Kahlaajat'
    
    # Raptors
    if order in ['ACCIPITRIFORMES', 'FALCONIFORMES', 'STRIGIFORMES']:
        return 'Petolinnut'
    
    # Songbirds (all Passerines)
    if order == 'PASSERIFORMES':
        return 'Varpuslinnut'
    
    # Gamebirds
    if order == 'GALLIFORMES':
        return 'Kanalinnut'
    
    # Woodpeckers
    if order == 'PICIFORMES':
        return 'Tikkalinnut'
    
    # Everything else
    return 'Muut'


def load_target_recordings():
    """Load target recording IDs from the new themes file."""
    rec_ids = []
    # Updated path to the new themes file
    input_path = 'output/analyysi_koodit/6525b5f3/themes_51x452.csv'
    
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) # skip header
        for row in reader:
            if row:
                rec_ids.append(row[0]) # First column is rec_id
    return rec_ids


def load_recording_metadata(target_rec_ids):
    """Load lon/lat for recordings."""
    metadata = {}
    with open('inputs/bird-metadata/recs_since_June25.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec_id = row['rec_id']
            if rec_id in target_rec_ids:
                try:
                    metadata[rec_id] = {
                        'lon': float(row['lon']),
                        'lat': float(row['lat'])
                    }
                except:
                    pass
    return metadata


def extract_bird_observations(target_rec_ids, threshold):
    """
    Extract bird observations from species_ids_since_June25.csv.
    
    Returns:
        dict: rec_id -> set of species (scientific names)
    """
    bird_observations = defaultdict(set)
    
    print(f"Loading bird observations (threshold >= {threshold})...")
    
    with open('inputs/bird-metadata/species_ids_since_June25.csv', 'r') as f:
        reader = csv.DictReader(f)
        total_lines = 0
        filtered_lines = 0
        
        for row in reader:
            total_lines += 1
            if total_lines % 1000000 == 0:
                print(f"  Processed {total_lines // 1000000}M lines...")
            
            rec_id = row['rec_id']
            if rec_id in target_rec_ids:
                try:
                    prob = float(row['prediction'])
                    if prob >= threshold:
                        species = row['species']
                        bird_observations[rec_id].add(species)
                        filtered_lines += 1
                except (ValueError, KeyError):
                    pass
    
    print(f"✓ Processed {total_lines:,} lines")
    print(f"✓ Found {filtered_lines:,} high-confidence detections")
    print()
    
    return dict(bird_observations)


def main():
    print("=" * 70)
    print("BIRD PRESENCE GROUPED MATRIX GENERATOR")
    print("=" * 70)
    print()
    
    # Step 1: Load IOC taxonomy
    print("Step 1: Loading IOC World Bird List...")
    taxonomy = load_ioc_taxonomy()
    print(f"✓ Loaded {len(taxonomy)} species from IOC")
    print()
    
    # Step 2: Load target recordings
    print("Step 2: Loading target recordings...")
    target_rec_ids = load_target_recordings()
    target_rec_ids_set = set(target_rec_ids)
    print(f"✓ Loaded {len(target_rec_ids)} target recordings")
    print()
    
    # Step 3: Load recording metadata
    print("Step 3: Loading recording metadata...")
    rec_metadata = load_recording_metadata(target_rec_ids_set)
    print(f"✓ Loaded metadata for {len(rec_metadata)} recordings")
    print()
    
    # Step 4: Extract bird observations
    print("Step 4: Extracting bird observations (>=50% confidence)...")
    bird_observations = extract_bird_observations(target_rec_ids_set, threshold=0.5)
    print(f"✓ Found observations for {len(bird_observations)} recordings")
    print()
    
    # Step 5: Find all unique species and classify them
    all_species = set()
    for species_set in bird_observations.values():
        all_species.update(species_set)
    
    all_species = sorted(all_species)
    print(f"Found {len(all_species)} unique species")
    print()
    
    # Classify each species into groups
    print("Classifying species into functional groups...")
    species_groups = {}
    group_counts = defaultdict(int)
    unmatched = []
    
    for species in all_species:
        group = classify_species(species, taxonomy)
        species_groups[species] = group
        group_counts[group] += 1
        if species not in taxonomy:
            unmatched.append(species)
    
    print()
    print("Species per group:")
    for group in ['Vesilinnut', 'Kahlaajat', 'Petolinnut', 'Varpuslinnut', 
                   'Kanalinnut', 'Tikkalinnut', 'Muut']:
        count = group_counts.get(group, 0)
        print(f"  {group:15s}: {count:3} species")
    
    if unmatched:
        print(f"\n⚠️  Warning: {len(unmatched)} species not in IOC masterlist")
        for s in unmatched[:5]:
            print(f"    - {s}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched) - 5} more")
    print()
    
    # Step 6: Create output directory
    output_dir = Path('inputs/bird-metadata-refined')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_latin = output_dir / 'bird_groups_latin.csv'
    output_finnish = output_dir / 'bird_groups_finnish.csv'
    
    print("Step 5: Creating grouped presence/absence matrices (2 versions)...")
    print()
    
    # Define group names (7 functional bird groups)
    group_names_finnish = ['Vesilinnut', 'Kahlaajat', 'Petolinnut', 'Varpuslinnut',
                           'Kanalinnut', 'Tikkalinnut', 'Muut']
    
    group_names_english = {
        'Vesilinnut': 'waterfowl',
        'Kahlaajat': 'waders',
        'Petolinnut': 'raptors',
        'Varpuslinnut': 'songbirds',
        'Kanalinnut': 'gamebirds',
        'Tikkalinnut': 'woodpeckers',
        'Muut': 'other'
    }
    
    # Prepare data rows (use ALL target recordings, even those without observations)
    data_rows = []
    for rec_id in target_rec_ids:
        lon, lat = None, None
        if rec_id in rec_metadata:
            lon = rec_metadata[rec_id]['lon']
            lat = rec_metadata[rec_id]['lat']
        
        species_present = bird_observations.get(rec_id, set())
        
        # Determine which groups are present
        groups_present = set()
        for species in species_present:
            group = species_groups.get(species, 'Muut')
            groups_present.add(group)
        
        data_rows.append({
            'rec_id': rec_id,
            'lon': lon,
            'lat': lat,
            'groups': groups_present
        })
    
    # 1. ENGLISH names (latin)
    with open(output_latin, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + [group_names_english[g] for g in group_names_finnish]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'] if row_data['lon'] is not None else '',
                'lat': row_data['lat'] if row_data['lat'] is not None else ''
            }
            for group_fi in group_names_finnish:
                group_en = group_names_english[group_fi]
                row[group_en] = 1 if group_fi in row_data['groups'] else 0
            writer.writerow(row)
    
    print(f"✓ Created: {output_latin.name}")
    
    # 2. FINNISH names
    with open(output_finnish, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + group_names_finnish
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'] if row_data['lon'] is not None else '',
                'lat': row_data['lat'] if row_data['lat'] is not None else ''
            }
            for group in group_names_finnish:
                row[group] = 1 if group in row_data['groups'] else 0
            writer.writerow(row)
    
    print(f"✓ Created: {output_finnish.name}")
    print()
    
    # Summary statistics
    print("=" * 70)
    print("✓ GROUPED BIRD PRESENCE MATRICES COMPLETE")
    print("=" * 70)
    print()
    print(f"Files created:")
    print(f"  - {output_latin.name}")
    print(f"  - {output_finnish.name}")
    print()
    print(f"Dimensions: {len(data_rows)} recordings × 7 bird groups")
    print()
    
    # Count recordings with each group
    print("Recording presence per group:")
    for group_fi in group_names_finnish:
        group_en = group_names_english[group_fi]
        count = sum(1 for row in data_rows if group_fi in row['groups'])
        pct = count / len(data_rows) * 100 if data_rows else 0
        print(f"  {group_fi:15s} ({group_en:10s}): {count:3d} ({pct:5.1f}%)")
    
    print()


if __name__ == '__main__':
    main()
