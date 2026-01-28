#!/usr/bin/env python3
"""
Create bird species presence matrices.

Generates presence/absence matrices for individual bird species detected
in recordings, using high confidence threshold for reliable identification.

Method:
1. Load species name mappings (scientific, English, Finnish)
2. Load target recording IDs and coordinates
3. Filter bird detections >= 90% confidence
4. Generate presence matrices with species as columns

Species are sorted alphabetically by scientific name (consistent order
across both language outputs). Column headers use common names.

Outputs: bird_species_english.csv, bird_species_finnish.csv
"""

import csv
from pathlib import Path
from collections import defaultdict

# Detection threshold
THRESHOLD = 0.9


def load_species_names():
    """Load species name mappings from classes_finland.csv.

    Returns:
        dict: scientific_name -> {'common': English, 'finnish': Finnish, ...}
    """
    species_map = {}
    with open('inputs/bird-metadata/classes_finland.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scientific = row['scientific_name']
            species_map[scientific] = {
                'code': row['species_code'],
                'common': row['common_name'],
                'finnish': row['suomenkielinen_nimi'],
                'scientific': scientific
            }
    return species_map


def load_target_recordings():
    """Load target recording IDs from themes file."""
    rec_ids = []
    input_path = 'output/analyysi_koodit/7176421e/themes_98x710.csv'

    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                rec_ids.append(row[0])
    return rec_ids


def load_recording_metadata(target_rec_ids):
    """Load lon/lat coordinates for recordings."""
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
                except (ValueError, KeyError):
                    pass
    return metadata


def extract_bird_observations(target_rec_ids, threshold):
    """Extract bird observations from species predictions.

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
    print(f"✓ Found {filtered_lines:,} detections")
    print()

    return dict(bird_observations)


def main():
    print("=" * 70)
    print("BIRD SPECIES PRESENCE MATRIX GENERATOR")
    print("=" * 70)
    print()
    print(f"Threshold: ≥{THRESHOLD*100:.0f}% confidence")
    print()

    # Step 1: Load species names
    print("Step 1: Loading species name mappings...")
    species_map = load_species_names()
    print(f"✓ Loaded {len(species_map)} species")
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
    print("Step 4: Extracting bird observations...")
    bird_observations = extract_bird_observations(target_rec_ids_set, threshold=THRESHOLD)
    print(f"✓ Found observations for {len(bird_observations)} recordings")
    print()

    # Step 5: Find all unique species and sort alphabetically
    all_species = set()
    for species_set in bird_observations.values():
        all_species.update(species_set)

    # Sort by scientific name (language-independent order)
    species_order = sorted(all_species)
    print(f"Found {len(species_order)} unique species")
    print()

    # Check for missing species in mapping
    missing_species = [s for s in species_order if s not in species_map]
    if missing_species:
        print(f"⚠️  Warning: {len(missing_species)} species not in classes_finland.csv")
        for s in missing_species[:5]:
            print(f"    - {s}")
        if len(missing_species) > 5:
            print(f"    ... and {len(missing_species) - 5} more")
        print()

    # Build display name dictionaries
    display_names_english = {}
    display_names_finnish = {}

    for species_sci in species_order:
        if species_sci in species_map:
            display_names_english[species_sci] = species_map[species_sci]['common']
            display_names_finnish[species_sci] = species_map[species_sci]['finnish']
        else:
            # Fallback for missing species: use scientific name
            display_names_english[species_sci] = species_sci
            display_names_finnish[species_sci] = species_sci.replace(' ', '_')

    # Step 6: Create output matrices
    output_dir = Path('inputs/bird-metadata-refined')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_english = output_dir / 'bird_species_english.csv'
    output_finnish = output_dir / 'bird_species_finnish.csv'

    print("Step 6: Creating species presence matrices...")
    print()

    # Prepare data rows (all target recordings, even without observations)
    data_rows = []
    for rec_id in target_rec_ids:
        lon, lat = None, None
        if rec_id in rec_metadata:
            lon = rec_metadata[rec_id]['lon']
            lat = rec_metadata[rec_id]['lat']

        species_present = bird_observations.get(rec_id, set())

        data_rows.append({
            'rec_id': rec_id,
            'lon': lon,
            'lat': lat,
            'species': species_present
        })

    # 1. English names (common names for publication)
    with open(output_english, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + [display_names_english[s] for s in species_order]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'] if row_data['lon'] is not None else '',
                'lat': row_data['lat'] if row_data['lat'] is not None else ''
            }
            for species_sci in species_order:
                display_name = display_names_english[species_sci]
                row[display_name] = 1 if species_sci in row_data['species'] else 0
            writer.writerow(row)

    print(f"✓ Created: {output_english.name}")

    # 2. Finnish names
    with open(output_finnish, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + [display_names_finnish[s] for s in species_order]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'] if row_data['lon'] is not None else '',
                'lat': row_data['lat'] if row_data['lat'] is not None else ''
            }
            for species_sci in species_order:
                display_name = display_names_finnish[species_sci]
                row[display_name] = 1 if species_sci in row_data['species'] else 0
            writer.writerow(row)

    print(f"✓ Created: {output_finnish.name}")
    print()

    # Summary statistics
    print("=" * 70)
    print("✓ MATRICES COMPLETE")
    print("=" * 70)
    print(f"Dimensions: {len(data_rows)} recordings × {len(species_order)} species")
    print()

    # Count presence for each species
    print("Most common species:")
    species_counts = {}
    for species_sci in species_order:
        count = sum(1 for row in data_rows if species_sci in row['species'])
        species_counts[species_sci] = count

    # Show top 10
    sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
    for species_sci, count in sorted_species[:10]:
        name_en = display_names_english[species_sci]
        name_fi = display_names_finnish[species_sci]
        pct = count / len(data_rows) * 100
        print(f"  {name_en:30s} ({name_fi:20s}): {count:3d} ({pct:5.1f}%)")

    print()


if __name__ == '__main__':
    main()
