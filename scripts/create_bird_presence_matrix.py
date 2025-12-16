#!/usr/bin/env python3
"""
Create bird species presence matrices.

This script:
1. Loads target recording IDs from koodit_16x452.csv
2. Filters species observations with prediction >= 50%
3. Creates two CSV files with bird presence (0/1) per recording:
   - bird_presence_latin.csv (scientific names)
   - bird_presence_finnish.csv (Finnish names)

Output: inputs/bird-metadata-refined/bird_presence_*.csv

Dimensions: ~450 recordings × ~126 species
"""

import csv
from pathlib import Path
from collections import defaultdict


def load_species_names():
    """Load species code mapping from classes_finland.csv."""
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
    print("BIRD SPECIES PRESENCE MATRIX GENERATOR")
    print("=" * 70)
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
    print("Step 4: Extracting bird observations (>=50% confidence)...")
    bird_observations = extract_bird_observations(target_rec_ids_set, threshold=0.5)
    print(f"✓ Found observations for {len(bird_observations)} recordings")
    print()
    
    # Step 5: Find all unique species
    all_species = set()
    for species_set in bird_observations.values():
        all_species.update(species_set)
    
    all_species = sorted(all_species)
    print(f"Found {len(all_species)} unique species")
    print()
    
    # Check for missing species in mapping
    missing_species = [s for s in all_species if s not in species_map]
    if missing_species:
        print(f"⚠️  Warning: {len(missing_species)} species not in classes_finland.csv")
        for s in missing_species[:5]:
            print(f"    - {s}")
        if len(missing_species) > 5:
            print(f"    ... and {len(missing_species) - 5} more")
        print()
    
    # Step 6: Create Finnish and Latin name lists
    species_finnish = []
    species_latin = []
    
    for species in all_species:
        if species in species_map:
            species_finnish.append(species_map[species]['finnish'])
            species_latin.append(species)
        else:
            # Fallback for missing species
            species_finnish.append(species.replace(' ', '_'))
            species_latin.append(species)
    
    # Step 7: Create output directory
    output_dir = Path('inputs/bird-metadata-refined')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_latin = output_dir / 'bird_presence_latin.csv'
    output_finnish = output_dir / 'bird_presence_finnish.csv'
    
    print("Step 5: Creating presence/absence matrices (2 versions)...")
    print()
    
    # Prepare data rows (use ALL target recordings, even those without observations)
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
    
    # 1. LATIN names (scientific)
    with open(output_latin, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + species_latin
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'] if row_data['lon'] is not None else '',
                'lat': row_data['lat'] if row_data['lat'] is not None else ''
            }
            for species in species_latin:
                row[species] = 1 if species in row_data['species'] else 0
            writer.writerow(row)
    
    print(f"✓ Created: {output_latin.name}")
    
    # 2. FINNISH names
    with open(output_finnish, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + species_finnish
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'] if row_data['lon'] is not None else '',
                'lat': row_data['lat'] if row_data['lat'] is not None else ''
            }
            for i, species in enumerate(species_latin):
                row[species_finnish[i]] = 1 if species in row_data['species'] else 0
            writer.writerow(row)
    
    print(f"✓ Created: {output_finnish.name}")
    print()
    
    # Summary statistics
    print("=" * 70)
    print("✓ BIRD PRESENCE MATRICES COMPLETE")
    print("=" * 70)
    print()
    print("Files created:")
    print(f"  1. {output_latin.name} - Scientific (Latin) names")
    print(f"  2. {output_finnish.name} - Finnish names")
    print()
    print(f"Dimensions: {len(data_rows)} recordings × {len(all_species)} species")
    print()
    
    # Count recordings with/without observations
    recs_with_birds = sum(1 for row in data_rows if len(row['species']) > 0)
    recs_without_birds = len(data_rows) - recs_with_birds
    
    print(f"Recordings with bird detections: {recs_with_birds} ({recs_with_birds/len(data_rows)*100:.1f}%)")
    print(f"Recordings without detections: {recs_without_birds} ({recs_without_birds/len(data_rows)*100:.1f}%)")
    print()
    
    # Top 10 most common species
    print("Top 10 most detected species:")
    species_counts = {}
    for row in data_rows:
        for species in row['species']:
            species_counts[species] = species_counts.get(species, 0) + 1
    
    for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = count / len(data_rows) * 100
        if species in species_map:
            finnish = species_map[species]['finnish']
            print(f"  {species:30s} ({finnish:20s}): {count:3} ({pct:5.1f}%)")
        else:
            print(f"  {species:30s} (unknown): {count:3} ({pct:5.1f}%)")
    
    print()
    print(f"All files in: {output_dir}/")
    print()
    print("✓ Ready for statistical analysis!")


if __name__ == '__main__':
    main()
