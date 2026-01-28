#!/usr/bin/env python3
"""
Create grouped bird presence matrices.

Groups bird species into 17 functional/habitat-based categories based on
taxonomic Order and Family from IOC World Bird List v14.2.

Method:
1. Load IOC taxonomy and target recording IDs
2. Filter bird observations (prediction >= 90% confidence)
3. Classify species into functional groups
4. Generate presence/absence matrices (0/1) for each recording

Functional groups (17 categories):
- Non-passerines: Aquatic Birds, Shorebirds, Raptors, Gamebirds,
  Woodpeckers, Cuckoos, Swifts, Pigeons
- Passerines (songbirds): Corvids, Tits, Thrushes, Finches, Warblers,
  Flycatchers, Sparrows, Other Songbirds
- Other: all remaining species

Groups prioritize ecological interpretability and statistical power over
strict taxonomic coherence. Some groups combine multiple Orders
(e.g., Aquatic Birds = waterfowl + grebes + cranes).

Outputs: bird_groups_english.csv, bird_groups_finnish.csv
"""

import csv
from pathlib import Path
from collections import defaultdict

# Taxonomic synonyms for species name changes in IOC taxonomy
SYNONYMS = {
    'Accipiter gentilis': 'Astur gentilis',
    'Corvus monedula': 'Coloeus monedula',
    'Sylvia communis': 'Curruca communis',
    'Sylvia curruca': 'Curruca curruca',
}

# Language-independent group keys (used internally)
GROUP_ORDER = [
    'aquatic_birds', 'shorebirds', 'raptors', 'gamebirds', 'woodpeckers',
    'cuckoos', 'swifts', 'pigeons', 'corvids', 'tits', 'thrushes', 'finches',
    'warblers', 'flycatchers', 'sparrows', 'other_songbirds', 'other'
]

# Display names for each language
DISPLAY_NAMES_ENGLISH = {
    'aquatic_birds': 'Aquatic Birds',
    'shorebirds': 'Shorebirds',
    'raptors': 'Raptors',
    'gamebirds': 'Gamebirds',
    'woodpeckers': 'Woodpeckers',
    'cuckoos': 'Cuckoos',
    'swifts': 'Swifts',
    'pigeons': 'Pigeons',
    'corvids': 'Corvids',
    'tits': 'Tits',
    'thrushes': 'Thrushes',
    'finches': 'Finches',
    'warblers': 'Warblers',
    'flycatchers': 'Flycatchers',
    'sparrows': 'Sparrows',
    'other_songbirds': 'Other Songbirds',
    'other': 'Other'
}

DISPLAY_NAMES_FINNISH = {
    'aquatic_birds': 'Vesilinnut',
    'shorebirds': 'Kahlaajat',
    'raptors': 'Petolinnut',
    'gamebirds': 'Kanalinnut',
    'woodpeckers': 'Tikkalinnut',
    'cuckoos': 'Käet',
    'swifts': 'Kiitäjät',
    'pigeons': 'Kyyhkyt',
    'corvids': 'Varikset',
    'tits': 'Tiaiset',
    'thrushes': 'Rastaat',
    'finches': 'Peipot',
    'warblers': 'Kertut',
    'flycatchers': 'Siepot',
    'sparrows': 'Varpuset',
    'other_songbirds': 'Muut varpuslinnut',
    'other': 'Muut'
}


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
    """Classify a species into a functional group based on Order and Family.
    
    Returns:
        str: Language-independent group key (e.g., 'aquatic_birds', 'corvids')
    """
    # Handle synonyms/renaming
    if scientific_name in SYNONYMS:
        scientific_name = SYNONYMS[scientific_name]

    if scientific_name not in taxonomy:
        return 'other'

    order = taxonomy[scientific_name]['order']
    family = taxonomy[scientific_name]['family']

    # --- NON-PASSERINES ---

    # Aquatic & Wetland Birds (Ducks, Geese, Loons, Grebes, Cranes, Herons, Cormorants)
    if order in ['ANSERIFORMES', 'GAVIIFORMES', 'PODICIPEDIFORMES', 'GRUIFORMES', 'PELECANIFORMES', 'SULIFORMES']:
        return 'aquatic_birds'

    # Shorebirds (Waders, Gulls, Terns, Auks)
    if order == 'CHARADRIIFORMES':
        return 'shorebirds'

    # Raptors (Hawks, Falcons, Owls)
    if order in ['ACCIPITRIFORMES', 'FALCONIFORMES', 'STRIGIFORMES']:
        return 'raptors'

    # Gamebirds (Grouse, Pheasants)
    if order == 'GALLIFORMES':
        return 'gamebirds'

    # Woodpeckers
    if order == 'PICIFORMES':
        return 'woodpeckers'

    # Cuckoos
    if order == 'CUCULIFORMES':
        return 'cuckoos'

    # Swifts
    if order == 'APODIFORMES':
        return 'swifts'

    # Pigeons & Doves
    if order == 'COLUMBIFORMES':
        return 'pigeons'

    # --- PASSERINES (Songbirds) - Split by Family ---
    if order == 'PASSERIFORMES':
        # Corvids (Crows, Jays, Magpies)
        if 'Corvidae' in family:
            return 'corvids'

        # Tits (Paridae)
        if 'Paridae' in family:
            return 'tits'

        # Thrushes (Turdidae)
        if 'Turdidae' in family:
            return 'thrushes'

        # Finches (Fringillidae)
        if 'Fringillidae' in family:
            return 'finches'

        # Warblers (Sylviidae, Phylloscopidae, Acrocephalidae, etc.)
        if any(f in family for f in ['Sylviidae', 'Phylloscopidae', 'Acrocephalidae', 'Locustellidae']):
            return 'warblers'

        # Flycatchers & Chats (Muscicapidae)
        if 'Muscicapidae' in family:
            return 'flycatchers'

        # Sparrows & Buntings
        if any(f in family for f in ['Passeridae', 'Emberizidae']):
            return 'sparrows'

        # Swallows, Wagtails, Larks (Open country/Aerial)
        if any(f in family for f in ['Hirundinidae', 'Motacillidae', 'Alaudidae']):
            return 'other_songbirds'

        # Other Passerines (Treecreepers, Goldcrests, etc.)
        return 'other_songbirds'

    # Everything else
    return 'other'


def load_target_recordings():
    """Load target recording IDs from the themes file."""
    rec_ids = []
    input_path = 'output/analyysi_koodit/7176421e/themes_98x710.csv'

    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:
                rec_ids.append(row[0])  # First column is rec_id
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
    print("Step 4: Extracting bird observations (>=90% confidence)...")
    bird_observations = extract_bird_observations(target_rec_ids_set, threshold=0.9)
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

        # Check if species is not in taxonomy (even after synonym mapping)
        mapped_name = SYNONYMS.get(species, species)
        if mapped_name not in taxonomy:
            unmatched.append(species)

    print()
    print("Species per group:")
    for group_key in GROUP_ORDER:
        count = group_counts.get(group_key, 0)
        name_en = DISPLAY_NAMES_ENGLISH[group_key]
        name_fi = DISPLAY_NAMES_FINNISH[group_key]
        print(f"  {name_en:20s} ({name_fi:20s}): {count:3} species")

    if unmatched:
        print(f"\n⚠️  Warning: {len(unmatched)} species not in IOC masterlist")
        for s in unmatched[:5]:
            print(f"    - {s}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched) - 5} more")
    print()

    # Step 6: Create output directory and files
    output_dir = Path('inputs/bird-metadata-refined')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_english = output_dir / 'bird_groups_english.csv'
    output_finnish = output_dir / 'bird_groups_finnish.csv'

    print("Step 6: Creating grouped presence/absence matrices...")
    print()

    # Prepare data rows (all target recordings, even without bird observations)
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
            group = species_groups.get(species, 'other')
            groups_present.add(group)

        data_rows.append({
            'rec_id': rec_id,
            'lon': lon,
            'lat': lat,
            'groups': groups_present
        })

    # 1. English names (for publication)
    with open(output_english, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + [DISPLAY_NAMES_ENGLISH[g] for g in GROUP_ORDER]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'] if row_data['lon'] is not None else '',
                'lat': row_data['lat'] if row_data['lat'] is not None else ''
            }
            for group_key in GROUP_ORDER:
                display_name = DISPLAY_NAMES_ENGLISH[group_key]
                row[display_name] = 1 if group_key in row_data['groups'] else 0
            writer.writerow(row)

    print(f"✓ Created: {output_english.name}")

    # 2. Finnish names
    with open(output_finnish, 'w', newline='') as f:
        fieldnames = ['rec_id', 'lon', 'lat'] + [DISPLAY_NAMES_FINNISH[g] for g in GROUP_ORDER]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_data in data_rows:
            row = {
                'rec_id': row_data['rec_id'],
                'lon': row_data['lon'] if row_data['lon'] is not None else '',
                'lat': row_data['lat'] if row_data['lat'] is not None else ''
            }
            for group_key in GROUP_ORDER:
                display_name = DISPLAY_NAMES_FINNISH[group_key]
                row[display_name] = 1 if group_key in row_data['groups'] else 0
            writer.writerow(row)

    print(f"✓ Created: {output_finnish.name}")
    print()

    # Summary statistics
    print("=" * 70)
    print("✓ MATRICES COMPLETE")
    print("=" * 70)
    print(f"Dimensions: {len(data_rows)} recordings × {len(GROUP_ORDER)} groups")
    print()

    # Count recordings with each group
    print("Recording presence per group:")
    for group_key in GROUP_ORDER:
        name_en = DISPLAY_NAMES_ENGLISH[group_key]
        name_fi = DISPLAY_NAMES_FINNISH[group_key]
        count = sum(1 for row in data_rows if group_key in row['groups'])
        pct = count / len(data_rows) * 100 if data_rows else 0
        print(f"  {name_en:20s} ({name_fi:20s}): {count:3d} ({pct:5.1f}%)")

    print()


if __name__ == '__main__':
    main()
