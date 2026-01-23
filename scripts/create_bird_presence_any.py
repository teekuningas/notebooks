#!/usr/bin/env python3
"""
Create binary bird presence matrix (any bird vs no bird) at a configurable confidence threshold.

This script:
1. Loads target recording IDs from themes analysis
2. Filters bird observations with prediction >= THRESHOLD
3. Creates binary presence column: any_bird (1 if any species detected, 0 otherwise)

Output: inputs/bird-metadata-refined/bird_presence_any.csv
Columns: rec_id, lon, lat, any_bird

Threshold rationale:
- Lower threshold (e.g., 10%) ensures "no bird" group is truly empty of bird sounds
- Higher threshold (e.g., 50%) ensures "bird present" group definitely has birds
"""

import csv
from pathlib import Path
from collections import defaultdict

# CONFIGURATION
# ---------------------------------------------------------------------------
# Confidence threshold for "any bird" detection (0.0 - 1.0)
# 0.1 = 10% confidence. If any species is detected with >= 10% confidence,
# the recording is marked as having a bird.
THRESHOLD = 0.1
# ---------------------------------------------------------------------------


def load_target_recordings():
    """Load target recording IDs from the themes analysis."""
    rec_ids = []
    input_path = 'output/analyysi_koodit/7176421e/themes_98x710.csv'
    
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
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


def detect_any_bird(target_rec_ids, threshold):
    """
    Detect if any bird was present in each recording.
    
    Returns:
        dict: rec_id -> bool (True if any bird detected above threshold)
    """
    bird_presence = {}
    
    print(f"Detecting bird presence (threshold >= {threshold})...")
    
    with open('inputs/bird-metadata/species_ids_since_June25.csv', 'r') as f:
        reader = csv.DictReader(f)
        total_lines = 0
        detections = 0
        
        for row in reader:
            total_lines += 1
            if total_lines % 1000000 == 0:
                print(f"  Processed {total_lines // 1000000}M lines...")
            
            rec_id = row['rec_id']
            if rec_id in target_rec_ids:
                try:
                    prob = float(row['prediction'])
                    if prob >= threshold:
                        bird_presence[rec_id] = True
                        detections += 1
                except (ValueError, KeyError):
                    pass
    
    print(f"✓ Processed {total_lines:,} lines")
    print(f"✓ Found {detections:,} bird detections above threshold")
    print()
    
    return bird_presence


def main():
    print("=" * 70)
    print("BIRD PRESENCE (ANY vs NONE) GENERATOR")
    print("=" * 70)
    print()
    print(f"Threshold: ≥{THRESHOLD*100:.0f}% confidence")
    print("Purpose: Perceptual analysis (was any bird detected?)")
    print()
    
    # Step 1: Load target recordings
    print("Step 1: Loading target recordings...")
    target_rec_ids = load_target_recordings()
    target_rec_ids_set = set(target_rec_ids)
    print(f"✓ Loaded {len(target_rec_ids)} target recordings")
    print()
    
    # Step 2: Load recording metadata
    print("Step 2: Loading recording metadata...")
    rec_metadata = load_recording_metadata(target_rec_ids_set)
    print(f"✓ Loaded metadata for {len(rec_metadata)} recordings")
    print()
    
    # Step 3: Detect bird presence
    print(f"Step 3: Detecting bird presence at {THRESHOLD*100:.0f}% threshold...")
    bird_presence = detect_any_bird(target_rec_ids_set, threshold=THRESHOLD)
    print(f"✓ Detected birds in {len(bird_presence)} recordings")
    print()
    
    # Step 4: Create output
    output_dir = Path('inputs/bird-metadata-refined')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'bird_presence_any.csv'
    
    print("Step 4: Creating binary presence file...")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rec_id', 'lon', 'lat', 'any_bird'])
        writer.writeheader()
        
        for rec_id in target_rec_ids:
            lon = rec_metadata.get(rec_id, {}).get('lon', '')
            lat = rec_metadata.get(rec_id, {}).get('lat', '')
            any_bird = 1 if rec_id in bird_presence else 0
            
            writer.writerow({
                'rec_id': rec_id,
                'lon': lon,
                'lat': lat,
                'any_bird': any_bird
            })
    
    print(f"✓ Created: {output_file.name}")
    print()
    
    # Summary statistics
    n_with_birds = len(bird_presence)
    n_without_birds = len(target_rec_ids) - n_with_birds
    
    print("=" * 70)
    print("✓ BIRD PRESENCE FILE COMPLETE")
    print("=" * 70)
    print()
    print(f"File: {output_file}")
    print(f"Dimensions: {len(target_rec_ids)} interviews × 1 binary variable")
    print()
    print(f"With birds:    {n_with_birds:3d} ({n_with_birds/len(target_rec_ids)*100:.1f}%)")
    print(f"Without birds: {n_without_birds:3d} ({n_without_birds/len(target_rec_ids)*100:.1f}%)")
    print()
    print(f"✓ Ready for statistical analysis (nb_stats_bird_presence_koodit.py)")


if __name__ == '__main__':
    main()
