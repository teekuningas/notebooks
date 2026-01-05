"""
Utilities for interview feature analysis.
Provides habitat extraction and interview data loading.
"""

import os
import re
import pandas as pd
from pathlib import Path


# Habitat extraction utilities

ESA_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
    80: "Permanent water bodies", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss and lichen"
}

HABITAT_MAPPING = {
    "Tree cover": "forest", "Shrubland": "shrubland", "Grassland": "grassland",
    "Cropland": "cropland", "Built-up": "urban", "Bare / sparse vegetation": "bare",
    "Permanent water bodies": "water", "Herbaceous wetland": "wetland",
    "Moss and lichen": "moss_lichen"
}


def generate_9point_circle(lon, lat):
    """Generate 9-point circular sampling grid at 100m radius."""
    import math
    LAT_100M = 100 / 111000
    LON_100M = 100 / 55000
    DIAG_LAT = 100 * math.cos(math.radians(45)) / 111000
    DIAG_LON = 100 * math.sin(math.radians(45)) / 55000
    
    return [
        (lon, lat), (lon, lat + LAT_100M), (lon, lat - LAT_100M),
        (lon + LON_100M, lat), (lon - LON_100M, lat),
        (lon + DIAG_LON, lat + DIAG_LAT), (lon + DIAG_LON, lat - DIAG_LAT),
        (lon - DIAG_LON, lat - DIAG_LAT), (lon - DIAG_LON, lat + DIAG_LAT)
    ]


def extract_habitat_batch(centers, esa_files):
    """Extract habitat data for a batch of coordinates from ESA WorldCover tiles."""
    import rasterio
    from rasterio.sample import sample_gen
    
    results = {rec_id: set() for rec_id in centers}
    
    for esa_file in esa_files:
        try:
            with rasterio.open(esa_file) as src:
                points_in_tile = []
                rec_map = []
                
                for rec_id, (lon, lat) in centers.items():
                    if (src.bounds.left <= lon <= src.bounds.right and 
                        src.bounds.bottom <= lat <= src.bounds.top):
                        circle = generate_9point_circle(lon, lat)
                        for pt in circle:
                            points_in_tile.append(pt)
                            rec_map.append(rec_id)
                
                if not points_in_tile:
                    continue
                
                for i, val in enumerate(sample_gen(src, points_in_tile)):
                    code = int(val[0])
                    if code in ESA_CLASSES:
                        rec_id = rec_map[i]
                        results[rec_id].add(ESA_CLASSES[code])
                        
        except Exception as e:
            print(f"Error reading {esa_file}: {e}")
            
    return results


def get_habitat_data(df_recs, esa_dir, cache_file):
    """
    Get habitat data for all recordings, using cache when available.
    
    Args:
        df_recs: DataFrame with rec_id, lon, lat columns
        esa_dir: Directory containing ESA WorldCover .tif files
        cache_file: Path to cache CSV file
    
    Returns:
        DataFrame with rec_id and habitat columns (urban, forest, etc.)
    """
    required_ids = set(df_recs['rec_id'])
    print(f"Targeting {len(required_ids)} recordings for habitat analysis")
    
    # Load cache
    cached_data = {}
    if os.path.exists(cache_file):
        print(f"Loading habitat cache from {cache_file}...")
        df_cache = pd.read_csv(cache_file)
        # Remove rows with missing rec_id
        df_cache = df_cache.dropna(subset=['rec_id'])
        # Remove duplicate rec_ids (keep first)
        df_cache = df_cache.drop_duplicates(subset=['rec_id'], keep='first')
        for _, row in df_cache.iterrows():
            cached_data[row['rec_id']] = row.to_dict()
    
    missing_ids = required_ids - set(cached_data.keys())
    
    if not missing_ids:
        print("All required habitat data found in cache.")
        rows = [cached_data[rid] for rid in required_ids if rid in cached_data]
        return pd.DataFrame(rows)
    
    print(f"Computing habitat for {len(missing_ids)} missing recordings...")
    
    # Prepare coordinates
    df_unique = df_recs.drop_duplicates(subset=['rec_id'])
    coords_lookup = df_unique.set_index('rec_id')[['lon', 'lat']].to_dict('index')
    centers = {rid: (coords_lookup[rid]['lon'], coords_lookup[rid]['lat']) 
               for rid in missing_ids if rid in coords_lookup}
    
    # Find ESA files
    esa_files = list(Path(esa_dir).glob('*.tif')) + list(Path(esa_dir).glob('*.tiff'))
    if not esa_files:
        print("WARNING: No ESA WorldCover files found.")
        return pd.DataFrame()
    
    print("Extracting from ESA rasters (this may take a few minutes)...")
    habitat_results = extract_habitat_batch(centers, esa_files)
    
    # Format and cache results
    new_rows = []
    for rid, types in habitat_results.items():
        row = {'rec_id': rid}
        for h_code in HABITAT_MAPPING.values():
            row[h_code] = 0
        for t in types:
            if t in HABITAT_MAPPING:
                row[HABITAT_MAPPING[t]] = 1
        new_rows.append(row)
        cached_data[rid] = row
    
    df_new = pd.DataFrame(new_rows)
    if os.path.exists(cache_file):
        df_new.to_csv(cache_file, mode='a', header=False, index=False)
    else:
        df_new.to_csv(cache_file, index=False)
    
    print(f"Updated cache with {len(new_rows)} new records.")
    
    final_rows = [cached_data[rid] for rid in required_ids if rid in cached_data]
    return pd.DataFrame(final_rows)


# Interview data loading utilities

def load_interviews_from_listing(listing_file, interview_type='0'):
    """
    Parse interview uploads from MinIO listing file.
    
    Args:
        listing_file: Path to listing.txt file
        interview_type: '0' for observation interviews
    
    Returns:
        DataFrame with rec_id and has_interview_upload columns
    """
    interviews = []
    with open(listing_file, 'r') as f:
        for line in f:
            match = re.search(r'STANDARD ([A-Z0-9-]+)/(\d)/([a-f0-9-]+)/', line)
            if match:
                user_id, itype, rec_id = match.groups()
                if itype == interview_type:
                    interviews.append({'rec_id': rec_id, 'has_interview_upload': 1})
    
    return pd.DataFrame(interviews).drop_duplicates(subset=['rec_id'])


def compute_user_habitat_profiles(df_habitat, df_recs):
    """
    Compute habitat usage ratios for each user.
    
    Args:
        df_habitat: DataFrame with rec_id and habitat columns
        df_recs: DataFrame with rec_id and user columns
    
    Returns:
        DataFrame with user, user_urban_ratio, user_forest_ratio columns
    """
    df_merged = df_habitat.merge(df_recs[['rec_id', 'user']], on='rec_id')
    
    user_profiles = df_merged.groupby('user').agg({
        'urban': 'mean',
        'forest': 'mean'
    }).rename(columns={'urban': 'user_urban_ratio', 'forest': 'user_forest_ratio'})
    
    return user_profiles
