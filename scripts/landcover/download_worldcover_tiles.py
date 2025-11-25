#!/usr/bin/env python3
"""
Download ESA WorldCover tiles for Finland and extract land cover types.
Compares with CORINE Land Cover classification.
"""

import csv
import os
import math
import requests
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

AWS_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map"


def get_required_tiles(csv_file, finland_only=True):
    """Determine which ESA WorldCover tiles are needed."""
    lats = []
    lons = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row['lat'])
                lon = float(row['lon'])
                
                # Filter to Finland if requested
                if finland_only and not (59 <= lat <= 71 and 19 <= lon <= 32):
                    continue
                    
                lats.append(lat)
                lons.append(lon)
            except (ValueError, KeyError):
                continue
    
    # Calculate tiles needed (3° x 3° tiles)
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    
    lat_start = math.floor(lat_min / 3) * 3
    lat_end = math.ceil(lat_max / 3) * 3
    lon_start = math.floor(lon_min / 3) * 3
    lon_end = math.ceil(lon_max / 3) * 3
    
    tiles = []
    for lat in range(lat_start, lat_end, 3):
        for lon in range(lon_start, lon_end, 3):
            tiles.append((lat, lon))
    
    print(f"Data range: {lat_min:.2f}°N to {lat_max:.2f}°N, {lon_min:.2f}°E to {lon_max:.2f}°E")
    print(f"Coordinates found: {len(lats):,}")
    print(f"Tiles needed: {len(tiles)}")
    
    return tiles, len(lats)


def download_tile(lat, lon, output_dir):
    """Download a single ESA WorldCover tile."""
    tile_name = f"ESA_WorldCover_10m_2020_v100_N{lat:02d}E{lon:03d}_Map.tif"
    url = f"{AWS_BASE_URL}/{tile_name}"
    output_path = output_dir / tile_name
    
    if output_path.exists():
        print(f"  ✓ Already downloaded: {tile_name}")
        return True
    
    print(f"  Downloading {tile_name}...", end=" ", flush=True)
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
        
        size_mb = total_size / 1024 / 1024
        print(f"{size_mb:.1f} MB ✓")
        return True
        
    except Exception as e:
        print(f"Failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def main():
    input_file = Path("inputs/bird-metadata/recs_since_June25.csv")
    tiles_dir = Path("data/worldcover_tiles")
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ESA WORLDCOVER TILE DOWNLOADER FOR FINLAND")
    print("="*70)
    print()
    
    # Determine required tiles
    tiles, coord_count = get_required_tiles(input_file, finland_only=True)
    
    total_size_mb = len(tiles) * 49
    print(f"\nTotal download size: ~{total_size_mb} MB (~{total_size_mb/1024:.2f} GB)")
    
    response = input(f"\nProceed with download of {len(tiles)} tiles? [y/N]: ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    print(f"\nDownloading to: {tiles_dir.absolute()}")
    print()
    
    # Download tiles
    successful = 0
    failed = 0
    
    for i, (lat, lon) in enumerate(tiles, 1):
        print(f"[{i}/{len(tiles)}]", end=" ")
        if download_tile(lat, lon, tiles_dir):
            successful += 1
        else:
            failed += 1
    
    print()
    print("="*70)
    print(f"Download complete: {successful} successful, {failed} failed")
    print("="*70)
    
    if successful > 0:
        print(f"\nTiles saved to: {tiles_dir.absolute()}")
        print(f"\nNext step: Run extract script to query land cover from these tiles")
        print("  (This will require rasterio library: pip install rasterio)")


if __name__ == '__main__':
    main()
