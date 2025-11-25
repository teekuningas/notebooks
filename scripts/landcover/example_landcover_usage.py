#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3
"""
Example usage of the landcover lookup system.

This demonstrates how to quickly lookup land cover types for coordinates
using the precomputed lookup maps.

Prerequisites:
1. Run scripts/precompute_esa_locations.py (requires ESA WorldCover tiles)
2. Run scripts/precompute_corine_locations.py (requires internet, takes ~30 min)

Then you can use this script for instant lookups!
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from lookup_landcover import LandCoverLookup


def main():
    print("=" * 70)
    print("LAND COVER LOOKUP EXAMPLE")
    print("=" * 70)
    print()
    
    # Initialize lookup system
    esa_file = Path("inputs/location_lookups/esa_locations_lookup.csv")
    corine_file = Path("inputs/location_lookups/corine_locations_lookup.csv")
    
    if not esa_file.exists() and not corine_file.exists():
        print("❌ Error: No lookup files found!")
        print(f"   Expected: {esa_file} or {corine_file}")
        print()
        print("Please run the precompute scripts first:")
        print("  1. scripts/precompute_esa_locations.py")
        print("  2. scripts/precompute_corine_locations.py")
        return
    
    print("Loading precomputed data...")
    lookup = LandCoverLookup(esa_file, corine_file)
    print()
    
    # Example 1: Single coordinate lookup
    print("Example 1: Single coordinate lookup")
    print("-" * 70)
    lon, lat = 25.0279813, 60.1670412
    result = lookup.lookup(lon, lat, source='both')
    
    print(f"📍 Location: Helsinki area ({lon}, {lat})")
    if result.get('esa'):
        esa = result['esa']
        print(f"   ESA: {esa['type']} (code: {esa['code']})")
    if result.get('corine'):
        corine = result['corine']
        print(f"   CORINE: {corine['type']} (code: {corine['code']})")
    print()
    
    # Example 2: Multiple coordinates
    print("Example 2: Batch lookup")
    print("-" * 70)
    
    test_coords = [
        (22.38, 62.96, "Seinäjoki area"),
        (24.94, 60.17, "Helsinki"),
        (25.47, 65.01, "Oulu area"),
    ]
    
    for lon, lat, name in test_coords:
        result = lookup.lookup(lon, lat, source='esa')
        if result.get('esa'):
            esa = result['esa']
            match_info = f"({esa['match']}, {esa['distance_m']}m)" if esa['match'] != 'exact' else "(exact)"
            print(f"   {name:20s}: {esa['type']} {match_info}")
        else:
            print(f"   {name:20s}: Not found")
    print()
    
    # Example 3: Demonstrate speed
    print("Example 3: Performance test")
    print("-" * 70)
    
    import time
    start = time.time()
    
    # Lookup 100 random points
    num_lookups = 100
    for i in range(num_lookups):
        lon = 24.0 + i * 0.01
        lat = 60.0 + i * 0.01
        result = lookup.lookup(lon, lat, source='esa')
    
    elapsed = time.time() - start
    per_lookup = elapsed / num_lookups * 1000
    
    print(f"   {num_lookups} lookups in {elapsed:.3f} seconds")
    print(f"   Average: {per_lookup:.2f} ms per lookup")
    print()
    
    print("✓ Examples complete! The lookup system is working.")
    print()
    print("For command-line usage, see: ./scripts/lookup_landcover.py --help")


if __name__ == '__main__':
    main()
