#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3
"""
Lookup land cover types from precomputed data.

This utility provides fast local lookup of land cover information
from the precomputed ESA WorldCover and CORINE datasets.

Usage:
    python lookup_landcover.py --lon 25.0279813 --lat 60.1670412
    python lookup_landcover.py --lon 24.5 --lat 61.0 --both
    python lookup_landcover.py --file coordinates.csv
"""

import csv
import argparse
from pathlib import Path
import math


class LandCoverLookup:
    """Fast lookup of land cover data from precomputed CSV files."""
    
    def __init__(self, esa_file=None, corine_file=None):
        self.esa_data = {}
        self.corine_data = {}
        
        # Load ESA data
        if esa_file and esa_file.exists():
            with open(esa_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lon = float(row['lon'])
                    lat = float(row['lat'])
                    self.esa_data[(lon, lat)] = {
                        'code': row['esa_code'],
                        'type': row['esa_type']
                    }
            print(f"✓ Loaded {len(self.esa_data):,} ESA locations")
        
        # Load CORINE data
        if corine_file and corine_file.exists():
            with open(corine_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lon = float(row['lon'])
                    lat = float(row['lat'])
                    if row['corine_code']:  # Skip errors
                        self.corine_data[(lon, lat)] = {
                            'code': row['corine_code'],
                            'type': row['corine_type']
                        }
            print(f"✓ Loaded {len(self.corine_data):,} CORINE locations")
    
    def find_nearest(self, target_lon, target_lat, data_dict, max_distance=0.001):
        """
        Find the nearest point in the dataset.
        
        max_distance: in degrees (~100m at Finland latitude)
        Returns: (distance, data) or (None, None)
        """
        nearest_dist = float('inf')
        nearest_data = None
        
        for (lon, lat), data in data_dict.items():
            # Euclidean distance (good enough for small distances)
            dist = math.sqrt((lon - target_lon)**2 + (lat - target_lat)**2)
            
            if dist < nearest_dist and dist <= max_distance:
                nearest_dist = dist
                nearest_data = data
        
        if nearest_data:
            # Convert distance to approximate meters
            dist_meters = nearest_dist * 111000  # Rough conversion at Finland latitude
            return (dist_meters, nearest_data)
        
        return (None, None)
    
    def lookup_esa(self, lon, lat):
        """
        Lookup ESA WorldCover land type.
        
        Returns: dict with 'code', 'type', 'match', 'distance_m'
        """
        # Try exact match first
        if (lon, lat) in self.esa_data:
            data = self.esa_data[(lon, lat)]
            return {
                'code': data['code'],
                'type': data['type'],
                'match': 'exact',
                'distance_m': 0
            }
        
        # Try nearest match (within 100m)
        distance, data = self.find_nearest(lon, lat, self.esa_data)
        
        if data:
            return {
                'code': data['code'],
                'type': data['type'],
                'match': 'nearest',
                'distance_m': round(distance, 1)
            }
        
        return None
    
    def lookup_corine(self, lon, lat):
        """
        Lookup CORINE Land Cover type.
        
        Returns: dict with 'code', 'type', 'match', 'distance_m'
        """
        # Try exact match first
        if (lon, lat) in self.corine_data:
            data = self.corine_data[(lon, lat)]
            return {
                'code': data['code'],
                'type': data['type'],
                'match': 'exact',
                'distance_m': 0
            }
        
        # Try nearest match (within 100m)
        distance, data = self.find_nearest(lon, lat, self.corine_data)
        
        if data:
            return {
                'code': data['code'],
                'type': data['type'],
                'match': 'nearest',
                'distance_m': round(distance, 1)
            }
        
        return None
    
    def lookup(self, lon, lat, source='both'):
        """
        Lookup land cover from specified source(s).
        
        source: 'esa', 'corine', or 'both'
        Returns: dict with results
        """
        result = {
            'lon': lon,
            'lat': lat
        }
        
        if source in ('esa', 'both'):
            esa_result = self.lookup_esa(lon, lat)
            result['esa'] = esa_result
        
        if source in ('corine', 'both'):
            corine_result = self.lookup_corine(lon, lat)
            result['corine'] = corine_result
        
        return result


def format_result(result):
    """Format lookup result for display."""
    lines = []
    lines.append(f"\n📍 Location: ({result['lon']:.7f}, {result['lat']:.7f})")
    lines.append("-" * 60)
    
    if 'esa' in result:
        if result['esa']:
            esa = result['esa']
            lines.append(f"🌍 ESA WorldCover:")
            lines.append(f"   Code: {esa['code']}")
            lines.append(f"   Type: {esa['type']}")
            lines.append(f"   Match: {esa['match']} ({esa['distance_m']}m)")
        else:
            lines.append("🌍 ESA WorldCover: ❌ Not found")
    
    if 'corine' in result:
        if result['corine']:
            corine = result['corine']
            lines.append(f"🗺️  CORINE Land Cover:")
            lines.append(f"   Code: {corine['code']}")
            lines.append(f"   Type: {corine['type']}")
            lines.append(f"   Match: {corine['match']} ({corine['distance_m']}m)")
        else:
            lines.append("🗺️  CORINE Land Cover: ❌ Not found")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Lookup land cover types from precomputed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --lon 25.0279813 --lat 60.1670412
  %(prog)s --lon 24.5 --lat 61.0 --both
  %(prog)s --lon 25.0 --lat 60.0 --source esa
  %(prog)s --file coordinates.csv
        """
    )
    
    parser.add_argument('--lon', type=float, help='Longitude')
    parser.add_argument('--lat', type=float, help='Latitude')
    parser.add_argument('--source', choices=['esa', 'corine', 'both'], default='both',
                        help='Data source (default: both)')
    parser.add_argument('--file', type=Path, help='CSV file with lon,lat columns')
    parser.add_argument('--esa-data', type=Path, default='inputs/location_lookups/esa_locations_lookup.csv',
                        help='ESA lookup file (default: inputs/location_lookups/esa_locations_lookup.csv)')
    parser.add_argument('--corine-data', type=Path, default='inputs/location_lookups/corine_locations_lookup.csv',
                        help='CORINE lookup file (default: inputs/location_lookups/corine_locations_lookup.csv)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.file and (args.lon is None or args.lat is None):
        parser.error("Either --lon/--lat or --file must be provided")
    
    # Initialize lookup
    print("Loading precomputed data...")
    lookup = LandCoverLookup(args.esa_data, args.corine_data)
    print()
    
    # Single coordinate lookup
    if args.lon is not None and args.lat is not None:
        result = lookup.lookup(args.lon, args.lat, args.source)
        print(format_result(result))
    
    # Batch file lookup
    elif args.file:
        if not args.file.exists():
            print(f"❌ Error: File not found: {args.file}")
            return
        
        print(f"Processing file: {args.file}")
        
        with open(args.file, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                try:
                    lon = float(row['lon'])
                    lat = float(row['lat'])
                    result = lookup.lookup(lon, lat, args.source)
                    print(format_result(result))
                except (ValueError, KeyError) as e:
                    print(f"❌ Error on row {i}: {e}")


if __name__ == '__main__':
    main()
