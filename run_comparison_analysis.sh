#!/bin/bash
#
# Run 452 vs 710 comparison analysis
# Uses existing theme files and runs all analyses cleanly
#

set -e

export MPLBACKEND=Agg

echo "============================================================"
echo "452 vs 710 COMPARISON ANALYSIS"
echo "============================================================"

# Clean and create output directories
rm -rf ./output/stats_comparison
mkdir -p ./output/stats_comparison/{452,710}

# Theme files
THEMES_452="./output/analyysi_koodit/88d43208/themes_98x452.csv"
THEMES_710="./output/analyysi_koodit/7176421e/themes_98x710.csv"

echo -e "\nUsing theme files:"
echo "  452: $THEMES_452 ($(wc -l < $THEMES_452) lines)"
echo "  710: $THEMES_710 ($(wc -l < $THEMES_710) lines)"

# Step 1: Run analyses on 452 dataset
echo -e "\n[1/2] Running analyses on 452 interviews..."

echo "  - ESA Habitats..."
python3 nb_stats_esa_koodit.py "$THEMES_452" "./output/stats_comparison/452/esa_koodit" 2>&1 | grep -E "(Using|Output|Filtering|Final|converged|Saved|Significant)"

echo "  - Bird Species..."
python3 nb_stats_bird_species_koodit.py "$THEMES_452" "./output/stats_comparison/452/bird_species_koodit" 2>&1 | grep -E "(Using|Output|Filtering|Final|converged|Saved|Significant)"

echo "  - Bird Presence..."
python3 nb_stats_bird_presence_koodit.py "$THEMES_452" "./output/stats_comparison/452/bird_presence_koodit" 2>&1 | grep -E "(Using|Output|Filtering|Final|converged|Saved|Significant)"

echo "  - Bird Groups..."
python3 nb_stats_bird_groups_koodit.py "$THEMES_452" "./output/stats_comparison/452/bird_groups_koodit" 2>&1 | grep -E "(Using|Output|Filtering|Final|converged|Saved|Significant)"

# Step 2: Run analyses on 710 dataset
echo -e "\n[2/2] Running analyses on 710 interviews..."

echo "  - ESA Habitats..."
python3 nb_stats_esa_koodit.py "$THEMES_710" "./output/stats_comparison/710/esa_koodit" 2>&1 | grep -E "(Using|Output|Filtering|Final|converged|Saved|Significant)"

echo "  - Bird Species..."
python3 nb_stats_bird_species_koodit.py "$THEMES_710" "./output/stats_comparison/710/bird_species_koodit" 2>&1 | grep -E "(Using|Output|Filtering|Final|converged|Saved|Significant)"

echo "  - Bird Presence..."
python3 nb_stats_bird_presence_koodit.py "$THEMES_710" "./output/stats_comparison/710/bird_presence_koodit" 2>&1 | grep -E "(Using|Output|Filtering|Final|converged|Saved|Significant)"

echo "  - Bird Groups..."
python3 nb_stats_bird_groups_koodit.py "$THEMES_710" "./output/stats_comparison/710/bird_groups_koodit" 2>&1 | grep -E "(Using|Output|Filtering|Final|converged|Saved|Significant)"

echo -e "\n✓ All analyses complete!"
echo "  452 results: ./output/stats_comparison/452/"
echo "  710 results: ./output/stats_comparison/710/"
echo ""
echo "Next step: Create comparison report"
