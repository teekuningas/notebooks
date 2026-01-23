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
echo ""
echo "Using 452 dataset prevalence for filtering both analyses (for comparability)"

# Step 1: Run analyses on 452 dataset (chisquared mode)
echo -e "\n[1/3] Running analyses on 452 interviews (chisquared mode)..."

echo "  - ESA Habitats..."
STATS_MODE=chisquared python3 nb_stats_esa_koodit.py "$THEMES_452" "./output/stats_comparison/452/esa_koodit" 2>&1 | grep -E "(Statistical mode|Using|Output|Filtering|Final|converged|Saved|Significant)"

echo "  - Bird Species..."
STATS_MODE=chisquared python3 nb_stats_bird_species_koodit.py "$THEMES_452" "./output/stats_comparison/452/bird_species_koodit" 2>&1 | grep -E "(Statistical mode|Using|Output|Filtering|Final|converged|Saved|Significant)"

echo "  - Bird Presence..."
STATS_MODE=chisquared python3 nb_stats_bird_presence_koodit.py "$THEMES_452" "./output/stats_comparison/452/bird_presence_koodit" 2>&1 | grep -E "(Statistical mode|Using|Output|Filtering|Final|converged|Saved|Significant)"

echo "  - Bird Groups..."
STATS_MODE=chisquared python3 nb_stats_bird_groups_koodit.py "$THEMES_452" "./output/stats_comparison/452/bird_groups_koodit" 2>&1 | grep -E "(Statistical mode|Using|Output|Filtering|Final|converged|Saved|Significant)"

# Step 2: Run analyses on 710 dataset (random_effects mode, using 452 prevalence filter)
echo -e "\n[2/3] Running analyses on 710 interviews (random_effects mode, using 452 prevalence)..."

echo "  - ESA Habitats..."
STATS_MODE=random_effects PREVALENCE_FILTER_DATASET="$THEMES_452" python3 nb_stats_esa_koodit.py "$THEMES_710" "./output/stats_comparison/710/esa_koodit" 2>&1 | grep -E "(Statistical mode|Using|Output|Filtering|Final|converged|Saved|Significant|Prevalence)"

echo "  - Bird Species..."
STATS_MODE=random_effects PREVALENCE_FILTER_DATASET="$THEMES_452" python3 nb_stats_bird_species_koodit.py "$THEMES_710" "./output/stats_comparison/710/bird_species_koodit" 2>&1 | grep -E "(Statistical mode|Using|Output|Filtering|Final|converged|Saved|Significant|Prevalence)"

echo "  - Bird Presence..."
STATS_MODE=random_effects PREVALENCE_FILTER_DATASET="$THEMES_452" python3 nb_stats_bird_presence_koodit.py "$THEMES_710" "./output/stats_comparison/710/bird_presence_koodit" 2>&1 | grep -E "(Statistical mode|Using|Output|Filtering|Final|converged|Saved|Significant|Prevalence)"

echo "  - Bird Groups..."
STATS_MODE=random_effects PREVALENCE_FILTER_DATASET="$THEMES_452" python3 nb_stats_bird_groups_koodit.py "$THEMES_710" "./output/stats_comparison/710/bird_groups_koodit" 2>&1 | grep -E "(Statistical mode|Using|Output|Filtering|Final|converged|Saved|Significant|Prevalence)"

# Step 3: Create comparison plots
echo -e "\n[3/3] Creating comparison plots..."
python3 nb_create_comparison_plots.py

echo -e "\n✓ All analyses and plots complete!"
echo "  452 results (chisquared): ./output/stats_comparison/452/"
echo "  710 results (random_effects): ./output/stats_comparison/710/"
echo "  Comparison plots: ./output/stats_comparison/comparison_*.png"
