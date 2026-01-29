# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% ═════════ Statistical Analysis: Bird Groups → Themes ═════════
# Examines whether bird functional groups predict thematic codes.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
warnings.filterwarnings('ignore')

from utils_stats import (
    run_chi_square_tests,
    run_mixed_effects_tests,
    calculate_cooccurrence_matrix,
    plot_cooccurrence_heatmap,
    plot_cooccurrence_percentage_heatmap,
    plot_effect_size_heatmap,
    plot_top_associations_barplot,
    print_summary_stats,
    print_data_summary,
    save_summary_table_image
)

sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

STANDARD_FIGSIZE = (12, 9)
FOOTNOTE_METHOD = "Groups based on IOC World Bird List taxonomy (Orders/Families). Species identification threshold: 90% confidence."

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# Command-line arguments: themes_file output_dir
if len(sys.argv) >= 3:
    THEMES_FILE = sys.argv[1]
    output_dir = sys.argv[2]
else:
    THEMES_FILE = './output/analyysi_koodit/7176421e/themes_98x710.csv'
    output_dir = './output/bird_groups_koodit'

# Statistical mode configuration
STATS_MODE = os.environ.get('STATS_MODE', 'random_effects')  # Default: random_effects

if STATS_MODE not in ['random_effects', 'chisquared']:
    raise ValueError(f"Invalid STATS_MODE: {STATS_MODE}. Must be 'random_effects' or 'chisquared'")

print(f"Statistical mode: {STATS_MODE}")

# Prevalence filter dataset: optional environment variable to use another dataset's 
# prevalence for filtering (e.g., to ensure same themes across 452 and 710 comparisons)
PREVALENCE_FILTER_DATASET = os.environ.get('PREVALENCE_FILTER_DATASET', None)

MIN_THEME_PREVALENCE = 0.20
MAX_THEME_PREVALENCE = 0.80

os.makedirs(output_dir, exist_ok=True)
print(f"Using themes: {THEMES_FILE}")
print(f"Output directory: {output_dir}")
if PREVALENCE_FILTER_DATASET:
    print(f"Prevalence filter dataset: {PREVALENCE_FILTER_DATASET}")

# %% ═════════ 1. Load and Prepare Data ═════════

# %%
birds_raw = pd.read_csv('./inputs/bird-metadata-refined/bird_groups_finnish.csv')
birds_raw = birds_raw.set_index('rec_id').drop(columns=['lon', 'lat'])

bird_group_cols = birds_raw.columns.tolist()
birds_raw = birds_raw[bird_group_cols]

koodit_raw = pd.read_csv(THEMES_FILE, index_col=0)
koodit_raw.columns = koodit_raw.columns.str.capitalize()

common_ids = birds_raw.index.intersection(koodit_raw.index)
predictor_binary = birds_raw.loc[common_ids].astype(int)
outcome_binary = (koodit_raw.loc[common_ids] >= 0.5).astype(int)

# Load prevalence filter dataset if specified
if PREVALENCE_FILTER_DATASET:
    filter_themes = pd.read_csv(PREVALENCE_FILTER_DATASET, index_col=0)
    filter_themes.columns = filter_themes.columns.str.capitalize()
    filter_outcome = (filter_themes >= 0.5).astype(int)
else:
    filter_outcome = outcome_binary

# Apply translation to both outcome and filter (if enabled)
if os.environ.get('ENABLE_TRANSLATION', '0') == '1':
    from utils_translate import translate_theme_names
    outcome_binary = translate_theme_names(outcome_binary)
    filter_outcome = translate_theme_names(filter_outcome)
    print(f"✓ Translated {len(outcome_binary.columns)} theme names to English")

# Calculate prevalence for filtering themes
prevalence = filter_outcome.mean()
if PREVALENCE_FILTER_DATASET:
    print(f"Using prevalence from: {PREVALENCE_FILTER_DATASET}")
    prevalence = filter_outcome.mean()
    print(f"Using prevalence from: {PREVALENCE_FILTER_DATASET}")
else:
    # Use prevalence from current dataset
    prevalence = outcome_binary.mean()

themes_to_keep = prevalence[(prevalence >= MIN_THEME_PREVALENCE) & (prevalence <= MAX_THEME_PREVALENCE)].index
print(f"Filtering themes: {len(outcome_binary.columns)} -> {len(themes_to_keep)} ({MIN_THEME_PREVALENCE*100:.0f}%-{MAX_THEME_PREVALENCE*100:.0f}% prevalence)")
outcome_binary = outcome_binary[themes_to_keep]

# Filter bird groups (min 10 occurrences)
min_occurrences = 10
pred_counts = predictor_binary.sum()
groups_to_keep = pred_counts[pred_counts >= min_occurrences].index
print(f"Filtering bird groups: {len(predictor_binary.columns)} -> {len(groups_to_keep)} (min {min_occurrences} occurrences)")
predictor_binary = predictor_binary[groups_to_keep]

print_data_summary(predictor_binary, outcome_binary, "Bird groups", "Themes")

# Load user IDs (only for random_effects mode)
if STATS_MODE == 'random_effects':
    recs_metadata = pd.read_csv('./inputs/bird-metadata/recs_since_June25.csv')
    recs_metadata = recs_metadata.set_index('rec_id')
    user_ids = recs_metadata.loc[common_ids, 'user']
    
    print(f"\nUser statistics:")
    print(f"  Total recordings: {len(user_ids)}")
    print(f"  Unique users: {user_ids.nunique()}")
    print(f"  Users with multiple recordings: {(user_ids.value_counts() > 1).sum()}")
    print(f"  Max recordings per user: {user_ids.value_counts().max()}")

# %% ═════════ 2. Predictor Overlap ═════════

# %%
predictor_counts = predictor_binary.sum(axis=1)

print("=" * 70)
print("PREDICTOR OVERLAP ANALYSIS")
print("=" * 70)
print(f"Interviews with 0 predictors:  {(predictor_counts == 0).sum():3d} ({(predictor_counts == 0).mean()*100:.1f}%)")
print(f"Interviews with 1 predictor:   {(predictor_counts == 1).sum():3d} ({(predictor_counts == 1).mean()*100:.1f}%)")
print(f"Interviews with 2+ predictors: {(predictor_counts >= 2).sum():3d} ({(predictor_counts >= 2).mean()*100:.1f}%)")
print(f"\nMean predictors per interview: {predictor_counts.mean():.2f}")

cooccurrence_matrix = calculate_cooccurrence_matrix(predictor_binary)

plot_cooccurrence_heatmap(
    cooccurrence_matrix,
    title=f'Bird group co-occurrence (n={len(predictor_binary)})',
    xlabel='Bird group', ylabel='Bird group',
    output_path=f'{output_dir}/01_predictor_overlap_counts.png',
    figsize=STANDARD_FIGSIZE,
    footnote=FOOTNOTE_METHOD
)

plot_cooccurrence_percentage_heatmap(
    cooccurrence_matrix,
    title='Bird group co-occurrence (%)',
    xlabel='Bird group', ylabel='Bird group',
    output_path=f'{output_dir}/02_predictor_overlap_percentage.png',
    figsize=STANDARD_FIGSIZE,
    footnote=FOOTNOTE_METHOD
)

# %% ═════════ 3. Statistical Tests ═════════

# %%
print("\n" + "=" * 70)
print("STATISTICAL TESTS")
print("=" * 70)

if STATS_MODE == 'random_effects':
    print(f"User IDs loaded: {user_ids.notna().sum()}/{len(user_ids)} valid")
    print("\nMethod: Mixed-effects logistic regression (glmer)")
    print("  - Accounts for non-independence (multiple recordings per user)")
    print("  - Effect sizes: log-odds coefficients (directional)")
    print("  - P-values from likelihood ratio test")
    print("  - Note: No Difference (%) - use coefficient for directional effect")
    
    print("\nFitting mixed-effects models (this may take a few minutes)...")
    results_df = run_mixed_effects_tests(predictor_binary, outcome_binary, user_ids)
    
    # Odds_Ratio already calculated by R script
    
    n_missing = results_df['p_value'].isna().sum()
    if n_missing > 0:
        print(f"\n  Note: {n_missing} associations excluded (model failed to converge)")
    
    print(f"\nResults:")
    print(f"  Total tests: {len(results_df)}")
    print(f"  Valid p-values: {(~results_df['p_value'].isna()).sum()}")
    print(f"  Significant (p < 0.05, uncorrected): {(results_df['p_value'] < 0.05).sum()}")
    print(f"  Significant (FDR q < 0.05):          {results_df['Significant'].sum()}")
    
    # Create temporary column for sorting by absolute coefficient
    results_df['abs_coef'] = results_df['Coefficient'].abs()
    results_df = results_df.sort_values('abs_coef', ascending=False)
    results_df = results_df.drop(columns=['abs_coef'])
    
    print(f"\nTop 20 associations by |coefficient|:")
    print(results_df[['Outcome', 'Predictor', 'Coefficient', 'Odds_Ratio', 'p_fdr']].head(20).to_string(index=False))
    
    # Configuration for plots
    effect_threshold = 0.5
    heatmap_title = 'Effect of bird group on theme (|log-odds|)'
    barplot_title = f'Strongest bird group-theme associations (|log-odds| ≥ {effect_threshold})'
    heatmap_vmax = 1.5

elif STATS_MODE == 'chisquared':
    print("\nMethod: Chi-squared test (uncorrected)")
    print("  - Tests for association between predictors and outcomes")
    print("  - Effect sizes from Cramér's V")
    print("  - P-values from chi-squared test (no user clustering correction)")
    
    results_df = run_chi_square_tests(predictor_binary, outcome_binary)
    
    print(f"\nResults:")
    print(f"  Total tests: {len(results_df)}")
    print(f"  Significant (p < 0.05, uncorrected): {(results_df['p_value'] < 0.05).sum()}")
    print(f"  Significant (FDR q < 0.05):          {results_df['Significant'].sum()}")
    print(f"  Medium+ effect size (V > 0.20):      {(results_df['Cramers_V'] > 0.2).sum()}")
    
    results_df = results_df.sort_values('Cramers_V', ascending=False)
    
    print(f"\nTop 20 associations by Cramér's V:")
    print(results_df[['Outcome', 'Predictor', 'Chi2', 'Cramers_V', 'p_fdr', 'Difference']].head(20).to_string(index=False))
    
    # Configuration for plots
    effect_threshold = 0.1
    heatmap_title = "Effect of bird group on theme (Cramér's V)"
    barplot_title = f"Strongest bird group-theme associations (V ≥ {effect_threshold})"
    heatmap_vmax = 0.4

# %% ═════════ 4. Heatmap ═════════

# %%
plot_effect_size_heatmap(
    results_df,
    title=heatmap_title,
    xlabel='Bird group',
    ylabel='Theme',
    output_path=f'{output_dir}/03_effect_size_significance.png',
    figsize=STANDARD_FIGSIZE,
    vmax=heatmap_vmax,
    footnote=f"{FOOTNOTE_METHOD} Significance: * q<0.05, ** q<0.01, *** q<0.001 (FDR)",
    stats_mode=STATS_MODE
)

# %% ═════════ 5. Significant Associations ═════════

# %%
significant = results_df[results_df['Significant']].copy()

print("\n" + "=" * 70)
print(f"DETAILS: {len(significant)} SIGNIFICANT ASSOCIATIONS (FDR q < 0.05)")
print("=" * 70)

if len(significant) > 0:
    if STATS_MODE == 'random_effects':
        for i, (_, row) in enumerate(significant.iterrows(), 1):
            print(f"\n{i}. {row['Outcome']} × {row['Predictor']}")
            print(f"   Log-odds = {row['Coefficient']:.3f}, OR = {row['Odds_Ratio']:.2f}")
            print(f"   p = {row['p_value']:.4g}, FDR q = {row['p_fdr']:.4g}")
    else:
        for i, (_, row) in enumerate(significant.iterrows(), 1):
            print(f"\n{i}. {row['Outcome']} × {row['Predictor']}")
            print(f"   Chi² = {row['Chi2']:.2f}, p = {row['p_value']:.4g}, FDR q = {row['p_fdr']:.4g}")
            print(f"   Cramér's V = {row['Cramers_V']:.3f}")
            print(f"   When {row['Predictor']}: {row['P(Outcome|Pred)']:.1f}%")
            print(f"   Otherwise:                {row['P(Outcome|~Pred)']:.1f}%")
            print(f"   → Difference: {row['Difference']:+.1f} percentage points")
else:
    print("No significant associations found.")

# %% ═════════ 6. Bar Plot ═════════

# %%
plot_top_associations_barplot(
    results_df,
    title=barplot_title,
    output_path=f'{output_dir}/04_top_associations.png',
    min_effect=effect_threshold,
    figsize=STANDARD_FIGSIZE,
    footnote=f"{FOOTNOTE_METHOD} Significance: * q<0.05, ** q<0.01, *** q<0.001 (FDR)",
    stats_mode=STATS_MODE
)

# %% ═════════ 7. Predictor Profiles ═════════

# %%
print("\n" + "=" * 70)
print("PREDICTOR PROFILES")
print("=" * 70)

if STATS_MODE == 'chisquared':
    # For chi-squared mode, show enrichment based on probabilities
    for predictor in predictor_binary.columns:
        n_pred = predictor_binary[predictor].sum()
        
        if n_pred < 10:
            continue
        
        pred_assocs = results_df[results_df['Predictor'] == predictor].copy()
        pred_assocs['Enrichment'] = pred_assocs['P(Outcome|Pred)'] / pred_assocs['P(Outcome|~Pred)']
        pred_assocs = pred_assocs.sort_values('Enrichment', ascending=False)
        
        enriched = pred_assocs[(pred_assocs['Significant']) & (pred_assocs['Enrichment'] > 1)].head(5)
        depleted = pred_assocs[(pred_assocs['Significant']) & (pred_assocs['Enrichment'] < 1)].tail(3)
        
        print(f"\n{predictor} (n = {n_pred} interviews):")
        
        if len(enriched) > 0:
            print("  Enriched outcomes:")
            for _, row in enriched.iterrows():
                print(f"    • {row['Outcome']:30s} {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
                      f"({row['Enrichment']:.2f}×, q={row['p_fdr']:.3f})")
        
        if len(depleted) > 0:
            print("  Depleted outcomes:")
            for _, row in depleted.iterrows():
                print(f"    • {row['Outcome']:30s} {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
                      f"({row['Enrichment']:.2f}×, q={row['p_fdr']:.3f})")
        
        if len(enriched) == 0 and len(depleted) == 0:
            print("  No significant enrichments or depletions")

elif STATS_MODE == 'random_effects':
    # For random effects, show based on odds ratios
    for predictor in predictor_binary.columns:
        n_pred = predictor_binary[predictor].sum()
        
        if n_pred < 10:
            continue
        
        pred_assocs = results_df[results_df['Predictor'] == predictor].copy()
        pred_assocs = pred_assocs.sort_values('Odds_Ratio', ascending=False)
        
        enriched = pred_assocs[(pred_assocs['Significant']) & (pred_assocs['Coefficient'] > 0)].head(5)
        depleted = pred_assocs[(pred_assocs['Significant']) & (pred_assocs['Coefficient'] < 0)].tail(3)
        
        print(f"\n{predictor} (n = {n_pred} interviews):")
        
        if len(enriched) > 0:
            print("  Positively associated outcomes:")
            for _, row in enriched.iterrows():
                print(f"    • {row['Outcome']:30s} OR={row['Odds_Ratio']:5.2f}, log-odds={row['Coefficient']:+.2f}, q={row['p_fdr']:.3f}")
        
        if len(depleted) > 0:
            print("  Negatively associated outcomes:")
            for _, row in depleted.iterrows():
                print(f"    • {row['Outcome']:30s} OR={row['Odds_Ratio']:5.2f}, log-odds={row['Coefficient']:+.2f}, q={row['p_fdr']:.3f}")
        
        if len(enriched) == 0 and len(depleted) == 0:
            print("  No significant associations")

# %% ═════════ 8. Summary ═════════

# %%
print_summary_stats(results_df, stats_mode=STATS_MODE)

sig_assocs = results_df[results_df['Significant']]
if len(sig_assocs) > 0:
    print(f"\nOutcomes most often associated with predictors:")
    outcome_counts = sig_assocs['Outcome'].value_counts().head(10)
    for outcome, count in outcome_counts.items():
        print(f"  {outcome:35s} {count} predictor(s)")
    
    print(f"\nPredictors most often associated with outcomes:")
    pred_counts = sig_assocs['Predictor'].value_counts().head(5)
    for pred, count in pred_counts.items():
        print(f"  {pred:25s} {count} outcome(s)")

# %% ═════════ 9. Save ═════════

# %%
save_summary_table_image(
    results_df,
    title="Bird Groups × Themes: Statistical Summary",
    output_path=f'{output_dir}/99_summary.png',
    stats_mode=STATS_MODE
)

output_file = f'{output_dir}/results.csv'
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nStatistical mode: {STATS_MODE}")
print(f"Analyzed: Bird groups × Themes")
print(f"Results: {output_file}")
print(f"Figures: {output_dir}/")

# %%
