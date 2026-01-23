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

# %% ═════════ Statistical Analysis: Bird Presence → Koodit ═════════
# Examines whether bird presence (any bird vs no birds) predicts thematic codes.

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
    plot_top_associations_barplot,
    print_summary_stats,
    print_data_summary,
    plot_binary_predictor_distribution,
    save_summary_table_image,
    plot_effect_size_heatmap
)

sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

STANDARD_FIGSIZE = (12, 9)
FOOTNOTE_METHOD = "Bird presence defined as >10% confidence for any species in the recording."

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# Command-line arguments: themes_file output_dir
if len(sys.argv) >= 3:
    THEMES_FILE = sys.argv[1]
    output_dir = sys.argv[2]
else:
    THEMES_FILE = './output/analyysi_koodit/7176421e/themes_98x710.csv'
    output_dir = './output/bird_presence_koodit'

MIN_THEME_PREVALENCE = 0.20
MAX_THEME_PREVALENCE = 0.80

os.makedirs(output_dir, exist_ok=True)
print(f"Using themes: {THEMES_FILE}")
print(f"Output directory: {output_dir}")

# %% ═════════ 1. Load and Prepare Data ═════════

# %%
birds_raw = pd.read_csv('./inputs/bird-metadata-refined/bird_presence_any.csv', index_col='rec_id')

predictor_binary = pd.DataFrame()
predictor_binary['Bird present'] = birds_raw['any_bird'].astype(int)

# Load themes
koodit_raw = pd.read_csv(THEMES_FILE, index_col=0)
koodit_raw.columns = koodit_raw.columns.str.capitalize()

# Load user IDs for clustering
recs_metadata = pd.read_csv('./inputs/bird-metadata/recs_since_June25.csv')
recs_metadata = recs_metadata.set_index('rec_id')

common_ids = predictor_binary.index.intersection(koodit_raw.index)
predictor_binary = predictor_binary.loc[common_ids]
outcome_binary = (koodit_raw.loc[common_ids] >= 0.5).astype(int)

# User IDs for clustering (aligned to common_ids)
user_ids = recs_metadata.loc[common_ids, 'user']
print(f"User IDs loaded: {user_ids.notna().sum()}/{len(user_ids)} valid")

prevalence = outcome_binary.mean()
themes_to_keep = prevalence[(prevalence >= MIN_THEME_PREVALENCE) & (prevalence <= MAX_THEME_PREVALENCE)].index
print(f"Filtering themes: {len(outcome_binary.columns)} -> {len(themes_to_keep)} ({MIN_THEME_PREVALENCE*100:.0f}%-{MAX_THEME_PREVALENCE*100:.0f}% prevalence)")
outcome_binary = outcome_binary[themes_to_keep]

print_data_summary(predictor_binary, outcome_binary, "Bird presence", "Themes")

# %% ═════════ 2. Predictor Distribution ═════════

# %%
n_with_bird = predictor_binary['Bird present'].sum()
n_without_bird = len(predictor_binary) - n_with_bird

print("=" * 70)
print("BIRD PRESENCE")
print("=" * 70)
print(f"Interviews with birds:    {n_with_bird:3d} ({n_with_bird/len(predictor_binary)*100:.1f}%)")
print(f"Interviews without birds: {n_without_bird:3d} ({n_without_bird/len(predictor_binary)*100:.1f}%)")

plot_binary_predictor_distribution(
    predictor_binary, 'Bird present', 
    'No bird', 'Bird present',
    'Bird presence in recordings',
    f'{output_dir}/01_bird_presence_distribution.png',
    figsize=STANDARD_FIGSIZE,
    footnote=FOOTNOTE_METHOD
)

# %% ═════════ 3. Statistical Tests with Robust Standard Errors ═════════

# %%
print("\n" + "=" * 70)
print("STATISTICAL TESTS")
print("=" * 70)
print("\nMethod: Mixed-effects logistic regression (glmer)")
print("  - Accounts for non-independence (multiple recordings per user)")
print("  - Effect sizes from chi-square (descriptive, unbiased)")
print("  - P-values from robust SEs (inferential, corrected)")

# Run chi-square for effect sizes (descriptive statistics, valid always)
chi_results_df = run_chi_square_tests(predictor_binary, outcome_binary)

# Run logistic regression with clustered SEs for p-values (inferential statistics, accounts for clustering)
logit_results_df = run_mixed_effects_tests(predictor_binary, outcome_binary, user_ids)

# Merge: Keep effect sizes from chi-square, p-values from mixed effects
# IMPORTANT: Merge on Outcome/Predictor, don't just copy columns (results may be in different order!)
results_df = chi_results_df.merge(
    logit_results_df[['Outcome', 'Predictor', 'p_value', 'p_fdr', 'Significant']],
    on=['Outcome', 'Predictor'],
    how='left',
    suffixes=('', '_glmer')
)

# Report excluded cases
n_excluded = results_df['p_value_glmer'].isna().sum()
n_total = len(results_df)
print(f"\nConvergence check: {n_excluded}/{n_total} tests excluded (perfect separation)")
print(f"Valid tests: {n_total - n_excluded}/{n_total} ({(n_total-n_excluded)/n_total*100:.1f}%)")

print(f"\nResults:")
print(f"  Significant (p < 0.05, uncorrected): {(results_df['p_value_glmer'] < 0.05).sum()}")
print(f"  Significant (FDR q < 0.05):          {results_df['Significant_glmer'].sum()}")
print(f"  Medium+ effect size (V > 0.20):      {(results_df['Cramers_V'] > 0.2).sum()}")

print(f"\nTop 20 associations:")
print(results_df[['Outcome', 'Predictor', 'Chi2', 'p_fdr_glmer', 'Cramers_V', 'Difference']].head(20).to_string(index=False))

# %% ═════════ 4. Heatmap ═════════

# %%
plot_effect_size_heatmap(
    results_df,
    title='Effect of bird presence on theme (effect size)',
    xlabel='Bird Presence',
    ylabel='Theme',
    output_path=f'{output_dir}/03_effect_size_significance.png',
    figsize=STANDARD_FIGSIZE,
    vmax=0.4,
    footnote=f"{FOOTNOTE_METHOD} Significance: * q<0.05, ** q<0.01, *** q<0.001 (FDR)",
    p_fdr_col='p_fdr_glmer'
)

# %% ═════════ 5. Significant Associations ═════════

# %%
significant = results_df[results_df['Significant_glmer']].copy()

print("\n" + "=" * 70)
print(f"DETAILS: {len(significant)} SIGNIFICANT ASSOCIATIONS (FDR q < 0.05)")
print("=" * 70)

if len(significant) > 0:
    for i, (_, row) in enumerate(significant.iterrows(), 1):
        print(f"\n{i}. {row['Outcome']}")
        print(f"   Chi² = {row['Chi2']:.2f}, p = {row['p_value_glmer']:.2e}, FDR q = {row['p_fdr_glmer']:.2e}")
        print(f"   Cramér's V = {row['Cramers_V']:.3f}")
        print(f"   When bird present: {row['P(Outcome|Pred)']:.1f}%")
        print(f"   When no bird:      {row['P(Outcome|~Pred)']:.1f}%")
        print(f"   → Difference: {row['Difference']:+.1f} percentage points")
        
        if row['Difference'] > 0:
            print(f"   → Theme more common when bird present")
        else:
            print(f"   → Theme more common when no bird")
else:
    print("No significant associations found.")

# %% ═════════ 6. Bar Plot ═════════

# %%
plot_top_associations_barplot(
    results_df,
    title='Effect of bird presence on themes (V ≥ 0.1)',
    output_path=f'{output_dir}/04_top_associations.png',
    min_effect=0.1,
    figsize=STANDARD_FIGSIZE,
    footnote=f"{FOOTNOTE_METHOD} Significance: * q<0.05, ** q<0.01, *** q<0.001 (FDR)",
    p_fdr_col='p_fdr_glmer',
    p_value_col='p_value_glmer'
)

# %% ═════════ 7. Summary ═════════

# %%
print_summary_stats(results_df)

sig_assocs = results_df[results_df['Significant_glmer']]
if len(sig_assocs) > 0:
    enriched = sig_assocs[sig_assocs['Difference'] > 0].sort_values('Cramers_V', ascending=False)
    depleted = sig_assocs[sig_assocs['Difference'] < 0].sort_values('Cramers_V', ascending=False)
    
    print(f"\nThemes more common WHEN BIRD PRESENT ({len(enriched)}):")
    for _, row in enriched.iterrows():
        print(f"  {row['Outcome']:30s}  {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
              f"(V={row['Cramers_V']:.3f}, q={row['p_fdr_glmer']:.3f})")
    
    print(f"\nThemes more common WHEN NO BIRD ({len(depleted)}):")
    for _, row in depleted.iterrows():
        print(f"  {row['Outcome']:30s}  {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
              f"(V={row['Cramers_V']:.3f}, q={row['p_fdr_glmer']:.3f})")

# %% ═════════ 7. Save ═════════

# %%
save_summary_table_image(
    results_df,
    title="Bird Presence × Themes: Statistical Summary",
    output_path=f'{output_dir}/99_summary.png',
    significant_col='Significant_glmer',
    p_value_col='p_value_glmer'
)

output_file = f'{output_dir}/mixed_effects_results.csv'
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nMethod: Mixed-effects logistic regression (random user intercepts)")
print("=" * 70)
print(f"\nAnalyzed: Bird presence × Themes")
print(f"  Total interviews: {len(predictor_binary)}")
print(f"  With birds: {n_with_bird} ({n_with_bird/len(predictor_binary)*100:.1f}%)")
print(f"  Without birds: {n_without_bird} ({n_without_bird/len(predictor_binary)*100:.1f}%)")
print(f"Results: {output_file}")
print(f"Figures: {output_dir}/")

# %%
