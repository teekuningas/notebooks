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
#
# Examines whether bird presence (any bird vs no birds) predicts thematic codes (koodit).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from utils_stats import (
    run_chi_square_tests,
    plot_top_associations_barplot,
    print_summary_stats,
    print_data_summary,
    plot_binary_predictor_distribution,
    plot_binary_predictor_prevalence,
    save_summary_table_image
)

sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

STANDARD_FIGSIZE = (12, 9)

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# Theme prevalence filtering (avoid extreme distributions)
MIN_THEME_PREVALENCE = 0.20  # 20%
MAX_THEME_PREVALENCE = 0.80  # 80%

output_dir = './output/bird_presence_koodit'
os.makedirs(output_dir, exist_ok=True)

# %% ═════════ 1. Load and Prepare Data ═════════

# %%
# Load "any bird" presence (50% confidence threshold)
birds_raw = pd.read_csv('./inputs/bird-metadata-refined/bird_presence_any.csv', index_col='rec_id')

# Extract the single binary predictor
predictor_binary = pd.DataFrame()
predictor_binary['Bird present'] = birds_raw['any_bird'].astype(int)

# Load outcomes from new themes file
koodit_raw = pd.read_csv('./output/analyysi_koodit/88d43208/themes_98x452.csv', index_col=0)
koodit_raw.columns = koodit_raw.columns.str.capitalize()

# Align by rec_id
common_ids = predictor_binary.index.intersection(koodit_raw.index)
predictor_binary = predictor_binary.loc[common_ids]
outcome_binary = (koodit_raw.loc[common_ids] >= 0.5).astype(int)

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

# Visualize distribution
plot_binary_predictor_distribution(
    predictor_binary, 'Bird present', 
    'No bird', 'Bird present',
    'Bird presence in interviews',
    f'{output_dir}/01_bird_presence_distribution.png',
    figsize=STANDARD_FIGSIZE
)


# %% ═════════ 3. Chi-Square Tests ═════════

# %%
print("\n" + "=" * 70)
print("CHI-SQUARE TESTS")
print("=" * 70)

results_df = run_chi_square_tests(predictor_binary, outcome_binary)

print(f"\nResults:")
print(f"  Significant (p < 0.05, uncorrected): {(results_df['p_value'] < 0.05).sum()}")
print(f"  Significant (FDR q < 0.05):          {results_df['Significant'].sum()}")
print(f"  Medium+ effect size (V > 0.20):      {(results_df['Cramers_V'] > 0.2).sum()}")

print(f"\nTop 20 associations:")
print(results_df[['Outcome', 'Predictor', 'Chi2', 'p_fdr', 'Cramers_V', 'Difference']].head(20).to_string(index=False))

# %% ═════════ 4. Effect Size Visualization ═════════

# %%
# Bar plot showing all associations with V >= 0.1
# Note: For binary predictor, labels include "Any_bird" explicitly
plot_top_associations_barplot(
    results_df,
    title='Effect of bird presence on themes (V ≥ 0.1)',
    output_path=f'{output_dir}/02_top_associations.png',
    min_effect=0.1,
    figsize=STANDARD_FIGSIZE
)


# %% ═════════ 5. Significant Associations ═════════

# %%
significant = results_df[results_df['Significant']].copy()

print("\n" + "=" * 70)
print(f"DETAILS: {len(significant)} SIGNIFICANT ASSOCIATIONS (FDR q < 0.05)")
print("=" * 70)

if len(significant) > 0:
    for i, (_, row) in enumerate(significant.iterrows(), 1):
        print(f"\n{i}. {row['Outcome']}")
        print(f"   Chi² = {row['Chi2']:.2f}, p = {row['p_value']:.2e}, FDR q = {row['p_fdr']:.2e}")
        print(f"   Cramér's V = {row['Cramers_V']:.3f}")
        print(f"   When bird present: {row['P(Outcome|Pred)']:.1f}%")
        print(f"   When no bird:      {row['P(Outcome|~Pred)']:.1f}%")
        print(f"   → Difference: {row['Difference']:+.1f} percentage points")
        
        # Interpret direction
        if row['Difference'] > 0:
            print(f"   → Theme more common when bird present")
        else:
            print(f"   → Theme more common when no bird")
else:
    print("No significant associations found.")

# %% ═════════ 6. Comparison Plot ═════════

# %%
# Show prevalence of outcomes with/without birds for significant associations
plot_binary_predictor_prevalence(
    results_df,
    'Bird present', 'No bird',
    'Theme prevalence by bird presence (top 15)',
    f'{output_dir}/03_prevalence_comparison.png',
    top_n=15,
    figsize=STANDARD_FIGSIZE
)


# %% ═════════ 7. Summary ═════════

# %%
print_summary_stats(results_df)

# Categorize by effect direction
sig_assocs = results_df[results_df['Significant']]
if len(sig_assocs) > 0:
    enriched = sig_assocs[sig_assocs['Difference'] > 0].sort_values('Cramers_V', ascending=False)
    depleted = sig_assocs[sig_assocs['Difference'] < 0].sort_values('Cramers_V', ascending=False)
    
    print(f"\nThemes more common WHEN BIRD PRESENT ({len(enriched)}):")
    for _, row in enriched.iterrows():
        print(f"  {row['Outcome']:30s}  {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
              f"(V={row['Cramers_V']:.3f}, q={row['p_fdr']:.3f})")
    
    print(f"\nThemes more common WHEN NO BIRD ({len(depleted)}):")
    for _, row in depleted.iterrows():
        print(f"  {row['Outcome']:30s}  {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
              f"(V={row['Cramers_V']:.3f}, q={row['p_fdr']:.3f})")

# %% ═════════ 8. Save ═════════

# %%
# Save summary image for PDF
save_summary_table_image(
    results_df,
    title="Bird Presence × Themes: Statistical Summary",
    output_path=f'{output_dir}/99_summary.png'
)

output_file = f'{output_dir}/chi_square_results.csv'
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nAnalyzed: Bird presence × Themes")
print(f"  Total interviews: {len(predictor_binary)}")
print(f"  With birds: {n_with_bird} ({n_with_bird/len(predictor_binary)*100:.1f}%)")
print(f"  Without birds: {n_without_bird} ({n_without_bird/len(predictor_binary)*100:.1f}%)")
print(f"Results: {output_file}")
print(f"Figures: {output_dir}/")

# %%
