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

# %% ═════════ Statistical Analysis: ESA Habitats → Themes ═════════
# Examines whether ESA WorldCover habitat types predict consolidated thematic codes.

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
FOOTNOTE_METHOD = "Data: ESA WorldCover 2020 10m. Method: Presence in 9-point grid (100m radius)."

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

THEMES_FILE = './output/analyysi_koodit/88d43208/themes_98x452.csv'
MIN_THEME_PREVALENCE = 0.20
MAX_THEME_PREVALENCE = 0.80

output_dir = './output/esa_koodit'
os.makedirs(output_dir, exist_ok=True)

# %% ═════════ 1. Load and Prepare Data ═════════

# %%
esa_raw = pd.read_csv('./inputs/bird-metadata-refined/esa_habitat_presence_en.csv')
esa_raw = esa_raw.set_index('rec_id').drop(columns=['lon', 'lat'])

themes_raw = pd.read_csv(THEMES_FILE, index_col=0)
themes_raw.columns = themes_raw.columns.str.capitalize()

common_ids = esa_raw.index.intersection(themes_raw.index)
predictor_binary = esa_raw.loc[common_ids].astype(int)
outcome_binary = (themes_raw.loc[common_ids] >= 0.5).astype(int)

prevalence = outcome_binary.mean()
themes_to_keep = prevalence[(prevalence >= MIN_THEME_PREVALENCE) & (prevalence <= MAX_THEME_PREVALENCE)].index
print(f"Filtering themes: {len(outcome_binary.columns)} -> {len(themes_to_keep)} ({MIN_THEME_PREVALENCE*100:.0f}%-{MAX_THEME_PREVALENCE*100:.0f}% prevalence)")
outcome_binary = outcome_binary[themes_to_keep]

# Filter habitats (min 10 occurrences)
min_occurrences = 10
pred_counts = predictor_binary.sum()
habitats_to_keep = pred_counts[pred_counts >= min_occurrences].index
print(f"Filtering habitats: {len(predictor_binary.columns)} -> {len(habitats_to_keep)} (min {min_occurrences} occurrences)")
predictor_binary = predictor_binary[habitats_to_keep]

print_data_summary(predictor_binary, outcome_binary, "ESA Habitats", "Themes")

# %% ═════════ 2. Predictor Overlap ═════════

# %%
predictor_counts = predictor_binary.sum(axis=1)

print("=" * 70)
print("HABITAT OVERLAP ANALYSIS")
print("=" * 70)
print(f"Interviews with 0 habitats:  {(predictor_counts == 0).sum():3d} ({(predictor_counts == 0).mean()*100:.1f}%)")
print(f"Interviews with 1 habitat:   {(predictor_counts == 1).sum():3d} ({(predictor_counts == 1).mean()*100:.1f}%)")
print(f"Interviews with 2+ habitats: {(predictor_counts >= 2).sum():3d} ({(predictor_counts >= 2).mean()*100:.1f}%)")
print(f"\nMean habitats per interview: {predictor_counts.mean():.2f}")

cooccurrence_matrix = calculate_cooccurrence_matrix(predictor_binary)

plot_cooccurrence_heatmap(
    cooccurrence_matrix,
    title=f'Habitat co-occurrence (n={len(predictor_binary)})',
    xlabel='Habitat', ylabel='Habitat',
    output_path=f'{output_dir}/01_predictor_overlap_counts.png',
    figsize=STANDARD_FIGSIZE,
    footnote=FOOTNOTE_METHOD
)

plot_cooccurrence_percentage_heatmap(
    cooccurrence_matrix,
    title='Habitat co-occurrence (%)',
    xlabel='Habitat', ylabel='Habitat',
    output_path=f'{output_dir}/02_predictor_overlap_percentage.png',
    figsize=STANDARD_FIGSIZE,
    footnote=FOOTNOTE_METHOD
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

# %% ═════════ 4. Heatmap ═════════

# %%
plot_effect_size_heatmap(
    results_df,
    title='Effect of habitat on theme (effect size)',
    xlabel='Habitat',
    ylabel='Theme',
    output_path=f'{output_dir}/03_effect_size_significance.png',
    figsize=STANDARD_FIGSIZE,
    vmax=0.4,
    footnote=f"{FOOTNOTE_METHOD} Significance: * q<0.05, ** q<0.01, *** q<0.001 (FDR)"
)

# %% ═════════ 5. Significant Associations ═════════

# %%
significant = results_df[results_df['Significant']].copy()

print("\n" + "=" * 70)
print(f"DETAILS: {len(significant)} SIGNIFICANT ASSOCIATIONS (FDR q < 0.05)")
print("=" * 70)

if len(significant) > 0:
    for i, (_, row) in enumerate(significant.iterrows(), 1):
        print(f"\n{i}. {row['Outcome']} × {row['Predictor']}")
        print(f"   Chi² = {row['Chi2']:.2f}, p = {row['p_value']:.2e}, FDR q = {row['p_fdr']:.2e}")
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
    title=f'Strongest habitat-theme associations (V ≥ 0.1)',
    output_path=f'{output_dir}/04_top_associations.png',
    min_effect=0.1,
    figsize=STANDARD_FIGSIZE,
    footnote=f"{FOOTNOTE_METHOD} Significance: * q<0.05, ** q<0.01, *** q<0.001 (FDR)"
)

# %% ═════════ 7. Predictor Profiles ═════════

# %%
print("\n" + "=" * 70)
print("HABITAT PROFILES")
print("=" * 70)

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
        print("  Enriched themes:")
        for _, row in enriched.iterrows():
            print(f"    • {row['Outcome']:30s} {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
                  f"({row['Enrichment']:.2f}×, q={row['p_fdr']:.3f})")
    
    if len(depleted) > 0:
        print("  Depleted themes:")
        for _, row in depleted.iterrows():
            print(f"    • {row['Outcome']:30s} {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
                  f"({row['Enrichment']:.2f}×, q={row['p_fdr']:.3f})")

# %% ═════════ 8. Summary ═════════

# %%
print_summary_stats(results_df)

# %% ═════════ 9. Save ═════════

# %%
save_summary_table_image(
    results_df,
    title="ESA Habitats × Themes: Statistical Summary",
    output_path=f'{output_dir}/99_summary.png'
)

output_file = f'{output_dir}/chi_square_results.csv'
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nAnalyzed: ESA Habitats × Themes")
print(f"Results: {output_file}")
print(f"Figures: {output_dir}/")

# %%
