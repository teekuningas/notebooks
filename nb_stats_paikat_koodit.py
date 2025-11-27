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

# %% ═════════ Statistical Analysis: Paikat → Koodit ═════════
#
# Examines whether location categories (paikat) predict thematic codes (koodit).

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
    print_data_summary
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

output_dir = './output/paikat_koodit'
os.makedirs(output_dir, exist_ok=True)

# %% ═════════ 1. Load and Prepare Data ═════════

# %%
paikat_raw = pd.read_csv('./inputs/llm-thematic-data/paikat_10x452.csv', index_col=0)
koodit_raw = pd.read_csv('./inputs/llm-thematic-data/koodit_16x452.csv', index_col=0)
koodit_raw = koodit_raw.drop(columns=['Metsä'], errors='ignore')

predictor_binary = (paikat_raw == 1.0).astype(int)
outcome_binary = (koodit_raw == 1.0).astype(int)

print_data_summary(predictor_binary, outcome_binary, "Paikat", "Koodit")

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
    title=f'Paikkojen päällekkäisyys (n={len(predictor_binary)})',
    xlabel='Paikka', ylabel='Paikka',
    output_path=f'{output_dir}/01_predictor_overlap_counts.png',
    figsize=STANDARD_FIGSIZE
)

# Percentage heatmap
plot_cooccurrence_percentage_heatmap(
    cooccurrence_matrix,
    title='Paikkojen päällekkäisyys (%)',
    xlabel='Paikka', ylabel='Paikka',
    output_path=f'{output_dir}/02_predictor_overlap_percentage.png',
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

# %% ═════════ 4. Heatmap ═════════

# %%
plot_effect_size_heatmap(
    results_df,
    title='Paikan vaikutus koodiin, efektikoko',
    xlabel='Paikka',
    ylabel='Koodi',
    output_path=f'{output_dir}/03_effect_size_significance.png',
    figsize=STANDARD_FIGSIZE,
    vmax=0.4
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
    title=f'Vahvimmat paikka-koodi -yhteydet (V ≥ 0.1)',
    output_path=f'{output_dir}/04_top_associations.png',
    min_effect=0.1,
    figsize=STANDARD_FIGSIZE
)

# %% ═════════ 7. Predictor Profiles ═════════

# %%
print("\n" + "=" * 70)
print("PREDICTOR PROFILES")
print("=" * 70)

for predictor in predictor_binary.columns:
    n_pred = predictor_binary[predictor].sum()
    
    if n_pred < 10:
        print(f"\n{predictor}: Skipped (only {n_pred} interviews)")
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

# %% ═════════ 8. Summary ═════════

# %%
print_summary_stats(results_df)

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
output_file = f'{output_dir}/chi_square_results.csv'
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nAnalyzed: Paikat × Koodit")
print(f"Results: {output_file}")
print(f"Figures: {output_dir}/")

# %%
