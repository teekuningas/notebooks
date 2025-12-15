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
#
# Examines whether ESA WorldCover habitat types predict consolidated thematic codes from interviews.

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

# Input: consolidated themes from analyysi_koodit
THEMES_FILE = './output/analyysi_koodit/99699c4b/themes_143x452.csv'

output_dir = './output/esa_koodit'
os.makedirs(output_dir, exist_ok=True)

# %% ═════════ 1. Load and Prepare Data ═════════

# %%
esa_raw = pd.read_csv('./inputs/bird-metadata-refined/esa_habitat_presence_fi.csv')
esa_raw = esa_raw.set_index('rec_id').drop(columns=['lon', 'lat'])

themes_raw = pd.read_csv(THEMES_FILE, index_col=0)

common_ids = esa_raw.index.intersection(themes_raw.index)
predictor_binary = esa_raw.loc[common_ids].astype(int)
outcome_binary = (themes_raw.loc[common_ids] >= 0.5).astype(int)

# Filter outcomes: keep themes present in 10%-90% of interviews
prevalence = outcome_binary.mean()
themes_to_keep = prevalence[(prevalence >= 0.10) & (prevalence <= 0.90)].index
print(f"Filtering themes: {len(outcome_binary.columns)} -> {len(themes_to_keep)} (10%-90% prevalence)")
outcome_binary = outcome_binary[themes_to_keep]

print_data_summary(predictor_binary, outcome_binary, "ESA habitaatit", "Teemat")

# %% ═════════ 2. Predictor Overlap ═════════

# %%
predictor_counts = predictor_binary.sum(axis=1)

print("=" * 70)
print("HABITAATTIEN PÄÄLLEKKÄISYYSANALYYSI")
print("=" * 70)
print(f"Haastatteluja 0 habitaattia:  {(predictor_counts == 0).sum():3d} ({(predictor_counts == 0).mean()*100:.1f}%)")
print(f"Haastatteluja 1 habitaatti:   {(predictor_counts == 1).sum():3d} ({(predictor_counts == 1).mean()*100:.1f}%)")
print(f"Haastatteluja 2+ habitaattia: {(predictor_counts >= 2).sum():3d} ({(predictor_counts >= 2).mean()*100:.1f}%)")
print(f"\nKeskimäärin habitaatteja per haastattelu: {predictor_counts.mean():.2f}")

cooccurrence_matrix = calculate_cooccurrence_matrix(predictor_binary)

plot_cooccurrence_heatmap(
    cooccurrence_matrix,
    title=f'Habitaattien päällekkäisyys (n={len(predictor_binary)})',
    xlabel='Habitaatti', ylabel='Habitaatti',
    output_path=f'{output_dir}/01_predictor_overlap_counts.png',
    figsize=STANDARD_FIGSIZE
)

plot_cooccurrence_percentage_heatmap(
    cooccurrence_matrix,
    title='Habitaattien päällekkäisyys (%)',
    xlabel='Habitaatti', ylabel='Habitaatti',
    output_path=f'{output_dir}/02_predictor_overlap_percentage.png',
    figsize=STANDARD_FIGSIZE
)

# %% ═════════ 3. Chi-Square Tests ═════════

# %%
print("\n" + "=" * 70)
print("CHI-NELIÖTESTIT")
print("=" * 70)

results_df = run_chi_square_tests(predictor_binary, outcome_binary)

print(f"\nTulokset:")
print(f"  Merkitseviä (p < 0.05, korjaamaton): {(results_df['p_value'] < 0.05).sum()}")
print(f"  Merkitseviä (FDR q < 0.05):          {results_df['Significant'].sum()}")
print(f"  Keskisuuri+ efektikoko (V > 0.20):   {(results_df['Cramers_V'] > 0.2).sum()}")

print(f"\n20 vahvinta yhteyttä:")
print(results_df[['Outcome', 'Predictor', 'Chi2', 'p_fdr', 'Cramers_V', 'Difference']].head(20).to_string(index=False))

# %% ═════════ 4. Heatmap ═════════

# %%
plot_effect_size_heatmap(
    results_df,
    title='Habitaatin vaikutus teemaan, efektikoko',
    xlabel='Habitaatti',
    ylabel='Teema',
    output_path=f'{output_dir}/03_effect_size_significance.png',
    figsize=STANDARD_FIGSIZE,
    vmax=0.4
)

# %% ═════════ 5. Significant Associations ═════════

# %%
significant = results_df[results_df['Significant']].copy()

print("\n" + "=" * 70)
print(f"YKSITYISKOHDAT: {len(significant)} MERKITSEVÄÄ YHTEYTTÄ (FDR q < 0.05)")
print("=" * 70)

if len(significant) > 0:
    for i, (_, row) in enumerate(significant.iterrows(), 1):
        print(f"\n{i}. {row['Outcome']} × {row['Predictor']}")
        print(f"   Chi² = {row['Chi2']:.2f}, p = {row['p_value']:.2e}, FDR q = {row['p_fdr']:.2e}")
        print(f"   Cramér's V = {row['Cramers_V']:.3f}")
        print(f"   Kun {row['Predictor']}: {row['P(Outcome|Pred)']:.1f}%")
        print(f"   Muuten:                 {row['P(Outcome|~Pred)']:.1f}%")
        print(f"   → Erotus: {row['Difference']:+.1f} prosenttiyksikköä")
else:
    print("Ei merkitseviä yhteyksiä.")

# %% ═════════ 6. Bar Plot ═════════

# %%
plot_top_associations_barplot(
    results_df,
    title=f'Vahvimmat habitaatti-teema -yhteydet (V ≥ 0.1)',
    output_path=f'{output_dir}/04_top_associations.png',
    min_effect=0.1,
    figsize=STANDARD_FIGSIZE
)

# %% ═════════ 7. Predictor Profiles ═════════

# %%
print("\n" + "=" * 70)
print("HABITAATTIPROFIILIT")
print("=" * 70)

for predictor in predictor_binary.columns:
    n_pred = predictor_binary[predictor].sum()
    
    if n_pred < 10:
        print(f"\n{predictor}: Ohitettu (vain {n_pred} haastattelua)")
        continue
    
    pred_assocs = results_df[results_df['Predictor'] == predictor].copy()
    pred_assocs['Enrichment'] = pred_assocs['P(Outcome|Pred)'] / pred_assocs['P(Outcome|~Pred)']
    pred_assocs = pred_assocs.sort_values('Enrichment', ascending=False)
    
    enriched = pred_assocs[(pred_assocs['Significant']) & (pred_assocs['Enrichment'] > 1)].head(5)
    depleted = pred_assocs[(pred_assocs['Significant']) & (pred_assocs['Enrichment'] < 1)].tail(3)
    
    print(f"\n{predictor} (n = {n_pred} haastattelua):")
    
    if len(enriched) > 0:
        print("  Rikastetut teemat:")
        for _, row in enriched.iterrows():
            print(f"    • {row['Outcome']:30s} {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
                  f"({row['Enrichment']:.2f}×, q={row['p_fdr']:.3f})")
    
    if len(depleted) > 0:
        print("  Köyhdytetyt teemat:")
        for _, row in depleted.iterrows():
            print(f"    • {row['Outcome']:30s} {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
                  f"({row['Enrichment']:.2f}×, q={row['p_fdr']:.3f})")
    
    if len(enriched) == 0 and len(depleted) == 0:
        print("  Ei merkitseviä rikastumisia tai köyhtymisiä")

# %% ═════════ 8. Summary ═════════

# %%
print_summary_stats(results_df)

sig_assocs = results_df[results_df['Significant']]
if len(sig_assocs) > 0:
    print(f"\nUseimmin habitaatteihin liittyvät teemat:")
    outcome_counts = sig_assocs['Outcome'].value_counts().head(10)
    for outcome, count in outcome_counts.items():
        print(f"  {outcome:35s} {count} habitaattia")
    
    print(f"\nUseimmin teemoihin liittyvät habitaatit:")
    pred_counts = sig_assocs['Predictor'].value_counts().head(5)
    for pred, count in pred_counts.items():
        print(f"  {pred:25s} {count} teemaa")

# %% ═════════ 9. Save ═════════

# %%
output_file = f'{output_dir}/chi_square_results.csv'
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("ANALYYSI VALMIS")
print("=" * 70)
print(f"\nAnalysoitu: ESA habitaatit × Teemat")
print(f"Tulokset: {output_file}")
print(f"Kuviot: {output_dir}/")

# %%
