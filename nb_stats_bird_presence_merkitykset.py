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

# %% ═════════ Statistical Analysis: Bird Presence → Merkitykset ═════════
#
# Examines whether bird presence (any bird vs no birds) predicts meanings (merkitykset).

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
    plot_effect_size_heatmap,
    plot_top_associations_barplot,
    print_summary_stats,
    print_data_summary,
    plot_binary_predictor_distribution,
    plot_binary_predictor_effects,
    plot_binary_predictor_prevalence
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

output_dir = './output/bird_presence_merkitykset'
os.makedirs(output_dir, exist_ok=True)

# %% ═════════ 1. Load and Prepare Data ═════════

# %%
# Load bird groups and create binary "any bird" variable
birds_raw = pd.read_csv('./inputs/bird-metadata-refined/bird_groups_finnish.csv')
birds_raw = birds_raw.set_index('rec_id').drop(columns=['lon', 'lat'])

# Create single binary predictor: bird present (1) or not (0)
bird_presence = (birds_raw.sum(axis=1) > 0).astype(int).to_frame(name='Lintu')

# Load outcomes
merkitykset_raw = pd.read_csv('./inputs/llm-thematic-data/merkitykset_10x452.csv', index_col=0)

# Align by rec_id
common_ids = bird_presence.index.intersection(merkitykset_raw.index)
predictor_binary = bird_presence.loc[common_ids]
outcome_binary = (merkitykset_raw.loc[common_ids] == 1.0).astype(int)

print_data_summary(predictor_binary, outcome_binary, "Linnun läsnäolo", "Merkitykset")

# %% ═════════ 2. Predictor Distribution ═════════

# %%
n_with_bird = predictor_binary['Lintu'].sum()
n_without_bird = len(predictor_binary) - n_with_bird

print("=" * 70)
print("LINNUN LÄSNÄOLO")
print("=" * 70)
print(f"Haastatteluja joissa lintu:     {n_with_bird:3d} ({n_with_bird/len(predictor_binary)*100:.1f}%)")
print(f"Haastatteluja joissa ei lintua:  {n_without_bird:3d} ({n_without_bird/len(predictor_binary)*100:.1f}%)")

# Visualize distribution
plot_binary_predictor_distribution(
    predictor_binary, 'Lintu', 
    'Ei lintua', 'Lintu läsnä',
    'Linnun läsnäolo havainnoissa',
    f'{output_dir}/01_bird_presence_distribution.png',
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

# %% ═════════ 4. Effect Size Visualization ═════════

# %%
# Since we only have one predictor, create a custom bar plot instead of heatmap
plot_binary_predictor_effects(
    results_df,
    'Linnun läsnäolon vaikutus merkityksiin (merkitsevät)',
    f'{output_dir}/02_effect_sizes.png',
    figsize=STANDARD_FIGSIZE
)

# %% ═════════ 5. Significant Associations ═════════

# %%
significant = results_df[results_df['Significant']].copy()

print("\n" + "=" * 70)
print(f"YKSITYISKOHDAT: {len(significant)} MERKITSEVÄÄ YHTEYTTÄ (FDR q < 0.05)")
print("=" * 70)

if len(significant) > 0:
    for i, (_, row) in enumerate(significant.iterrows(), 1):
        print(f"\n{i}. {row['Outcome']}")
        print(f"   Chi² = {row['Chi2']:.2f}, p = {row['p_value']:.2e}, FDR q = {row['p_fdr']:.2e}")
        print(f"   Cramér's V = {row['Cramers_V']:.3f}")
        print(f"   Kun lintu läsnä: {row['P(Outcome|Pred)']:.1f}%")
        print(f"   Kun ei lintua:   {row['P(Outcome|~Pred)']:.1f}%")
        print(f"   → Erotus: {row['Difference']:+.1f} prosenttiyksikköä")
        
        # Interpret direction
        if row['Difference'] > 0:
            print(f"   → Merkitys yleisempi kun lintu läsnä")
        else:
            print(f"   → Merkitys yleisempi kun ei lintua")
else:
    print("Ei merkitseviä yhteyksiä.")

# %% ═════════ 6. Comparison Plot ═════════

# %%
# Show prevalence of outcomes with/without birds for significant associations
plot_binary_predictor_prevalence(
    results_df,
    'Lintu läsnä', 'Ei lintua',
    'Merkitysten esiintyvyys linnun läsnäolon mukaan (top 15)',
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
    
    print(f"\nMerkitykset jotka ovat yleisempiä KUN LINTU LÄSNÄ ({len(enriched)} kpl):")
    for _, row in enriched.iterrows():
        print(f"  {row['Outcome']:30s}  {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
              f"(V={row['Cramers_V']:.3f}, q={row['p_fdr']:.3f})")
    
    print(f"\nMerkitykset jotka ovat yleisempiä KUN EI LINTUA ({len(depleted)} kpl):")
    for _, row in depleted.iterrows():
        print(f"  {row['Outcome']:30s}  {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
              f"(V={row['Cramers_V']:.3f}, q={row['p_fdr']:.3f})")

# %% ═════════ 8. Save ═════════

# %%
output_file = f'{output_dir}/chi_square_results.csv'
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("ANALYYSI VALMIS")
print("=" * 70)
print(f"\nAnalysoitu: Linnun läsnäolo × Merkitykset")
print(f"  Haastatteluja yhteensä: {len(predictor_binary)}")
print(f"  Lintu läsnä: {n_with_bird} ({n_with_bird/len(predictor_binary)*100:.1f}%)")
print(f"  Ei lintua: {n_without_bird} ({n_without_bird/len(predictor_binary)*100:.1f}%)")
print(f"\nTulokset: {output_file}")
print(f"Kuviot: {output_dir}/")

# %%
