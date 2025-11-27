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

output_dir = './output/bird_presence_koodit'
os.makedirs(output_dir, exist_ok=True)

# %% ═════════ 1. Load and Prepare Data ═════════

# %%
# Load bird groups and create binary "any bird" variable
birds_raw = pd.read_csv('./inputs/bird-metadata-refined/bird_groups_finnish.csv')
birds_raw = birds_raw.set_index('rec_id').drop(columns=['lon', 'lat'])

# Create single binary predictor: bird present (1) or not (0)
bird_presence = (birds_raw.sum(axis=1) > 0).astype(int).to_frame(name='Lintu')

# Load outcomes
koodit_raw = pd.read_csv('./inputs/llm-thematic-data/koodit_16x452.csv', index_col=0)

# Align by rec_id
common_ids = bird_presence.index.intersection(koodit_raw.index)
predictor_binary = bird_presence.loc[common_ids]
outcome_binary = (koodit_raw.loc[common_ids] == 1.0).astype(int)

print_data_summary(predictor_binary, outcome_binary, "Linnun läsnäolo", "Koodit")

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
fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
counts = predictor_binary['Lintu'].value_counts().sort_index()
bars = ax.bar(['Ei lintua', 'Lintu läsnä'], [counts[0], counts[1]], 
              color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=1.5)

# Add counts on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(predictor_binary)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Haastattelujen määrä', fontsize=12, fontweight='bold')
ax.set_title('Linnun läsnäolo havainnoissa', fontsize=13, fontweight='bold', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.1)
plt.savefig(f'{output_dir}/01_bird_presence_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

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
significant = results_df[results_df['Significant']].copy()
significant = significant.sort_values('Cramers_V', ascending=True)

if len(significant) > 0:
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
    
    y_pos = np.arange(len(significant))
    colors = ['#2ecc71' if diff > 0 else '#e74c3c' 
              for diff in significant['Difference']]
    
    bars = ax.barh(y_pos, significant['Cramers_V'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add significance stars
    for i, (_, row) in enumerate(significant.iterrows()):
        stars = '***' if row['p_fdr'] < 0.001 else '**' if row['p_fdr'] < 0.01 else '*'
        ax.text(row['Cramers_V'] + 0.01, i, stars, va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(significant['Outcome'])
    ax.set_xlabel("Cramér's V (efektikoko)", fontsize=12, fontweight='bold')
    ax.set_title('Linnun läsnäolon vaikutus koodeihin (merkitsevät)', fontsize=13, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.subplots_adjust(left=0.3, right=0.95, top=0.92, bottom=0.08)
    plt.savefig(f'{output_dir}/02_effect_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
else:
    print("Ei merkitseviä yhteyksiä visualisoitavaksi.")

# %% ═════════ 5. Significant Associations ═════════

# %%
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
            print(f"   → Koodi yleisempi kun lintu läsnä")
        else:
            print(f"   → Koodi yleisempi kun ei lintua")
else:
    print("Ei merkitseviä yhteyksiä.")

# %% ═════════ 6. Comparison Plot ═════════

# %%
# Show prevalence of outcomes with/without birds for significant associations
if len(significant) > 0:
    top_n = min(15, len(significant))
    top_sig = significant.nlargest(top_n, 'Cramers_V')
    
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
    
    y_pos = np.arange(len(top_sig))
    width = 0.35
    
    with_bird = top_sig['P(Outcome|Pred)'].values
    without_bird = top_sig['P(Outcome|~Pred)'].values
    
    bars1 = ax.barh(y_pos - width/2, with_bird, width, label='Lintu läsnä', 
                    color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2 = ax.barh(y_pos + width/2, without_bird, width, label='Ei lintua',
                    color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_sig['Outcome'])
    ax.set_xlabel('Esiintyvyys (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Koodien esiintyvyys linnun läsnäolon mukaan (top {top_n})', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, fancybox=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(left=0.3, right=0.95, top=0.92, bottom=0.08)
    plt.savefig(f'{output_dir}/03_prevalence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %% ═════════ 7. Summary ═════════

# %%
print_summary_stats(results_df)

# Categorize by effect direction
sig_assocs = results_df[results_df['Significant']]
if len(sig_assocs) > 0:
    enriched = sig_assocs[sig_assocs['Difference'] > 0].sort_values('Cramers_V', ascending=False)
    depleted = sig_assocs[sig_assocs['Difference'] < 0].sort_values('Cramers_V', ascending=False)
    
    print(f"\nKoodit jotka ovat yleisempiä KUN LINTU LÄSNÄ ({len(enriched)} kpl):")
    for _, row in enriched.iterrows():
        print(f"  {row['Outcome']:30s}  {row['P(Outcome|Pred)']:5.1f}% vs {row['P(Outcome|~Pred)']:5.1f}%  " +
              f"(V={row['Cramers_V']:.3f}, q={row['p_fdr']:.3f})")
    
    print(f"\nKoodit jotka ovat yleisempiä KUN EI LINTUA ({len(depleted)} kpl):")
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
print(f"\nAnalysoitu: Linnun läsnäolo × Koodit")
print(f"  Haastatteluja yhteensä: {len(predictor_binary)}")
print(f"  Lintu läsnä: {n_with_bird} ({n_with_bird/len(predictor_binary)*100:.1f}%)")
print(f"  Ei lintua: {n_without_bird} ({n_without_bird/len(predictor_binary)*100:.1f}%)")
print(f"\nTulokset: {output_file}")
print(f"Kuviot: {output_dir}/")

# %%
