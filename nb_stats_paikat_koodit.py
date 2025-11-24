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

# %% ═════════ Statistical Analysis: How Locations Affect Codes and Meanings ═════════
#
# This notebook examines whether location categories (paikat) have significant 
# associations with the presence of thematic codes (koodit) and emotional meanings 
# (merkitykset) in interview data.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, false_discovery_control
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for clean, professional plots
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# Standard figure size for all plots (A4 landscape proportions)
STANDARD_FIGSIZE = (12, 9)

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION: Choose which outcomes to analyze

import os
import sys

# Allow configuration via environment variable or command line argument
# Usage: ANALYSIS_MODE=merkitykset python nb_stats_paikat_koodit.py
#    or: python nb_stats_paikat_koodit.py merkitykset
# Options: 'both' (default), 'koodit', 'merkitykset'

# Determine analysis mode
if len(sys.argv) > 1:
    OUTCOME_TYPE = sys.argv[1]
elif 'ANALYSIS_MODE' in os.environ:
    OUTCOME_TYPE = os.environ['ANALYSIS_MODE']
else:
    OUTCOME_TYPE = 'both'

# Validate
if OUTCOME_TYPE not in ['both', 'koodit', 'merkitykset']:
    raise ValueError(f"Invalid OUTCOME_TYPE: {OUTCOME_TYPE}. Must be 'both', 'koodit', or 'merkitykset'")

print(f"═══ Running analysis with OUTCOME_TYPE = '{OUTCOME_TYPE}' ═══\n")

# Create output directory for figures
output_dir = f'./output/figures_paikat_{OUTCOME_TYPE}'
os.makedirs(output_dir, exist_ok=True)

# %% ═════════ 1. Load and Prepare Data ═════════

# %%
# Load data
locations_raw = pd.read_csv('./koodidata/correlation-data/paikat_10x452.csv', index_col=0)
codes_raw = pd.read_csv('./koodidata/correlation-data/koodit_16x452.csv', index_col=0)
meanings_raw = pd.read_csv('./koodidata/correlation-data/merkitykset_10x452.csv', index_col=0)

# Remove codes that duplicate location names
codes_raw = codes_raw.drop(columns=['Metsä'], errors='ignore')

# Select outcomes based on configuration
if OUTCOME_TYPE == 'both':
    outcomes_raw = pd.concat([codes_raw, meanings_raw], axis=1)
    outcome_label = 'koodit ja merkitykset'
elif OUTCOME_TYPE == 'koodit':
    outcomes_raw = codes_raw.copy()
    outcome_label = 'koodit'
elif OUTCOME_TYPE == 'merkitykset':
    outcomes_raw = meanings_raw.copy()
    outcome_label = 'merkitykset'
else:
    raise ValueError(f"Invalid OUTCOME_TYPE: {OUTCOME_TYPE}. Must be 'both', 'koodit', or 'merkitykset'")

print("=" * 70)
print(f"ANALYZING: {outcome_label.upper()}")
print("=" * 70)
print(f"Paikat (locations):   {locations_raw.shape[0]} interviews × {locations_raw.shape[1]} location types")
print(f"Outcomes analyzed:    {outcomes_raw.shape[0]} interviews × {outcomes_raw.shape[1]} variables")
print(f"Output directory:     {output_dir}")
print()

# Convert to binary: only 1.0 counts as "present"
locations_binary = (locations_raw == 1.0).astype(int)
outcomes_binary = (outcomes_raw == 1.0).astype(int)

# Create summary tables
locations_prevalence = (locations_binary.mean() * 100).sort_values(ascending=False)
outcomes_prevalence = (outcomes_binary.mean() * 100).sort_values(ascending=False)

print("Location Prevalence (%):")
print(locations_prevalence.to_string())
print()
print("Outcome Prevalence (%) - Top 15:")
print(outcomes_prevalence.head(15).to_string())

# %% ═════════ 2. Exploratory Analysis: Location Overlap ═════════

# %%
# Examine location overlap
location_counts = locations_binary.sum(axis=1)

print("\n" + "=" * 70)
print("LOCATION OVERLAP ANALYSIS")
print("=" * 70)
print(f"Interviews with 0 locations:  {(location_counts == 0).sum():3d} ({(location_counts == 0).mean()*100:.1f}%)")
print(f"Interviews with 1 location:   {(location_counts == 1).sum():3d} ({(location_counts == 1).mean()*100:.1f}%)")
print(f"Interviews with 2+ locations: {(location_counts >= 2).sum():3d} ({(location_counts >= 2).mean()*100:.1f}%)")
print(f"\nMean locations per interview: {location_counts.mean():.2f}")
print(f"Max locations per interview:  {location_counts.max()}")

# Co-occurrence matrix (showing overlaps between locations)
# Diagonal shows how many interviews have each location
cooccurrence_matrix = locations_binary.T @ locations_binary

# Figure 1a: Absolute counts
fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
sns.heatmap(cooccurrence_matrix, annot=True, fmt='d', cmap='YlGn', 
            cbar_kws={'label': 'Yhteiset esiintymät'}, ax=ax)
total_n = len(locations_binary)
ax.set_title(f'Paikkojen päällekkäisyys (n={total_n})', 
             fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('Paikka', fontsize=12, fontweight='bold')
ax.set_ylabel('Paikka', fontsize=12, fontweight='bold')
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
plt.savefig(f'{output_dir}/01_paallekkaisyys_maara.png', dpi=300)
plt.show()
plt.close()

# Figure 1b: Conditional probabilities (row-normalized percentages)
# Shows: "Of interviews mentioning location X (row), what % also mention Y (column)?"
cooccurrence_pct = cooccurrence_matrix.div(cooccurrence_matrix.values.diagonal(), axis=0) * 100

fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
sns.heatmap(cooccurrence_pct, annot=True, fmt='.0f', cmap='YlGn', 
            vmin=0, vmax=100, cbar_kws={'label': 'Osuus (%)'}, ax=ax)
ax.set_title('Paikkojen päällekkäisyys (%)', 
             fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('Paikka', fontsize=12, fontweight='bold')
ax.set_ylabel('Paikka', fontsize=12, fontweight='bold')
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
fig.text(0.5, 0.05,
             "Tulkinta: Kukin rivi näyttää, monessako prosentissa sen paikan haastatteluista esiintyy myös sarakkeen paikka.",
             ha='center', fontsize=9, style='italic', wrap=True)
plt.savefig(f'{output_dir}/02_paallekkaisyys_prosentti.png', dpi=300)
plt.show()
plt.close()

# %% ═════════ 3. Chi-Square Tests of Independence ═════════

# %%
def cramers_v(chi2, n):
    """Calculate Cramér's V effect size."""
    return np.sqrt(chi2 / n)

# Perform chi-square tests for all pairs
print("\n" + "=" * 70)
print("RUNNING CHI-SQUARE TESTS")
print("=" * 70)
n_tests = len(outcomes_binary.columns) * len(locations_binary.columns)
print(f"Testing {len(outcomes_binary.columns)} outcomes × {len(locations_binary.columns)} locations = {n_tests} pairs...")

results = []
for outcome in outcomes_binary.columns:
    for location in locations_binary.columns:
        # Create 2×2 contingency table
        contingency = pd.crosstab(outcomes_binary[outcome], locations_binary[location])
        
        # Ensure complete 2×2 structure
        for val in [0, 1]:
            if val not in contingency.index:
                contingency.loc[val] = 0
            if val not in contingency.columns:
                contingency[val] = 0
        contingency = contingency.loc[[0, 1], [0, 1]]
        
        # Run chi-square test
        chi2, p_value, _, _ = chi2_contingency(contingency)
        
        # Calculate effect size and conditional probabilities
        n = len(outcomes_binary)
        cramers = cramers_v(chi2, n)
        
        loc_mask = locations_binary[location].astype(bool)
        if loc_mask.sum() > 0:
            p_outcome_given_loc = outcomes_binary[outcome][loc_mask].mean()
            p_outcome_given_not_loc = outcomes_binary[outcome][~loc_mask].mean()
            diff = p_outcome_given_loc - p_outcome_given_not_loc
        else:
            p_outcome_given_loc = p_outcome_given_not_loc = diff = np.nan
        
        results.append({
            'Outcome': outcome,
            'Location': location,
            'Chi2': chi2,
            'p_value': p_value,
            'Cramers_V': cramers,
            'P(Outcome|Loc)': p_outcome_given_loc * 100,
            'P(Outcome|~Loc)': p_outcome_given_not_loc * 100,
            'Difference': diff * 100
        })

chi_square_df = pd.DataFrame(results)

# Apply FDR correction (Benjamini-Hochberg)
chi_square_df['p_fdr'] = false_discovery_control(chi_square_df['p_value'], method='bh')
chi_square_df['Significant'] = chi_square_df['p_fdr'] < 0.05
chi_square_df = chi_square_df.sort_values('p_value')

# Display summary
n_sig_uncorr = (chi_square_df['p_value'] < 0.05).sum()
n_sig_fdr = chi_square_df['Significant'].sum()
n_medium_effect = (chi_square_df['Cramers_V'] > 0.2).sum()

print(f"\nResults:")
print(f"  Significant (p < 0.05, uncorrected): {n_sig_uncorr}")
print(f"  Significant (FDR q < 0.05):          {n_sig_fdr}")
print(f"  Medium+ effect size (V > 0.20):      {n_medium_effect}")

# Show top associations
print(f"\nTop 20 associations:")
display_cols = ['Outcome', 'Location', 'Chi2', 'p_fdr', 'Cramers_V', 'Difference']
print(chi_square_df[display_cols].head(20).to_string(index=False))

# %% ═════════ 4. Heatmap Visualization ═════════

# %%
# Prepare data for heatmap
pivot_pval = chi_square_df.pivot(index='Outcome', columns='Location', values='p_fdr')
pivot_cramers = chi_square_df.pivot(index='Outcome', columns='Location', values='Cramers_V')

# Get dimensions for borders
n_outcomes = len(pivot_pval.index)
n_locations = len(pivot_pval.columns)

# Create annotations with significance markers
annot_matrix = np.empty(pivot_cramers.shape, dtype=object)
for i in range(pivot_cramers.shape[0]):
    for j in range(pivot_cramers.shape[1]):
        v = pivot_cramers.iloc[i, j]
        p = pivot_pval.iloc[i, j]
        
        # Add significance markers
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''
        
        annot_matrix[i, j] = f'{v:.2f}{sig}'

# Combined figure: Effect Size with Significance
fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
sns.heatmap(pivot_cramers, annot=annot_matrix, fmt='', cmap='YlGn',
            vmin=0, vmax=0.4, ax=ax,
            cbar_kws={'label': "Cramérin V"},
            yticklabels=True, annot_kws={'fontsize': 8})
ax.axhline(y=0, color='k', linewidth=2)
ax.axhline(y=n_outcomes, color='k', linewidth=2)
ax.axvline(x=0, color='k', linewidth=2)
ax.axvline(x=n_locations, color='k', linewidth=2)
ax.set_title('Paikan vaikutus muuttujaan, efektikoko', 
              fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('Paikka', fontsize=12, fontweight='bold')
ax.set_ylabel('Muuttuja', fontsize=11, fontweight='bold')
# Ensure y-tick labels are horizontal
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=9)
# Rotate x-tick labels for consistency
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, va='top', ha='center')
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
plt.savefig(f'{output_dir}/03_vaikutuskoko_merkitsevyys.png', dpi=300)
plt.show()
plt.close()

# %% ═════════ 5. Detailed Results: Significant Associations ═════════

# %%
significant = chi_square_df[chi_square_df['Significant']].copy()

print("\n" + "=" * 70)
print(f"DETAILED RESULTS: {len(significant)} SIGNIFICANT ASSOCIATIONS (FDR q < 0.05)")
print("=" * 70)

if len(significant) > 0:
    for i, (_, row) in enumerate(significant.iterrows(), 1):
        print(f"\n{i}. {row['Outcome']} × {row['Location']}")
        print(f"   Chi² = {row['Chi2']:.2f}, p = {row['p_value']:.2e}, FDR q = {row['p_fdr']:.2e}")
        print(f"   Cramér's V = {row['Cramers_V']:.3f}")
        print(f"   When {row['Location']}: {row['P(Outcome|Loc)']:.1f}%")
        print(f"   Otherwise:            {row['P(Outcome|~Loc)']:.1f}%")
        print(f"   → Difference: {row['Difference']:+.1f} percentage points")
else:
    print("No significant associations found.")

# %% ═════════ 6. Top Associations Bar Plot ═════════

# %%
# Filter for effect size >= 0.1
strong_effects = chi_square_df[chi_square_df['Cramers_V'] >= 0.1].copy()
strong_effects = strong_effects.sort_values('p_fdr')

if len(strong_effects) > 0:
    # Create figure with more space
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)
    
    # Create labels
    labels = [f"{row['Outcome']} × {row['Location']}" 
              for _, row in strong_effects.iterrows()]
    
    # Create diverging colormap based on direction: purple for negative, green for positive
    from matplotlib.colors import TwoSlopeNorm
    differences = strong_effects['Difference'].values
    
    # Symmetrical limits based on rounding max absolute value up to the next 10
    abs_max = np.abs(differences).max()
    sym_limit = np.ceil(abs_max / 10) * 10 if abs_max > 0 else 10
    vmin, vmax = -sym_limit, sym_limit
    
    # Handle edge case where all differences have the same sign
    vmin_orig, vmax_orig = differences.min(), differences.max()
    if vmin_orig >= 0:  # All positive
        norm = plt.Normalize(vmin=0, vmax=vmax) # Use symmetrical max
        colors = plt.cm.Greens(norm(differences))
    elif vmax_orig <= 0:  # All negative
        norm = plt.Normalize(vmin=vmin, vmax=0) # Use symmetrical min
        colors = plt.cm.Purples(norm(differences))
    else:  # Mixed signs
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) # Use symmetrical limits
        colors = plt.cm.PRGn(norm(differences))
    
    # Plot horizontal bars
    y_pos = np.arange(len(strong_effects))
    bars = ax.barh(y_pos, -np.log10(strong_effects['p_fdr']), color=colors, edgecolor='black', linewidth=0.5)
    
    # Add difference annotations
    for i, (_, row) in enumerate(strong_effects.iterrows()):
        diff_text = f"{row['Difference']:+.1f}%"
        ax.text(-np.log10(row['p_fdr']) + 0.1, i, diff_text, 
               va='center', fontsize=9, fontweight='bold')
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('-log₁₀(FDR q-arvo)', fontsize=12, fontweight='bold')
    ax.set_title(f'Vahvimmat paikka-muuttuja -yhteydet (V ≥ 0.1)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(1.3, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='q = 0.05')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15) # Extend x-axis to prevent text overlap
    ax.grid(axis='x', alpha=0.3)
    
    # Add colorbar for direction
    sm = plt.cm.ScalarMappable(cmap=plt.cm.PRGn, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, aspect=30, shrink=0.8)
    cbar.set_label('Erotus (%)', rotation=270, labelpad=15, fontsize=10)
    
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
    plt.savefig(f'{output_dir}/04_vahvimmat_yhteydet.png', dpi=300)
    plt.show()
    plt.close()
else:
    print("No associations with V >= 0.1 found.")

# %% ═════════ 7. Location Profiles ═════════

# %%
print("\n" + "=" * 70)
print("LOCATION PROFILES")
print("=" * 70)

for location in locations_binary.columns:
    n_loc = locations_binary[location].sum()
    
    # Skip very rare locations
    if n_loc < 10:
        print(f"\n{location}: Skipped (only {n_loc} interviews)")
        continue
    
    # Get all associations for this location
    loc_assocs = chi_square_df[chi_square_df['Location'] == location].copy()
    loc_assocs['Enrichment'] = loc_assocs['P(Outcome|Loc)'] / loc_assocs['P(Outcome|~Loc)']
    loc_assocs = loc_assocs.sort_values('Enrichment', ascending=False)
    
    # Find significant enrichments and depletions
    enriched = loc_assocs[(loc_assocs['Significant']) & (loc_assocs['Enrichment'] > 1)].head(5)
    depleted = loc_assocs[(loc_assocs['Significant']) & (loc_assocs['Enrichment'] < 1)].tail(3)
    
    print(f"\n{location} (n = {n_loc} interviews):")
    
    if len(enriched) > 0:
        print("  Enriched outcomes:")
        for _, row in enriched.iterrows():
            print(f"    • {row['Outcome']:30s} {row['P(Outcome|Loc)']:5.1f}% vs {row['P(Outcome|~Loc)']:5.1f}%  " +
                  f"({row['Enrichment']:.2f}×, q={row['p_fdr']:.3f})")
    
    if len(depleted) > 0:
        print("  Depleted outcomes:")
        for _, row in depleted.iterrows():
            print(f"    • {row['Outcome']:30s} {row['P(Outcome|Loc)']:5.1f}% vs {row['P(Outcome|~Loc)']:5.1f}%  " +
                  f"({row['Enrichment']:.2f}×, q={row['p_fdr']:.3f})")
    
    if len(enriched) == 0 and len(depleted) == 0:
        print("  No significant enrichments or depletions")

# %% ═════════ 8. Summary Statistics ═════════

# %%
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

# Overall summary
print(f"\nTotal tests performed:              {len(chi_square_df)}")
print(f"Significant (uncorrected p<0.05):   {(chi_square_df['p_value'] < 0.05).sum()}")
print(f"Significant (FDR q<0.05):           {chi_square_df['Significant'].sum()}")

# Effect size distribution
print(f"\nEffect size distribution:")
print(f"  Large (V > 0.50):        {(chi_square_df['Cramers_V'] > 0.5).sum()}")
print(f"  Medium (0.30 < V < 0.50): {((chi_square_df['Cramers_V'] > 0.3) & (chi_square_df['Cramers_V'] <= 0.5)).sum()}")
print(f"  Small (0.10 < V < 0.30):  {((chi_square_df['Cramers_V'] > 0.1) & (chi_square_df['Cramers_V'] <= 0.3)).sum()}")
print(f"  Negligible (V < 0.10):    {(chi_square_df['Cramers_V'] <= 0.1).sum()}")

# Practical significance
sig_assocs = chi_square_df[chi_square_df['Significant']]
if len(sig_assocs) > 0:
    print(f"\nAmong significant associations:")
    print(f"  Mean effect size (Cramér's V):    {sig_assocs['Cramers_V'].mean():.3f}")
    print(f"  Mean percentage point difference:  {sig_assocs['Difference'].abs().mean():.1f} pp")
    print(f"  Largest difference:                {sig_assocs['Difference'].abs().max():.1f} pp")

# Most affected outcomes and locations
print(f"\nOutcomes most often significantly associated with locations:")
outcome_counts = sig_assocs['Outcome'].value_counts().head(10)
for outcome, count in outcome_counts.items():
    print(f"  {outcome:35s} {count} location(s)")

print(f"\nLocations most often significantly associated with codes:")
loc_counts = sig_assocs['Location'].value_counts().head(5)
for loc, count in loc_counts.items():
    print(f"  {loc:25s} {count} code(s)")

# %%
# Save complete results
output_file = f'./koodidata/correlation-data/chi_square_results_{OUTCOME_TYPE}.csv'
chi_square_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nAnalyzed: {outcome_label}")
print(f"Results saved to: {output_file}")
print(f"\nFigures saved to: {output_dir}/")
# %%
