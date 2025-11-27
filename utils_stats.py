"""
Statistical utility functions for association analysis.

Pure, composable functions for chi-square tests, effect sizes, and visualizations.
Used by nb_stats_*.py notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, false_discovery_control
from matplotlib.colors import TwoSlopeNorm


# ═══════════════════════════════════════════════════════════════════════════
# Core Statistical Functions
# ═══════════════════════════════════════════════════════════════════════════

def cramers_v(chi2, n):
    """
    Calculate Cramér's V effect size from chi-square statistic.
    
    Args:
        chi2: Chi-square test statistic
        n: Sample size
        
    Returns:
        float: Cramér's V (0-1 scale, where 0=no association, 1=perfect)
    """
    return np.sqrt(chi2 / n)


def run_chi_square_tests(predictors, outcomes):
    """
    Run chi-square tests for all predictor-outcome pairs.
    
    Args:
        predictors: Binary DataFrame (rec_id × predictor variables)
        outcomes: Binary DataFrame (rec_id × outcome variables)
        
    Returns:
        DataFrame with columns: Predictor, Outcome, Chi2, p_value, Cramers_V,
                                P(Outcome|Pred), P(Outcome|~Pred), Difference
    """
    results = []
    n = len(predictors)
    
    for outcome in outcomes.columns:
        for predictor in predictors.columns:
            # Create 2×2 contingency table
            contingency = pd.crosstab(outcomes[outcome], predictors[predictor])
            
            # Ensure complete 2×2 structure
            for val in [0, 1]:
                if val not in contingency.index:
                    contingency.loc[val] = 0
                if val not in contingency.columns:
                    contingency[val] = 0
            contingency = contingency.loc[[0, 1], [0, 1]]
            
            # Run chi-square test
            chi2, p_value, _, _ = chi2_contingency(contingency)
            
            # Calculate effect size
            cramers = cramers_v(chi2, n)
            
            # Calculate conditional probabilities
            pred_mask = predictors[predictor].astype(bool)
            if pred_mask.sum() > 0:
                p_outcome_given_pred = outcomes[outcome][pred_mask].mean()
                p_outcome_given_not_pred = outcomes[outcome][~pred_mask].mean()
                diff = p_outcome_given_pred - p_outcome_given_not_pred
            else:
                p_outcome_given_pred = p_outcome_given_not_pred = diff = np.nan
            
            results.append({
                'Outcome': outcome,
                'Predictor': predictor,
                'Chi2': chi2,
                'p_value': p_value,
                'Cramers_V': cramers,
                'P(Outcome|Pred)': p_outcome_given_pred * 100,
                'P(Outcome|~Pred)': p_outcome_given_not_pred * 100,
                'Difference': diff * 100
            })
    
    df = pd.DataFrame(results)
    
    # Apply FDR correction
    df['p_fdr'] = false_discovery_control(df['p_value'], method='bh')
    df['Significant'] = df['p_fdr'] < 0.05
    df = df.sort_values('p_value')
    
    return df


def calculate_cooccurrence_matrix(binary_df):
    """
    Calculate co-occurrence matrix for binary features.
    
    Args:
        binary_df: Binary DataFrame (rec_id × features)
        
    Returns:
        DataFrame: Feature × Feature co-occurrence counts
    """
    return binary_df.T @ binary_df


# ═══════════════════════════════════════════════════════════════════════════
# Annotation & Display Helpers
# ═══════════════════════════════════════════════════════════════════════════

def create_significance_annotations(cramers_matrix, pval_matrix):
    """
    Create annotation matrix with effect sizes and significance stars.
    
    Args:
        cramers_matrix: DataFrame of Cramér's V values
        pval_matrix: DataFrame of FDR-corrected p-values
        
    Returns:
        numpy array of strings like "0.25***"
    """
    annot_matrix = np.empty(cramers_matrix.shape, dtype=object)
    
    for i in range(cramers_matrix.shape[0]):
        for j in range(cramers_matrix.shape[1]):
            v = cramers_matrix.iloc[i, j]
            p = pval_matrix.iloc[i, j]
            
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
    
    return annot_matrix


def print_summary_stats(results_df):
    """
    Print summary statistics from chi-square results.
    
    Args:
        results_df: DataFrame from run_chi_square_tests()
    """
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Overall summary
    print(f"\nTotal tests performed:              {len(results_df)}")
    print(f"Significant (uncorrected p<0.05):   {(results_df['p_value'] < 0.05).sum()}")
    print(f"Significant (FDR q<0.05):           {results_df['Significant'].sum()}")
    
    # Effect size distribution
    print(f"\nEffect size distribution:")
    print(f"  Large (V > 0.50):         {(results_df['Cramers_V'] > 0.5).sum()}")
    print(f"  Medium (0.30 < V < 0.50): {((results_df['Cramers_V'] > 0.3) & (results_df['Cramers_V'] <= 0.5)).sum()}")
    print(f"  Small (0.10 < V < 0.30):  {((results_df['Cramers_V'] > 0.1) & (results_df['Cramers_V'] <= 0.3)).sum()}")
    print(f"  Negligible (V < 0.10):    {(results_df['Cramers_V'] <= 0.1).sum()}")
    
    # Practical significance
    sig_assocs = results_df[results_df['Significant']]
    if len(sig_assocs) > 0:
        print(f"\nAmong significant associations:")
        print(f"  Mean effect size (Cramér's V):    {sig_assocs['Cramers_V'].mean():.3f}")
        print(f"  Mean percentage point difference:  {sig_assocs['Difference'].abs().mean():.1f} pp")
        print(f"  Largest difference:                {sig_assocs['Difference'].abs().max():.1f} pp")


# ═══════════════════════════════════════════════════════════════════════════
# Visualization Functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_cooccurrence_heatmap(cooccurrence_matrix, title, xlabel, ylabel, 
                               output_path, figsize=(12, 9)):
    """
    Plot co-occurrence heatmap with absolute counts.
    
    Args:
        cooccurrence_matrix: Square DataFrame of co-occurrence counts
        title: Plot title
        xlabel, ylabel: Axis labels
        output_path: Where to save figure
        figsize: Figure dimensions
    """
    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(cooccurrence_matrix, annot=True, fmt='d', cmap='YlGn',
                     cbar_kws={'label': ''}, ax=ax)
    
    # Style colorbar to match axis labels
    cbar = hm.collections[0].colorbar
    cbar.set_label('Yhteiset esiintymät', fontsize=12, labelpad=15)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Rotate tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, va='top', ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.2)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_cooccurrence_percentage_heatmap(cooccurrence_matrix, title, xlabel, ylabel,
                                          output_path, figsize=(12, 9)):
    """
    Plot co-occurrence heatmap with conditional percentages.
    Shows: P(column | row) = percentage of interviews with row feature that also have column feature.
    
    Args:
        cooccurrence_matrix: Square DataFrame of co-occurrence counts
        title: Plot title
        xlabel, ylabel: Axis labels
        output_path: Where to save figure
        figsize: Figure dimensions
    """
    # Convert to conditional probabilities
    cooccurrence_pct = cooccurrence_matrix.div(cooccurrence_matrix.values.diagonal(), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(cooccurrence_pct, annot=True, fmt='.0f', cmap='YlGn',
                     vmin=0, vmax=100, cbar_kws={'label': ''}, ax=ax)
    
    # Style colorbar to match axis labels
    cbar = hm.collections[0].colorbar
    cbar.set_label('Osuus (%)', fontsize=12, labelpad=15)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Rotate tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, va='top', ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.2)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_effect_size_heatmap(results_df, title, xlabel, ylabel, 
                              output_path, figsize=(12, 9), vmax=0.4):
    """
    Plot heatmap of effect sizes with significance annotations.
    
    Args:
        results_df: DataFrame from run_chi_square_tests()
        title: Plot title
        xlabel, ylabel: Axis labels
        output_path: Where to save figure
        figsize: Figure dimensions
        vmax: Maximum value for color scale
    """
    # Pivot to matrices
    pivot_pval = results_df.pivot(index='Outcome', columns='Predictor', values='p_fdr')
    pivot_cramers = results_df.pivot(index='Outcome', columns='Predictor', values='Cramers_V')
    
    # Create annotations
    annot_matrix = create_significance_annotations(pivot_cramers, pivot_pval)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(pivot_cramers, annot=annot_matrix, fmt='', cmap='YlGn',
                     vmin=0, vmax=vmax, ax=ax,
                     cbar_kws={'label': ''},
                     yticklabels=True, annot_kws={'fontsize': 8})
    
    # Style colorbar to match axis labels
    cbar = hm.collections[0].colorbar
    cbar.set_label("Cramérin V", fontsize=12, labelpad=15)
    
    # Borders
    n_outcomes, n_predictors = pivot_cramers.shape
    ax.axhline(y=0, color='k', linewidth=2)
    ax.axhline(y=n_outcomes, color='k', linewidth=2)
    ax.axvline(x=0, color='k', linewidth=2)
    ax.axvline(x=n_predictors, color='k', linewidth=2)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Tick labels
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, va='top', ha='center')
    
    plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.2)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_binary_predictor_distribution(predictor_binary, predictor_name, 
                                       label_0, label_1, title, output_path,
                                       figsize=(12, 9)):
    """
    Plot distribution of a binary predictor variable.
    
    Args:
        predictor_binary: Binary DataFrame with one column
        predictor_name: Name of the predictor column
        label_0, label_1: Labels for 0 and 1 values
        title: Plot title
        output_path: Where to save figure
        figsize: Figure dimensions
    """
    fig, ax = plt.subplots(figsize=figsize)
    counts = predictor_binary[predictor_name].value_counts().sort_index()
    bars = ax.bar([label_0, label_1], [counts[0], counts[1]], 
                  color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add counts on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(predictor_binary)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Haastattelujen määrä', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(left=0.25, right=0.85, top=0.85, bottom=0.2)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_binary_predictor_effects(results_df, title, output_path,
                                    figsize=(12, 9)):
    """
    Plot effect sizes for a binary predictor (horizontal bars).
    
    Args:
        results_df: DataFrame from run_chi_square_tests()
        title: Plot title
        output_path: Where to save figure
        figsize: Figure dimensions
    """
    significant = results_df[results_df['Significant']].copy()
    significant = significant.sort_values('Cramers_V', ascending=True)
    
    if len(significant) == 0:
        print("Ei merkitseviä yhteyksiä visualisoitavaksi.")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(significant))
    colors = ['#2ecc71' if diff > 0 else '#e74c3c' 
              for diff in significant['Difference']]
    
    bars = ax.barh(y_pos, significant['Cramers_V'], color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    
    # Add significance stars
    for i, (_, row) in enumerate(significant.iterrows()):
        stars = '***' if row['p_fdr'] < 0.001 else '**' if row['p_fdr'] < 0.01 else '*'
        ax.text(row['Cramers_V'] + 0.01, i, stars, va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(significant['Outcome'])
    ax.set_xlabel("Cramér's V (efektikoko)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.subplots_adjust(left=0.25, right=0.85, top=0.85, bottom=0.2)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_binary_predictor_prevalence(results_df, label_pred, label_not_pred, 
                                      title, output_path, top_n=15,
                                      figsize=(12, 9)):
    """
    Plot prevalence comparison for binary predictor (grouped horizontal bars).
    
    Args:
        results_df: DataFrame from run_chi_square_tests() 
        label_pred: Label for predictor=1 condition
        label_not_pred: Label for predictor=0 condition
        title: Plot title
        output_path: Where to save figure
        top_n: Number of top associations to show
        figsize: Figure dimensions
    """
    significant = results_df[results_df['Significant']].copy()
    
    if len(significant) == 0:
        print("Ei merkitseviä yhteyksiä visualisoitavaksi.")
        return
    
    top_n = min(top_n, len(significant))
    top_sig = significant.nlargest(top_n, 'Cramers_V')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(top_sig))
    width = 0.35
    
    with_pred = top_sig['P(Outcome|Pred)'].values
    without_pred = top_sig['P(Outcome|~Pred)'].values
    
    bars1 = ax.barh(y_pos - width/2, with_pred, width, label=label_pred, 
                    color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2 = ax.barh(y_pos + width/2, without_pred, width, label=label_not_pred,
                    color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_sig['Outcome'])
    ax.set_xlabel('Esiintyvyys (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., frameon=True, fancybox=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(left=0.25, right=0.75, top=0.85, bottom=0.2)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_top_associations_barplot(results_df, title, output_path, 
                                    min_effect=0.1, figsize=(12, 9)):
    """
    Plot ranked bar plot of top associations.
    
    Args:
        results_df: DataFrame from run_chi_square_tests()
        title: Plot title
        output_path: Where to save figure
        min_effect: Minimum Cramér's V to include
        figsize: Figure dimensions
    """
    # Filter for strong effects
    strong_effects = results_df[results_df['Cramers_V'] >= min_effect].copy()
    strong_effects = strong_effects.sort_values('p_fdr')
    
    if len(strong_effects) == 0:
        print(f"No associations with V >= {min_effect} found.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Labels
    labels = [f"{row['Outcome']} × {row['Predictor']}" 
              for _, row in strong_effects.iterrows()]
    
    # Color by direction
    differences = strong_effects['Difference'].values
    abs_max = np.abs(differences).max()
    sym_limit = np.ceil(abs_max / 10) * 10 if abs_max > 0 else 10
    vmin, vmax = -sym_limit, sym_limit
    
    # Handle edge cases for colormap
    vmin_orig, vmax_orig = differences.min(), differences.max()
    if vmin_orig >= 0:  # All positive
        norm = plt.Normalize(vmin=0, vmax=vmax)
        colors = plt.cm.Greens(norm(differences))
    elif vmax_orig <= 0:  # All negative
        norm = plt.Normalize(vmin=vmin, vmax=0)
        colors = plt.cm.Purples(norm(differences))
    else:  # Mixed
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        colors = plt.cm.PRGn(norm(differences))
    
    # Plot bars
    y_pos = np.arange(len(strong_effects))
    ax.barh(y_pos, -np.log10(strong_effects['p_fdr']), 
            color=colors, edgecolor='black', linewidth=0.5)
    
    # Annotations
    for i, (_, row) in enumerate(strong_effects.iterrows()):
        diff_text = f"{row['Difference']:+.1f}%"
        ax.text(-np.log10(row['p_fdr']) + 0.1, i, diff_text,
                va='center', fontsize=9, fontweight='bold')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('-log₁₀(FDR q-arvo)', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.axvline(1.3, color='gray', linestyle='--', linewidth=1, 
               alpha=0.7, label='q = 0.05')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    ax.grid(axis='x', alpha=0.3)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.PRGn, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, aspect=30, shrink=0.8)
    cbar.set_label('Erotus (%)', rotation=270, labelpad=20, fontsize=12)
    
    plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.2)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Data Preparation Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_and_align_data(predictor_path, outcome_path, 
                        drop_predictor_cols=None, drop_outcome_cols=None):
    """
    Load predictor and outcome CSVs, align by rec_id, convert to binary.
    
    Args:
        predictor_path: Path to predictor CSV (must have rec_id column or index)
        outcome_path: Path to outcome CSV (must have rec_id column or index)
        drop_predictor_cols: Columns to drop from predictor (e.g., ['lon', 'lat'])
        drop_outcome_cols: Columns to drop from outcome
        
    Returns:
        tuple: (predictor_binary, outcome_binary) - aligned binary DataFrames
    """
    # Load data
    pred_raw = pd.read_csv(predictor_path)
    out_raw = pd.read_csv(outcome_path, index_col=0)
    
    # Set index if needed
    if 'rec_id' in pred_raw.columns:
        pred_raw = pred_raw.set_index('rec_id')
    
    # Drop unwanted columns
    if drop_predictor_cols:
        pred_raw = pred_raw.drop(columns=drop_predictor_cols, errors='ignore')
    if drop_outcome_cols:
        out_raw = out_raw.drop(columns=drop_outcome_cols, errors='ignore')
    
    # Align by index (inner join)
    common_ids = pred_raw.index.intersection(out_raw.index)
    predictor_aligned = pred_raw.loc[common_ids]
    outcome_aligned = out_raw.loc[common_ids]
    
    # Convert to binary
    predictor_binary = (predictor_aligned >= 0.5).astype(int)
    outcome_binary = (outcome_aligned == 1.0).astype(int)
    
    return predictor_binary, outcome_binary


def print_data_summary(predictor_df, outcome_df, predictor_label, outcome_label):
    """
    Print summary of loaded data.
    
    Args:
        predictor_df: Predictor DataFrame
        outcome_df: Outcome DataFrame
        predictor_label: Human-readable name for predictors
        outcome_label: Human-readable name for outcomes
    """
    print("=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"{predictor_label}: {predictor_df.shape[0]} interviews × {predictor_df.shape[1]} variables")
    print(f"{outcome_label}: {outcome_df.shape[0]} interviews × {outcome_df.shape[1]} variables")
    print()
    
    # Prevalence
    pred_prev = (predictor_df.mean() * 100).sort_values(ascending=False)
    out_prev = (outcome_df.mean() * 100).sort_values(ascending=False)
    
    print(f"{predictor_label} Prevalence (%):")
    print(pred_prev.to_string())
    print()
    print(f"{outcome_label} Prevalence (%) - Top 15:")
    print(out_prev.head(15).to_string())
    print()
