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
import statsmodels.formula.api as smf


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





def run_mixed_effects_tests(predictors, outcomes, user_ids, r_script_path='scripts/run_glmer_tests.R'):
    """
    Run mixed-effects logistic regression (glmer) via R for all predictor-outcome pairs.
    Accounts for non-independence by explicitly modeling user-level random intercepts.
    
    This properly handles user concentration effects by explicitly modeling between-user
    variance, preventing spurious associations driven by repeated measurements from
    a small number of users.
    
    Args:
        predictors: Binary DataFrame (rec_id × predictor variables)
        outcomes: Binary DataFrame (rec_id × outcome variables)
        user_ids: Series or array of user IDs (same index as predictors/outcomes)
        r_script_path: Path to R script that runs glmer models
        
    Returns:
        DataFrame with columns: Outcome, Predictor, Coefficient, Odds_Ratio, p_value,
                                p_fdr, Significant
        Note: No Difference (%) column - use Coefficient (log-odds) for directional effect
    """
    import subprocess
    import tempfile
    import os
    
    # Prepare data for R
    # Merge predictors, outcomes, and user_ids into single dataframe
    data_for_r = pd.DataFrame({
        'user': user_ids
    }, index=predictors.index)
    
    # Add all predictor columns with prefix
    for col in predictors.columns:
        data_for_r[f'pred_{col}'] = predictors[col].astype(int)
    
    # Add all outcome columns with prefix
    for col in outcomes.columns:
        data_for_r[f'out_{col}'] = outcomes[col].astype(int)
    
    # Save to temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_data_file = f.name
        data_for_r.to_csv(f, index=True)
    
    # Create temporary output file
    temp_output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False).name
    
    try:
        # Call R script
        result = subprocess.run([
            r_script_path,
            temp_data_file,
            temp_output_file,
            str(len(predictors.columns)),
            str(len(outcomes.columns))
        ], check=True, capture_output=True, text=True)
        
        # Print R output for monitoring
        if result.stdout:
            print(result.stdout)
        
        # Read results from R
        df = pd.read_csv(temp_output_file)
        
        # Apply FDR correction
        valid_pvals = df['p_value'].notna()
        if valid_pvals.sum() > 0:
            df.loc[valid_pvals, 'p_fdr'] = false_discovery_control(
                df.loc[valid_pvals, 'p_value'], method='bh'
            )
        else:
            df['p_fdr'] = np.nan
        
        df['Significant'] = df['p_fdr'] < 0.05
        df = df.sort_values('p_value')
        
        return df
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_data_file):
            os.unlink(temp_data_file)
        if os.path.exists(temp_output_file):
            os.unlink(temp_output_file)


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

def create_significance_annotations(effect_matrix, pval_matrix, stats_mode='chisquared'):
    """
    Create annotation matrix with effect sizes and significance stars.
    
    Args:
        effect_matrix: DataFrame of effect size values (Cramér's V or Coefficient)
        pval_matrix: DataFrame of FDR-corrected p-values
        stats_mode: 'chisquared' or 'random_effects'
        
    Returns:
        numpy array of strings like "0.25***" or "0.75***"
    """
    annot_matrix = np.empty(effect_matrix.shape, dtype=object)
    
    for i in range(effect_matrix.shape[0]):
        for j in range(effect_matrix.shape[1]):
            v = effect_matrix.iloc[i, j]
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


def print_summary_stats(results_df, stats_mode='chisquared'):
    """
    Print summary statistics from chi-square or mixed effects results.
    
    Args:
        results_df: DataFrame from run_chi_square_tests() or run_mixed_effects_tests()
        stats_mode: 'chisquared' or 'random_effects'
    """
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Overall summary
    print(f"\nTotal tests performed:              {len(results_df)}")
    print(f"Significant (uncorrected p<0.05):   {(results_df['p_value'] < 0.05).sum()}")
    print(f"Significant (FDR q<0.05):           {results_df['Significant'].sum()}")
    
    # Effect size distribution
    if stats_mode == 'chisquared':
        print(f"\nEffect size distribution (Cramér's V):")
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
    
    elif stats_mode == 'random_effects':
        print(f"\nEffect size distribution (log-odds):")
        print(f"  Large (|log-odds| > 1.0):        {(results_df['Coefficient'].abs() > 1.0).sum()}")
        print(f"  Medium (0.6 < |log-odds| < 1.0): {((results_df['Coefficient'].abs() > 0.6) & (results_df['Coefficient'].abs() <= 1.0)).sum()}")
        print(f"  Small (0.5 < |log-odds| < 0.6):  {((results_df['Coefficient'].abs() > 0.5) & (results_df['Coefficient'].abs() <= 0.6)).sum()}")
        print(f"  Negligible (|log-odds| < 0.5):   {(results_df['Coefficient'].abs() <= 0.5).sum()}")
        
        # Practical significance
        sig_assocs = results_df[results_df['Significant']]
        if len(sig_assocs) > 0:
            print(f"\nAmong significant associations:")
            print(f"  Mean |log-odds|:                   {sig_assocs['Coefficient'].abs().mean():.3f}")
            print(f"  Mean odds ratio:                   {sig_assocs['Odds_Ratio'].mean():.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# Visualization Functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_cooccurrence_heatmap(cooccurrence_matrix, title, xlabel, ylabel, 
                               output_path, figsize=(12, 9), footnote=None):
    """
    Plot co-occurrence heatmap with absolute counts.
    
    Args:
        cooccurrence_matrix: Square DataFrame of co-occurrence counts
        title: Plot title
        xlabel, ylabel: Axis labels
        output_path: Where to save figure
        figsize: Figure dimensions
        footnote: Optional footnote text
    """
    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(cooccurrence_matrix, annot=True, fmt='d', cmap='YlGn',
                     cbar_kws={'label': ''}, ax=ax)
    
    # Style colorbar to match axis labels
    cbar = hm.collections[0].colorbar
    cbar.set_label('Co-occurrences', fontsize=12, labelpad=15)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    
    # Rotate tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    if footnote:
        fig.text(0.99, 0.01, footnote, ha='right', va='bottom', fontsize=8, style='italic', color='gray')
    
    plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.35)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_cooccurrence_percentage_heatmap(cooccurrence_matrix, title, xlabel, ylabel,
                                          output_path, figsize=(12, 9), footnote=None):
    """
    Plot co-occurrence heatmap with conditional percentages.
    Shows: P(column | row) = percentage of interviews with row feature that also have column feature.
    
    Args:
        cooccurrence_matrix: Square DataFrame of co-occurrence counts
        title: Plot title
        xlabel, ylabel: Axis labels
        output_path: Where to save figure
        figsize: Figure dimensions
        footnote: Optional footnote text
    """
    # Convert to conditional probabilities
    cooccurrence_pct = cooccurrence_matrix.div(cooccurrence_matrix.values.diagonal(), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(cooccurrence_pct, annot=True, fmt='.0f', cmap='YlGn',
                     vmin=0, vmax=100, cbar_kws={'label': ''}, ax=ax)
    
    # Style colorbar to match axis labels
    cbar = hm.collections[0].colorbar
    cbar.set_label('Percentage (%)', fontsize=12, labelpad=15)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    
    # Rotate tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    if footnote:
        fig.text(0.99, 0.01, footnote, ha='right', va='bottom', fontsize=8, style='italic', color='gray')
    
    plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.35)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_effect_size_heatmap(results_df, title, xlabel, ylabel, 
                              output_path, figsize=(12, 9), vmax=0.4, footnote=None, 
                              stats_mode='chisquared'):
    """
    Plot heatmap of effect sizes with significance annotations.
    
    Args:
        results_df: DataFrame from run_chi_square_tests() or run_mixed_effects_tests()
        title: Plot title
        xlabel, ylabel: Axis labels
        output_path: Where to save figure
        figsize: Figure dimensions
        vmax: Maximum value for color scale
        footnote: Optional footnote text
        stats_mode: 'chisquared' or 'random_effects'
    """
    # Choose columns based on stats mode
    if stats_mode == 'random_effects':
        effect_col = 'Coefficient'
        effect_label = 'Log-odds'
    else:  # chisquared
        effect_col = 'Cramers_V'
        effect_label = "Cramér's V"
    
    # Pivot to matrices
    pivot_pval = results_df.pivot(index='Outcome', columns='Predictor', values='p_fdr')
    pivot_effect = results_df.pivot(index='Outcome', columns='Predictor', values=effect_col)
    
    # Dynamic font sizing based on number of outcomes
    n_outcomes = len(pivot_effect.index)
    n_predictors = len(pivot_effect.columns)
    
    if n_outcomes <= 30:
        ylabel_fontsize = 9
        annot_fontsize = 8
    elif n_outcomes <= 50:
        ylabel_fontsize = 7
        annot_fontsize = 6
    elif n_outcomes <= 70:
        ylabel_fontsize = 6
        annot_fontsize = 5
    else:
        ylabel_fontsize = 5
        annot_fontsize = 4
    
    # Adjust figure height based on number of outcomes
    height = max(figsize[1], n_outcomes * 0.12)  # At least 0.12 inches per row
    figsize = (figsize[0], height)
    
    # Calculate relative margins for constant absolute margins
    # Top: ~1.0 inch for title
    # Bottom: ~2.0 inches for x-axis labels (rotated)
    top_margin = 1.0
    bottom_margin = 2.0
    
    top_frac = 1.0 - (top_margin / height)
    bottom_frac = bottom_margin / height
    
    # Create annotations
    annot_matrix = create_significance_annotations(pivot_effect, pivot_pval, stats_mode)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use YlGn colormap for both modes for consistency
    # For random_effects, show absolute values in heatmap (magnitude of effect)
    if stats_mode == 'random_effects':
        # Show absolute coefficient values (magnitude matters more than direction for heatmap)
        pivot_effect_abs = pivot_effect.abs()
        hm = sns.heatmap(pivot_effect_abs, annot=annot_matrix, fmt='', cmap='YlGn',
                         vmin=0, vmax=vmax, ax=ax,
                         cbar_kws={'label': ''},
                         yticklabels=True, annot_kws={'fontsize': annot_fontsize})
    else:  # chisquared
        hm = sns.heatmap(pivot_effect, annot=annot_matrix, fmt='', cmap='YlGn',
                         vmin=0, vmax=vmax, ax=ax,
                         cbar_kws={'label': ''},
                         yticklabels=True, annot_kws={'fontsize': annot_fontsize})
    
    # Style colorbar to match axis labels
    cbar = hm.collections[0].colorbar
    cbar.set_label(effect_label, fontsize=12, labelpad=15)
    
    # Borders
    ax.axhline(y=0, color='k', linewidth=2)
    ax.axhline(y=n_outcomes, color='k', linewidth=2)
    ax.axvline(x=0, color='k', linewidth=2)
    ax.axvline(x=n_predictors, color='k', linewidth=2)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=20)
    
    # Tick labels with dynamic sizing
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=ylabel_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add footnote
    if footnote:
        fig.text(0.99, 0.01, footnote, ha='right', va='bottom', fontsize=8, style='italic', color='gray')
    
    plt.subplots_adjust(left=0.25, right=0.95, top=top_frac, bottom=bottom_frac)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_binary_predictor_distribution(predictor_binary, predictor_name, 
                                       label_0, label_1, title, output_path,
                                       figsize=(12, 9), footnote=None):
    """
    Plot distribution of a binary predictor variable.
    
    Args:
        predictor_binary: Binary DataFrame with one column
        predictor_name: Name of the predictor column
        label_0, label_1: Labels for 0 and 1 values
        title: Plot title
        output_path: Where to save figure
        figsize: Figure dimensions
        footnote: Optional footnote text
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
    
    ax.set_ylabel('Number of interviews', fontsize=12, labelpad=15)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if footnote:
        fig.text(0.99, 0.01, footnote, ha='right', va='bottom', fontsize=8, style='italic', color='gray')
    
    plt.subplots_adjust(left=0.25, right=0.85, top=0.85, bottom=0.2)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_binary_predictor_effects(results_df, title, output_path,
                                    min_effect=0.1, figsize=(12, 9), footnote=None,
                                    significant_col='Significant', p_fdr_col='p_fdr'):
    """
    Plot effect sizes for a binary predictor (horizontal bars).
    
    Args:
        results_df: DataFrame from run_chi_square_tests()
        title: Plot title
        output_path: Where to save figure
        min_effect: Minimum Cramér's V to include (default: 0.1)
        figsize: Figure dimensions
        footnote: Optional footnote text
        significant_col: Name of the Significant column (default: 'Significant')
        p_fdr_col: Name of the p_fdr column (default: 'p_fdr')
    """
    # Filter by effect size (not just significance)
    strong_effects = results_df[results_df['Cramers_V'] >= min_effect].copy()
    strong_effects = strong_effects.sort_values('Cramers_V', ascending=True)
    
    if len(strong_effects) == 0:
        print(f"No associations with V >= {min_effect} found.")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(strong_effects))
    colors = ['#2ecc71' if diff > 0 else '#e74c3c' 
              for diff in strong_effects['Difference']]
    
    bars = ax.barh(y_pos, strong_effects['Cramers_V'], color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    
    # Add significance stars only for significant results
    for i, (_, row) in enumerate(strong_effects.iterrows()):
        if row[significant_col]:
            stars = '***' if row[p_fdr_col] < 0.001 else '**' if row[p_fdr_col] < 0.01 else '*'
            ax.text(row['Cramers_V'] + 0.01, i, stars, va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(strong_effects['Outcome'])
    ax.set_xlabel("Cramér's V (effect size)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=min_effect, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    if footnote:
        fig.text(0.99, 0.01, footnote, ha='right', va='bottom', fontsize=8, style='italic', color='gray')
        
    plt.subplots_adjust(left=0.25, right=0.85, top=0.85, bottom=0.2)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_binary_predictor_prevalence(results_df, label_pred, label_not_pred, 
                                      title, output_path, top_n=15,
                                      figsize=(12, 9), significant_col='Significant'):
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
        significant_col: Name of the Significant column (default: 'Significant')
    """
    significant = results_df[results_df[significant_col]].copy()
    
    if len(significant) == 0:
        print("No significant associations to visualize.")
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
    ax.set_xlabel('Prevalence (%)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., frameon=True, fancybox=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(left=0.25, right=0.75, top=0.85, bottom=0.2)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_top_associations_barplot(results_df, title, output_path, 
                                    min_effect=0.1, figsize=(12, 9), footnote=None,
                                    stats_mode='chisquared'):
    """
    Plot ranked bar plot of top associations.
    
    Args:
        results_df: DataFrame from run_chi_square_tests() or run_mixed_effects_tests()
        title: Plot title
        output_path: Where to save figure
        min_effect: Minimum effect size to include (V for chisquared, |coef| for random_effects)
        figsize: Figure dimensions
        footnote: Optional footnote text
        stats_mode: 'chisquared' or 'random_effects'
    """
    # Choose filtering based on stats mode
    if stats_mode == 'random_effects':
        effect_col = 'Coefficient'
        strong_effects = results_df[results_df[effect_col].abs() >= min_effect].copy()
        # For random effects, also require at least nominal significance
        # This prevents showing huge effect sizes that are just noise
        strong_effects = strong_effects[strong_effects['p_value'] < 0.05]
    else:  # chisquared
        effect_col = 'Cramers_V'
        strong_effects = results_df[results_df[effect_col] >= min_effect].copy()
        # Chi-squared: V alone is sufficient (test is built-in)
    
    strong_effects = strong_effects.sort_values('p_value')
    
    if len(strong_effects) == 0:
        print(f"No associations with effect >= {min_effect} found.")
        return
    
    # Dynamic font sizing based on number of items
    n_items = len(strong_effects)
    if n_items <= 20:
        label_fontsize = 10
        annot_fontsize = 9
    elif n_items <= 40:
        label_fontsize = 8
        annot_fontsize = 7
    elif n_items <= 60:
        label_fontsize = 7
        annot_fontsize = 6
    else:
        label_fontsize = 6
        annot_fontsize = 5
    
    # Adjust figure height dynamically
    # Scale height based on number of items to avoid huge bars for small N
    # Base height ~3.5 inches for margins/title, plus ~0.35 inches per item
    calculated_height = n_items * 0.35 + 3.5
    height = max(4.5, calculated_height)
    figsize = (figsize[0], height)
    
    # Calculate relative margins to maintain constant absolute margins
    # We want ~1.0 inch top margin and ~1.5 inch bottom margin regardless of height
    # This ensures consistency between small (few items) and large (many items) plots
    top_margin = 1.0
    bottom_margin = 1.5
    
    top_frac = 1.0 - (top_margin / height)
    bottom_frac = bottom_margin / height
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Labels
    labels = [f"{row['Outcome']} × {row['Predictor']}" 
              for _, row in strong_effects.iterrows()]
    
    # Color by direction (use Coefficient for random_effects, Difference for chisquared)
    if stats_mode == 'random_effects':
        color_values = strong_effects['Coefficient'].values
        color_label = 'Log-odds'
    else:
        color_values = strong_effects['Difference'].values
        color_label = 'Difference (%)'
    
    abs_max = np.abs(color_values).max()
    
    # Use sensible color scale limits based on actual data
    # For log-odds: typical range is -3 to +3, extend slightly for visibility
    # For difference: typical range is -50 to +50, extend slightly
    if stats_mode == 'random_effects':
        # Round to nearest integer, with minimum range of ±2
        sym_limit = max(2.0, np.ceil(abs_max))
    else:
        # For percentage differences, round to nearest 10
        sym_limit = max(10.0, np.ceil(abs_max / 10) * 10)
    
    vmin, vmax = -sym_limit, sym_limit
    
    # Handle edge cases for colormap
    vmin_orig, vmax_orig = color_values.min(), color_values.max()
    if vmin_orig >= 0:  # All positive
        norm = plt.Normalize(vmin=0, vmax=vmax)
        colors = plt.cm.Greens(norm(color_values))
    elif vmax_orig <= 0:  # All negative
        norm = plt.Normalize(vmin=vmin, vmax=0)
        colors = plt.cm.Purples(norm(color_values))
    else:  # Mixed
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        colors = plt.cm.PRGn(norm(color_values))
    
    # Plot bars
    y_pos = np.arange(len(strong_effects))
    
    # Use raw p-value for bar length instead of FDR q-value
    # This prevents the "staircase" effect where many items have identical q=1.0
    # We still use FDR for significance testing/stars
    bar_lengths = -np.log10(strong_effects['p_value'])
    
    ax.barh(y_pos, bar_lengths, 
            color=colors, edgecolor='black', linewidth=0.5)
    
    # Annotations
    for i, (_, row) in enumerate(strong_effects.iterrows()):
        if stats_mode == 'random_effects':
            # Show log-odds coefficient directly (matches color scale)
            coef_val = row['Coefficient']
            annot_text = f"{coef_val:+.2f}"
        else:
            # Show difference percentage
            annot_text = f"{row['Difference']:+.1f}%"
        
        # Add stars if FDR significant
        stars = ""
        if row['p_fdr'] < 0.001: stars = "***"
        elif row['p_fdr'] < 0.01: stars = "**"
        elif row['p_fdr'] < 0.05: stars = "*"
        
        # Position text
        text_pos = bar_lengths.iloc[i] + 0.1
        ax.text(text_pos, i, f"{annot_text} {stars}",
                va='center', fontsize=annot_fontsize, fontweight='bold')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.set_xlabel('-log₁₀(Raw p-value)', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    # Add threshold lines for raw p-values
    ax.axvline(-np.log10(0.05), color='gray', linestyle='--', linewidth=1, 
               alpha=0.7, label='p = 0.05')
    ax.axvline(-np.log10(0.01), color='gray', linestyle=':', linewidth=1, 
               alpha=0.5, label='p = 0.01')
               
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    ax.grid(axis='x', alpha=0.3)
    
    # Add footnote
    if footnote:
        fig.text(0.99, 0.01, footnote, ha='right', va='bottom', fontsize=8, style='italic', color='gray')
    else:
        # Default footnote if none provided
        default_note = "Significance: * q<0.05, ** q<0.01, *** q<0.001 (Benjamini-Hochberg FDR correction)"
        fig.text(0.99, 0.01, default_note, ha='right', va='bottom', fontsize=8, style='italic', color='gray')
    
    # Colorbar
    # Dynamic aspect ratio to keep bar width roughly constant regardless of plot height
    # Standard height=9 -> aspect=30 implies width ~0.24 inches
    # We want to maintain that width: aspect = (height * shrink) / width
    cbar_aspect = height * 3.0
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.PRGn, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, aspect=cbar_aspect, shrink=0.8)
    cbar.set_label(color_label, rotation=270, labelpad=20, fontsize=12)
    
    plt.subplots_adjust(left=0.35, right=0.95, top=top_frac, bottom=bottom_frac)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def save_summary_table_image(results_df, title, output_path, top_n=20, figsize=(12, 12),
                            stats_mode='chisquared'):
    """
    Render a summary table of statistical results as an image.
    Includes overall stats and a table of top significant associations.
    
    Args:
        results_df: DataFrame from run_chi_square_tests() or run_mixed_effects_tests()
        title: Title for the summary page
        output_path: Where to save the PNG
        top_n: Number of top associations to list
        figsize: Figure dimensions
        stats_mode: 'chisquared' or 'random_effects'
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # 1. Header
    ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=18, fontweight='bold')
    ax.text(0.5, 0.91, "Statistical Summary", ha='center', va='top', fontsize=14, color='gray')
    
    # 2. Overall Statistics
    n_total = len(results_df)
    n_sig_unc = (results_df['p_value'] < 0.05).sum()
    n_sig_fdr = results_df['Significant'].sum()
    
    if stats_mode == 'chisquared':
        max_effect = results_df['Cramers_V'].max()
        effect_label = "Strongest effect size (V):"
        effect_str = f"{max_effect:.3f}"
    else:  # random_effects
        max_effect = results_df['Coefficient'].abs().max()
        effect_label = "Strongest effect (|coef|):"
        effect_str = f"{max_effect:.3f}"
    
    stats_text = (
        f"Total tests performed:        {n_total}\n"
        f"Significant (raw p < 0.05):   {n_sig_unc} ({n_sig_unc/n_total*100:.1f}%)\n"
        f"Significant (FDR q < 0.05):   {n_sig_fdr} ({n_sig_fdr/n_total*100:.1f}%)\n"
        f"{effect_label:30s}{effect_str}"
    )
    
    # Draw a box for stats
    ax.text(0.1, 0.85, "Overview", fontsize=14, fontweight='bold')
    ax.text(0.1, 0.82, stats_text, fontsize=12, va='top', linespacing=1.6, family='monospace')
    
    # 3. Top Associations Table
    sig_df = results_df[results_df['Significant']].copy()
    
    if stats_mode == 'chisquared':
        sort_col = 'Cramers_V'
    else:  # random_effects
        # Sort by absolute coefficient
        sig_df = sig_df.copy()
        sig_df['abs_coef'] = sig_df['Coefficient'].abs()
        sort_col = 'abs_coef'
    
    if len(sig_df) > 0:
        table_title = f"Top {top_n} Significant Associations (by Effect Size)"
        display_df = sig_df.sort_values(sort_col, ascending=False).head(top_n)
        footer_note = "Showing statistically significant results (FDR q < 0.05)."
    else:
        # No significant results - don't show a table
        ax.text(0.5, 0.5, "No significant associations found (FDR q < 0.05)", 
                ha='center', va='center', fontsize=16, style='italic', color='#666666')
        plt.savefig(output_path, dpi=300)
        plt.close()
        return
    
    ax.text(0.1, 0.65, table_title, fontsize=14, fontweight='bold')
    
    if len(display_df) > 0:
        table_data = []
        
        if stats_mode == 'chisquared':
            # Columns: Outcome, Predictor, Chi2, p, V, Diff
            col_labels = ['Outcome', 'Predictor', 'Chi²', 'p-value', "V", 'Diff']
            for _, row in display_df.iterrows():
                table_data.append([
                    row['Outcome'][:20],  # Truncate long names
                    row['Predictor'][:20],
                    f"{row['Chi2']:.1f}",
                    f"{row['p_value']:.2e}",
                    f"{row['Cramers_V']:.2f}",
                    f"{row['Difference']:+.1f}%"
                ])
        else:  # random_effects
            # Columns: Outcome, Predictor, Coef, OR, p-value
            col_labels = ['Outcome', 'Predictor', 'Coef', 'OR', 'p-value']
            for _, row in display_df.iterrows():
                table_data.append([
                    row['Outcome'][:20],  # Truncate long names
                    row['Predictor'][:20],
                    f"{row['Coefficient']:.2f}",
                    f"{row['Odds_Ratio']:.2f}",
                    f"{row['p_value']:.2e}"
                ])
        
        # Create table with specific column widths to avoid cutting off text
        # Widths sum to ~0.9 (leaving margins)
        if stats_mode == 'chisquared':
            col_widths = [0.35, 0.20, 0.08, 0.09, 0.08, 0.10]
        else:  # random_effects - one fewer column
            col_widths = [0.35, 0.25, 0.10, 0.10, 0.10]
        
        table = ax.table(cellText=table_data, colLabels=col_labels, colWidths=col_widths,
                         loc='center', cellLoc='left', bbox=[0.05, 0.05, 0.9, 0.58])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.6) # Increase row height
        
        # Style headers
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e6e6e6')
                cell.set_height(0.05)
    
    # Footnote
    ax.text(0.5, 0.02, footer_note, ha='center', fontsize=10, style='italic', color='#444444')
    
    # IMPORTANT: Do NOT use bbox_inches='tight' to ensure exact 12-inch width matching other plots
    plt.savefig(output_path, dpi=300)
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
