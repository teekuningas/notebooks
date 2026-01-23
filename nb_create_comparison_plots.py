"""
Create clean comparison plots: 452 vs 710 interviews
One plot per analysis type, showing side-by-side bars for associations
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Match settings from utils_stats.py
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
FIGSIZE = (12, 9)

def get_significance_marker(p_fdr):
    """Return asterisk marker based on FDR q-value"""
    if pd.isna(p_fdr):
        return ''
    elif p_fdr < 0.001:
        return '***'
    elif p_fdr < 0.01:
        return '**'
    elif p_fdr < 0.05:
        return '*'
    else:
        return ''

def create_comparison_plot(analysis_name, analysis_title, output_path, 
                          min_effect=0.1, max_p_raw=0.10):
    """
    Create side-by-side comparison plot for one analysis type.
    Shows -log10(p_value) for associations meeting criteria in either dataset.
    
    Args:
        analysis_name: Directory name (e.g., 'esa_koodit')
        analysis_title: Display title (e.g., 'ESA Habitats × Themes')
        output_path: Where to save PNG
        min_effect: Minimum effect size to show
        max_p_raw: Maximum uncorrected p-value to show
    """
    # Load results
    results_452 = pd.read_csv(f'./output/stats_comparison/452/{analysis_name}/mixed_effects_results.csv')
    results_710 = pd.read_csv(f'./output/stats_comparison/710/{analysis_name}/mixed_effects_results.csv')
    
    # Create keys
    results_452['key'] = results_452['Predictor'] + ' → ' + results_452['Outcome']
    results_710['key'] = results_710['Predictor'] + ' → ' + results_710['Outcome']
    
    # Filter for "interesting" associations (would show in top associations plot)
    interesting_452 = results_452[
        (results_452['Cramers_V'] >= min_effect) & 
        (results_452['p_value'] < max_p_raw)
    ]['key'].tolist()
    
    interesting_710 = results_710[
        (results_710['Cramers_V'] >= min_effect) & 
        (results_710['p_value'] < max_p_raw)
    ]['key'].tolist()
    
    # Union of interesting associations
    all_interesting = sorted(set(interesting_452 + interesting_710))
    
    if len(all_interesting) == 0:
        print(f"  {analysis_title}: No associations meet criteria in either dataset")
        # Create empty placeholder
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.text(0.5, 0.5, 'No associations meet criteria\n(V ≥ 0.1, p < 0.10)\nin either dataset',
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Get data for all interesting associations
    comparison_data = []
    for key in all_interesting:
        row_452 = results_452[results_452['key'] == key]
        row_710 = results_710[results_710['key'] == key]
        
        # Get p-values (use _glmer version for inference)
        if len(row_452) > 0:
            p_val_452 = row_452['p_value_glmer'].values[0] if 'p_value_glmer' in row_452.columns else row_452['p_value'].values[0]
            p_fdr_452 = row_452['p_fdr_glmer'].values[0] if 'p_fdr_glmer' in row_452.columns else row_452['p_fdr'].values[0]
        else:
            p_val_452 = 1.0
            p_fdr_452 = 1.0
            
        if len(row_710) > 0:
            p_val_710 = row_710['p_value_glmer'].values[0] if 'p_value_glmer' in row_710.columns else row_710['p_value'].values[0]
            p_fdr_710 = row_710['p_fdr_glmer'].values[0] if 'p_fdr_glmer' in row_710.columns else row_710['p_fdr'].values[0]
        else:
            p_val_710 = 1.0
            p_fdr_710 = 1.0
        
        comparison_data.append({
            'key': key,
            'p_val_452': p_val_452,
            'p_val_710': p_val_710,
            'p_fdr_452': p_fdr_452,
            'p_fdr_710': p_fdr_710,
            'sig_452': p_fdr_452 < 0.05,
            'sig_710': p_fdr_710 < 0.05,
            'min_p': min(p_val_452, p_val_710)  # For sorting
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('min_p', ascending=True)  # Most significant first (will be at top)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    y_pos = np.arange(len(df))
    bar_height = 0.35
    
    # Calculate -log10(p) for bar lengths
    bar_452 = -np.log10(df['p_val_452'].clip(lower=1e-300))  # Avoid log(0)
    bar_710 = -np.log10(df['p_val_710'].clip(lower=1e-300))
    
    # Plot bars
    bars_452 = ax.barh(y_pos - bar_height/2, bar_452, bar_height,
                       label='452 interviews', color='#7fc97f', edgecolor='black', linewidth=0.5)
    bars_710 = ax.barh(y_pos + bar_height/2, bar_710, bar_height,
                       label='710 interviews', color='#beaed4', edgecolor='black', linewidth=0.5)
    
    # Add significance markers at the end of bars
    for idx, row in df.iterrows():
        i = list(df.index).index(idx)
        
        sig_452 = get_significance_marker(row['p_fdr_452'])
        sig_710 = get_significance_marker(row['p_fdr_710'])
        
        if sig_452:
            text_pos = bar_452.iloc[i] + 0.2
            ax.text(text_pos, i - bar_height/2, sig_452,
                   va='center', ha='left', fontsize=9, fontweight='bold')
        
        if sig_710:
            text_pos = bar_710.iloc[i] + 0.2
            ax.text(text_pos, i + bar_height/2, sig_710,
                   va='center', ha='left', fontsize=9, fontweight='bold')
    
    # Labels and styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['key'], fontsize=9)
    ax.set_xlabel("-log₁₀(p-value)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'{analysis_title}: Comparison of Datasets\n452 vs 710 Interviews',
                fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add reference line for p=0.05
    p_05_line = -np.log10(0.05)
    ax.axvline(p_05_line, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='p=0.05')
    
    # Footnote
    footnote = f"Associations with V≥{min_effect} and p<{max_p_raw} in either dataset. Significance: * q<0.05, ** q<0.01, *** q<0.001 (FDR)"
    fig.text(0.99, 0.01, footnote, ha='right', va='bottom', fontsize=8, style='italic', color='gray')
    
    plt.subplots_adjust(left=0.35, right=0.95, top=0.92, bottom=0.08)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    n_sig_452 = df['sig_452'].sum()
    n_sig_710 = df['sig_710'].sum()
    n_both = (df['sig_452'] & df['sig_710']).sum()
    
    print(f"  {analysis_title}:")
    print(f"    Total associations shown: {len(df)}")
    print(f"    Significant in 452: {n_sig_452}, in 710: {n_sig_710}, in both: {n_both}")


if __name__ == '__main__':
    import os
    os.environ['MPLBACKEND'] = 'Agg'
    
    output_dir = './output/stats_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*70)
    print()
    
    # Create one plot per analysis
    create_comparison_plot(
        'esa_koodit',
        'ESA Habitats × Themes',
        f'{output_dir}/comparison_esa.png',
        min_effect=0.1
    )
    
    create_comparison_plot(
        'bird_species_koodit',
        'Bird Species × Themes',
        f'{output_dir}/comparison_bird_species.png',
        min_effect=0.1
    )
    
    create_comparison_plot(
        'bird_presence_koodit',
        'Bird Presence × Themes',
        f'{output_dir}/comparison_bird_presence.png',
        min_effect=0.1
    )
    
    create_comparison_plot(
        'bird_groups_koodit',
        'Bird Groups × Themes',
        f'{output_dir}/comparison_bird_groups.png',
        min_effect=0.1
    )
    
    print()
    print("="*70)
    print("✓ COMPARISON PLOTS COMPLETE")
    print("="*70)
    print(f"\nFiles saved in: {output_dir}/")
    print("  - comparison_esa.png")
    print("  - comparison_bird_species.png")
    print("  - comparison_bird_presence.png")
    print("  - comparison_bird_groups.png")
