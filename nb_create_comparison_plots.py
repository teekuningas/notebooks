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

def create_comparison_plot(analysis_name, analysis_title, output_path):
    """
    Create side-by-side comparison plot for one analysis type.
    Shows -log10(p_value) for associations meeting criteria in either dataset.
    
    Filtering strategy:
    - Find worst (largest) p-value among FDR-significant associations in each dataset
    - Include any association that meets this threshold in EITHER dataset
    - This ensures all FDR-significant associations are shown, plus similar strength ones
    
    Args:
        analysis_name: Directory name (e.g., 'esa_koodit')
        analysis_title: Display title (e.g., 'ESA Habitats × Themes')
        output_path: Where to save PNG
    """
    # Load results - now using results.csv instead of mixed_effects_results.csv
    results_452 = pd.read_csv(f'./output/stats_comparison/452/{analysis_name}/results.csv')
    results_710 = pd.read_csv(f'./output/stats_comparison/710/{analysis_name}/results.csv')
    
    # Create keys
    results_452['key'] = results_452['Predictor'] + ' → ' + results_452['Outcome']
    results_710['key'] = results_710['Predictor'] + ' → ' + results_710['Outcome']
    
    # Find p-value thresholds: worst p-value among FDR-significant associations
    sig_452 = results_452[results_452['Significant'] == True]
    sig_710 = results_710[results_710['Significant'] == True]
    
    # If NEITHER dataset has significant associations, skip this plot
    if len(sig_452) == 0 and len(sig_710) == 0:
        print(f"  {analysis_title}: No FDR-significant associations in either dataset - skipping plot")
        return
    
    # Use worst p-value among significant associations (or 0.001 if one dataset has none)
    threshold_452 = sig_452['p_value'].max() if len(sig_452) > 0 else 0.001
    threshold_710 = sig_710['p_value'].max() if len(sig_710) > 0 else 0.001
    
    # Include associations that meet threshold in EITHER dataset
    interesting_452 = results_452[results_452['p_value'] <= threshold_452]['key'].tolist()
    interesting_710 = results_710[results_710['p_value'] <= threshold_710]['key'].tolist()
    
    # Union of interesting associations
    all_interesting = sorted(set(interesting_452 + interesting_710))
    
    if len(all_interesting) == 0:
        print(f"  {analysis_title}: No associations meet criteria in either dataset")
        # Create empty placeholder
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.text(0.5, 0.5, f'No FDR-significant associations\nin either dataset',
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
        
        # Get p-values (now both datasets use unified p_value and p_fdr columns)
        if len(row_452) > 0:
            p_val_452 = row_452['p_value'].values[0]
            p_fdr_452 = row_452['p_fdr'].values[0]
        else:
            p_val_452 = 1.0
            p_fdr_452 = 1.0
            
        if len(row_710) > 0:
            p_val_710 = row_710['p_value'].values[0]
            p_fdr_710 = row_710['p_fdr'].values[0]
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
        })
    
    df = pd.DataFrame(comparison_data)
    # Sort by 710 dataset's p-value (most significant first, will be at top)
    df = df.sort_values('p_val_710', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    y_pos = np.arange(len(df))
    bar_height = 0.35
    
    # Calculate -log10(p) for bar lengths
    bar_452 = -np.log10(df['p_val_452'].clip(lower=1e-300))  # Avoid log(0)
    bar_710 = -np.log10(df['p_val_710'].clip(lower=1e-300))
    
    # Plot bars (452 on top, 710 on bottom)
    # Use light purple (old/past) and light green (new/fresh)
    bars_452 = ax.barh(y_pos + bar_height/2, bar_452, bar_height,
                       label='452 interviews', color='#B19CD9', edgecolor='black', linewidth=0.5)
    bars_710 = ax.barh(y_pos - bar_height/2, bar_710, bar_height,
                       label='710 interviews', color='#90EE90', edgecolor='black', linewidth=0.5)
    
    # Add significance markers at the end of bars
    for idx, row in df.iterrows():
        i = list(df.index).index(idx)
        
        sig_452 = get_significance_marker(row['p_fdr_452'])
        sig_710 = get_significance_marker(row['p_fdr_710'])
        
        if sig_452:
            text_pos = bar_452.iloc[i] + 0.2
            ax.text(text_pos, i + bar_height/2, sig_452,
                   va='center', ha='left', fontsize=9, fontweight='bold')
        
        if sig_710:
            text_pos = bar_710.iloc[i] + 0.2
            ax.text(text_pos, i - bar_height/2, sig_710,
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
    
    plt.subplots_adjust(left=0.35, right=0.95, top=0.92, bottom=0.08)
    plt.savefig(output_path, dpi=300)
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
    # Filtering: Include all FDR-significant + associations with similar p-values
    # Sorting: By 710 dataset p-value
    create_comparison_plot(
        'esa_koodit',
        'ESA Habitats × Themes',
        f'{output_dir}/comparison_esa.png'
    )
    
    create_comparison_plot(
        'bird_species_koodit',
        'Bird Species × Themes',
        f'{output_dir}/comparison_bird_species.png'
    )
    
    create_comparison_plot(
        'bird_presence_koodit',
        'Bird Presence × Themes',
        f'{output_dir}/comparison_bird_presence.png'
    )
    
    create_comparison_plot(
        'bird_groups_koodit',
        'Bird Groups × Themes',
        f'{output_dir}/comparison_bird_groups.png'
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
