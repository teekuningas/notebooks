# %% [markdown]
# # Comprehensive Interview Participation Analysis
# 
# This analysis examines user engagement with an optional audio interview feature 
# in a bird observation mobile application.
# 
# **Research Questions:**
# 1. **Adoption:** What is the participation rate among users?
# 2. **Event-Level:** What recording characteristics predict interview participation?
# 3. **User-Level:** What user characteristics predict interview frequency?

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

from utils_interview import (
    get_habitat_data, 
    load_interviews_from_listing,
    compute_user_habitat_profiles
)

# Configuration
DATE_START_FEATURE = '2025-06-11'
INPUT_DIR = 'inputs/bird-metadata'
LISTING_FILE = 'inputs/minio-listing/listing.txt'
TRANSCRIPT_FILE = 'inputs/minio-listing/transcript-listing.txt'
ESA_DIR = 'inputs/esa_worldcover'
OUTPUT_DIR = 'output/interview_comprehensive'
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Data Loading

# %%
def load_data():
    print("--- Loading Data ---")
    
    # Recordings
    print("Loading recordings...")
    df_recs = pd.read_csv(os.path.join(INPUT_DIR, 'recs_since_June25.csv'))
    df_recs['datetime'] = pd.to_datetime(df_recs['date'] + ' ' + df_recs['time'])
    df_recs = df_recs[df_recs['date'] >= DATE_START_FEATURE].copy()
    
    # Load interviews with valid transcripts
    print("Loading interviews (valid transcripts only)...")
    df_interviews = load_interviews_from_listing(TRANSCRIPT_FILE, interview_type='0')
    df_interviews = df_interviews.rename(columns={'has_interview_upload': 'has_interview'})
    print(f"Total Interviews: {len(df_interviews):,}")
    
    # Align temporal coverage: use only recordings from period when interviews exist
    df_recs_temp = df_recs[['rec_id', 'date']].copy()
    df_int_with_dates = df_interviews.merge(df_recs_temp, on='rec_id', how='inner')
    if len(df_int_with_dates) > 0:
        last_interview_date = df_int_with_dates['date'].max()
        print(f"Last interview date: {last_interview_date}")
        print(f"Filtering recordings to {last_interview_date}")
        df_recs = df_recs[df_recs['date'] <= last_interview_date].copy()
    
    # Deduplicate recordings
    n_before = len(df_recs)
    df_recs = df_recs.drop_duplicates(subset=['rec_id'], keep='first')
    n_after = len(df_recs)
    if n_before > n_after:
        print(f"WARNING: Removed {n_before - n_after:,} duplicate rec_ids")
    
    print(f"Recordings (deduplicated): {len(df_recs):,}")

    # Habitat data
    print("Loading habitat data...")
    cache_file = os.path.join(CACHE_DIR, 'habitat_sample.csv')
    df_habitat = get_habitat_data(df_recs, ESA_DIR, cache_file)
    print(f"Habitat data loaded: {len(df_habitat)} records")

    # User habitat profiles
    print("Generating user habitat profiles...")
    user_profiles = compute_user_habitat_profiles(df_habitat, df_recs)
    print("User profiles generated.")

    # Species
    print("Loading species data...")
    df_species = pd.read_csv(os.path.join(INPUT_DIR, 'species_ids_since_June25.csv'))
    
    return df_recs, df_interviews, df_habitat, df_species, user_profiles

df_recs, df_interviews, df_habitat, df_species, user_profiles = load_data()

# %% [markdown]
# ## 2. Feature Engineering
# 
# Construct predictor variables for statistical models.

# %%
print("\n--- Feature Engineering ---")

# Merge habitat data onto recordings
# Event analysis uses only recordings with habitat data to avoid confounding
# missing data with habitat effects
print("Merging data...")
df = df_habitat.merge(df_recs, on='rec_id', how='inner')
print(f"Analysis Dataset Size: {len(df):,}")

df = df.merge(df_interviews, on='rec_id', how='left')
df['has_interview'] = df['has_interview'].fillna(0).astype(int)
df['has_interview_upload'] = df['has_interview']

# Temporal features
launch_date = pd.to_datetime(DATE_START_FEATURE)
df['days_since_launch'] = (df['datetime'] - launch_date).dt.days
df['hour'] = df['datetime'].dt.hour

# User history features
# Compute on full dataset to get accurate user-level sequences 
print("Calculating user history on full dataset...")
df_full = df_recs.copy()
df_full = df_full.sort_values(['user', 'datetime'])

# Species recognition indicator
rec_ids_with_species = set(df_species['rec_id'])
df_full['has_species'] = df_full['rec_id'].isin(rec_ids_with_species).astype(int)

# User sequence features
df_full['recs_since_launch'] = df_full.groupby('user').cumcount() + 1

# Prior interview count (excludes current recording)
df_full_merged = df_full.merge(df_interviews, on='rec_id', how='left')
df_full_merged['has_interview'] = df_full_merged['has_interview'].fillna(0).astype(int)
df_full_merged['interviews_since_launch'] = df_full_merged.groupby('user')['has_interview'].cumsum().shift(1).fillna(0).astype(int)

# Merge history back to analysis dataset
history_cols = ['rec_id', 'has_species', 'recs_since_launch', 'interviews_since_launch']
df_history = df_full_merged[history_cols].drop_duplicates(subset=['rec_id'])
n_dups = len(df_full_merged[history_cols]) - len(df_history)
if n_dups > 0:
    print(f"WARNING: Removed {n_dups:,} duplicate rec_ids from history")

df = df.merge(df_history, on='rec_id', how='left')

# Final deduplication check
n_before_dedup = len(df)
df = df.drop_duplicates(subset=['rec_id'])
n_after_dedup = len(df)
if n_before_dedup > n_after_dedup:
    print(f"WARNING: Removed {n_before_dedup - n_after_dedup:,} duplicate rec_ids")

print("Feature engineering complete.")

# %% [markdown]
# ## 3. Descriptive Statistics

# %%
def generate_descriptive_stats(df_full, df_interviews):
    print("\n--- Descriptive Statistics ---")

    n_users = df_full['user'].nunique()
    n_recs = len(df_full)
    n_interviews = len(df_interviews)

    print(f"Time Period: {df_full['date'].min()} to {df_full['date'].max()}")
    print(f"Unique Users: {n_users:,}")
    print(f"Total Recordings: {n_recs:,}")
    print(f"Interviews (with valid transcripts): {n_interviews:,} ({n_interviews/n_recs*100:.3f}% of recordings)")

    # User Participation
    # We need to merge interview info onto full dataset to count users
    df_merged = df_full.merge(df_interviews, on='rec_id', how='left')
    df_merged['has_interview'] = df_merged['has_interview'].fillna(0)

    user_upload_counts = df_merged.groupby('user')['has_interview'].sum()
    users_with_interview = (user_upload_counts > 0).sum()

    print(f"\nUsers who interviewed: {users_with_interview:,} ({users_with_interview/n_users*100:.2f}% of all users)")

    # Distribution of interviews per participating user
    interviewers = user_upload_counts[user_upload_counts > 0]
    print("\nInterviews per participating user:")
    print(interviewers.describe().to_string())
    
    # Plot Histogram
    max_interviews = int(interviewers.max())
    plt.figure(figsize=(10, 6))
    plt.hist(interviewers, bins=range(1, max_interviews + 2), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    plt.title('Distribution of Interviews per User')
    plt.xlabel('Number of Interviews')
    plt.ylabel('Count of Users')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'dist_interviews_per_user.png'))
    plt.close()

    return interviewers

interviewers = generate_descriptive_stats(df_recs, df_interviews)

# %% [markdown]
# ## 4. Variable Distributions
#
# Plot distributions of all explanatory variables used in both analyses.

# %%
def plot_variable_distributions(df_event, df_full, df_species, user_profiles):
    """
    Plot distributions of predictor variables for event-level and user-level analyses.
    Uses log scale on y-axis for highly skewed distributions.
    """
    print("\n--- Variable Distributions ---")
    
    # Print Event-Level Variable Statistics
    print("\n** Event-Level Variables **")
    print(f"days_since_launch: min={df_event['days_since_launch'].min()}, max={df_event['days_since_launch'].max()}, mean={df_event['days_since_launch'].mean():.1f}")
    print(f"has_species: {(df_event['has_species']==1).sum():,} ({(df_event['has_species']==1).sum()/len(df_event)*100:.1f}%) have recognized species")
    print(f"recs_since_launch: min={df_event['recs_since_launch'].min()}, max={df_event['recs_since_launch'].max()}, median={df_event['recs_since_launch'].median():.0f}")
    print(f"interviews_since_launch: min={df_event['interviews_since_launch'].min()}, max={df_event['interviews_since_launch'].max()}, mean={df_event['interviews_since_launch'].mean():.2f}")
    print(f"hour: range={df_event['hour'].min()}-{df_event['hour'].max()}, median={df_event['hour'].median():.0f}h")
    
    # Print habitat statistics (only urban)
    if 'urban' in df_event.columns:
        print(f"urban: {df_event['urban'].sum():,} ({df_event['urban'].sum()/len(df_event)*100:.1f}%) recordings")

    # Event-Level Variables Plot (grid 3x2 for 6 variables)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('Event-Level Analysis: Variable Distributions', fontsize=16, y=0.995)

    # 1. days_since_launch
    ax = axes[0, 0]
    ax.hist(df_event['days_since_launch'], bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel('Days Since Launch')
    ax.set_ylabel('Count')
    ax.set_title('Days Since Launch\n(days_since_launch)')
    ax.grid(axis='y', alpha=0.3)

    # 2. has_species
    ax = axes[0, 1]
    has_species_counts = df_event['has_species'].value_counts().sort_index()
    ax.bar(has_species_counts.index, has_species_counts.values, color=['lightcoral', 'skyblue'], edgecolor='black')
    ax.set_xlabel('Has Species Recognized')
    ax.set_ylabel('Count')
    ax.set_title('Species Recognition\n(has_species)')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])
    ax.grid(axis='y', alpha=0.3)

    # 3. recs_since_launch
    ax = axes[1, 0]
    ax.hist(df_event['recs_since_launch'], bins=100, color='skyblue', edgecolor='black')
    ax.set_xlabel('Recording Number (Since Launch)')
    ax.set_ylabel('Count')
    ax.set_title('User Recording Rank\n(recs_since_launch, log-transformed in model)')
    ax.set_yscale('log')  # Highly skewed distribution - log y-axis to see full range
    ax.grid(axis='y', alpha=0.3)

    # 4. interviews_since_launch
    ax = axes[1, 1]
    max_interviews = int(df_event['interviews_since_launch'].max())
    ax.hist(df_event['interviews_since_launch'], bins=range(0, max_interviews + 2), 
            color='skyblue', edgecolor='black', align='left')
    ax.set_xlabel('Interviews Since Launch')
    ax.set_ylabel('Count')
    ax.set_title('User Interview Experience\n(interviews_since_launch)')
    ax.set_yscale('log')  # Extremely zero-inflated - log y-axis to see non-zero cases
    ax.grid(axis='y', alpha=0.3)

    # 5. hour
    ax = axes[2, 0]
    ax.hist(df_event['hour'], bins=24, color='skyblue', edgecolor='black', range=(0, 24))
    ax.set_xlabel('Hour of Day (UTC)')
    ax.set_ylabel('Count')
    ax.set_title('Recording Time\n(hour)')
    ax.grid(axis='y', alpha=0.3)

    # 6. Habitat: urban
    ax = axes[2, 1]
    if 'urban' in df_event.columns:
        urban_counts = df_event['urban'].value_counts().sort_index()
        ax.bar(urban_counts.index, urban_counts.values, color=['lightcoral', 'skyblue'], edgecolor='black')
        ax.set_xlabel('Urban Habitat')
        ax.set_ylabel('Count')
        ax.set_title('Urban Habitat Presence\n(urban)')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(OUTPUT_DIR, 'event_variable_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # User-Level Variables
    # First, prepare user-level dataset
    df_merged = df_full.merge(df_interviews, on='rec_id', how='left')
    df_merged['has_interview'] = df_merged['has_interview'].fillna(0)
    
    df_sp_meta = df_species.merge(df_merged[['rec_id', 'user', 'datetime']], on='rec_id')
    df_sp_meta = df_sp_meta.sort_values('datetime')
    discoveries = df_sp_meta.drop_duplicates(subset=['user', 'species'])
    user_discoveries = discoveries.groupby('user').size().reset_index(name='n_discoveries')
    
    user_stats = df_merged.groupby('user').agg({
        'rec_id': 'count',
        'has_interview': 'sum',
        'datetime': lambda x: (x.max() - x.min()).days
    }).rename(columns={
        'rec_id': 'total_recordings',
        'has_interview': 'n_uploads',
        'datetime': 'tenure_days'
    })
    
    user_stats = user_stats.merge(user_discoveries, on='user', how='left')
    user_stats['n_discoveries'] = user_stats['n_discoveries'].fillna(0)
    user_stats = user_stats.merge(user_profiles, on='user', how='left')
    user_stats['user_urban_ratio'] = user_stats['user_urban_ratio'].fillna(0)
    
    # Print User-Level Variable Statistics
    print("\n** User-Level Variables **")
    print(f"N users: {len(user_stats):,}")
    print(f"total_recordings: min={user_stats['total_recordings'].min()}, max={user_stats['total_recordings'].max()}, median={user_stats['total_recordings'].median():.0f}")
    print(f"n_discoveries: min={user_stats['n_discoveries'].min():.0f}, max={user_stats['n_discoveries'].max():.0f}, mean={user_stats['n_discoveries'].mean():.1f}")
    print(f"tenure_days: min={user_stats['tenure_days'].min():.0f}, max={user_stats['tenure_days'].max():.0f}, mean={user_stats['tenure_days'].mean():.1f}")
    print(f"user_urban_ratio: mean={user_stats['user_urban_ratio'].mean():.3f}, median={user_stats['user_urban_ratio'].median():.3f}")
    print(f"n_interviews (outcome): {(user_stats['n_uploads']==0).sum():,} ({(user_stats['n_uploads']==0).sum()/len(user_stats)*100:.1f}%) users with 0 interviews")
    
    # User-Level Variables (4 plots now - 2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('User-Level Analysis: Variable Distributions', fontsize=16, y=0.995)
    
    # 1. total_recordings
    ax = axes[0, 0]
    ax.hist(user_stats['total_recordings'], bins=100, color='skyblue', edgecolor='black')
    ax.set_xlabel('Total Recordings')
    ax.set_ylabel('Count of Users')
    ax.set_title('User Activity Level\n(total_recordings, log-transformed in model)')
    ax.set_yscale('log')  # Highly skewed (median=4, max=5036) - log y-axis to see full range
    ax.grid(axis='y', alpha=0.3)
    
    # 2. n_discoveries
    ax = axes[0, 1]
    ax.hist(user_stats['n_discoveries'], bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel('Number of Discoveries')
    ax.set_ylabel('Count of Users')
    ax.set_title('Species Discovery Count\n(n_discoveries)')
    ax.set_yscale('log')  # Highly skewed - log y-axis to see tail
    ax.grid(axis='y', alpha=0.3)
    
    # 3. tenure_days
    ax = axes[1, 0]
    ax.hist(user_stats['tenure_days'], bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel('Tenure (Days)')
    ax.set_ylabel('Count of Users')
    ax.set_title('User Tenure\n(tenure_days)')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. user_urban_ratio
    ax = axes[1, 1]
    ax.hist(user_stats['user_urban_ratio'], bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel('Urban Ratio')
    ax.set_ylabel('Count of Users')
    ax.set_title('User Urban Habitat Ratio\n(user_urban_ratio)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(OUTPUT_DIR, 'user_variable_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Variable distribution plots saved.")

plot_variable_distributions(df, df_recs, df_species, user_profiles)

# %% [markdown]
# ## 5. Event-Level Analysis
# 
# Logistic regression to predict interview participation at the recording level.

# %%
def run_event_analysis(df, target_col, label):
    print(f"\n--- Event Analysis: {label} ---")
    
    # Only urban is considered now
    possible_habitats = ['urban']
    valid_habs = []
    for c in possible_habitats:
        if c in df.columns:
            n_positive = df[c].sum()
            n_negative = len(df) - n_positive
            if n_positive >= 50 and n_negative >= 50:
                valid_habs.append(c)
                print(f"Including habitat: {c} ({n_positive} positive, {n_negative} negative)")
    
    hab_terms = " + ".join(valid_habs)
    
    # Model formula
    # Log transform for recs_since_launch captures diminishing marginal effects
    base_formula = f"{target_col} ~ days_since_launch + has_species + np.log(recs_since_launch) + interviews_since_launch"
    
    if hab_terms:
        formula = f"{base_formula} + {hab_terms}"
    else:
        formula = base_formula
    
    print(f"Formula: {formula}")
    
    # Multicollinearity check
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    df_clean = df.dropna(subset=['days_since_launch', 'has_species', 'recs_since_launch', 'interviews_since_launch'])
    
    X = df_clean[['days_since_launch', 'has_species', 'recs_since_launch', 'interviews_since_launch']].copy()
    X['log_recs'] = np.log(X['recs_since_launch'])
    X = X.drop(columns=['recs_since_launch'])
    X['Intercept'] = 1
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    
    print("\n--- Variance Inflation Factors (VIF) ---")
    print(vif_data)
    
    # Fit logistic regression
    model = smf.logit(formula=formula, data=df).fit(disp=0)
    
    # Extract Odds Ratios
    params = model.params
    conf = model.conf_int()
    conf['OR'] = np.exp(params)
    conf['Lower'] = np.exp(conf[0])
    conf['Upper'] = np.exp(conf[1])
    conf['p-value'] = model.pvalues
    
    results = conf[['OR', 'Lower', 'Upper', 'p-value']]
    print(results)
    
    return results, model.summary()

print("\nFitting logistic regression...")
res_interviews, event_model_summary = run_event_analysis(df, 'has_interview_upload', "Interviews")

def plot_odds_ratios(results):
    """Plot odds ratios with confidence intervals."""
    r = results.drop('Intercept')
    r = r[(r['OR'] > 1e-6) & (r['OR'] < 1e6)]  # Filter numerical issues
    
    if len(r) == 0:
        print("Warning: No plottable odds ratios")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y = np.arange(len(r))
    ax.barh(y, r['OR'], xerr=[r['OR']-r['Lower'], r['Upper']-r['OR']], 
            capsize=5, color='skyblue', edgecolor='black')
    
    ax.set_yticks(y)
    ax.set_yticklabels(r.index)
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Odds Ratio')
    ax.set_xscale('log')
    ax.set_title('Event-Level Predictors of Interview Participation')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'odds_ratios.png'))
    plt.close()

plot_odds_ratios(res_interviews)

# %% [markdown]
# ## 6. User-Level Analysis
# 
# Negative Binomial regression to model interview count per user.
# Includes log(total_recordings) as predictor to learn activity-interview relationship.

# %%
def run_user_analysis(df_full, df_interviews, df_species, user_profiles):
    print("\n--- User-Level Analysis ---")
    
    # Merge interview data
    df_merged = df_full.merge(df_interviews, on='rec_id', how='left')
    df_merged['has_interview'] = df_merged['has_interview'].fillna(0)
    
    if 'days_since_launch' not in df_merged.columns:
        launch_date = pd.to_datetime(DATE_START_FEATURE)
        df_merged['days_since_launch'] = (df_merged['datetime'] - launch_date).dt.days
    
    # Aggregate user-level statistics
    print("Aggregating user statistics...")
    df_merged = df_merged.sort_values(['user', 'datetime'])
    
    # Species discoveries per user
    df_sp_meta = df_species.merge(df_merged[['rec_id', 'user', 'datetime']], on='rec_id')
    df_sp_meta = df_sp_meta.sort_values('datetime')
    discoveries = df_sp_meta.drop_duplicates(subset=['user', 'species'])
    user_discoveries = discoveries.groupby('user').size().reset_index(name='n_discoveries')
    
    # Core user metrics
    user_stats = df_merged.groupby('user').agg({
        'rec_id': 'count',
        'has_interview': 'sum',
        'days_since_launch': lambda x: (x.max() - x.min())
    }).rename(columns={
        'rec_id': 'total_recordings',
        'has_interview': 'n_uploads',
        'days_since_launch': 'tenure_days'
    })
    
    user_stats = user_stats.merge(user_discoveries, on='user', how='left')
    user_stats['n_discoveries'] = user_stats['n_discoveries'].fillna(0)
    
    # Merge habitat profiles
    user_stats = user_stats.merge(user_profiles, on='user', how='left')
    user_stats['user_urban_ratio'] = user_stats['user_urban_ratio'].fillna(0)
    
    print(f"Analyzing {len(user_stats):,} users")
    
    # Model formula: include log(total_recordings) as predictor
    # Coefficient <1: Diminishing returns, >1: Accelerating engagement
    formula = "n_uploads ~ np.log(total_recordings) + n_discoveries + tenure_days + user_urban_ratio"
    
    print(f"Formula: {formula}")
    
    # Multicollinearity check
    print("\n--- Variance Inflation Factors ---")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_user = user_stats[['total_recordings', 'n_discoveries', 'tenure_days', 'user_urban_ratio']].copy()
    X_user['log_total_recordings'] = np.log(X_user['total_recordings'])
    X_user = X_user.drop(columns=['total_recordings'])
    X_user['Intercept'] = 1
    X_user_clean = X_user.dropna()
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_user_clean.columns
    vif_data["VIF"] = [variance_inflation_factor(X_user_clean.values, i) for i in range(len(X_user_clean.columns))]
    print(vif_data)
    
    # Fit model with log(total_recordings) as predictor
    model = smf.glm(formula=formula, data=user_stats, 
                    family=sm.families.NegativeBinomial()).fit()
        
    print(model.summary())
    
    # Create interpretation table showing effects over realistic ranges
    params = model.params.drop('Intercept')
    conf_int = model.conf_int().drop('Intercept')
    pvalues = model.pvalues.drop('Intercept')
    
    # Calculate interpretable effects
    effects_table = pd.DataFrame({
        'Variable': params.index,
        'Coefficient': params.values,
        'CI_lower': conf_int[0].values,
        'CI_upper': conf_int[1].values,
        'p-value': pvalues.values
    })
    
    # Add interpretation column (effect over typical range)
    interpretations = []
    for var, coef in zip(params.index, params.values):
        if 'log(total_recordings)' in var:
            # Doubling recordings effect
            effect = np.exp(coef * np.log(2)) - 1
            interpretations.append(f"Doubling recordings → {effect*100:+.0f}% change in count")
        elif var == 'n_discoveries':
            # Effect of 50 discoveries
            effect = np.exp(coef * 50) - 1
            interpretations.append(f"50 discoveries → {effect*100:+.0f}% change in count")
        elif var == 'tenure_days':
            # Effect of 100 days
            effect = np.exp(coef * 100) - 1
            interpretations.append(f"100 days tenure → {effect*100:+.0f}% change in count")
        elif var == 'user_urban_ratio':
            # Effect of going from rural (0) to urban (1)
            effect = np.exp(coef) - 1
            interpretations.append(f"0% urban→100% urban → {effect*100:+.0f}% change in count")
        else:
            interpretations.append("")
    
    effects_table['Interpretation'] = interpretations
    
    print("\n--- Interpretable Effects (Interview Count) ---")
    print(effects_table.to_string(index=False))
    
    # Save table as image for report
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Format table data
    table_data = []
    table_data.append(['Variable', 'Coefficient', '95% CI', 'p-value', 'Effect over typical range'])
    for _, row in effects_table.iterrows():
        table_data.append([
            row['Variable'],
            f"{row['Coefficient']:.3f}",
            f"[{row['CI_lower']:.3f}, {row['CI_upper']:.3f}]",
            f"{row['p-value']:.3f}" if row['p-value'] >= 0.001 else "<0.001",
            row['Interpretation']
        ])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.18, 0.12, 0.18, 0.10, 0.42])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('User-Level Analysis: Factors Influencing Interview Count',
              fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'user_coefficients.png'))
    plt.close()
    
    return model.summary()

user_model_summary = run_user_analysis(df_recs, df_interviews, df_species, user_profiles)

# %% [markdown]
# ## 7. User-Level Analysis: Interview Participation (Binary)
# 
# Logistic regression to predict whether a user does at least one interview.
# Uses the same user-level predictors as the count model.

# %%
def run_user_participation_analysis(df_full, df_interviews, df_species, user_profiles):
    print("\n--- User-Level Participation Analysis ---")
    
    # Merge interview data
    df_merged = df_full.merge(df_interviews, on='rec_id', how='left')
    df_merged['has_interview'] = df_merged['has_interview'].fillna(0)
    
    if 'days_since_launch' not in df_merged.columns:
        launch_date = pd.to_datetime(DATE_START_FEATURE)
        df_merged['days_since_launch'] = (df_merged['datetime'] - launch_date).dt.days
    
    # Aggregate user-level statistics
    print("Aggregating user statistics...")
    df_merged = df_merged.sort_values(['user', 'datetime'])
    
    # Species discoveries per user
    df_sp_meta = df_species.merge(df_merged[['rec_id', 'user', 'datetime']], on='rec_id')
    df_sp_meta = df_sp_meta.sort_values('datetime')
    discoveries = df_sp_meta.drop_duplicates(subset=['user', 'species'])
    user_discoveries = discoveries.groupby('user').size().reset_index(name='n_discoveries')
    
    # Core user metrics
    user_stats = df_merged.groupby('user').agg({
        'rec_id': 'count',
        'has_interview': 'sum',
        'days_since_launch': lambda x: (x.max() - x.min())
    }).rename(columns={
        'rec_id': 'total_recordings',
        'has_interview': 'n_uploads',
        'days_since_launch': 'tenure_days'
    })
    
    user_stats = user_stats.merge(user_discoveries, on='user', how='left')
    user_stats['n_discoveries'] = user_stats['n_discoveries'].fillna(0)
    
    # Merge habitat profiles
    user_stats = user_stats.merge(user_profiles, on='user', how='left')
    user_stats['user_urban_ratio'] = user_stats['user_urban_ratio'].fillna(0)
    
    # Create binary outcome: at least one interview
    user_stats['has_any_interview'] = (user_stats['n_uploads'] > 0).astype(int)
    
    print(f"Analyzing {len(user_stats):,} users")
    print(f"Users with at least one interview: {user_stats['has_any_interview'].sum():,} ({user_stats['has_any_interview'].mean()*100:.1f}%)")
    
    # Model formula: same predictors as count model
    formula = "has_any_interview ~ np.log(total_recordings) + n_discoveries + tenure_days + user_urban_ratio"
    
    print(f"Formula: {formula}")
    
    # Multicollinearity check
    print("\n--- Variance Inflation Factors ---")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_user = user_stats[['total_recordings', 'n_discoveries', 'tenure_days', 'user_urban_ratio']].copy()
    X_user['log_total_recordings'] = np.log(X_user['total_recordings'])
    X_user = X_user.drop(columns=['total_recordings'])
    X_user['Intercept'] = 1
    X_user_clean = X_user.dropna()
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_user_clean.columns
    vif_data["VIF"] = [variance_inflation_factor(X_user_clean.values, i) for i in range(len(X_user_clean.columns))]
    print(vif_data)
    
    # Fit logistic regression
    model = smf.logit(formula=formula, data=user_stats).fit(disp=0)
    
    print(model.summary())
    
    # Extract Odds Ratios
    params = model.params
    conf = model.conf_int()
    conf['OR'] = np.exp(params)
    conf['Lower'] = np.exp(conf[0])
    conf['Upper'] = np.exp(conf[1])
    conf['p-value'] = model.pvalues
    
    results = conf[['OR', 'Lower', 'Upper', 'p-value']]
    print("\n--- Odds Ratios ---")
    print(results)
    
    # Create interpretation table
    params_no_int = model.params.drop('Intercept')
    conf_int = model.conf_int().drop('Intercept')
    pvalues = model.pvalues.drop('Intercept')
    
    effects_table = pd.DataFrame({
        'Variable': params_no_int.index,
        'Coefficient': params_no_int.values,
        'OR': np.exp(params_no_int.values),
        'CI_lower': np.exp(conf_int[0].values),
        'CI_upper': np.exp(conf_int[1].values),
        'p-value': pvalues.values
    })
    
    # Add interpretation column
    interpretations = []
    for var, coef in zip(params_no_int.index, params_no_int.values):
        if 'log(total_recordings)' in var:
            # Doubling recordings effect on odds
            or_double = np.exp(coef * np.log(2))
            interpretations.append(f"Doubling recordings → {or_double:.2f}× odds")
        elif var == 'n_discoveries':
            # Effect of 50 discoveries
            or_50 = np.exp(coef * 50)
            interpretations.append(f"50 discoveries → {or_50:.2f}× odds")
        elif var == 'tenure_days':
            # Effect of 100 days
            or_100 = np.exp(coef * 100)
            interpretations.append(f"100 days tenure → {or_100:.2f}× odds")
        elif var == 'user_urban_ratio':
            # Effect of going from rural to urban
            or_urban = np.exp(coef)
            interpretations.append(f"0% urban→100% urban → {or_urban:.2f}× odds")
        else:
            interpretations.append("")
    
    effects_table['Interpretation'] = interpretations
    
    print("\n--- Interpretable Effects (Participation Odds) ---")
    print(effects_table.to_string(index=False))
    
    # Save table as image
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Variable', 'Coefficient', 'Odds Ratio', '95% CI', 'p-value', 'Effect over typical range'])
    for _, row in effects_table.iterrows():
        table_data.append([
            row['Variable'],
            f"{row['Coefficient']:.3f}",
            f"{row['OR']:.3f}",
            f"[{row['CI_lower']:.3f}, {row['CI_upper']:.3f}]",
            f"{row['p-value']:.3f}" if row['p-value'] >= 0.001 else "<0.001",
            row['Interpretation']
        ])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.18, 0.10, 0.10, 0.18, 0.08, 0.36])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('User-Level Analysis: Factors Influencing Interview Participation',
              fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'user_participation_coefficients.png'))
    plt.close()
    
    # Plot odds ratios
    r = results.drop('Intercept')
    r = r[(r['OR'] > 1e-6) & (r['OR'] < 1e6)]
    
    if len(r) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y = np.arange(len(r))
        ax.barh(y, r['OR'], xerr=[r['OR']-r['Lower'], r['Upper']-r['OR']], 
                capsize=5, color='skyblue', edgecolor='black')
        
        ax.set_yticks(y)
        ax.set_yticklabels(r.index)
        ax.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Odds Ratio')
        ax.set_xscale('log')
        ax.set_title('User-Level Predictors of Interview Participation (Binary)')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'user_participation_odds_ratios.png'))
        plt.close()
    
    return results, model.summary()

participation_results, participation_model_summary = run_user_participation_analysis(df_recs, df_interviews, df_species, user_profiles)

# %% [markdown]
# ## 8. Report Generation

# %%
def generate_markdown_report(df_recs, df_interviews, res_interviews, event_model_summary, 
                            user_model_summary, participation_results, participation_model_summary):
    """Generate comprehensive markdown report with tables and figures."""
    report_path = os.path.join(OUTPUT_DIR, 'interview_analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Interview Participation Analysis\n\n")
        
        f.write("## 1. Data Summary\n\n")
        f.write(f"* **Period:** {df_recs['date'].min()} to {df_recs['date'].max()}\n")
        f.write(f"* **Recordings:** {len(df_recs):,}\n")
        f.write(f"* **Users:** {df_recs['user'].nunique():,}\n")
        f.write(f"* **Interviews:** {len(df_interviews):,} ({len(df_interviews)/len(df_recs)*100:.2f}%)\n\n")
        
        f.write("![Distribution](dist_interviews_per_user.png)\n\n")
        
        f.write("\\clearpage\n\n")
        
        f.write("## 2. Event-Level Analysis\n\n")
        f.write("Logistic regression predicting interview participation from recording characteristics.\n\n")
        
        f.write("### Variables\n\n")
        f.write("* `days_since_launch` - Days since feature launch\n")
        f.write("* `has_species` - Species recognition indicator\n")
        f.write("* `recs_since_launch` - User's recording count since launch (log-transformed)\n")
        f.write("* `interviews_since_launch` - User's prior interview count\n")
        f.write("* `urban` - Urban habitat indicator (ESA WorldCover)\n\n")
        
        f.write("![Event Variables](event_variable_distributions.png)\n\n")
        
        f.write("\\clearpage\n\n")
        
        f.write("### Model Summary\n\n")
        f.write("```\n")
        f.write(str(event_model_summary))
        f.write("\n```\n\n")

        f.write("### Odds Ratios\n\n")
        f.write("| Factor | Odds Ratio | 95% CI | p-value |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        for idx in res_interviews.index:
            if idx == 'Intercept': continue
            r = res_interviews.loc[idx]
            name = idx.replace('np.log(', '').replace(')', '')
            f.write(f"| {name} | {r['OR']:.2f} | [{r['Lower']:.2f}, {r['Upper']:.2f}] | {r['p-value']:.3f} |\n")
            
        f.write("\n")
        
        f.write("## 3. User-Level Analysis: Interview Count\n\n")
        f.write("Negative Binomial regression predicting interview count from user characteristics.\n\n")
        
        f.write("### Variables\n\n")
        f.write("* `total_recordings` - Total recording count (log-transformed)\n")
        f.write("* `n_discoveries` - Unique species count\n")
        f.write("* `tenure_days` - Activity span\n")
        f.write("* `user_urban_ratio` - Urban habitat preference\n\n")
        
        f.write("![User Variables](user_variable_distributions.png)\n\n")
        
        f.write("\\clearpage\n\n")
        
        f.write("### Model Summary\n\n")
        f.write("```\n")
        f.write(str(user_model_summary))
        f.write("\n```\n\n")
        
        f.write("## 4. User-Level Analysis: Participation (Binary)\n\n")
        f.write("Logistic regression predicting whether a user does at least one interview.\n\n")
        f.write("Uses the same predictors as the count model above.\n\n")
        
        f.write("### Model Summary\n\n")
        f.write("```\n")
        f.write(str(participation_model_summary))
        f.write("\n```\n\n")

        f.write("### Odds Ratios\n\n")
        f.write("| Factor | Odds Ratio | 95% CI | p-value |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        for idx in participation_results.index:
            if idx == 'Intercept': continue
            r = participation_results.loc[idx]
            name = idx.replace('np.log(', '').replace(')', '')
            f.write(f"| {name} | {r['OR']:.2f} | [{r['Lower']:.2f}, {r['Upper']:.2f}] | {r['p-value']:.3f} |\n")
        
    print(f"Report generated: {report_path}")

generate_markdown_report(df_recs, df_interviews, res_interviews, event_model_summary, 
                        user_model_summary, participation_results, participation_model_summary)

# %%
print("\n--- Analysis Complete ---")
print(f"Plots saved to {OUTPUT_DIR}/")
