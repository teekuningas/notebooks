# %% [markdown]
# # Comprehensive Interview Participation Report
# 
# This report analyses the user engagement with the optional "Interview" feature introduced around June 11, 2025.
# 
# **Key Questions:**
# 1. **Adoption:** How many users participate? What is the conversion rate?
# 2. **Data Quality:** How many interviews yield valid transcripts vs. empty audio?
# 3. **Event Analysis:** *When* does a user choose to interview? (Contextual factors)
# 4. **User Analysis:** *Who* chooses to interview? (User profile factors)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime
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
    
    # Load interviews (only those with valid transcripts ~795)
    print("Loading interviews (valid transcripts only)...")
    df_interviews = load_interviews_from_listing(TRANSCRIPT_FILE, interview_type='0')
    df_interviews = df_interviews.rename(columns={'has_interview_upload': 'has_interview'})
    print(f"Total Interviews: {len(df_interviews):,}")
    
    # Find last interview date to filter recordings
    df_recs_temp = df_recs[['rec_id', 'date']].copy()
    df_int_with_dates = df_interviews.merge(df_recs_temp, on='rec_id', how='inner')
    if len(df_int_with_dates) > 0:
        last_interview_date = df_int_with_dates['date'].max()
        print(f"Last interview date: {last_interview_date}")
        print(f"Filtering recordings to {last_interview_date} (interview feature availability)")
        df_recs = df_recs[df_recs['date'] <= last_interview_date].copy()
    
    # Deduplicate recordings (keep first occurrence)
    n_before = len(df_recs)
    df_recs = df_recs.drop_duplicates(subset=['rec_id'], keep='first')
    n_after = len(df_recs)
    if n_before > n_after:
        print(f"WARNING: Removed {n_before - n_after:,} duplicate rec_ids from recordings")
    
    print(f"Recordings (post-launch, deduplicated, feature-available): {len(df_recs):,}")

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
# We construct variables to capture:
# *   **Novelty:** `days_since_launch`
# *   **Context:** `hour`, `urban`, `forest`

# %%
print("\n--- Feature Engineering ---")

# Merge everything onto recordings
# IMPORTANT: For Event Analysis, we only use the subset where we have habitat data!
# Otherwise we are comparing "Habitat=0 (missing)" vs "Habitat=1 (present)" which is wrong.
# So we filter df to only those in df_habitat.

print("Merging data...")
df = df_habitat.merge(df_recs, on='rec_id', how='inner')
print(f"Analysis Dataset Size (Sampled): {len(df)}")

df = df.merge(df_interviews, on='rec_id', how='left')

# Fill NaNs for interview labels (since we merged left onto the sample)
df['has_interview'] = df['has_interview'].fillna(0).astype(int)

# Rename for compatibility with analysis code
df['has_interview_upload'] = df['has_interview']

# 1. Temporal Features
launch_date = pd.to_datetime(DATE_START_FEATURE)
df['days_since_launch'] = (df['datetime'] - launch_date).dt.days
df['hour'] = df['datetime'].dt.hour

# 2. User History Features
# We need history for ALL users in the sample.
# But to compute history correctly, we need the FULL recording history of those users, 
# not just the sampled rows.
# So we compute history on df_recs FIRST, then merge it in.

print("Calculating User History on full dataset...")
df_full = df_recs.copy()
df_full = df_full.sort_values(['user', 'datetime'])

# A. Species Recognition
# Has Species? (Did the app recognize anything?)
rec_ids_with_species = set(df_species['rec_id'])
df_full['has_species'] = df_full['rec_id'].isin(rec_ids_with_species).astype(int)

# B. User-Centric Sequence Features
# Rank: 1st, 2nd, 3rd recording for this user IN THIS PERIOD (since launch)
# NOTE: This is NOT the user's lifetime rank, but rank since the feature became available.
df_full['recs_since_launch'] = df_full.groupby('user').cumcount() + 1

# Interviews So Far: How many interviews has this user done BEFORE this recording?
# We need to merge interview info onto df_full first to calculate this
df_full_merged = df_full.merge(df_interviews, on='rec_id', how='left')
df_full_merged['has_interview'] = df_full_merged['has_interview'].fillna(0).astype(int)

# Shift to get "previous" count (don't count current interview)
df_full_merged['interviews_since_launch'] = df_full_merged.groupby('user')['has_interview'].cumsum().shift(1).fillna(0).astype(int)

# Now merge history features back to our sampled dataset
# We need to be careful to merge from df_full_merged which has the new cols
history_cols = ['rec_id', 'has_species', 'recs_since_launch', 'interviews_since_launch']

# Ensure no duplicates in history features (safety check)
df_history = df_full_merged[history_cols].drop_duplicates(subset=['rec_id'])
n_dups = len(df_full_merged[history_cols]) - len(df_history)
if n_dups > 0:
    print(f"WARNING: Removed {n_dups:,} duplicate rec_ids from history features")

df = df.merge(df_history, on='rec_id', how='left')

# Final safety check: ensure df itself has no duplicates
n_before_dedup = len(df)
df = df.drop_duplicates(subset=['rec_id'])
n_after_dedup = len(df)
if n_before_dedup > n_after_dedup:
    print(f"WARNING: Removed {n_before_dedup - n_after_dedup:,} duplicate rec_ids from final analysis dataset")

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
    Plot distributions of all variables used in the analysis.
    
    Visualization principles:
    - X-axis: Always show raw values (never log-transformed)
    - Y-axis: Use log scale when histogram counts span ≥3 orders of magnitude
      This makes highly skewed distributions readable while preserving actual values on x-axis
    """
    print("\n--- Variable Distributions ---")
    
    # Print Event-Level Variable Statistics
    print("\n** Event-Level Variables **")
    print(f"days_since_launch: min={df_event['days_since_launch'].min()}, max={df_event['days_since_launch'].max()}, mean={df_event['days_since_launch'].mean():.1f}")
    print(f"has_species: {(df_event['has_species']==1).sum():,} ({(df_event['has_species']==1).sum()/len(df_event)*100:.1f}%) have recognized species")
    print(f"recs_since_launch: min={df_event['recs_since_launch'].min()}, max={df_event['recs_since_launch'].max()}, median={df_event['recs_since_launch'].median():.0f}")
    print(f"interviews_since_launch: min={df_event['interviews_since_launch'].min()}, max={df_event['interviews_since_launch'].max()}, mean={df_event['interviews_since_launch'].mean():.2f}")
    print(f"hour: range={df_event['hour'].min()}-{df_event['hour'].max()}, median={df_event['hour'].median():.0f}h")
    
    # Print habitat statistics - only those used in the model
    for hab in ['forest', 'water', 'urban', 'grassland']:
        if hab in df_event.columns:
            print(f"{hab}: {df_event[hab].sum():,} ({df_event[hab].sum()/len(df_event)*100:.1f}%) recordings")

    # Event-Level Variables Plot
    fig, axes = plt.subplots(5, 2, figsize=(12, 18))
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

    # 6. Habitat: forest
    ax = axes[2, 1]
    if 'forest' in df_event.columns:
        forest_counts = df_event['forest'].value_counts().sort_index()
        ax.bar(forest_counts.index, forest_counts.values, color=['lightcoral', 'skyblue'], edgecolor='black')
        ax.set_xlabel('Forest Habitat')
        ax.set_ylabel('Count')
        ax.set_title('Forest Habitat Presence\n(forest)')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.grid(axis='y', alpha=0.3)

    # 7. Habitat: water
    ax = axes[3, 0]
    if 'water' in df_event.columns:
        water_counts = df_event['water'].value_counts().sort_index()
        ax.bar(water_counts.index, water_counts.values, color=['lightcoral', 'skyblue'], edgecolor='black')
        ax.set_xlabel('Water Habitat')
        ax.set_ylabel('Count')
        ax.set_title('Water Habitat Presence\n(water)')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.grid(axis='y', alpha=0.3)

    # 8. Habitat: urban
    ax = axes[3, 1]
    if 'urban' in df_event.columns:
        urban_counts = df_event['urban'].value_counts().sort_index()
        ax.bar(urban_counts.index, urban_counts.values, color=['lightcoral', 'skyblue'], edgecolor='black')
        ax.set_xlabel('Urban Habitat')
        ax.set_ylabel('Count')
        ax.set_title('Urban Habitat Presence\n(urban)')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.grid(axis='y', alpha=0.3)

    # 9. Habitat: grassland
    ax = axes[4, 0]
    if 'grassland' in df_event.columns:
        grassland_counts = df_event['grassland'].value_counts().sort_index()
        ax.bar(grassland_counts.index, grassland_counts.values, color=['lightcoral', 'skyblue'], edgecolor='black')
        ax.set_xlabel('Grassland Habitat')
        ax.set_ylabel('Count')
        ax.set_title('Grassland Habitat Presence\n(grassland)')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.grid(axis='y', alpha=0.3)
    
    # Hide the unused subplot
    axes[4, 1].axis('off')
    
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
    user_stats['user_forest_ratio'] = user_stats['user_forest_ratio'].fillna(0)
    
    # Print User-Level Variable Statistics
    print("\n** User-Level Variables **")
    print(f"N users: {len(user_stats):,}")
    print(f"total_recordings: min={user_stats['total_recordings'].min()}, max={user_stats['total_recordings'].max()}, median={user_stats['total_recordings'].median():.0f}")
    print(f"n_discoveries: min={user_stats['n_discoveries'].min():.0f}, max={user_stats['n_discoveries'].max():.0f}, mean={user_stats['n_discoveries'].mean():.1f}")
    print(f"tenure_days: min={user_stats['tenure_days'].min():.0f}, max={user_stats['tenure_days'].max():.0f}, mean={user_stats['tenure_days'].mean():.1f}")
    print(f"user_urban_ratio: mean={user_stats['user_urban_ratio'].mean():.3f}, median={user_stats['user_urban_ratio'].median():.3f}")
    print(f"user_forest_ratio: mean={user_stats['user_forest_ratio'].mean():.3f}, median={user_stats['user_forest_ratio'].median():.3f}")
    print(f"n_interviews (outcome): {(user_stats['n_uploads']==0).sum():,} ({(user_stats['n_uploads']==0).sum()/len(user_stats)*100:.1f}%) users with 0 interviews")
    
    # User-Level Variables (5 plots, no outcome)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
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
    ax.set_xlabel('Urban Ratio (0=Rural, 1=Urban)')
    ax.set_ylabel('Count of Users')
    ax.set_title('User Urban Habitat Ratio\n(user_urban_ratio)')
    ax.grid(axis='y', alpha=0.3)
    
    # 5. user_forest_ratio
    ax = axes[2, 0]
    ax.hist(user_stats['user_forest_ratio'], bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel('Forest Ratio')
    ax.set_ylabel('Count of Users')
    ax.set_title('User Forest Habitat Ratio\n(user_forest_ratio)')
    ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplot
    axes[2, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(OUTPUT_DIR, 'user_variable_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Variable distribution plots saved.")

plot_variable_distributions(df, df_recs, df_species, user_profiles)

# %% [markdown]
# ## 5. Analysis A: Event-Level (When?)
# 
# We use Logistic Regression to predict `has_interview_upload` (interviews with valid transcripts).

# %%
def run_event_analysis(df, target_col, label):
    print(f"\n--- Event Analysis: {label} ---")
    
    # Formula
    # We include urban/forest/wetland/water if they exist
    # Check for perfect separation or low variance in habitat columns
    # Use the 4 most common habitat types to avoid numerical instability
    # with rare habitats (based on data exploration)
    possible_habitats = ['forest', 'water', 'urban', 'grassland']
    valid_habs = []
    for c in possible_habitats:
        if c in df.columns:
            n_positive = df[c].sum()
            n_negative = len(df) - n_positive
            if n_positive >= 50 and n_negative >= 50:
                valid_habs.append(c)
                print(f"Including habitat: {c} ({n_positive} positive, {n_negative} negative)")
    
    hab_terms = " + ".join(valid_habs)
    
    # Formula construction:
    # Log transforms are used for variables with diminishing marginal effects:
    # - np.log(recs_since_launch): Effect of 1st→2nd recording differs from 100th→101st
    #   The novelty/learning effect diminishes with experience
    # Other variables are used in raw form as they have linear relationships with outcome
    base_formula = f"{target_col} ~ days_since_launch + has_species + np.log(recs_since_launch) + interviews_since_launch"
    
    if hab_terms:
        formula = f"{base_formula} + {hab_terms}"
    else:
        formula = base_formula
    
    print(f"Formula: {formula}")
    
    # Check Collinearity (VIF)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Create design matrix
    # We need to drop NAs first
    df_clean = df.dropna(subset=['days_since_launch', 'has_species', 'recs_since_launch', 'interviews_since_launch'])
    
    # Construct a temporary dataframe for VIF
    X = df_clean[['days_since_launch', 'has_species', 'recs_since_launch', 'interviews_since_launch']]
    X['log_recs'] = np.log(X['recs_since_launch'])
    X = X.drop(columns=['recs_since_launch'])
    X['Intercept'] = 1
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    
    print("\n--- Variance Inflation Factors (VIF) ---")
    print(vif_data)
    
    # Logistic Regression
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
    
    return results

print("\nRunning Logit for Interviews...")
res_interviews = run_event_analysis(df, 'has_interview_upload', "Interviews")

# Create simple odds ratio plot
def plot_odds_ratios(results):
    r = results.drop('Intercept')
    
    # Filter out extreme values that cause plotting issues
    # (These indicate near-perfect separation or numerical issues)
    r = r[(r['OR'] > 1e-6) & (r['OR'] < 1e6)]
    
    if len(r) == 0:
        print("Warning: No plottable odds ratios (all extreme values)")
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
    ax.set_title('Factors Influencing Interview Probability')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'odds_ratios.png'))
    plt.close()

plot_odds_ratios(res_interviews)

# %% [markdown]
# ## 6. Analysis B: User-Level (Who?)
# 
# We aggregate data by user and model the *count* of interviews.
# This handles the "super-user" issue (some give 20, some 1).

# %%
def run_user_analysis(df_full, df_interviews, df_species, user_profiles):
    print("\n--- User Analysis: Who Interviews? ---")
    
    # We need to aggregate from the FULL dataset, not the sampled one.
    # We now have user_profiles (Urban/Forest ratios) to include!
    
    # 1. Merge Interview Counts
    df_merged = df_full.merge(df_interviews, on='rec_id', how='left')
    df_merged['has_interview'] = df_merged['has_interview'].fillna(0)
    
    # Ensure days_since_launch is present
    if 'days_since_launch' not in df_merged.columns:
        launch_date = pd.to_datetime(DATE_START_FEATURE)
        df_merged['days_since_launch'] = (df_merged['datetime'] - launch_date).dt.days
    
    # 2. Calculate User Stats
    print("Aggregating user stats...")
    # Sort
    df_merged = df_merged.sort_values(['user', 'datetime'])
    
    # New Species Count
    # Merge species
    df_sp_meta = df_species.merge(df_merged[['rec_id', 'user', 'datetime']], on='rec_id')
    df_sp_meta = df_sp_meta.sort_values('datetime')
    discoveries = df_sp_meta.drop_duplicates(subset=['user', 'species'])
    user_discoveries = discoveries.groupby('user').size().reset_index(name='n_discoveries')
    
    # Base Aggregation
    user_stats = df_merged.groupby('user').agg({
        'rec_id': 'count',
        'has_interview': 'sum',
        'days_since_launch': lambda x: (x.max() - x.min()) # Tenure
    }).rename(columns={
        'rec_id': 'total_recordings',
        'has_interview': 'n_uploads',
        'days_since_launch': 'tenure_days'
    })
    
    # Merge discoveries
    user_stats = user_stats.merge(user_discoveries, on='user', how='left')
    user_stats['n_discoveries'] = user_stats['n_discoveries'].fillna(0)
    
    # Merge Habitat Profiles
    user_stats = user_stats.merge(user_profiles, on='user', how='left')
    # Fill missing profiles (users with <5 recs might be missing if we filtered them out in sampling, 
    # or if sampling failed. We'll fill with mean or 0)
    user_stats['user_urban_ratio'] = user_stats['user_urban_ratio'].fillna(0)
    user_stats['user_forest_ratio'] = user_stats['user_forest_ratio'].fillna(0)
    
    # Filter: We include ALL users, as we have habitat data for everyone.
    # We add a flag for 'is_active' to see if the behavior differs for frequent recorders.
    user_stats['is_active'] = (user_stats['total_recordings'] >= 5).astype(int)
    
    print(f"Analyzing {len(user_stats)} users (All).")
    
    # Formula construction for user-level count model with OFFSET:
    # We use total_recordings as an offset to model INTERVIEW RATE, not raw count
    # This controls for the mechanical correlation: more recordings → more opportunities to interview
    # Now we're asking: "Given the same activity level, what factors increase interview propensity?"
    # 
    # Other variables capture user engagement and context:
    # - n_discoveries: Species seeking behavior
    # - tenure_days: Long-term engagement
    # - user_urban_ratio, user_forest_ratio: Habitat context
    formula = "n_uploads ~ n_discoveries + tenure_days + user_urban_ratio + user_forest_ratio"
    
    print(f"Formula (with offset): {formula}")
    print(f"Offset: log(total_recordings)")
    
    # Add VIF check for multicollinearity
    print("\n--- Checking Multicollinearity (VIF) ---")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_user = user_stats[['n_discoveries', 'tenure_days', 'user_urban_ratio', 'user_forest_ratio']].copy()
    X_user['Intercept'] = 1
    X_user_clean = X_user.dropna()
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_user_clean.columns
    vif_data["VIF"] = [variance_inflation_factor(X_user_clean.values, i) for i in range(len(X_user_clean.columns))]
    print(vif_data)
    
    # Fit model with offset
    model = smf.glm(formula=formula, data=user_stats, 
                    family=sm.families.NegativeBinomial(),
                    offset=np.log(user_stats['total_recordings'])).fit()
        
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
        if var == 'n_discoveries':
            # Effect of 50 discoveries
            effect = np.exp(coef * 50) - 1
            interpretations.append(f"50 discoveries → {effect*100:+.0f}% change in rate")
        elif var == 'tenure_days':
            # Effect of 100 days
            effect = np.exp(coef * 100) - 1
            interpretations.append(f"100 days tenure → {effect*100:+.0f}% change in rate")
        elif var == 'user_urban_ratio':
            # Effect of going from rural (0) to urban (1)
            effect = np.exp(coef) - 1
            interpretations.append(f"Rural→Urban → {effect*100:+.0f}% change in rate")
        elif var == 'user_forest_ratio':
            # Effect of going from no forest to all forest
            effect = np.exp(coef) - 1
            interpretations.append(f"No forest→All forest → {effect*100:+.0f}% change in rate")
        else:
            interpretations.append("")
    
    effects_table['Interpretation'] = interpretations
    
    print("\n--- Interpretable Effects (Interview Rate) ---")
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
    
    plt.title('User-Level Analysis: Factors Influencing Interview Rate\n(Controlling for activity level via offset)',
              fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'user_coefficients.png'))
    plt.close()
    
    return model.summary()

user_model_summary = run_user_analysis(df_recs, df_interviews, df_species, user_profiles)

# Generate Report
def generate_markdown_report(df_recs, df_interviews, res_interviews, user_model_summary):
    report_path = os.path.join(OUTPUT_DIR, 'interview_analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Interview Analysis\n\n")
        
        f.write("## 1. Data\n")
        f.write(f"* **Period:** {df_recs['date'].min()} to {df_recs['date'].max()}\n")
        f.write(f"* **Recordings:** {len(df_recs):,}\n")
        f.write(f"* **Users:** {df_recs['user'].nunique():,}\n")
        f.write(f"* **Interviews:** {len(df_interviews):,} ({len(df_interviews)/len(df_recs)*100:.2f}%)\n\n")
        
        f.write("![Distribution](dist_interviews_per_user.png)\n\n")
        
        f.write("## 2. Variables\n\n")
        
        f.write("**Event-Level (Analysis A):**\n\n")
        f.write("* `days_since_launch` - Days since June 11, 2025\n\n")
        f.write("* `has_species` - Was any bird species recognized?\n\n")
        f.write("* `recs_since_launch` - User's recording count since launch\n\n")
        f.write("* `interviews_since_launch` - User's prior interview count\n\n")
        f.write("* `hour` - Hour of day (UTC)\n\n")
        f.write("* `forest`, `water`, `urban`, `grassland` - Habitat presence (ESA WorldCover)\n\n")
        
        f.write("**User-Level (Analysis B):**\n\n")
        f.write("* `total_recordings` - User's total recording count\n\n")
        f.write("* `n_discoveries` - Unique species discovered by user\n\n")
        f.write("* `tenure_days` - Days between user's first and last recording\n\n")
        f.write("* `user_urban_ratio` - Proportion of user's recordings in urban habitat\n\n")
        f.write("* `user_forest_ratio` - Proportion of user's recordings in forest habitat\n\n")
        
        f.write("![Event Variables](event_variable_distributions.png)\n\n")
        f.write("![User Variables](user_variable_distributions.png)\n\n")
        
        f.write("## 3. Analysis A: When?\n\n")
        f.write("*When do users interview?*\n\n")
        f.write("**Method:** Logistic regression\n\n")
        f.write("**Outcome:** Interview (0/1)\n\n")
        
        f.write("**Odds Ratios** (>1 = increased probability, <1 = decreased):\n\n")
        f.write("| Factor | Odds Ratio | 95% CI | p-value |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        # Create results table
        for idx in res_interviews.index:
            if idx == 'Intercept': continue
            r = res_interviews.loc[idx]
            name = idx.replace('np.log(', '').replace(')', '')
            f.write(f"| {name} | {r['OR']:.2f} | [{r['Lower']:.2f}, {r['Upper']:.2f}] | {r['p-value']:.3f} |\n")
            
        f.write("\n![Odds Ratios](odds_ratios.png)\n\n")
        
        f.write("## 4. Analysis B: Who?\n\n")
        f.write("*Who interviews more frequently?*\n\n")
        f.write("**Method:** Negative Binomial (offset: log recordings)\n\n")
        f.write("**Outcome:** Interview rate\n\n")
        
        f.write("**Model:**\n")
        f.write("```\n")
        f.write(str(user_model_summary))
        f.write("\n```\n\n")
        
        f.write("![User Effects Table](user_coefficients.png)\n")
        
    print(f"Report generated: {report_path}")

# %% [markdown]
# ## 7. Report Generation

# %%
generate_markdown_report(df_recs, df_interviews, res_interviews, user_model_summary)

# %%
print("\n--- Analysis Complete ---")
print(f"Plots saved to {OUTPUT_DIR}/")
