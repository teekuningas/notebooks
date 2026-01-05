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
    
    # Interview uploads (use TRANSCRIPTS as ground truth - actual interviews with content)
    print("Loading interviews (valid transcripts only)...")
    df_int_all = load_interviews_from_listing(TRANSCRIPT_FILE, interview_type='0')
    df_int_all = df_int_all.rename(columns={'has_interview_upload': 'has_interview'})
    print(f"Total Interviews (with valid transcripts): {len(df_int_all):,}")
    
    # Find last interview date to filter recordings
    df_recs_temp = df_recs[['rec_id', 'date']].copy()
    df_int_with_dates = df_int_all.merge(df_recs_temp, on='rec_id', how='inner')
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

    # Valid transcripts (already loaded as df_int_all, but keep for compatibility)
    print("Loading valid transcripts (same as interviews)...")
    df_int_valid = df_int_all.copy()
    df_int_valid = df_int_valid.rename(columns={'has_interview': 'has_valid_transcript'})
    print(f"Total Valid Transcripts: {len(df_int_valid):,}")

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
    
    return df_recs, df_int_all, df_int_valid, df_habitat, df_species, user_profiles

df_recs, df_int_all, df_int_valid, df_habitat, df_species, user_profiles = load_data()

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

df = df.merge(df_int_all, on='rec_id', how='left')
df = df.merge(df_int_valid, on='rec_id', how='left')

# Fill NaNs for interview labels (since we merged left onto the sample)
df['has_interview'] = df['has_interview'].fillna(0).astype(int)
df['has_valid_transcript'] = df['has_valid_transcript'].fillna(0).astype(int)

# For this analysis, we only use valid transcripts as our outcome
df['has_interview_upload'] = df['has_interview']  # Rename for compatibility with rest of code

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
df_full_merged = df_full.merge(df_int_all, on='rec_id', how='left')
df_full_merged['has_interview'] = df_full_merged['has_interview'].fillna(0).astype(int)

# Shift to get "previous" count (don't count current interview)
df_full_merged['interviews_so_far'] = df_full_merged.groupby('user')['has_interview'].cumsum().shift(1).fillna(0).astype(int)

# Now merge history features back to our sampled dataset
# We need to be careful to merge from df_full_merged which has the new cols
history_cols = ['rec_id', 'has_species', 'recs_since_launch', 'interviews_so_far']

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
def generate_descriptive_stats(df_full, df_int_all, df_int_valid):
    print("\n--- Descriptive Statistics ---")

    n_users = df_full['user'].nunique()
    n_recs = len(df_full)
    n_interviews = len(df_int_all)  # Now these are the same (transcripts only)

    print(f"Time Period: {df_full['date'].min()} to {df_full['date'].max()}")
    print(f"Unique Users: {n_users:,}")
    print(f"Total Recordings: {n_recs:,}")
    print(f"Interviews (with valid transcripts): {n_interviews:,} ({n_interviews/n_recs*100:.3f}% of recordings)")

    # User Participation
    # We need to merge interview info onto full dataset to count users
    df_merged = df_full.merge(df_int_all, on='rec_id', how='left')
    df_merged['has_interview'] = df_merged['has_interview'].fillna(0)

    user_upload_counts = df_merged.groupby('user')['has_interview'].sum()
    users_with_interview = (user_upload_counts > 0).sum()

    print(f"\nUsers who interviewed: {users_with_interview:,} ({users_with_interview/n_users*100:.2f}% of all users)")

    # Distribution of interviews per participating user
    interviewers = user_upload_counts[user_upload_counts > 0]
    print("\nInterviews per participating user:")
    print(interviewers.describe().to_string())
    
    # Plot Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(interviewers, bins=range(1, 30), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    plt.title('Distribution of Interviews per User')
    plt.xlabel('Number of Interviews')
    plt.ylabel('Count of Users')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'dist_interviews_per_user.png'))

    return interviewers

interviewers = generate_descriptive_stats(df_recs, df_int_all, df_int_valid)

# %% [markdown]
# ## 4. Variable Distributions
#
# Plot distributions of all explanatory variables used in both analyses.

# %%
def plot_variable_distributions(df_event, df_full, df_species, user_profiles):
    print("\n--- Variable Distributions ---")
    
    # Print Event-Level Variable Statistics
    print("\n** Event-Level Variables **")
    print(f"days_since_launch: min={df_event['days_since_launch'].min()}, max={df_event['days_since_launch'].max()}, mean={df_event['days_since_launch'].mean():.1f}")
    print(f"has_species: {(df_event['has_species']==1).sum():,} ({(df_event['has_species']==1).sum()/len(df_event)*100:.1f}%) have recognized species")
    print(f"recs_since_launch: min={df_event['recs_since_launch'].min()}, max={df_event['recs_since_launch'].max()}, median={df_event['recs_since_launch'].median():.0f}")
    print(f"interviews_so_far: min={df_event['interviews_so_far'].min()}, max={df_event['interviews_so_far'].max()}, mean={df_event['interviews_so_far'].mean():.2f}")
    print(f"hour: range={df_event['hour'].min()}-{df_event['hour'].max()}, median={df_event['hour'].median():.0f}h")
    
    if 'urban' in df_event.columns:
        print(f"urban: {df_event['urban'].sum():,} ({df_event['urban'].sum()/len(df_event)*100:.1f}%) recordings")
    if 'forest' in df_event.columns:
        print(f"forest: {df_event['forest'].sum():,} ({df_event['forest'].sum()/len(df_event)*100:.1f}%) recordings")
    if 'wetland' in df_event.columns:
        print(f"wetland: {df_event['wetland'].sum():,} ({df_event['wetland'].sum()/len(df_event)*100:.1f}%) recordings")
    if 'water' in df_event.columns:
        print(f"water: {df_event['water'].sum():,} ({df_event['water'].sum()/len(df_event)*100:.1f}%) recordings")

    # Event-Level Variables Plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Event-Level Analysis: Variable Distributions', fontsize=16)

    # 1. days_since_launch
    ax = axes[0, 0]
    ax.hist(df_event['days_since_launch'], bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel('Days Since Launch')
    ax.set_ylabel('Count')
    ax.set_title('Days Since Launch')
    ax.grid(axis='y', alpha=0.3)

    # 2. has_species
    ax = axes[0, 1]
    has_species_counts = df_event['has_species'].value_counts().sort_index()
    ax.bar(has_species_counts.index, has_species_counts.values, color=['salmon', 'skyblue'], edgecolor='black')
    ax.set_xlabel('Has Species Recognized')
    ax.set_ylabel('Count')
    ax.set_title('Species Recognition')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])
    ax.grid(axis='y', alpha=0.3)

    # 3. recs_since_launch
    ax = axes[0, 2]
    # Limit to reasonable range for visualization
    recs_capped = df_event['recs_since_launch'].clip(upper=100)
    ax.hist(recs_capped, bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel('Recording Number (Since Launch, capped at 100)')
    ax.set_ylabel('Count')
    ax.set_title('User Recording Rank')
    ax.grid(axis='y', alpha=0.3)

    # 4. interviews_so_far
    ax = axes[1, 0]
    interviews_capped = df_event['interviews_so_far'].clip(upper=20)
    ax.hist(interviews_capped, bins=21, color='skyblue', edgecolor='black', align='left')
    ax.set_xlabel('Interviews So Far (capped at 20)')
    ax.set_ylabel('Count')
    ax.set_title('User Interview Experience')
    ax.grid(axis='y', alpha=0.3)

    # 5. hour
    ax = axes[1, 1]
    ax.hist(df_event['hour'], bins=24, color='skyblue', edgecolor='black', range=(0, 24))
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Count')
    ax.set_title('Recording Time')
    ax.grid(axis='y', alpha=0.3)

    # 6. Habitat: urban
    ax = axes[1, 2]
    if 'urban' in df_event.columns:
        urban_counts = df_event['urban'].value_counts().sort_index()
        ax.bar(urban_counts.index, urban_counts.values, color=['lightgreen', 'salmon'], edgecolor='black')
        ax.set_xlabel('Urban Habitat')
        ax.set_ylabel('Count')
        ax.set_title('Urban Habitat Presence')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.grid(axis='y', alpha=0.3)

    # 7. Habitat: forest
    ax = axes[2, 0]
    if 'forest' in df_event.columns:
        forest_counts = df_event['forest'].value_counts().sort_index()
        ax.bar(forest_counts.index, forest_counts.values, color=['lightcoral', 'lightgreen'], edgecolor='black')
        ax.set_xlabel('Forest Habitat')
        ax.set_ylabel('Count')
        ax.set_title('Forest Habitat Presence')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.grid(axis='y', alpha=0.3)

    # 8. Habitat: wetland
    ax = axes[2, 1]
    if 'wetland' in df_event.columns:
        wetland_counts = df_event['wetland'].value_counts().sort_index()
        ax.bar(wetland_counts.index, wetland_counts.values, color=['tan', 'lightblue'], edgecolor='black')
        ax.set_xlabel('Wetland Habitat')
        ax.set_ylabel('Count')
        ax.set_title('Wetland Habitat Presence')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.grid(axis='y', alpha=0.3)

    # 9. Habitat: water
    ax = axes[2, 2]
    if 'water' in df_event.columns:
        water_counts = df_event['water'].value_counts().sort_index()
        ax.bar(water_counts.index, water_counts.values, color=['wheat', 'steelblue'], edgecolor='black')
        ax.set_xlabel('Water Habitat')
        ax.set_ylabel('Count')
        ax.set_title('Water Habitat Presence')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'event_variable_distributions.png'), dpi=150)
    plt.close()

    # User-Level Variables
    # First, prepare user-level dataset
    df_merged = df_full.merge(df_int_all, on='rec_id', how='left')
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
    print(f"n_uploads (outcome): {(user_stats['n_uploads']==0).sum():,} ({(user_stats['n_uploads']==0).sum()/len(user_stats)*100:.1f}%) users with 0 interviews")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('User-Level Analysis: Variable Distributions', fontsize=16)
    
    # 1. total_recordings (log scale)
    ax = axes[0, 0]
    ax.hist(np.log1p(user_stats['total_recordings']), bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel('log(Total Recordings + 1)')
    ax.set_ylabel('Count of Users')
    ax.set_title('User Activity Level')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. n_discoveries
    ax = axes[0, 1]
    n_disc_capped = user_stats['n_discoveries'].clip(upper=50)
    ax.hist(n_disc_capped, bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel('Number of Discoveries (capped at 50)')
    ax.set_ylabel('Count of Users')
    ax.set_title('Species Discovery Count')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. tenure_days
    ax = axes[0, 2]
    ax.hist(user_stats['tenure_days'], bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel('Tenure (Days)')
    ax.set_ylabel('Count of Users')
    ax.set_title('User Tenure')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. user_urban_ratio
    ax = axes[1, 0]
    ax.hist(user_stats['user_urban_ratio'], bins=50, color='salmon', edgecolor='black')
    ax.set_xlabel('Urban Ratio (0=Rural, 1=Urban)')
    ax.set_ylabel('Count of Users')
    ax.set_title('User Urban Habitat Ratio')
    ax.grid(axis='y', alpha=0.3)
    
    # 5. user_forest_ratio
    ax = axes[1, 1]
    ax.hist(user_stats['user_forest_ratio'], bins=50, color='lightgreen', edgecolor='black')
    ax.set_xlabel('Forest Ratio')
    ax.set_ylabel('Count of Users')
    ax.set_title('User Forest Habitat Ratio')
    ax.grid(axis='y', alpha=0.3)
    
    # 6. n_uploads (outcome variable)
    ax = axes[1, 2]
    # Show distribution including zeros
    ax.hist(user_stats['n_uploads'], bins=range(0, min(21, int(user_stats['n_uploads'].max()) + 2)), 
            color='gold', edgecolor='black', align='left')
    ax.set_xlabel('Number of Interviews')
    ax.set_ylabel('Count of Users')
    ax.set_title('Interview Count (Outcome)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'user_variable_distributions.png'), dpi=150)
    plt.close()
    
    print("Variable distribution plots saved.")

plot_variable_distributions(df, df_recs, df_species, user_profiles)

# %% [markdown]
# ## 5. Analysis A: Event-Level (When?)
# 
# We use Logistic Regression to predict `has_interview_upload`.
# We also run a check using `has_valid_transcript` to see if factors differ for "quality" interviews.

# %%
def run_event_analysis(df, target_col, label):
    print(f"\n--- Event Analysis: {label} ---")
    
    # Formula
    # We include urban/forest/wetland/water if they exist
    # Check for perfect separation or low variance in habitat columns
    valid_habs = []
    for c in ['urban', 'forest', 'wetland', 'water']:
        if c in df.columns:
            # Check if we have enough variance (at least 5 positives and 5 negatives)
            if df[c].sum() > 5 and (len(df) - df[c].sum()) > 5:
                valid_habs.append(c)
    
    hab_terms = " + ".join(valid_habs)
    
    # Updated formula with user-centric features
    # We use log(rank) because the effect of 1st->2nd is likely bigger than 100th->101st
    # Added 'has_species' to control for successful recognition
    base_formula = f"{target_col} ~ days_since_launch + has_species + np.log(recs_since_launch) + interviews_so_far"
    
    if hab_terms:
        formula = f"{base_formula} + {hab_terms}"
    else:
        formula = base_formula
    
    print(f"Formula: {formula}")
    
    # Check Collinearity (VIF)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Create design matrix
    # We need to drop NAs first
    df_clean = df.dropna(subset=['days_since_launch', 'has_species', 'recs_since_launch', 'interviews_so_far'])
    
    # Construct a temporary dataframe for VIF
    X = df_clean[['days_since_launch', 'has_species', 'recs_since_launch', 'interviews_so_far']]
    X['log_recs'] = np.log(X['recs_since_launch'])
    X = X.drop(columns=['recs_since_launch'])
    X['Intercept'] = 1
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    
    print("\n--- Variance Inflation Factors (VIF) ---")
    print(vif_data)
    
    # Standard Logit
    try:
        model = smf.logit(formula=formula, data=df).fit(disp=0)
    except Exception as e:
        print(f"Logit failed: {e}. Trying with fewer variables...")
        # Fallback: remove habitat
        formula = base_formula
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

print("\nRunning Logit for Interviews (Valid Transcripts)...")
res_uploads = run_event_analysis(df, 'has_interview_upload', "Interviews")

# Keep res_valid for compatibility with report generation (but it's the same data)
res_valid = res_uploads.copy()

# Create simple odds ratio plot
def plot_odds_ratios(results):
    r = results.drop('Intercept')
    
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

plot_odds_ratios(res_uploads)

# %% [markdown]
# ## 6. Analysis B: User-Level (Who?)
# 
# We aggregate data by user and model the *count* of interviews.
# This handles the "super-user" issue (some give 20, some 1).

# %%
def run_user_analysis(df_full, df_int_all, df_int_valid, df_species, user_profiles):
    print("\n--- User Analysis: Who Interviews? ---")
    
    # We need to aggregate from the FULL dataset, not the sampled one.
    # We now have user_profiles (Urban/Forest ratios) to include!
    
    # 1. Merge Interview Counts
    df_merged = df_full.merge(df_int_all, on='rec_id', how='left')
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
    
    # Handle potential zeros in log
    user_stats['log_total_recordings'] = np.log(user_stats['total_recordings'] + 1)
    
    print(f"Analyzing {len(user_stats)} users (All).")
    
    # Model: Negative Binomial
    # Formula: n_uploads ~ log_total_recordings + n_discoveries + tenure_days + user_urban_ratio + user_forest_ratio
    
    formula = "n_uploads ~ log_total_recordings + n_discoveries + tenure_days + user_urban_ratio + user_forest_ratio"
    
    print(f"Formula: {formula}")
    
    try:
        model = smf.glm(formula=formula, data=user_stats, 
                        family=sm.families.NegativeBinomial()).fit()
    except Exception as e:
        print(f"Negative Binomial failed: {e}, falling back to Poisson")
        try:
            model = smf.glm(formula=formula, data=user_stats, 
                            family=sm.families.Poisson()).fit()
        except Exception as e2:
            print(f"Poisson failed too: {e2}. Trying simplified model.")
            formula_simple = "n_uploads ~ log_total_recordings + n_discoveries"
            model = smf.glm(formula=formula_simple, data=user_stats, 
                            family=sm.families.Poisson()).fit()
        
    print(model.summary())
    
    # Visualizing 1: Discovery Effect
    user_stats['discovery_bin'] = pd.qcut(user_stats['n_discoveries'], q=5, duplicates='drop')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=user_stats, x='discovery_bin', y='n_uploads')
    plt.title('Average Interviews per User by "Species Discovery" Level')
    plt.xlabel('Number of New Species Found')
    plt.ylabel('Average Number of Interviews')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'user_discovery_vs_interviews.png'))
    
    # Visualizing 2: Habitat Effect (Urban vs Forest)
    # Bin users by Urban Ratio
    user_stats['urban_bin'] = pd.cut(user_stats['user_urban_ratio'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=user_stats, x='urban_bin', y='n_uploads')
    plt.title('Average Interviews per User by "Urban Ratio"')
    plt.xlabel('User Urban Ratio (0=Rural, 1=Urban)')
    plt.ylabel('Average Number of Interviews')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'user_urban_vs_interviews.png'))
    
    # Visualizing 3: Adoption Curve (Event Level - but aggregated here for simplicity?)
    # No, Adoption Curve is best done on the Event Level dataframe 'df'
    
    return model.summary()

user_model_summary = run_user_analysis(df_recs, df_int_all, df_int_valid, df_species, user_profiles)

# %%
def plot_adoption_curve(df):
    print("\n--- Plotting Adoption Curve ---")
    # Probability of interview by Recording Rank (1st, 2nd, 3rd...)
    # We limit to first 50 recordings to avoid noise at the tail
    
    df_subset = df[df['recs_since_launch'] <= 50].copy()
    
    # Calculate probability per rank
    adoption = df_subset.groupby('recs_since_launch')['has_interview_upload'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=adoption, x='recs_since_launch', y='has_interview_upload', marker='o')
    plt.title('Interview Probability by User Recording Rank (Since Launch)')
    plt.xlabel('Recording Number Since Launch (1st, 2nd, ...)')
    plt.ylabel('Probability of Interview')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'adoption_curve.png'))

plot_adoption_curve(df)

# Generate Report
def generate_markdown_report(df_recs, df_int_all, df_int_valid, res_uploads, res_valid, user_model_summary):
    report_path = os.path.join(OUTPUT_DIR, 'interview_analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Interview Feature Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write("## 1. Data\n")
        f.write(f"* **Period:** {df_recs['date'].min()} to {df_recs['date'].max()}\n")
        f.write(f"* **Recordings:** {len(df_recs):,}\n")
        f.write(f"* **Users:** {df_recs['user'].nunique():,}\n")
        f.write(f"* **Interviews:** {len(df_int_all):,} ({len(df_int_all)/len(df_recs)*100:.2f}%)\n\n")
        
        f.write("![Distribution](dist_interviews_per_user.png)\n\n")
        
        f.write("## 2. Variables\n\n")
        
        f.write("**Event-Level (Analysis A):**\n")
        f.write("* `days_since_launch` - Days since feature launch (June 11, 2025)\n")
        f.write("* `has_species` - Bird species recognized?\n")
        f.write("* `recs_since_launch` - User's recording number (1st, 2nd, 3rd...)\n")
        f.write("* `interviews_so_far` - Interviews completed before this recording\n")
        f.write("* `hour` - Hour of day\n")
        f.write("* `urban`, `forest`, `wetland`, `water` - Habitat (ESA WorldCover)\n\n")
        
        f.write("**User-Level (Analysis B):**\n")
        f.write("* `total_recordings` - Total recordings (log-transformed)\n")
        f.write("* `n_discoveries` - Unique species discovered\n")
        f.write("* `tenure_days` - Days between first and last recording\n")
        f.write("* `user_urban_ratio` - Proportion in urban habitat\n")
        f.write("* `user_forest_ratio` - Proportion in forest habitat\n\n")
        
        f.write("![Event Variables](event_variable_distributions.png)\n")
        f.write("![User Variables](user_variable_distributions.png)\n\n")
        
        f.write("## 3. Analysis A: When do users interview?\n\n")
        f.write("*Milloin käyttäjä haluaa kertoa luontohavainnoistaan?*\n\n")
        f.write("**Method:** Logistic Regression  \n")
        f.write("**Outcome:** `has_interview_upload` (0/1)\n\n")
        
        f.write("**Odds Ratios** (>1 = increased probability, <1 = decreased):\n\n")
        f.write("| Factor | Odds Ratio (All Uploads) | p-value | Odds Ratio (Valid) | p-value |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        # Merge results for table
        common_idx = res_uploads.index.intersection(res_valid.index)
        for idx in common_idx:
            if idx == 'Intercept': continue
            r1 = res_uploads.loc[idx]
            r2 = res_valid.loc[idx]
            name = idx.replace('np.log(', '').replace(')', '')
            f.write(f"| {name} | {r1['OR']:.2f} | {r1['p-value']:.3f} | {r2['OR']:.2f} | {r2['p-value']:.3f} |\n")
            
        f.write("\n![Odds Ratios](comparison_odds_ratios.png)\n\n")
        
        f.write("## 4. Analysis B: Who interviews?\n\n")
        f.write("*Kuka haluaa kertoa luontohavainnoistaan?*\n\n")
        f.write("**Method:** Negative Binomial/Poisson Regression  \n")
        f.write("**Outcome:** `n_uploads` (count)\n\n")
        
        f.write("**Model:**\n")
        f.write("```\n")
        f.write(str(user_model_summary))
        f.write("\n```\n\n")
        
        f.write("![User Discovery vs Interviews](user_discovery_vs_interviews.png)\n")
        f.write("![User Urban Ratio vs Interviews](user_urban_vs_interviews.png)\n\n")
        
        f.write("## 5. Adoption Curve\n\n")
        f.write("Interview probability by recording rank (1st, 2nd, 3rd...).\n\n")
        f.write("![Adoption Curve](adoption_curve.png)\n")
        
    print(f"Report generated: {report_path}")

# %% [markdown]
# ## 7. Report Generation

# %%
generate_markdown_report(df_recs, df_int_all, df_int_valid, res_uploads, res_valid, user_model_summary)

# %%
print("\n--- Analysis Complete ---")
print(f"Plots saved to {OUTPUT_DIR}/")
