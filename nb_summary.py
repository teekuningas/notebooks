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

# %% [markdown]
# # Data Summary Notebook
#
# This notebook provides a summary of the interview data, including statistics on users and recordings.

# %%
import os
from collections import defaultdict
from utils import read_interview_data

# %% [markdown]
# ## Data Loading
#
# First, we load both "observation" (type 0) and "feedback" (type 1) recordings from the `data/birdinterview` directory.

# %%
# Load all data
observations = read_interview_data("data/birdinterview", "observation")
feedbacks = read_interview_data("data/birdinterview", "feedback")
all_data = observations + feedbacks

# %% [markdown]
# ## Summary Function
#
# This function calculates and prints summary statistics for a given dataset.

# %%
def print_summary(data, title):
    """
    Calculates and prints summary statistics for the given data.
    """
    print(f"--- {title} ---")

    if not data:
        print("No data to summarize.")
        print("-" * (len(title) + 8))
        return

    users = defaultdict(lambda: {'observations': [], 'feedback': []})
    for meta, content in data:
        user_id = meta['user_id']
        if meta['type'] == 'observation':
            users[user_id]['observations'].append((meta, content))
        elif meta['type'] == 'feedback':
            users[user_id]['feedback'].append((meta, content))

    num_total_users = len(users)
    print(f"Total number of transcripts: {len(data)}")
    print(f"Total number of unique users: {num_total_users}")

    users_with_feedback = sum(1 for user_id in users if users[user_id]['feedback'])
    print(f"Number of users who have given feedback: {users_with_feedback}")

    users_with_one_observation = sum(1 for user_id in users if len(users[user_id]['observations']) == 1)
    print(f"Number of users with exactly one observation: {users_with_one_observation}")

    if num_total_users > 0:
        total_observations = sum(len(user_data['observations']) for user_data in users.values())
        avg_observations_per_user = total_observations / num_total_users if num_total_users > 0 else 0
        print(f"Average number of observations per user: {avg_observations_per_user:.2f}")

    gaps = []
    for user_id, user_data in users.items():
        if user_data['observations'] and user_data['feedback']:
            # Sort observations by timestamp to find the earliest one
            first_observation = min(user_data['observations'], key=lambda x: int(x[0]['timestamp']))
            feedback_time = int(user_data['feedback'][0][0]['timestamp'])
            
            first_observation_time = int(first_observation[0]['timestamp'])
            
            gap = feedback_time - first_observation_time
            if gap >= 0:
                gaps.append(gap)

    if gaps:
        print("\n--- Time Gap Between First Observation and Feedback ---")
        min_gap = min(gaps)
        max_gap = max(gaps)
        avg_gap = sum(gaps) / len(gaps)

        def format_gap(seconds):
            days, rem = divmod(seconds, 86400)
            hours, rem = divmod(rem, 3600)
            minutes, seconds = divmod(rem, 60)
            return f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"

        print(f"Shortest gap: {format_gap(min_gap)}")
        print(f"Longest gap: {format_gap(max_gap)}")
        print(f"Average gap: {format_gap(avg_gap)}")
    else:
        print("\nNo users found with both observation and feedback to calculate gaps.")
    
    print("-" * (len(title) + 8))
    print()

# %% [markdown]
# ## Summary of All Data
#
# Here is the summary for all the loaded data without any filtering.

# %%
print_summary(all_data, "Summary of All Data")

# %% [markdown]
# ## Filtering
#
# Now, we filter the data. A recording is kept if it has at least 10 words and at least 10 unique words. This is based on the logic from `filter_interview_simple` in `utils.py`.

# %%
def filter_data_by_word_count(data, min_words=10, min_unique_words=10):
    """
    Filters data based on word count and unique word count.
    """
    filtered_data = []
    for metadata, content in data:
        stripped_text = content.strip()
        words = stripped_text.split()
        if len(words) >= min_words and len(set(words)) >= min_unique_words:
            filtered_data.append((metadata, content))
    return filtered_data

filtered_data = filter_data_by_word_count(all_data)

# %% [markdown]
# ## Summary of Filtered Data
#
# This summary is based on the data that has passed the word count filter.

# %%
print_summary(filtered_data, "Summary of Filtered Data")

# %%
