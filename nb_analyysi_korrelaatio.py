# ---
# jupyter:
#   jupytext:
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
# # Correlational Analysis of Interview Data and Metadata
#
# This notebook aims to explore correlations between qualitative interview data and associated metadata.
# We will:
# 1. Load interview texts from the `data/luontokokemus_simulaatio` directory.
# 2. Load corresponding metadata from `data/luontokokemus_simulaatio/metadata.csv`.
# 3. Prepare the data for further analysis.

# %%
import pandas as pd
import os
from utils import read_files, strip_webvtt_to_plain_text

# %% [markdown]
# ## Load Interview Data
#
# We read all files starting with "interview" from the specified folder.

# %%
# Define the folder and prefix for interview files
interview_folder = "data/luontokokemus_simulaatio"
interview_prefix = "interview" # Based on files like interview1.txt

# Read the interview files
# The read_files function returns a list of tuples (filename, content)
interview_contents = read_files(folder=interview_folder, prefix=interview_prefix)

# Clean up texts if they were in WebVTT format (though these are .txt, it's a good general step)
interview_contents = [(fname, strip_webvtt_to_plain_text(text)) for fname, text in interview_contents]

# Print a summary of loaded interviews
print(f"Loaded {len(interview_contents)} interviews:")
for fname, text in interview_contents:
    print(f"- {fname} (length: {len(text)} characters)")

# Display the first few lines of the first interview as an example
if interview_contents:
    print(f"\nExample from {interview_contents[0][0]}:\n")
    print(interview_contents[0][1][:300] + "...") # Print first 300 characters
else:
    print("\nNo interview files found.")

# %% [markdown]
# ## Load Metadata
#
# We load the metadata from the CSV file. The metadata contains information associated with each interview, such as location type, distance to city, etc.

# %%
# Define the path to the metadata file
metadata_file_path = os.path.join(interview_folder, "metadata.csv")

# Read the metadata CSV file into a pandas DataFrame
try:
    metadata_df = pd.read_csv(metadata_file_path)
    print("\nMetadata loaded successfully:")
    print(metadata_df.info())
    print("\nFirst 5 rows of metadata:")
    print(metadata_df.head())
except FileNotFoundError:
    print(f"\nError: Metadata file not found at {metadata_file_path}")
    metadata_df = pd.DataFrame() # Create an empty DataFrame if file not found

# %% [markdown]
# ## Next Steps
#
# With both interview texts and metadata loaded, we can proceed to:
# 1. Preprocess the text data (e.g., feature extraction, sentiment analysis, topic modeling).
# 2. Merge or align text-derived features with the metadata based on `interview_id`.
# 3. Perform statistical correlational analyses.
# 4. Visualize findings.

# %%
# Placeholder for future analysis
print("\nSetup complete. Ready for correlational analysis.")

