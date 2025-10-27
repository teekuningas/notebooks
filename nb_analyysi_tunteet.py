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

# %%
from utils import read_files
from utils import strip_webvtt_to_plain_text

# Define the emotions of interest
emotions = ["ilo", "viha", "suru", "hämmästys", "pelko", "inho"]

# And read the texts of interest from the file system
contents = read_files(folder="data/linnut-03", prefix="inputfile")
# contents = read_files(folder="data/linnut", prefix="nayte")

# Remove timestamps if present
contents = [(fname, strip_webvtt_to_plain_text(text)) for fname, text in contents]

# Print to check that texts are correctly read
for fname, text in contents:
    print(f"{fname}:\n\n")
    print(f"{text}\n\n")

# %%
import json
import requests
from pprint import pprint

from llm import generate_simple

# Here, we will go through every text and every code for some number of iterations, and ask the llm to decide whether the emotion applies to the text or not.
# Asking the same question multiple gives us a reliability estimate.

# Define a machine-readable output format that the llm should produce, i.e. a boolean value "emotion_present".
output_format = {
    "type": "object",
    "properties": {
        "emotion_present": {
            "type": "boolean"
        }
    },
    "required": [
      "emotion_present"
    ]
}

# For every text and code, generate {n_iter} decisions.
n_iter = 2

results = []
for fname, text in contents:
    for emotion in emotions:
        for idx in range(n_iter):
            
            # Define the instructions. 
            instruction = """
            Olet laadullisen tutkimuksen avustaja. Saat tekstinäytteen sekä yhden ihmisen perustunteista. Lue teksti huolella ja päätä esiintyykö tunne tekstinäytteessä.
            """

            # Define the content snippet given to the llm.
            content = f"Tunne: {emotion} \n\n Tekstinäyte: \n\n {text}"

            # Generate the answer
            result = generate_simple(instruction, content, seed=idx, output_format=output_format)

            # Extract the result
            present = json.loads(result)['emotion_present']

            # Store it
            results.append({
                "fname": fname,
                "emotion": emotion,
                "iter": idx,
                "result": present
            })

# %%
import pandas as pd

# Here we use pandas to construct a simple colored table out of the results.

# Create a dictionary to store the results
transformed_data = {}

# From the flat results structure, create hierarchical easier structure
for item in results:
    fname = item['fname']
    emotion = item['emotion']
    
    if fname not in transformed_data:
        transformed_data[fname] = {}
    
    if emotion not in transformed_data[fname]:
        transformed_data[fname][emotion] = []
    
    transformed_data[fname][emotion].append(item['result'])

# Calculate percentages of ("yes" / all) for each text and code.
for fname in transformed_data:
    for emotion in transformed_data[fname]:
        true_count = transformed_data[fname][emotion].count(True)
        total_count = len(transformed_data[fname][emotion])
        transformed_data[fname][emotion] = (true_count / total_count) * 100

# Create DataFrame
df = pd.DataFrame.from_dict(transformed_data, orient='index')

# Add averages
df['total'] = df.mean(axis=1)
df.loc['total'] = df.mean()

# Format percentages
df = df.round(2)

# Define the styling function
def color_high_values(val):
    color = 'background-color: rgba(144, 238, 144, 0.3)' if val >= 80 else ''
    return color

# Apply the styling
styled_df = df.style.map(color_high_values).format("{:.2f}")

# Display the styled DataFrame
styled_df

# %%
