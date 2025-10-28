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

# Define the codes that are used
codes = ['Luonto', 'Rauha', 'Sää', 'Kylmyys', 'ympäristö', 'Melu', 'Liikenne', 'Aurinko', 'Luistelu', 'tuli', 'Metsä']

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

from llm import generate_simple

# Here, we will go through every text and every code for some number of iterations, and ask the llm to decide whether the code applies to the text or not.
# Asking the same question multiple gives us a reliability estimate.

# Define a machine-readable output format that the llm should produce, i.e. a boolean value "code_present".
output_format = {
    "type": "object",
    "properties": {
        "code_present": {
            "type": "boolean"
        }
    },
    "required": [
      "code_present"
    ]
}

# For every text and code, generate {n_iter} decisions.
n_iter = 2

results = []
for fname, text in contents:
    for code in codes:
        for idx in range(n_iter):

            # Define the instructions. 
            instruction = """
            Olet laadullisen tutkimuksen avustaja. Saat tekstinäytteen sekä aineiston pohjalta rakennetun koodikirjan yksittäisen koodin. Lue teksti huolella ja päätä kuvaako koodi tekstinäytettä.
            """

            # Define the content snippet given to the llm.
            content = f"Koodi: {code} \n\n Tekstinäyte: \n\n {text}"

            # Generate the answer
            result = generate_simple(instruction, content, seed=idx, output_format=output_format, provider="llamacpp")

            # Extract the result
            code_present = json.loads(result)['code_present']
            
            # Store it
            results.append({
                "fname": fname,
                "code": code,
                "iter": idx,
                "result": code_present
            })

# %%
import pandas as pd

# Here we use pandas to construct a simple colored table out of the results.

# Create a dictionary to store the results
transformed_data = {}

# From the flat results structure, create hierarchical easier structure
for item in results:
    fname = item['fname']
    code = item['code']
    
    if fname not in transformed_data:
        transformed_data[fname] = {}
    
    if code not in transformed_data[fname]:
        transformed_data[fname][code] = []
    
    transformed_data[fname][code].append(item['result'])

# Calculate percentages of ("yes" / all) for each text and code.
for fname in transformed_data:
    for code in transformed_data[fname]:
        true_count = transformed_data[fname][code].count(True)
        total_count = len(transformed_data[fname][code])
        transformed_data[fname][code] = (true_count / total_count) * 100

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
