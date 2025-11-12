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
from utils import read_interview_data
from utils import strip_webvtt_to_plain_text

from utils import filter_interview_simple

# Define the codes that are used
codes = ['Rauha', 'Ilo', 'Huoli', 'Yhteys luontoon', 'Nostalgia', 'Turvallisuus', 'Vapaus', 'Ylpeys', 'Kaipuu', 'Uteliaisuus']

# And read the texts of interest from the file system
#contents = read_files(folder="data/linnut", prefix="nayte")
contents = read_interview_data("data/birdinterview", "observation")

# Filter to a sensible subset
contents = filter_interview_simple(contents)

# Convert to (filename, content) tuples for now
contents = [(meta["rec_id"], text) for meta, text in contents]

## First a smaller sample
#codes = codes[:20]
contents = contents[:50]

print(f"len(contents): {len(contents)}")
print(f"len(codes): {len(codes)}")

## Print to check that texts are correctly read
#for fname, text in contents:
#    print(f"{fname}:\n\n")
#    print(f"{text}\n\n")

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
n_iter = 1

results = []
for idx, (fname, text) in enumerate(contents):
    print(f"Generating row {idx+1} for {fname}..")
    for code in codes:
        idx = 0
        seed = 0
        while idx < n_iter:
            # It is easier for llms to do the decision in two steps: first request a free-formatted answer and then request it in the correct format. 
            # This also allows different roles: we could use a reasoning model (which is bad at formatting) to do the first step and a formatting model (which is bad at reasoning) to do the second step.

            # Define the instructions for the first, free-form step.
            instruction = """
            Päätä, liittyykö tekstinäyte annettuun koodiin. Vastaa "kyllä" tai "ei".
            """

            # Define the content snippet given to the llm.
            content = f"Koodi:\n\n\"{code}\"\n\nTekstinäyte: \n\n\"{text}\""

            # Generate the answer
            free_form_result = generate_simple(instruction, content, seed=idx, provider="llamacpp")

            if not free_form_result:
                print("Trying again.. (no result)")
                print(f"Code was: {code}")
                print(f"Text was: {text}")
                print(f"Seed was: {seed}")
                seed += 1
                continue

            # Now, we use a second LLM call to format the free-form answer into the desired JSON format.
            # This is more robust than trying to parse the free-form text manually.
            formatting_instruction = '''
            Saat syötteenä vapaamuotoisen viestin, joka on joko myönteinen tai kielteinen. Muotoile se uudelleen JSONiksi niin että code_present = true jos syöteteksti on myönteinen ja code_present = false jos syöteteksti on kielteinen.
            '''

            json_result_str = generate_simple(formatting_instruction, free_form_result, seed=10, output_format=output_format, provider="llamacpp")

            try:
                # Extract the boolean value from the JSON result.
                code_present = json.loads(json_result_str)['code_present']
            except (json.JSONDecodeError, KeyError):
                print(f"Trying again.. (invalid JSON result: '{json_result_str}')")
                print(f"Code was: {code}")
                print(f"Text was: {text}")
                print(f"Seed was: {seed}")
                seed += 1
                continue

            # Store it
            results.append({
                "fname": fname,
                "code": code,
                "iter": idx,
                "result": code_present
            })
            idx += 1
            seed += 1
# %%
import pandas as pd
from uuid import uuid4
import os

# Here we use pandas to construct a simple colored table out of the results.

run_id = str(uuid4())[:8]

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

# Calculate true percentages for each text and code.
for fname in transformed_data:
    for code in transformed_data[fname]:
        true_count = transformed_data[fname][code].count(True)
        total_count = len(transformed_data[fname][code])
        transformed_data[fname][code] = (true_count / total_count)

# Create DataFrame
df = pd.DataFrame.from_dict(transformed_data, orient='index')

# Save the DataFrame with values between 0 and 1
output_dir = f"output/analyysi_koodit/{run_id}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/koodit_{len(codes)}x{len(contents)}.csv"
df.to_csv(filename)

# Add averages and format for visualization
df_display = df.copy()
df_display['total'] = df_display.mean(axis=1)
df_display.loc['total'] = df_display.mean()
df_display = df_display.round(2) * 100

# Define the styling function
def color_high_values(val):
    color = 'background-color: rgba(144, 238, 144, 0.3)' if val >= 50 else ''
    return color

# Apply the styling
styled_df = df_display.style.map(color_high_values).format("{:.2f}")

print(f"Results saved to: {filename}")

# Display the styled DataFrame
styled_df

# %%
