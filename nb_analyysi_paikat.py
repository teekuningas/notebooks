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

# Define the paikat that are used
paikat = ["Metsä", "Järvi", "Meri", "Ranta", "Suo", "Kaupunki", "Maaseutu", "Suojelualue", "Piha", "Mökki"]

# And read the texts of interest from the file system
#contents = read_files(folder="data/linnut", prefix="nayte")
contents = read_interview_data("data/birdinterview", "observation")

# Filter to a sensible subset
contents = filter_interview_simple(contents)

# Convert to (filename, content) tuples for now
contents = [(meta["rec_id"], text) for meta, text in contents]

## First a smaller sample
paikat = paikat[:2]
contents = contents[:50]

print(f"len(contents): {len(contents)}")
print(f"len(paikat): {len(paikat)}")

## Print to check that texts are correctly read
#for fname, text in contents:
#    print(f"{fname}:\n\n")
#    print(f"{text}\n\n")

# %%
import json

from llm import generate_simple

# Here, we will go through every text and every paikka for some number of iterations, and ask the llm to decide whether the paikka applies to the text or not.
# Asking the same question multiple gives us a reliability estimate.

# Define a machine-readable output format that the llm should produce, i.e. a boolean value "paikka_present".
output_format = {
    "type": "object",
    "properties": {
        "paikka_present": {
            "type": "boolean"
        }
    },
    "required": [
      "paikka_present"
    ]
}

# For every text and paikka, generate {n_iter} decisions.
n_iter = 3

results = []
for idx, (fname, text) in enumerate(contents):
    print(f"Generating row {idx+1} for {fname}..")
    for paikka in paikat:
        idx = 0
        seed = 0
        while idx < n_iter:
            # It is easier for llms to do the decision in two steps: first request a free-formatted answer and then request it in the correct format. 
            # This also allows different roles: we could use a reasoning model (which is bad at formatting) to do the first step and a formatting model (which is bad at reasoning) to do the second step.

            # Define the instructions for the first, free-form step.
            instruction = """
            Päätä, liittyykö tekstinäyte annettuun paikkaan. Molempien ehtojen tulee täyttyä:

            1) Paikan täytyy eksplisiittisesti esiintyä tekstissä.
            2) Teksti täytyy olla lähtöisin kyseisestä paikasta.

            Vastaa "kyllä" tai "ei".
            """

            # Define the content snippet given to the llm.
            content = f"Paikka:\n\n\"{paikka}\"\n\nTekstinäyte: \n\n\"{text}\""

            # Generate the answer
            free_form_result = generate_simple(instruction, content, seed=idx, provider="llamacpp")

            if not free_form_result:
                print("Trying again.. (no result)")
                print(f"Paikka was: {paikka}")
                print(f"Text was: {text}")
                print(f"Seed was: {seed}")
                seed += 1
                continue

            # Now, we use a second LLM call to format the free-form answer into the desired JSON format.
            # This is more robust than trying to parse the free-form text manually.
            formatting_instruction = '''
            Saat syötteenä vapaamuotoisen viestin, joka on joko myönteinen tai kielteinen. Muotoile se uudelleen JSONiksi niin että paikka_present = true jos syöteteksti on myönteinen ja paikka_present = false jos syöteteksti on kielteinen.
            '''

            json_result_str = generate_simple(formatting_instruction, free_form_result, seed=10, output_format=output_format, provider="llamacpp")

            try:
                # Extract the boolean value from the JSON result.
                paikka_present = json.loads(json_result_str)['paikka_present']
            except (json.JSONDecodeError, KeyError):
                print(f"Trying again.. (invalid JSON result: '{json_result_str}')")
                print(f"Paikka was: {paikka}")
                print(f"Text was: {text}")
                print(f"Seed was: {seed}")
                seed += 1
                continue

            # Store it
            results.append({
                "fname": fname,
                "paikka": paikka,
                "iter": idx,
                "result": paikka_present
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
    paikka = item['paikka']
    
    if fname not in transformed_data:
        transformed_data[fname] = {}
    
    if paikka not in transformed_data[fname]:
        transformed_data[fname][paikka] = []
    
    transformed_data[fname][paikka].append(item['result'])

# Calculate true percentages for each text and paikka.
for fname in transformed_data:
    for paikka in transformed_data[fname]:
        true_count = transformed_data[fname][paikka].count(True)
        total_count = len(transformed_data[fname][paikka])
        transformed_data[fname][paikka] = (true_count / total_count)

# Create DataFrame
df = pd.DataFrame.from_dict(transformed_data, orient='index')

# Save the DataFrame with values between 0 and 1
output_dir = f"output/analyysi_paikat/{run_id}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/paikat_{len(paikat)}x{len(contents)}.csv"
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
