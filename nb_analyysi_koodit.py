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
import json
from utils import read_interview_data, filter_interview_simple

# --- Configuration ---
INPUT_FILE = "output/consolidation/77db8e4e_seed11/final_themes.json"

# --- Load Themes ---
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    themes = json.load(f)

codes = [theme['name'] for theme in themes]

print(f"Loaded {len(codes)} themes from: {INPUT_FILE}")

# --- Load Interview Data ---
contents = read_interview_data("data/birdinterview", "observation")
contents = filter_interview_simple(contents)
contents = [(meta["rec_id"], text) for meta, text in contents]

print(f"Interviews: {len(contents)}")
print(f"Themes: {len(codes)}")

# %%
# --- Code Presence Analysis ---

from llm import generate_simple
import os
from uuid import uuid4

# Create output directory and debug log file
run_id = str(uuid4())[:8]
output_dir = f"output/analyysi_koodit/{run_id}"
os.makedirs(output_dir, exist_ok=True)
debug_log_path = os.path.join(output_dir, "debug_log.txt")

output_format = {
    "type": "object",
    "properties": {
        "theme_present": {"type": "boolean"},
        "reason": {"type": "string"}
    },
    "required": ["theme_present", "reason"]
}

N_ITERATIONS = 1

results = []
with open(debug_log_path, 'w', encoding='utf-8') as debug_file:
    for row_idx, (rec_id, text) in enumerate(contents):
        print(f"Processing interview {row_idx+1}/{len(contents)}: {rec_id}")
        
        for code in codes:
            iter_idx = 0
            seed = 0
            
            while iter_idx < N_ITERATIONS:
                prompt = f"""Päätä esiintyykö seuraava teema tekstinäytteessä.

TEEMA: {code}

TEKSTINÄYTE:
{text}

Vastaa TRUE jos teema esiintyy tekstissä.
Vastaa FALSE jos teema ei esiinny tekstissä.

Anna lyhyt perustelu päätöksellesi.

Vastaa JSON: {{ "theme_present": true, "reason": "lyhyt perustelu" }} tai {{ "theme_present": false, "reason": "lyhyt perustelu" }}"""

                try:
                    response = generate_simple(prompt, "Arvioi objektiivisesti.", seed=seed, 
                                             output_format=output_format, provider="llamacpp")
                    
                    # Debug: Write to file instead of print
                    debug_file.write(f"[{rec_id}] [{code}] {response}\n")
                    debug_file.flush()
                    
                    data = json.loads(response)
                    theme_present = data['theme_present']
                    
                    results.append({
                        "fname": rec_id,
                        "code": code,
                        "iter": iter_idx,
                        "result": theme_present
                    })
                    
                    iter_idx += 1
                    seed += 1
                    
                except (json.JSONDecodeError, KeyError) as e:
                    debug_file.write(f"[{rec_id}] [{code}] ERROR: {e}\n")
                    debug_file.flush()
                    seed += 1
                    continue

print(f"\nGenerated {len(results)} results")
print(f"Debug log saved to: {debug_log_path}")
# %%
# --- Results Table ---

import pandas as pd

# Aggregate results by interview and theme
table = {}
for item in results:
    rec_id = item['fname']
    code = item['code']
    
    if rec_id not in table:
        table[rec_id] = {}
    if code not in table[rec_id]:
        table[rec_id][code] = []
    
    table[rec_id][code].append(item['result'])

# Calculate presence percentage
for rec_id in table:
    for code in table[rec_id]:
        true_count = table[rec_id][code].count(True)
        total = len(table[rec_id][code])
        table[rec_id][code] = true_count / total

df = pd.DataFrame.from_dict(table, orient='index')

# Save results
output_path = os.path.join(output_dir, f"themes_{len(codes)}x{len(contents)}.csv")
df.to_csv(output_path)

print(f"Saved to: {output_path}")

# Display with formatting
df_display = df.copy()
df_display['total'] = df_display.mean(axis=1)
df_display.loc['total'] = df_display.mean()
df_display = (df_display * 100).round(2)

def highlight_present(val):
    return 'background-color: rgba(144, 238, 144, 0.3)' if val >= 50 else ''

styled_df = df_display.style.map(highlight_present).format("{:.2f}")
styled_df

# %%
