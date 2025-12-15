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
INPUT_FILE = "output/consolidation/435b3eaa_seed10/final_themes.json"

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

output_format = {
    "type": "object",
    "properties": {
        "code_present": {"type": "boolean"}
    },
    "required": ["code_present"]
}

N_ITERATIONS = 1

results = []
for row_idx, (rec_id, text) in enumerate(contents):
    print(f"Processing interview {row_idx+1}/{len(contents)}: {rec_id}")
    
    for code in codes:
        iter_idx = 0
        seed = 0
        
        while iter_idx < N_ITERATIONS:
            prompt = f"""Päätä, esiintyykö seuraava teema tekstinäytteessä.

TEEMA: {code}

TEKSTINÄYTE:
{text}

Vastaa "kyllä" jos teema esiintyy, "ei" jos ei esiinny."""

            try:
                free_form = generate_simple(prompt, "", seed=seed, provider="llamacpp")
                
                if not free_form:
                    seed += 1
                    continue

                format_prompt = """Muotoile vapaamuotoinen vastaus JSON-muotoon.
Jos vastaus on myönteinen: code_present = true
Jos vastaus on kielteinen: code_present = false"""

                json_str = generate_simple(format_prompt, free_form, seed=10, 
                                          output_format=output_format, provider="llamacpp")
                
                code_present = json.loads(json_str)['code_present']
                
                results.append({
                    "fname": rec_id,
                    "code": code,
                    "iter": iter_idx,
                    "result": code_present
                })
                
                iter_idx += 1
                seed += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                seed += 1
                continue

print(f"\nGenerated {len(results)} results")
# %%
# --- Results Table ---

import pandas as pd
import os
from uuid import uuid4

run_id = str(uuid4())[:8]

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
output_dir = f"output/analyysi_koodit/{run_id}"
os.makedirs(output_dir, exist_ok=True)
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
