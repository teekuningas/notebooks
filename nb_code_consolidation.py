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
import re
import os
import json
import numpy as np
from uuid import uuid4
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

from llm import embed, generate_simple

# --- Configuration ---
INPUT_FILE = "output/koodit/487ef9e9/koodit_raw_rec_id_anonymized.txt" # Adjust to point to your data
SIMILARITY_MERGE_THRESHOLD = 0.45 # Vector similarity lower bound (speed filter)
LLM_MERGE_THRESHOLD = 0.70 # LLM score lower bound (quality filter). 0.0-1.0. Lower = more aggressive merging.
TARGET_THEMES = 20 # Soft target. Loop will stop if this is reached OR if no pairs > threshold exist.
SIZE_PENALTY_WEIGHT = 3.0 # Soft penalty for large themes. Higher = stronger preference for balanced merges.
RUN_ID = str(uuid4())[:8]
OUTPUT_DIR = f"output/consolidation/{RUN_ID}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Run ID: {RUN_ID}")
print(f"Vector Threshold: {SIMILARITY_MERGE_THRESHOLD}")
print(f"LLM Threshold: {LLM_MERGE_THRESHOLD}")
print(f"Size Penalty Weight: {SIZE_PENALTY_WEIGHT}")

# %%
# --- Step 1: Robust Parsing ---

def parse_code_file(filepath):
    """
    Parses the text file using a state-machine approach.
    Robust to multi-line explanations and varying indentation.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    all_codes = []
    
    # State variables
    current_rec_id = "unknown"
    current_iter = -1
    
    current_code = None # The code string
    current_explanation_lines = [] # List to accumulate explanation text
    
    def save_current_entry():
        nonlocal current_code, current_explanation_lines
        if current_code and current_explanation_lines:
            full_explanation = " ".join([l.strip() for l in current_explanation_lines]).strip()
            all_codes.append({
                "code": current_code,
                "explanation": full_explanation,
                "rec_id": current_rec_id,
                "iter_idx": current_iter,
                "original_text": f"{current_code}: {full_explanation}"
            })
        # Reset
        current_code = None
        current_explanation_lines = []

    for line in lines:
        clean_line = line.strip()
        
        if not clean_line:
            continue
            
        # Detect Record ID
        if "Rec ID Anonymized:" in line:
            save_current_entry() # Save pending work before switching context
            match = re.search(r'Rec ID Anonymized:\s*([a-f0-9\-]+)', line)
            if match:
                current_rec_id = match.group(1)
                current_iter = -1 # Reset iter
            continue

        # Detect Iteration
        if clean_line.startswith("Iter") and clean_line.endswith(":"):
            save_current_entry()
            match = re.search(r'Iter\s+(\d+):', clean_line)
            if match:
                current_iter = int(match.group(1))
            continue

        # Detect Code Start
        # Matches: "  - Code: Some Code Name"
        if "- Code:" in line:
            save_current_entry() # Save previous code
            # Extract code name (remove "- Code:" part)
            parts = line.split("- Code:", 1)
            if len(parts) > 1:
                current_code = parts[1].strip()
            continue

        # Detect Explanation Start
        # Matches: "    Explanation: Some text..."
        if "Explanation:" in line:
            # Extract the first part of the explanation
            parts = line.split("Explanation:", 1)
            if len(parts) > 1:
                current_explanation_lines.append(parts[1].strip())
            continue
            
        # If we are here, it's likely a continuation of the explanation
        # OR it's separator lines like "-----------------"
        if "-----------------" in line:
            continue
            
        if current_code is not None:
            # Append to explanation
            current_explanation_lines.append(clean_line)

    # Save the very last entry
    save_current_entry()
            
    return all_codes

# Check if file exists, if not try to find it in a likely location
if not os.path.exists(INPUT_FILE):
    base_dir = "output/koodit"
    if os.path.exists(base_dir):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if subdirs:
            # Pick the most recent one based on name/time
            subdirs.sort()
            INPUT_FILE = os.path.join(base_dir, subdirs[-1], "koodit_raw_rec_id_anonymized.txt")
            print(f"Found input file at: {INPUT_FILE}")

raw_codes = parse_code_file(INPUT_FILE)
TOTAL_CODES = len(raw_codes)
print(f"Parsed {TOTAL_CODES} codes.")

# %%
# --- Step 2: String-Based Deduplication (The "Obvious" Phase) ---

# Before expensive embeddings, merge everything that has the same name.
# "Luonto" == "luonto" == "Luonto "

grouped_codes = {}

for item in raw_codes:
    # Normalize key
    key = item['code'].strip().lower()
    if key not in grouped_codes:
        grouped_codes[key] = []
    grouped_codes[key].append(item)

print(f"Found {len(grouped_codes)} unique code names.")

# Convert these groups into initial Theme objects
themes = []

for key, items in grouped_codes.items():
    # Find most common display name (e.g., "Luonto" vs "luonto")
    names = [item['code'].strip() for item in items]
    most_common_name = max(set(names), key=names.count)
    
    # Find longest explanation to use as the "canonical" definition for now
    best_explanation = max([item['explanation'] for item in items], key=len)
    
    theme = {
        "id": str(uuid4()),
        "name": most_common_name,
        "definition": best_explanation,
        "source_codes": items,
        "embedding": None
    }
    themes.append(theme)

# %%
# --- Step 3: Embedding Initial Themes ---

print(f"Embedding {len(themes)} initial themes...")

for theme in themes:
    # Now we embed the consolidated concept
    text_to_embed = f"{theme['name']}: {theme['definition']}"
    theme['embedding'] = embed(text_to_embed, provider="llamacpp")

print(f"Ready to start Symmetric Loop with {len(themes)} themes.")

# %%
# --- Step 4: The LLM-Verified Symmetric Loop ---

def get_cosine_similarity_matrix(current_themes):
    embs = np.array([t['embedding'] for t in current_themes])
    
    # Raw Cosine Similarity
    sim = np.dot(embs, embs.T)
    norms = np.linalg.norm(embs, axis=1) + 1e-9
    sim /= np.outer(norms, norms)
    
    # Set diagonal to -1
    np.fill_diagonal(sim, -1)
    return sim

def apply_size_penalty(sim_matrix, current_themes):
    # Apply Zen Balance Penalty (Soft Size Penalty)
    sizes = np.array([len(t['source_codes']) for t in current_themes])
    total_n = TOTAL_CODES
    
    # Matrix of combined sizes
    combined_sizes = sizes[:, None] + sizes[None, :]
    combined_proportions = combined_sizes / total_n
    
    penalty_matrix = 1.0 / (1.0 + SIZE_PENALTY_WEIGHT * combined_proportions)
    
    # Apply penalty
    priority_matrix = sim_matrix * penalty_matrix
    return priority_matrix

def merge_themes(theme_a, theme_b):
    # Combine source codes
    combined_sources = theme_a['source_codes'] + theme_b['source_codes']
    
    # LLM Prompt to synthesize new name/definition
    explanations = [item['explanation'] for item in combined_sources]
    sample_explanations = list(set(explanations))[:10] 
    
    prompt = f"""
    Yhdistä nämä kaksi käsitettä yhdeksi ylemmän tason kategoriaksi.
    
    KÄSITE 1: {theme_a['name']} ({theme_a['definition']})
    KÄSITE 2: {theme_b['name']} ({theme_b['definition']})
    
    Luo uusi nimi ja määritelmä.
    
    OHJEET:
    1. Nimen on oltava yksinkertainen, yleiskielinen substantiivi tai perusmuotoinen fraasi.
    2. Älä keksi uusia, runollisia tai monimutkaisia termejä.
    3. Jos toinen alkuperäisistä nimistä kuvaa hyvin koko yhdistelmää, käytä sitä.
    4. Määritelmän tulee olla yksi selkeä ja toteava virke.
    
    Vastaa JSON-muodossa: {{ "name": "...", "definition": "..." }}
    """
    
    output_format = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "definition": {"type": "string"}
        },
        "required": ["name", "definition"]
    }
    
    try:
        response = generate_simple(prompt, "Yhdistä.", output_format=output_format, provider="llamacpp")
        data = json.loads(response)
    except Exception as e:
        print(f"Merge LLM error: {e}. Fallback to combining names.")
        data = {
            "name": f"{theme_a['name']} & {theme_b['name']}",
            "definition": f"Yhdistelmä teemoista {theme_a['name']} ja {theme_b['name']}."
        }

    # Compute new embedding for the combined theme
    new_emb = embed(f"{data['name']}: {data['definition']}", provider="llamacpp")
    
    new_theme = {
        "id": str(uuid4()),
        "name": data['name'],
        "definition": data['definition'],
        "source_codes": combined_sources,
        "embedding": new_emb
    }
    return new_theme
def check_merge_validity(theme_a, theme_b):
    # Include samples of original explanations for context
    explanations_a = list(set([x['explanation'] for x in theme_a['source_codes']]))[:15]
    explanations_b = list(set([x['explanation'] for x in theme_b['source_codes']]))[:15]
    
    context_str = ""
    context_str += f"Esimerkkejä teeman '{theme_a['name']}' selityksistä:\n" + "\n".join([f"- {e}" for e in explanations_a])
    context_str += f"\n\nEsimerkkejä teeman '{theme_b['name']}' selityksistä:\n" + "\n".join([f"- {e}" for e in explanations_b])

    prompt = f"""
    Arvioi kahden teeman samankaltaisuutta.
    
    TEEMA 1: {theme_a['name']}
    Määritelmä: {theme_a['definition']}
    
    TEEMA 2: {theme_b['name']}
    Määritelmä: {theme_b['definition']}
    
    {context_str}
    
    Anna samankaltaisuuspisteet (0.0 - 1.0) seuraavin perustein:
    
    - Korkea (0.8 - 1.0): Teemat ovat sama asia tai toinen sisältyy toiseen.
    - Keskitaso (0.5 - 0.8): Teemat ovat läheisiä, mutta niillä on selkeä vivahde-ero.
    - Matala (0.0 - 0.5): Teemat ovat eri asioita.
    
    Vastaa vain JSON: {{ "score": 0.00 }}
    """
    
    output_format = {
        "type": "object",
        "properties": {
            "score": {"type": "number"}
        },
        "required": ["score"]
    }
    
    try:
        response = generate_simple(prompt, "Analysoi.", output_format=output_format, provider="llamacpp")
        return float(json.loads(response)['score'])
    except:
        return 0.0 # Fail safe

# --- The Loop ---

ignore_pairs = set() # Store tuples of (id_a, id_b) that LLM rejected

while len(themes) > TARGET_THEMES:
    print(f"Themes remaining: {len(themes)}")
    
    # 1. Calculate Raw Similarity
    sim_matrix = get_cosine_similarity_matrix(themes)
    
    # 2. Calculate Merge Priority (Similarity weighted by Balance)
    priority_matrix = apply_size_penalty(sim_matrix, themes)
    
    # 3. Find best candidate pair based on PRIORITY
    found_pair = False
    
    # Get indices sorted by PRIORITY (descending)
    flat_indices = np.argsort(priority_matrix, axis=None)[::-1]
    
    for idx in flat_indices:
        i, j = np.unravel_index(idx, priority_matrix.shape)
        
        if i >= j: continue # process upper triangle only
        
        # Check raw threshold on SIMILARITY (not priority) to avoid merging garbage
        if sim_matrix[i, j] < SIMILARITY_MERGE_THRESHOLD:
            print(f"No pairs above raw similarity threshold {SIMILARITY_MERGE_THRESHOLD}. Stopping.")
            found_pair = False 
            break 
            
        t_a = themes[i]
        t_b = themes[j]
        
        # Check if previously rejected
        pair_key = tuple(sorted((t_a['id'], t_b['id'])))
        if pair_key in ignore_pairs:
            continue
            
        print(f"Candidate: '{t_a['name']}' <--> '{t_b['name']}' (Raw: {sim_matrix[i,j]:.2f}, Prio: {priority_matrix[i,j]:.2f})")
        
        # 4. LLM Check
        score = check_merge_validity(t_a, t_b)
        print(f"  -> LLM Score: {score:.2f}")
        
        if score >= LLM_MERGE_THRESHOLD:
            print(f"  -> Merging...")
            new_theme = merge_themes(t_a, t_b)
            print(f"     => Created: '{new_theme['name']}'")
            
            # Update list: Remove old, add new
            new_theme_list = [t for k, t in enumerate(themes) if k != i and k != j]
            new_theme_list.append(new_theme)
            themes = new_theme_list
            
            found_pair = True
            break
        else:
            print(f"  -> Rejected (Score < {LLM_MERGE_THRESHOLD}). Ignoring pair.")
            ignore_pairs.add(pair_key)
            
    if not found_pair:
        print("Loop terminated: No valid merges found or threshold reached.")
        break

print(f"Final theme count: {len(themes)}")

# %%
# --- Step 5: Save Results ---

output_path = os.path.join(OUTPUT_DIR, "final_themes.json")
simple_output_path = os.path.join(OUTPUT_DIR, "final_themes_list.txt")

# Save full JSON (excluding embedding vectors)
with open(output_path, 'w') as f:
    serializable_themes = []
    for t in themes:
        t_copy = t.copy()
        if 'embedding' in t_copy:
            del t_copy['embedding']
        serializable_themes.append(t_copy)
    json.dump(serializable_themes, f, indent=2, ensure_ascii=False)

# Save simple text list for review
with open(simple_output_path, 'w') as f:
    f.write(f"Final {len(themes)} Themes (Threshold: {SIMILARITY_MERGE_THRESHOLD}):\n\n")
    for idx, t in enumerate(themes):
        f.write(f"{idx+1}. {t['name']}\n")
        f.write(f"   Def: {t['definition']}\n")
        f.write(f"   (Merged from {len(t['source_codes'])} source codes)\n\n")

print(f"Saved results to {OUTPUT_DIR}")

# Print preview
for idx, t in enumerate(themes):
    print(f"{idx+1}: {t['name']} ({len(t['source_codes'])})")