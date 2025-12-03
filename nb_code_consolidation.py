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

from llm import embed, generate_simple

# --- Configuration ---
INPUT_FILE = "output/koodit/487ef9e9/koodit_raw_rec_id_anonymized.txt"

# The Zen Parameter: controls merge resistance based on size imbalance.
# penalty = exp(-λ × imbalance)
# where:
#   imbalance = (ratio - 1), ratio = max(size_a, size_b) / min(size_a, size_b)
#
# This prevents "big eating small" - a large theme absorbing tiny ones.
# We intentionally do NOT penalize large balanced merges, because:
# - If two large themes are semantically the same, they should merge
# - The self-healing nature of the algorithm means incorrectly merged
#   themes will re-emerge as smaller clusters that can't be absorbed
SIZE_PENALTY_LAMBDA = 3.0

MERGE_THRESHOLD = 0.50  # Lower threshold to allow more merging
MAX_REJECTIONS_PER_ITERATION = 30  # More attempts per iteration
ARTIFACT_BATCH_SIZE = 20  # Batch size for artifact detection

RUN_ID = str(uuid4())[:8]
OUTPUT_DIR = f"output/consolidation/{RUN_ID}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Run ID: {RUN_ID}")
print(f"Size Penalty λ: {SIZE_PENALTY_LAMBDA}")
print(f"Merge Threshold: {MERGE_THRESHOLD}")

# %%
# --- Step 1: Robust Parsing ---

def parse_code_file(filepath):
    """
    Parses the text file using a state-machine approach.
    Robust to multi-line explanations and varying indentation.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
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
print(f"Parsed {len(raw_codes)} codes.")

# %%
# --- Step 2: Deduplication and Artifact Removal ---
#
# 1. Group codes by normalized name (case-insensitive)
# 2. Detect and remove technical artifacts (numbered codes like K1, L3)
# 3. Build initial themes

def detect_artifacts_batch(code_names):
    """Ask LLM to identify artifacts from a batch of code names."""
    codes_list = "\n".join([f"- {name}" for name in code_names])
    
    prompt = f"""Tunnista seuraavista koodeista VAIN selkeät tekniset artefaktit.

Artefakteja ovat AINOASTAAN:
- Numerokoodit tai tunnisteet (esim. "1.1", "K1", "L3", "T2")
- Lyhenteet joissa numero (esim. "LS1", "M1")

EI artefakteja:
- Oikeat suomenkieliset sanat tai fraasit (esim. "Lintubongaus", "Metsätalous", "Rauha ja hiljaisuus")
- Käsitteet jotka kuvaavat teemoja

KOODIT:
{codes_list}

Palauta VAIN selkeät numerokoodit/tunnisteet. Tyhjä lista jos ei ole.
Vastaa JSON: {{ "artifacts": ["koodi1", "koodi2"] }}"""

    output_format = {
        "type": "object",
        "properties": {
            "artifacts": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["artifacts"]
    }
    
    try:
        response = generate_simple(prompt, "Tunnista.", output_format=output_format, provider="llamacpp")
        return json.loads(response).get('artifacts', [])
    except:
        return []

# Group by normalized name
grouped_codes = {}
for item in raw_codes:
    key = item['code'].strip().lower()
    if key not in grouped_codes:
        grouped_codes[key] = []
    grouped_codes[key].append(item)

print(f"Unique code names: {len(grouped_codes)} (from {len(raw_codes)} total instances)")

# Detect artifacts (use original case names for LLM)
unique_names = [items[0]['code'].strip() for items in grouped_codes.values()]
all_artifacts = set()

for i in range(0, len(unique_names), ARTIFACT_BATCH_SIZE):
    batch = unique_names[i:i+ARTIFACT_BATCH_SIZE]
    artifacts = detect_artifacts_batch(batch)
    all_artifacts.update(a.lower() for a in artifacts)  # normalize
    if artifacts:
        print(f"  Batch {i//ARTIFACT_BATCH_SIZE + 1}: found {len(artifacts)} artifacts")

if all_artifacts:
    print(f"Artifacts removed ({len(all_artifacts)}):")
    for key in sorted(all_artifacts):
        if key in grouped_codes:
            expl = grouped_codes[key][0]['explanation'][:60]
            print(f"  - {key}: {expl}...")
            del grouped_codes[key]

print(f"Clean unique codes: {len(grouped_codes)}")

# Build themes
themes = []
TOTAL_CODES = 0

for key, items in grouped_codes.items():
    names = [item['code'].strip() for item in items]
    most_common_name = max(set(names), key=names.count)
    best_explanation = max([item['explanation'] for item in items], key=len)
    
    theme = {
        "id": str(uuid4()),
        "name": most_common_name,
        "definition": best_explanation,
        "source_codes": items,
        "embedding": None
    }
    themes.append(theme)
    TOTAL_CODES += len(items)

print(f"Initial themes: {len(themes)}, total source codes: {TOTAL_CODES}")

# %%
# --- Step 3: Embedding Initial Themes ---

print(f"Embedding {len(themes)} initial themes...")

for theme in themes:
    # Now we embed the consolidated concept
    text_to_embed = f"{theme['name']}: {theme['definition']}"
    theme['embedding'] = embed(text_to_embed, provider="llamacpp")

print(f"Ready to start Symmetric Loop with {len(themes)} themes.")

# %%
# --- Step 4: The Zen Merge Loop ---
#
# Philosophy:
# - Embeddings guide exploration priority (what to try next)
# - LLM is the judge (should this pair merge?)
# - Cache remembers tested pairs to skip them
# - Size imbalance penalty prevents "big eating small"
#
# The Formula:
#   penalty = exp(-λ × imbalance)
#   where imbalance = ratio - 1, ratio = max(size_a, size_b) / min(size_a, size_b)
#
# Note: We do NOT penalize large balanced merges. If two large themes
# are semantically the same (e.g., "Luonto" + "Luonto"), they should merge.
# The self-healing nature of the algorithm handles incorrect merges.
#
# Cache Strategy (simple set, not dict):
# - If pair in cache → already tested and rejected → skip
# - If pair merged, it wouldn't be in cache (new theme = new key)
# - When themes merge, old cache entries become irrelevant

def get_cosine_similarity_matrix(current_themes):
    """Compute pairwise cosine similarity between theme embeddings."""
    embs = np.array([t['embedding'] for t in current_themes])
    sim = np.dot(embs, embs.T)
    norms = np.linalg.norm(embs, axis=1) + 1e-9
    sim /= np.outer(norms, norms)
    np.fill_diagonal(sim, -1)
    return sim


def merge_themes(theme_a, theme_b):
    """Ask LLM to synthesize a new theme name and definition from two merged themes."""
    combined_sources = theme_a['source_codes'] + theme_b['source_codes']
    
    prompt = f"""Yhdistä nämä kaksi käsitettä yhdeksi ylemmän tason kategoriaksi.

KÄSITE 1: {theme_a['name']} ({theme_a['definition']})
KÄSITE 2: {theme_b['name']} ({theme_b['definition']})

Luo uusi nimi ja määritelmä.

OHJEET:
1. Nimen on oltava yksinkertainen, yleiskielinen substantiivi tai perusmuotoinen fraasi.
2. Älä keksi uusia, runollisia tai monimutkaisia termejä.
3. Jos toinen alkuperäisistä nimistä kuvaa hyvin koko yhdistelmää, käytä sitä.
4. Määritelmän tulee olla yksi selkeä ja toteava virke.

Vastaa JSON-muodossa: {{ "name": "...", "definition": "..." }}"""
    
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
    """
    Ask LLM to evaluate semantic similarity between two themes.
    Returns raw score in [0, 1].
    """
    explanations_a = list(set([x['explanation'] for x in theme_a['source_codes']]))[:15]
    explanations_b = list(set([x['explanation'] for x in theme_b['source_codes']]))[:15]
    
    context_str = f"Esimerkkejä teeman '{theme_a['name']}' selityksistä:\n"
    context_str += "\n".join([f"- {e}" for e in explanations_a])
    context_str += f"\n\nEsimerkkejä teeman '{theme_b['name']}' selityksistä:\n"
    context_str += "\n".join([f"- {e}" for e in explanations_b])

    prompt = f"""Arvioi kahden teeman samankaltaisuutta.

TEEMA 1: {theme_a['name']}
Määritelmä: {theme_a['definition']}

TEEMA 2: {theme_b['name']}
Määritelmä: {theme_b['definition']}

{context_str}

Anna samankaltaisuuspisteet (0.0 - 1.0):
- Korkea (0.8 - 1.0): Teemat ovat sama asia tai toinen sisältyy toiseen.
- Keskitaso (0.5 - 0.8): Teemat ovat läheisiä, mutta niillä on selkeä vivahde-ero.
- Matala (0.0 - 0.5): Teemat ovat eri asioita.

Vastaa vain JSON: {{ "score": 0.00 }}"""
    
    output_format = {
        "type": "object",
        "properties": {"score": {"type": "number"}},
        "required": ["score"]
    }
    
    try:
        response = generate_simple(prompt, "Analysoi.", output_format=output_format, provider="llamacpp")
        return float(json.loads(response)['score'])
    except:
        return 0.0


# --- The Zen Loop ---

log_lines = []

def log(msg):
    """Print and record to log."""
    print(msg)
    log_lines.append(msg)

# Log header with run parameters
log(f"{'='*60}")
log(f"CODE CONSOLIDATION RUN")
log(f"{'='*60}")
log(f"Run ID: {RUN_ID}")
log(f"Input file: {INPUT_FILE}")
log(f"Size Penalty λ: {SIZE_PENALTY_LAMBDA}")
log(f"Merge Threshold: {MERGE_THRESHOLD}")
log(f"Initial themes: {len(themes)}")
log(f"Total source codes: {TOTAL_CODES}")
log(f"{'='*60}")

# Cache for tested pairs: set of (content_key_a, content_key_b)
# If in cache → already tested and rejected (if it passed, it would have merged)
tested_pairs = set()
total_merges = 0

def get_content_key(theme):
    """Create a simple string key from theme's source codes."""
    # Sort source code identifiers to make a deterministic key
    # Include code name and explanation to ensure uniqueness even within same rec_id/iter
    ids = sorted(f"{item['rec_id']}_{item['iter_idx']}_{item['code']}_{item['explanation']}" for item in theme['source_codes'])
    return tuple(ids)

def get_cache_key(theme_a, theme_b):
    """Create a cache key for a pair of themes."""
    key_a = get_content_key(theme_a)
    key_b = get_content_key(theme_b)
    # Sort the two keys to ensure (A,B) == (B,A)
    if key_a <= key_b:
        return (key_a, key_b)
    return (key_b, key_a)

while True:
    n_themes = len(themes)
    
    log(f"\n{'='*60}")
    log(f"--- Iteration: {n_themes} themes ---")
    
    # 1. Compute base similarity from embeddings
    sim_matrix = get_cosine_similarity_matrix(themes)
    
    # 2. Compute size penalty (imbalance only - no proportion penalty)
    sizes = np.array([len(t['source_codes']) for t in themes])
    size_max = np.maximum(sizes[:, None], sizes[None, :])
    size_min = np.minimum(sizes[:, None], sizes[None, :])
    ratio_matrix = size_max / np.maximum(size_min, 1)
    imbalance_matrix = ratio_matrix - 1
    
    penalty_matrix = np.exp(-SIZE_PENALTY_LAMBDA * imbalance_matrix)
    
    # 3. Build priority matrix: ALWAYS use embeddings for priority
    #    (Cache is for skipping, not for ordering)
    priority_matrix = sim_matrix * penalty_matrix
    np.fill_diagonal(priority_matrix, -np.inf)
    
    # 4. Iterate pairs in order of embedding priority (most promising first)
    flat_indices = np.argsort(priority_matrix, axis=None)[::-1]
    
    found_merge = False
    rejections_this_iteration = 0
    
    for idx in flat_indices:
        i, j = np.unravel_index(idx, priority_matrix.shape)
        if i >= j:
            continue
            
        t_a, t_b = themes[i], themes[j]
        cache_key = get_cache_key(t_a, t_b)
        penalty = penalty_matrix[i, j]
        
        # Skip already tested pairs
        if cache_key in tested_pairs:
            continue
        
        # New pair: ask LLM
        llm_score = check_merge_validity(t_a, t_b)
        tested_pairs.add(cache_key)
        
        # Apply penalty to decision
        effective_score = llm_score * penalty
        
        size_a, size_b = len(t_a['source_codes']), len(t_b['source_codes'])
        raw_sim = sim_matrix[i, j]
        
        # Detailed logging (commented out for cleaner output, enable for debugging)
        # log(f"[Test {rejections_this_iteration + 1}/{MAX_REJECTIONS_PER_ITERATION}] '{t_a['name']}' ({size_a}) <-> '{t_b['name']}' ({size_b})")
        # log(f"  Embed: {raw_sim:.2f} × Penalty: {penalty:.2f} = {raw_sim * penalty:.2f}")
        # log(f"  LLM: {llm_score:.2f} × {penalty:.2f} = {effective_score:.2f}")
        
        if effective_score >= MERGE_THRESHOLD:
            new_theme = merge_themes(t_a, t_b)
            log(f"  ✓ '{t_a['name']}' + '{t_b['name']}' → '{new_theme['name']}' ({len(new_theme['source_codes'])} codes) [LLM: {llm_score:.2f}]")
            
            themes = [t for k, t in enumerate(themes) if k != i and k != j]
            themes.append(new_theme)
            total_merges += 1
            found_merge = True
            break
        else:
            # Detailed rejection logging (commented out)
            # log(f"  ✗ Rejected (need ≥ {MERGE_THRESHOLD})")
            rejections_this_iteration += 1
            if rejections_this_iteration >= MAX_REJECTIONS_PER_ITERATION:
                break
    
    # Iteration summary
    if found_merge:
        log(f"  Tests: {rejections_this_iteration + 1}, Merges: 1 → {len(themes)} themes remaining")
    else:
        log(f"  Tests: {rejections_this_iteration}, Merges: 0 → stopping")
        log(f"\n{'='*60}")
        log(f"ALGORITHM COMPLETE")
        log(f"  Total pairs tested: {len(tested_pairs)}")
        log(f"  Total merges: {total_merges}")
        log(f"  Total rejections: {len(tested_pairs) - total_merges}")
        log(f"  Final theme count: {len(themes)}")
        break

log(f"\n{'='*60}")
log(f"FINAL SUMMARY: {len(themes)} themes from {TOTAL_CODES} original codes")

# %%
# --- Step 5: Save Results ---

# Deduplicate theme names by appending size for disambiguation
name_counts = {}
for t in themes:
    name = t['name']
    name_counts[name] = name_counts.get(name, 0) + 1

if any(c > 1 for c in name_counts.values()):
    seen = {}
    for t in themes:
        name = t['name']
        if name_counts[name] > 1:
            count = seen.get(name, 0) + 1
            seen[name] = count
            t['name'] = f"{name} ({count})"

output_path = os.path.join(OUTPUT_DIR, "final_themes.json")
simple_output_path = os.path.join(OUTPUT_DIR, "final_themes_list.txt")

# Save full JSON (excluding embedding vectors)
with open(output_path, 'w', encoding='utf-8') as f:
    serializable_themes = []
    for t in themes:
        t_copy = t.copy()
        if 'embedding' in t_copy:
            del t_copy['embedding']
        serializable_themes.append(t_copy)
    json.dump(serializable_themes, f, indent=2, ensure_ascii=False)

# Save simple text list for review
with open(simple_output_path, 'w', encoding='utf-8') as f:
    f.write(f"Code Consolidation Results\n")
    f.write(f"==========================\n")
    f.write(f"λ (size penalty): {SIZE_PENALTY_LAMBDA}\n")
    f.write(f"Merge threshold: {MERGE_THRESHOLD}\n")
    f.write(f"Final themes: {len(themes)}\n\n")
    
    # Sort by size for readability
    sorted_themes = sorted(themes, key=lambda t: len(t['source_codes']), reverse=True)
    for idx, t in enumerate(sorted_themes):
        f.write(f"{idx+1}. {t['name']} ({len(t['source_codes'])} codes)\n")
        f.write(f"   {t['definition']}\n\n")

# Save action log for debugging
log_path = os.path.join(OUTPUT_DIR, "merge_log.txt")
with open(log_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(log_lines))

print(f"Saved results to {OUTPUT_DIR}")

# Print size distribution
sorted_themes = sorted(themes, key=lambda t: len(t['source_codes']), reverse=True)
print("\nSize distribution:")
for idx, t in enumerate(sorted_themes[:20]):
    print(f"  {t['name']}: {len(t['source_codes'])}")