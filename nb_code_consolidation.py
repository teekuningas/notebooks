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

MAX_REJECTIONS_PER_ITERATION = 100
ARTIFACT_BATCH_SIZE = 20
RANDOM_SEED = 11

# Set random seed for reproducibility
if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

uuid_part = str(uuid4())[:8]
if RANDOM_SEED is not None:
    RUN_ID = f"{uuid_part}_seed{RANDOM_SEED}"
else:
    RUN_ID = uuid_part

OUTPUT_DIR = f"output/consolidation/{RUN_ID}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Run ID: {RUN_ID}")
print(f"Random Seed: {RANDOM_SEED}")
# %%
# --- Parse Input Codes ---

def parse_code_file(filepath):
    """Parse code file using state-machine approach."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    all_codes = []
    current_rec_id = "unknown"
    current_iter = -1
    current_code = None
    current_explanation_lines = []
    
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
        current_code = None
        current_explanation_lines = []

    for line in lines:
        clean_line = line.strip()
        
        if not clean_line:
            continue
            
        if "Rec ID Anonymized:" in line:
            save_current_entry()
            match = re.search(r'Rec ID Anonymized:\s*([a-f0-9\-]+)', line)
            if match:
                current_rec_id = match.group(1)
                current_iter = -1
            continue

        if clean_line.startswith("Iter") and clean_line.endswith(":"):
            save_current_entry()
            match = re.search(r'Iter\s+(\d+):', clean_line)
            if match:
                current_iter = int(match.group(1))
            continue

        if "- Code:" in line:
            save_current_entry()
            parts = line.split("- Code:", 1)
            if len(parts) > 1:
                current_code = parts[1].strip()
            continue

        if "Explanation:" in line:
            parts = line.split("Explanation:", 1)
            if len(parts) > 1:
                current_explanation_lines.append(parts[1].strip())
            continue
            
        if "-----------------" in line:
            continue
            
        if current_code is not None:
            current_explanation_lines.append(clean_line)

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
# --- Remove Singleton Codes ---

code_interviews = {}
for item in raw_codes:
    code_key = item['code'].strip().lower()
    if code_key not in code_interviews:
        code_interviews[code_key] = set()
    code_interviews[code_key].add(item['rec_id'])

filtered_raw_codes = [
    item for item in raw_codes 
    if len(code_interviews[item['code'].strip().lower()]) >= 2
]

print(f"Original codes: {len(raw_codes)}")
print(f"Codes in 2+ interviews: {len(filtered_raw_codes)}")
print(f"Singletons removed: {len(raw_codes) - len(filtered_raw_codes)}")

raw_codes = filtered_raw_codes

# %%
# --- Deduplicate and Remove Artifacts ---

def detect_artifacts_batch(code_names):
    """Ask LLM to identify technical artifacts (K1, L3, etc.)."""
    codes_list = "\n".join([f"- {name}" for name in code_names])
    
    prompt = f"""Tunnista seuraavista koodeista VAIN selkeät tekniset artefaktit.

Artefakteja ovat AINOASTAAN:
- Numerokoodit tai tunnisteet (esim. "1.1", "K1", "L3", "T2")
- Lyhenteet joissa numero (esim. "LS1", "M1")

EI artefakteja:
- Oikeat suomenkieliset sanat tai fraasit
- Käsitteet jotka kuvaavat teemoja

KOODIT:
{codes_list}

Palauta VAIN numerokoodit. Tyhjä lista jos ei ole.
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
        response = generate_simple(prompt, "Tunnista.", output_format=output_format, provider="llamacpp", seed=RANDOM_SEED or 0)
        return json.loads(response).get('artifacts', [])
    except:
        return []

grouped_codes = {}
for item in raw_codes:
    key = item['code'].strip().lower()
    if key not in grouped_codes:
        grouped_codes[key] = []
    grouped_codes[key].append(item)

print(f"Unique code names: {len(grouped_codes)}")

# Detect and remove artifacts
# Sort keys for deterministic order
sorted_keys = sorted(grouped_codes.keys())
unique_names = [grouped_codes[key][0]['code'].strip() for key in sorted_keys]
all_artifacts = set()

for i in range(0, len(unique_names), ARTIFACT_BATCH_SIZE):
    batch = unique_names[i:i+ARTIFACT_BATCH_SIZE]
    artifacts = detect_artifacts_batch(batch)
    all_artifacts.update(a.lower() for a in artifacts)

if all_artifacts:
    print(f"Artifacts removed: {len(all_artifacts)}")
    for key in all_artifacts:
        grouped_codes.pop(key, None)

print(f"Clean unique codes: {len(grouped_codes)}")

# Build initial themes
themes = []

print("Building initial themes...")

# Sort keys for deterministic order
sorted_keys = sorted(grouped_codes.keys())
for key in sorted_keys:
    items = grouped_codes[key]
    names = [item['code'].strip() for item in items]
    most_common_name = max(set(names), key=names.count)

    theme = {
        "id": str(uuid4()),
        "name": most_common_name,
        "source_codes": items,
        "embedding": None
    }
    themes.append(theme)

print(f"\nInitial themes: {len(themes)}")

# %%
# --- Embed Themes ---

for theme in themes:
    text_to_embed = theme['name'] # Embed only name
    theme['embedding'] = embed(text_to_embed, provider="llamacpp")

print(f"Embedded {len(themes)} themes")

# %%
# --- Merge Loop: Consolidate Synonyms ---
# Embeddings guide which pairs to test; LLM judges if they're truly synonyms.

def get_cosine_similarity_matrix(current_themes):
    """Compute pairwise cosine similarity between theme embeddings."""
    embs = np.array([t['embedding'] for t in current_themes])
    sim = np.dot(embs, embs.T)
    norms = np.linalg.norm(embs, axis=1) + 1e-9
    sim /= np.outer(norms, norms)
    np.fill_diagonal(sim, -np.inf)  # Exclude self-similarity
    return sim


def generate_merged_theme_name(theme_a, theme_b):
    """
    Ask LLM to generate a merged theme name from two themes.
    Returns the proposed name as a string.
    """
    prompt = f"""Yhdistä nämä kaksi teemaa yhdeksi uudeksi teemaksi.

TEEMA 1: {theme_a['name']}
TEEMA 2: {theme_b['name']}

Ehdota uudelle teemalle nimi, joka kuvaa parhaiten näiden kahden käsitteen yhteistä merkitystä.

OHJEET:
1. Nimen on oltava yksinkertainen, yleiskielinen ilmaisu.
2. Jos toinen alkuperäisistä nimistä on hyvä ja kuvaa molempia, käytä sitä.
3. Vältä liian yleisiä yläkäsitteitä.
4. Uuden käsitteen tulisi säilyttää alkuperäisten käsitteiden "kokoluokka".

Vastaa JSON-muodossa: {{ "name": "..." }}"""

    output_format = {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    }
    
    try:
        response = generate_simple(prompt, "Ehdota.", output_format=output_format, provider="llamacpp", seed=RANDOM_SEED or 0)
        data = json.loads(response)
        return data['name']
    except Exception as e:
        print(f"Name generation error: {e}. Fallback to combining names.")
        return f"{theme_a['name']} & {theme_b['name']}"


def check_merge_validity(theme_a, theme_b, proposed_name):
    """
    Ask LLM to decide if two themes should be merged to the proposed name.
    Returns True if should merge, False otherwise.
    """
    prompt = f"""Arvioi ovatko kaksi teemaa sama käsite.

TEEMA 1: {theme_a['name']}
TEEMA 2: {theme_b['name']}

EHDOTETTU YHDISTETTY NIMI: {proposed_name}

TÄRKEÄÄ: Arvioi ENSIN ovatko alkuperäiset teemat tarpeeksi samankaltaisia.
Vasta sitten arvioi onko ehdotettu nimi hyvä.

VAIHE 1 - Tarkista alkuperäiset:
Ovatko TEEMA 1 ja TEEMA 2 synonyymejä tai tarkoittavat käytännössä samaa asiaa?

JOS EI → Vastaa FALSE (vaikka ehdotettu nimi olisi kuinka hyvä tahansa!)

VAIHE 2 - Jos alkuperäiset ovat samankaltaisia:
Kuvaako ehdotettu nimi hyvin molempia?

Vastaa FALSE jos:
  - Alkuperäiset teemat ovat ERI ASIOITA (tärkein ehto!)
  - Toinen teema on yleisempi tai spesifimpi kuin toinen
  - Teemat ovat eri kokoluokkaa
  - Teemat kuvaavat eri puolia samasta ilmiöstä
  - Ehdotettu nimi on liian yleinen tai menettää alkuperäisten merkityksen
  - On VÄHÄNKÄÄN epävarmuutta!

Ovatko teemat sama käsite?

Vastaa JSON: {{ "are_same": true, "reason": "lyhyt perustelu" }} tai {{ "are_same": false, "reason": "lyhyt perustelu" }}"""
    
    output_format = {
        "type": "object",
        "properties": {
            "are_same": {"type": "boolean"},
            "reason": {"type": "string"}
        },
        "required": ["are_same", "reason"]
    }
    
    try:
        response = generate_simple(prompt, "Arvioi objektiivisesti.", output_format=output_format, provider="llamacpp", seed=RANDOM_SEED or 0)
        
        # Debug: Print the raw JSON response
        print(f"    [LLM] {response}")
        
        data = json.loads(response)
        return data['are_same']
    except Exception as e:
        print(f"    [LLM] ERROR: {e}")
        return False


def create_merged_theme(theme_a, theme_b, merged_name):
    """Create a new merged theme object with combined source codes."""
    combined_sources = theme_a['source_codes'] + theme_b['source_codes']
    
    # Compute new embedding for the combined theme
    new_emb = embed(merged_name, provider="llamacpp")
    
    new_theme = {
        "id": str(uuid4()),
        "name": merged_name,
        "source_codes": combined_sources,
        "embedding": new_emb
    }
    return new_theme


# --- Iterative Merge Loop ---

log_lines = []

def log(msg):
    """Print and record to log."""
    print(msg)
    log_lines.append(msg)

# Log header
log(f"{ '='*60}")
log(f"Code Consolidation")
log(f"{ '='*60}")
log(f"Run ID: {RUN_ID}")
log(f"Random Seed: {RANDOM_SEED}")
log(f"Initial themes: {len(themes)}")
log(f"{ '='*60}")

tested_pairs = set()
total_merges = 0

def get_content_key(theme):
    """Create cache key from theme's source codes."""
    ids = sorted(f"{item['rec_id']}_{item['iter_idx']}_{item['code']}_{item['explanation']}" 
                 for item in theme['source_codes'])
    return tuple(ids)

def get_cache_key(theme_a, theme_b):
    """Create cache key for theme pair."""
    key_a = get_content_key(theme_a)
    key_b = get_content_key(theme_b)
    if key_a <= key_b:
        return (key_a, key_b)
    return (key_b, key_a)

while True:
    n_themes = len(themes)
    
    log(f"\n{ '='*60}")
    log(f"--- Iteration: {n_themes} themes ---")
    
    # Compute embedding similarity and use as priority
    sim_matrix = get_cosine_similarity_matrix(themes)
    
    # Test pairs in order of similarity (most promising first)
    flat_indices = np.argsort(sim_matrix, axis=None)[::-1]
    
    found_merge = False
    rejections_this_iteration = 0
    
    for idx in flat_indices:
        i, j = np.unravel_index(idx, sim_matrix.shape)
        if i >= j:
            continue
            
        t_a, t_b = themes[i], themes[j]
        cache_key = get_cache_key(t_a, t_b)
        
        if cache_key in tested_pairs:
            continue
        
        # Step 1: Generate proposed merged name
        proposed_name = generate_merged_theme_name(t_a, t_b)
        
        # Step 2: Ask LLM to decide if should merge to this name
        should_merge = check_merge_validity(t_a, t_b, proposed_name)
        tested_pairs.add(cache_key)
        
        if should_merge:
            new_theme = create_merged_theme(t_a, t_b, proposed_name)
            log(f"  ✓ '{t_a['name']}' + '{t_b['name']}' → '{proposed_name}' ({len(new_theme['source_codes'])} codes) [LLM: YES]")
            themes = [t for k, t in enumerate(themes) if k != i and k != j]
            themes.append(new_theme)
            total_merges += 1
            found_merge = True
            break
        else:
            rejections_this_iteration += 1
            if rejections_this_iteration >= MAX_REJECTIONS_PER_ITERATION:
                break
    
    if found_merge:
        log(f"  Tests: {rejections_this_iteration + 1}, Merges: 1 → {len(themes)} themes remaining")
    else:
        log(f"  Tests: {rejections_this_iteration}, Merges: 0 → stopping")
        log(f"\n{ '='*60}")
        log(f"  Total pairs tested: {len(tested_pairs)}")
        log(f"  Total merges: {total_merges}")
        log(f"  Final themes: {len(themes)}")
        break

log(f"\n{ '='*60}")
log(f"Final: {len(themes)} themes")

# %%
# --- Save Results ---

# Deduplicate theme names
name_counts = {}
for t in themes:
    name_counts[t['name']] = name_counts.get(t['name'], 0) + 1

if any(c > 1 for c in name_counts.values()):
    seen = {}
    for t in themes:
        name = t['name']
        if name_counts[name] > 1:
            count = seen.get(name, 0) + 1
            seen[name] = count
            t['name'] = f"{name} ({count})"

# Save JSON
output_path = os.path.join(OUTPUT_DIR, "final_themes.json")
with open(output_path, 'w', encoding='utf-8') as f:
    serializable_themes = []
    for t in themes:
        t_copy = t.copy()
        if 'embedding' in t_copy:
            del t_copy['embedding']
        serializable_themes.append(t_copy)
    json.dump(serializable_themes, f, indent=2, ensure_ascii=False)

# Save readable list
simple_output_path = os.path.join(OUTPUT_DIR, "final_themes_list.txt")
with open(simple_output_path, 'w', encoding='utf-8') as f:
    f.write(f"Code Consolidation Results\n")
    f.write(f"==========================\n")
    f.write(f"Run ID: {RUN_ID}\n")
    f.write(f"Random seed: {RANDOM_SEED}\n")
    f.write(f"Final themes: {len(themes)}\n\n")
    
    sorted_themes = sorted(themes, key=lambda t: len(t['source_codes']), reverse=True)
    for idx, t in enumerate(sorted_themes):
        f.write(f"{idx+1}. {t['name']} ({len(t['source_codes'])} codes)\n\n")

# Save log
log_path = os.path.join(OUTPUT_DIR, "merge_log.txt")
with open(log_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(log_lines))

print(f"\nSaved to {OUTPUT_DIR}")

# Print size distribution
sorted_themes = sorted(themes, key=lambda t: len(t['source_codes']), reverse=True)
print("\nTop themes:")
for t in sorted_themes[:30]:
    print(f"  {t['name']}: {len(t['source_codes'])}")

# %%
