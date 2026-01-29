import os
import glob
import pandas as pd
import re
import sys

# Statistical mode configuration
STATS_MODE = os.environ.get('STATS_MODE', 'random_effects')  # Default: random_effects
if STATS_MODE not in ['random_effects', 'chisquared']:
    raise ValueError(f"Invalid STATS_MODE: {STATS_MODE}. Must be 'random_effects' or 'chisquared'")

def find_latest_dir(base_path):
    dirs = [d for d in glob.glob(os.path.join(base_path, "*")) if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def parse_merge_log(log_path):
    stats = {
        "initial_themes": 0,
        "final_themes": 0,
        "total_merges": 0,
        "run_id": "unknown"
    }

    if not os.path.exists(log_path):
        return stats

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract Run ID
    match = re.search(r"Run ID: ([^\n]+)", content)
    if match:
        stats["run_id"] = match.group(1).strip()

    # Extract Initial themes
    match = re.search(r"Initial themes: (\d+)", content)
    if match:
        stats["initial_themes"] = int(match.group(1))

    # Extract Final themes
    match = re.search(r"Final themes: (\d+)", content)
    if match:
        stats["final_themes"] = int(match.group(1))

    # Extract Total merges
    match = re.search(r"Total merges: (\d+)", content)
    if match:
        stats["total_merges"] = int(match.group(1))

    return stats

def get_all_themes(list_path):
    themes = []
    if not os.path.exists(list_path):
        return themes

    with open(list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        # Match format: "1. Theme Name (123 codes)"
        match = re.match(r"\d+\.\s+(.+?)\s+\((\d+)\s+codes\)", line.strip())
        if match:
            themes.append(match.group(1))

    return themes

def analyze_presence_csv(csv_path):
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path, index_col=0)

    # Calculate prevalence for each theme (column)
    prevalence = df.mean()

    # Filter for 0.2 - 0.8 range
    valid_themes = prevalence[(prevalence >= 0.2) & (prevalence <= 0.8)].index.tolist()

    stats = {
        "n_interviews": len(df),
        "n_themes_analyzed": len(df.columns),
        "avg_themes_per_interview": df.sum(axis=1).mean(),
        "valid_themes": valid_themes,
        "total_themes": len(prevalence)
    }

    return stats

def parse_raw_codes(filepath):
    """Parse raw code file to calculate pre-consolidation stats."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    all_codes = []
    current_rec_id = None

    for line in lines:
        line = line.strip()
        if "Rec ID Anonymized:" in line:
            match = re.search(r'Rec ID Anonymized:\s*([a-f0-9\-]+)', line)
            if match:
                current_rec_id = match.group(1)
        elif "- Code:" in line:
            parts = line.split("- Code:", 1)
            if len(parts) > 1 and current_rec_id:
                code = parts[1].strip()
                all_codes.append({"code": code, "rec_id": current_rec_id})

    # 1. Total raw codes
    total_raw = len(all_codes)

    # 2. Filter singletons (appear in < 2 interviews)
    code_counts = {}
    for item in all_codes:
        c = item['code'].lower().strip()
        if c not in code_counts:
            code_counts[c] = set()
        code_counts[c].add(item['rec_id'])

    # Codes that appear in at least 2 interviews
    kept_codes = [c for c, ids in code_counts.items() if len(ids) >= 2]
    n_kept_unique = len(kept_codes)

    # Calculate how many raw entries were removed
    # (This is an approximation since we are counting unique names vs raw entries, 
    # but the user asked for "remove 1255 themes while keeping 1579")
    # Let's count raw entries kept vs removed
    kept_raw_entries = [item for item in all_codes if len(code_counts[item['code'].lower().strip()]) >= 2]
    n_kept_raw = len(kept_raw_entries)
    n_removed_raw = total_raw - n_kept_raw

    # Get top 10 raw themes
    sorted_codes = sorted(code_counts.items(), key=lambda x: len(x[1]), reverse=True)
    top_10_raw = [{"name": c.capitalize(), "count": len(ids)} for c, ids in sorted_codes[:10]]

    return {
        "total_raw": total_raw,
        "n_removed_raw": n_removed_raw,
        "n_kept_raw": n_kept_raw,
        "n_kept_unique": n_kept_unique,
        "top_10": top_10_raw
    }

def generate_report():
    # Parse command-line arguments for custom output directory
    custom_output = None
    for i, arg in enumerate(sys.argv):
        if arg == '--output' and i + 1 < len(sys.argv):
            custom_output = sys.argv[i + 1]

    # Statistical mode configuration
    stats_mode = os.environ.get('STATS_MODE', 'random_effects')  # Default: random_effects
    if stats_mode not in ['random_effects', 'chisquared']:
        raise ValueError(f"Invalid STATS_MODE: {stats_mode}. Must be 'random_effects' or 'chisquared'")

    # 1. Consolidation Stats
    log_path = "output/consolidation/77db8e4e_seed11/merge_log.txt"
    list_path = "output/consolidation/77db8e4e_seed11/final_themes_list.txt"
    raw_codes_path = "output/koodit/487ef9e9/koodit_raw_rec_id_anonymized.txt"

    cons_stats = parse_merge_log(log_path)
    all_themes = get_all_themes(list_path)
    raw_stats = parse_raw_codes(raw_codes_path)

    # Calculate artifacts removed
    # We know the "Initial themes" for merging (from log)
    # We know "n_kept_unique" from raw parsing
    # Difference is artifacts
    artifacts_removed = 0
    if raw_stats and cons_stats['initial_themes'] > 0:
        artifacts_removed = raw_stats['n_kept_unique'] - cons_stats['initial_themes']

    # 2. Analysis Stats - Load both datasets
    csv_path_710 = "output/analyysi_koodit/7176421e/themes_98x710.csv"
    csv_path_452 = "output/analyysi_koodit/88d43208/themes_98x452.csv"

    analysis_stats_710 = analyze_presence_csv(csv_path_710)
    analysis_stats_452 = analyze_presence_csv(csv_path_452)

    # 3. Load theme translations if available
    translation_table = ""
    trans_dict = {}
    try:
        from utils_translate import load_translation_dict
        trans_dict = load_translation_dict()
        # Filter to only include capitalized versions (the actual theme names)
        theme_translations = [(fi, en) for fi, en in trans_dict.items() if fi[0].isupper()]
        # Sort by Finnish name
        theme_translations.sort(key=lambda x: x[0])

        translation_table = "\n## Theme Translations\n\n"
        translation_table += "**Note**: English translations are provided for readability. All analysis was performed using the original Finnish theme names.\n\n"
        translation_table += "The following table shows all 98 themes with their Finnish and English names:\n\n"
        translation_table += "| Finnish | English |\n"
        translation_table += "| :--- | :--- |\n"
        for fi, en in theme_translations:
            translation_table += f"| {fi.capitalize()} | {en.capitalize()} |\n"
        translation_table += "\n"
    except Exception as e:
        print(f"Note: Could not load translations: {e}")

    # Format lists with capitalization (use 710 dataset as primary)
    formatted_all_themes = ", ".join([t.capitalize() for t in all_themes])
    formatted_valid_themes_710 = ", ".join([t.capitalize() for t in analysis_stats_710['valid_themes']])

    # Translate filtered themes for display
    formatted_valid_themes_710_en = ""
    if trans_dict:
        valid_themes_710_en = [trans_dict.get(t.capitalize(), t.capitalize()).capitalize() for t in analysis_stats_710['valid_themes']]
        formatted_valid_themes_710_en = ", ".join(valid_themes_710_en)

    # Format top 10 table with translations
    top_10_rows = ""
    for t in raw_stats['top_10']:
        finnish = t['name']
        english = trans_dict.get(finnish, finnish) if trans_dict else finnish
        top_10_rows += f"| {finnish} | {english.capitalize()} | {t['count']} |\n"

    # Statistical method - always show both methods since both datasets are analyzed
    stats_method = """### Statistical method

Two different statistical approaches were used for the two datasets:

**710 dataset** (multiple interviews per user): Mixed-effects logistic regression with random user intercepts (R package lme4::glmer) to account for repeated measures:

```
glmer(theme ~ predictor + (1|user), family=binomial)
```

This controls for user-level confounding and within-user correlation.

**452 dataset** (one interview per user): Chi-squared tests with Cramér's V as the effect size measure, providing descriptive associations without user clustering.

Both methods use False Discovery Rate correction (FDR, Benjamini-Hochberg) for multiple comparisons."""

    # 3. Generate Markdown
    report = f"""# Themes

## 1. Initial extraction
We extracted themes and concepts from interviews using a local LLM (llama3.3:70b). Each interview was processed with the instruction:

> "Olet laadullisen tutkimuksen avustaja. Tehtäväsi on tunnistaa tekstistä teemat ja käsitteet ja muodostaa niistä koodikirja. Perustele selkeästi jokaiselle koodille erikseen millä tavoin se ilmenee tekstissä. Vältä lyhenteitä. Vastaa suomeksi."

This produced a total of {raw_stats['total_raw']} raw codes (approx. {raw_stats['total_raw']/analysis_stats_710['n_interviews']:.1f} per interview).

The most frequent themes in this initial set were:

| Finnish | English | Interviews |
| :--- | :--- | :--- |
{top_10_rows}

## 2. Cleaning up
Before merging, we cleaned the raw data:

1. Singleton removal: We removed {raw_stats['n_removed_raw']} codes that appeared in only one interview, keeping {raw_stats['n_kept_raw']} codes.
2. Deduplication: The remaining {raw_stats['n_kept_raw']} codes were grouped by name, resulting in {raw_stats['n_kept_unique']} unique themes.
3. Artifact removal: From these {raw_stats['n_kept_unique']} unique themes, we removed {artifacts_removed} technical artifacts (e.g., numbering or nonsense codes).

This left {cons_stats['initial_themes']} unique themes for the consolidation phase.

## 3. Merging similar themes
The themes were consolidated to remove duplicates and synonyms through an iterative process.

In each iteration:

1. We calculated embeddings for all themes to identify the most similar pairs.
2. These pairs were evaluated by a local LLM (llama3.3:70b) with the instruction:

> "Arvioi ovatko kaksi teemaa sama käsite."

3. If the local LLM confirmed they were the same concept, they were merged.

The process repeated until 50 consecutive pairs were rejected. This resulted in {cons_stats['final_themes']} consolidated themes.

{translation_table}

**Note**: English translations are provided for readability. All analysis was performed using the original Finnish theme names.

## 4. Re-evaluating themes in interviews
The consolidated themes were then checked against the original interview texts to determine their presence or absence. The instruction used was:

> "Päätä esiintyykö seuraava teema tekstinäytteessä."

## 5. Datasets

Two datasets were created:

- **452 dataset**: One interview per user (n={analysis_stats_452['n_interviews']} interviews)
- **710 dataset**: All interviews (n={analysis_stats_710['n_interviews']} interviews, multiple per user)

## 6. Statistical Analysis

For the statistical analysis, we filtered the themes to include only those with a prevalence between 20% and 80%. This ensures sufficient variation for statistical testing.

**Filtering based on 710 dataset**: {len(analysis_stats_710['valid_themes'])} themes met this criterion:

{formatted_valid_themes_710_en}

**Note**: This same filtered theme list was used for both the 452 and 710 dataset analyses to ensure comparability.

{stats_method}"""

    # Determine output file path
    if custom_output:
        os.makedirs(custom_output, exist_ok=True)
        output_file = os.path.join(custom_output, "METHODOLOGY_SUMMARY.md")
    else:
        output_file = "output/METHODOLOGY_SUMMARY.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report generated: {output_file}")

if __name__ == "__main__":
    generate_report()
