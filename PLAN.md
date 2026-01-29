# Publication Materials Preparation Plan

## Overview
This plan outlines the steps to create publication-ready materials from the bird observation and interview theme analysis. All outputs will be in English and organized in `./output/ossi_all/` directory.

**Key Principle**: Minimally destructive - keep original functionality intact, use parameterization to enable both old and new workflows.

---

## 📊 Progress Tracker

### ✅ Phase 1: Theme Translation Infrastructure - COMPLETED
- **Theme Translation File**: `./inputs/translations/theme_translations.csv` (98 themes)
- **Translation Utility**: `utils_translate.py` with full functionality
- **Testing**: Verified with actual 710×98 dataset
- **Quality Review**: 20 improvements through 3 review rounds
  - Round 1: 16 improvements (critical: "Loneliness" → "Solitude")
  - Round 2: 3 improvements from external review
  - Round 3: 1 accessibility improvement ("Olfactory experiences" → "Smells and scents")
- **Status**: All 98 themes translated with publication-ready academic English
- **Key Files Created**:
  - `./inputs/translations/theme_translations.csv` - Translation mapping
  - `utils_translate.py` - Translation utility with caching

**How to verify translations work:**
```python
from utils_translate import load_translation_dict, translate_theme_names
import pandas as pd

# Test with actual data
themes = pd.read_csv('./output/analyysi_koodit/7176421e/themes_98x710.csv', index_col=0)
themes_en = translate_theme_names(themes)
print(f"✓ Translated {len(themes_en.columns)} theme columns")

# Test individual translations
trans = load_translation_dict()
print(f"✓ Loaded {len(trans)} translations")
print(f"Example: {list(trans.items())[0]}")
```

**Critical implementation notes:**
- Translation utility uses `@lru_cache` - must call `load_translation_dict.cache_clear()` after updating CSV
- Theme names in dataset are capitalized (e.g., "Aika" not "aika") - utility handles both
- Returns original name if translation not found (fail-safe behavior)
- All 98 translations are unique (no duplicates in English)

### ✅ Phase 2: Update Statistical Analysis Scripts - COMPLETED
- **Modified 4 scripts**: All `nb_stats_*.py` scripts now support optional translation
- **Implementation**: Added conditional translation after theme loading (~line 97-103 in each)
- **Activation**: Only when `ENABLE_TRANSLATION=1` environment variable is set
- **Testing**: Verified with both Finnish (default) and English (translated) outputs
- **Key Files Modified**:
  - `nb_stats_esa_koodit.py` - ESA habitat analysis
  - `nb_stats_bird_groups_koodit.py` - Bird functional groups analysis
  - `nb_stats_bird_presence_koodit.py` - Bird presence analysis
  - `nb_stats_bird_species_koodit.py` - Bird species analysis

### ✅ Phase 3: Update Supporting Scripts - COMPLETED
- **Modified 4 scripts**: All supporting scripts now have English output and command-line parameters
- **Changes made**:
  - `nb_create_comparison_plots.py`: Added `--base-dir` and `--output-dir` parameters
  - `nb_methodology_summary.py`: Always shows both 452 & 710 datasets, added `--output` parameter, includes theme translations table
  - `nb_kartat.py`: Changed labels to English ("prevalence", "Ratio"), added `--all-themes` and `--output` flags, supports translation
  - `nb_interview_comprehensive.py`: Added output directory as command-line argument
- **English labels**: All hardcoded labels now in English (e.g., "yleisyys" → "prevalence", "Osuus" → "Ratio")
- **Translation**: Map and theme names translated when `ENABLE_TRANSLATION=1`

### ✅ Phase 4: Create Orchestration Files - COMPLETED
- **Created Makefile.ossi**: Complete orchestration for publication workflow
- **Key features**:
  - Uses 710 prevalence filter for both datasets
  - ENABLE_TRANSLATION=1 set globally
  - Outputs to ./output/ossi_all/
  - Direct Python script calls (no bash wrapper)
  - Individual and combined targets available
- **Targets created**: ossi-run-452, ossi-run-710, ossi-run-comparison, ossi-run-maps, ossi-run-interview, ossi-run-methodology, ossi-pdf-*, ossi-all
- **Simplification**: Removed redundant bash wrapper, Makefile calls Python scripts directly

### ⏳ Phase 5: Theme Translation Execution - SKIPPED (Done in Phase 1)

### ✅ Phase 6: Testing and Validation - COMPLETED
- **Tested full pipeline**: Successfully ran `make -f Makefile.ossi ossi-all`
- **All 12 PDFs generated** in `./output/ossi_all/`
- **English output verified**: Theme names, labels, all in English
- **Bug fixed**: Translation now applies to both outcome and prevalence filter datasets
- **Cache optimization**: Interview analysis cache copied to new location

### 🎯 Phase 7: Quality Improvements - IN PROGRESS

Based on initial review, the following improvements identified:

#### ✅ 7.1: Translation Completeness - COMPLETED
**Changes made (2026-01-29):**
- ✅ **ESA habitat names**: Updated English display names to be less technical
  - "Bare / sparse vegetation" → "Bare ground"
  - "Built-up" → "Urban"
  - "Permanent water bodies" → "Water"
  - "Herbaceous wetland" → "Wetland"
  - Updated `scripts/create_habitat_presence_esa.py` and regenerated metadata files
- ✅ **nb_stats_esa_koodit.py**: Now uses English habitat file when `ENABLE_TRANSLATION=1`
- ✅ **nb_stats_bird_groups_koodit.py**: Now uses English bird group file when `ENABLE_TRANSLATION=1`
- ✅ **nb_stats_bird_species_koodit.py**: Now uses English bird species file when `ENABLE_TRANSLATION=1`
- ✅ **Comparison plots**: Automatically use translated names from results.csv files
- ✅ **Tested**: Verified translations work correctly

**Next step**: Run `make -f Makefile.ossi ossi-all` to regenerate all outputs with improved translations

#### ✅ 7.2: Visual Consistency - COMPLETED
**Changes made (2026-01-29):**
- ✅ **Bird presence colors**: Changed from green/red to green/purple in `utils_stats.py`
  - `plot_binary_predictor_distribution()`: Purple for "no bird", green for "bird present"
  - `plot_binary_predictor_effects()`: Green for positive difference, purple for negative
  - `plot_binary_predictor_prevalence()`: Green for "with predictor", purple for "without"
  - Colors now match comparison plots: `#90EE90` (green), `#B19CD9` (purple)
- ✅ **Interview analysis colors**: Changed from blue/red to green/purple in `nb_interview_comprehensive.py`
  - Binary bar charts: Purple for 0, green for 1
  - Odds ratio reference line: Purple instead of red
  - Bar colors: Green for main bars
- ✅ **Theme maps**: Added footnote explaining proportion calculation in `nb_kartat.py`
  - Changed colorbar label from "Ratio" to "Proportion"
  - Added footnote: "Proportion: Ratio of interviews within each grid cell where this theme was present."

**Result**: All visualizations now use consistent green/purple color scheme throughout the project

#### ✅ 7.3: Plot Clarity - COMPLETED
**Changes made (2026-01-29):**
- ✅ **Bar chart filtering**: Modified `plot_top_associations_barplot()` in `utils_stats.py`
  - Removed effect size threshold filtering (V >= 0.1, |coef| >= threshold)
  - Now filters by p-value: only shows associations with **p ≤ 0.01**
  - **Rationale**: Too many bars to fit on A4 paper with old filtering
  - **Result**: Cleaner plots with fewer, more significant associations

#### ✅ 7.4: Methodology Documentation - COMPLETED
**Changes made (2026-01-29) in `nb_methodology_summary.py`:**
- ✅ **Top 10 themes table**: Now shows both Finnish and English names
- ✅ **Translation note added**: "English translations are provided for readability in international publications. All analysis was performed using the original Finnish theme names."
- ✅ **Removed technical details**: 
  - Removed "nb_code_consolidation.py (seed=11)" reference
  - Now just says "through an iterative process"
- ✅ **Prevalence filtering section improved**:
  - Shows all 98 consolidated themes in Finnish
  - Shows filtered list (710 prevalence, 20-80%) in Finnish
  - Shows same filtered list translated to English
  - Clarifies this list used for both 710 and 452 analyses
  - Removed confusing 452-specific filtered list section
- ✅ **Better structure**: "Statistical Analysis" section now clearer

**Result**: Methodology documentation is now clearer, less technical, and shows bilingual theme information appropriately.

---

## 🎉 Phase 7 Complete!

All quality improvements have been implemented:
- ✅ 7.1: Translation Completeness
- ✅ 7.2: Visual Consistency
- ✅ 7.3: Plot Clarity
- ✅ 7.4: Methodology Documentation

**Status (2026-01-29)**: Successfully regenerated all outputs. Analysis completed, but found a few minor refinements needed.

---

## 🔧 Phase 7.5: Final Polish (Minor Tweaks Identified)

After running `make -f Makefile.ossi ossi-all`, the following minor issues were identified:

#### 7.5.1: Interview Analysis Colors - Not All Updated
- **Issue**: Most plots in `nb_interview_comprehensive.py` still use blue color instead of green
- **Location**: Histogram plots (lines ~195, 239, 258, 269, 278, 340, 349, 358, 366, 483, 823)
- **Fix needed**: Change `color='skyblue'` to `color='#90EE90'` (green) for consistency

#### 7.5.2: Methodology - Missing Translations
- **Issue**: Not all of the top 10 most frequent themes have translations in the translation file
- **Fix needed**: 
  1. Check which themes are missing from `./inputs/translations/theme_translations.csv`
  2. Add missing translations to the file

#### 7.5.3: Methodology - Restructure Theme Display
- **Current**: Translation table shown at the end, 98 themes listed as comma-separated
- **Desired**: 
  1. After "This resulted in 98 consolidated themes", show the **translation table** (with bilingual note)
  2. In "Filtering based on 710 dataset" section, show **only English names** (no Finnish list)
- **Rationale**: Better flow, readers see translations immediately after theme introduction

#### 7.5.4: Theme Maps - Image Quality
- **Issue**: Maps appear pixelated (currently saved at 300 DPI)
- **Fix needed**: Investigate doubling the resolution (e.g., 600 DPI or larger figure size)
- **Location**: `nb_kartat.py` - `plt.savefig(..., dpi=300)`

---

## Current Status Assessment

### Existing Infrastructure
- ✅ 4 statistical analysis scripts: `nb_stats_esa_koodit.py`, `nb_stats_bird_groups_koodit.py`, `nb_stats_bird_presence_koodit.py`, `nb_stats_bird_species_koodit.py`
- ✅ Comparison analysis script: `nb_create_comparison_plots.py` + `run_comparison_analysis.sh`
- ✅ Methodology summary: `nb_methodology_summary.py`
- ✅ Maps script: `nb_kartat.py`
- ✅ Interview analysis: `nb_interview_comprehensive.py`
- ✅ Makefile.stats for orchestration (will remain unchanged)
- ✅ English bird metadata already exists in `./inputs/bird-metadata-refined/`

### Key Datasets
- **452 dataset**: `./output/analyysi_koodit/88d43208/themes_98x452.csv` (one interview per user, chi-squared tests)
- **710 dataset**: `./output/analyysi_koodit/7176421e/themes_98x710.csv` (multiple interviews per user, mixed-effects models)
- **98 themes** in Finnish (need English translation)

### Theme Extraction Process (Verified)
From code inspection and merge log analysis:

1. **nb_koodit.py** (run_id: 487ef9e9): LLM extracts codes from interviews in Finnish
   - Uses llamacpp provider
   - Generates codes + explanations for each interview
   - N_ITERATIONS = 1

2. **Consolidation - Which script was used?**
   - ✅ **nb_code_consolidation.py** (RANDOM_SEED=11, no MERGE_THRESHOLD)
   - ❌ **nb_code_explanation_consolidation.py** (RANDOM_SEED=10, MERGE_THRESHOLD=0.75)
   - **Evidence**: merge_log.txt shows "Random Seed: 11" (matches nb_code_consolidation.py)
   - Method: Embeds code names only → finds similar pairs → LLM judges if synonyms
   - Process: 268 initial → 170 merges → 98 final themes
   - Run ID: 77db8e4e_seed11

3. **nb_analyysi_koodit.py**: Applies 98 themes to interviews
   - LLM judges presence of each theme in each interview
   - Generates presence matrices: themes_98x452.csv, themes_98x710.csv

## Required Deliverables

### 1. Statistical Analyses - 452 Dataset (4 PDFs)
- `ossi_all/esa_themes_452.pdf` - ESA habitats × themes
- `ossi_all/bird_groups_themes_452.pdf` - Bird functional groups × themes
- `ossi_all/bird_presence_themes_452.pdf` - Bird presence × themes
- `ossi_all/bird_species_themes_452.pdf` - Bird species × themes

### 2. Statistical Analyses - 710 Dataset (4 PDFs)
- `ossi_all/esa_themes_710.pdf` - ESA habitats × themes
- `ossi_all/bird_groups_themes_710.pdf` - Bird functional groups × themes
- `ossi_all/bird_presence_themes_710.pdf` - Bird presence × themes
- `ossi_all/bird_species_themes_710.pdf` - Bird species × themes

### 3. Comparison Analysis (1 PDF)
- `ossi_all/stats_comparison.pdf` - 452 vs 710 comparison

### 4. Methodology Summary (1 PDF)
- `ossi_all/methodology_summary.pdf` - Complete methods description

### 5. Theme Maps (1 PDF)
- `ossi_all/theme_maps.pdf` - Maps for all themes

### 6. Interview Analysis (1 PDF)
- `ossi_all/interview_analysis.pdf` - User participation analysis

**Total: 12 PDFs**

---

## Design Decisions

### Prevalence Filtering Strategy
**Current approach** (Makefile.stats): Uses 452 dataset prevalence to filter themes in both 452 and 710 analyses
```bash
PREVALENCE_FILTER_DATASET=$(THEMES_452) python3 nb_stats_esa_koodit.py $(THEMES_710) ...
```

**New approach** (Makefile.ossi): Use 710 dataset prevalence as the base for filtering
- More comprehensive (includes more interviews)
- Both 452 and 710 analyses use same theme list from 710x98 prevalence
- Better represents the full dataset characteristics

### Minimal Disruption Strategy
1. **Keep original scripts unchanged** - add optional parameters only
2. **Keep Makefile.stats unchanged** - it still works as before
3. **Create new Makefile.ossi** - contains all publication-specific targets
4. **Add translation layer** - optional, only activated when needed
5. **Parameterize output paths** - scripts already support this via command-line args

## Implementation Tasks

### Phase 1: Theme Translation Infrastructure

#### Task 1.1: Create Theme Translation File
**File**: `./inputs/theme_translations.csv`
- Extract all 98 theme names from `themes_98x710.csv`
- Create CSV with columns: `finnish`, `english`
- Use LLM to generate high-quality English translations
- Manual review and refinement

#### Task 1.2: Create Translation Utility
**File**: `utils_translate.py`
- Function to load translation dictionary
- Function to translate theme names in dataframes
- Function to translate theme names in lists
- Cache translations for performance

### Phase 2: Update Statistical Analysis Scripts

#### Task 2.1: Modify `nb_stats_*.py` Scripts
All four scripts need these changes:
- ✅ Already use English bird metadata
- ✅ Already support `STATS_MODE` environment variable
- ✅ Already support command-line args for themes_file and output_dir
- ✅ Already support `PREVALENCE_FILTER_DATASET` for comparability
- 🔧 **Need to add**: Optional theme name translation (only when ENABLE_TRANSLATION=1)
- 🔧 **Need to add**: Ensure all plot labels/titles are in English (already are!)
- 🔧 **Need to add**: Ensure summary tables use English (already are!)

**Scripts to modify:**
1. `nb_stats_esa_koodit.py`
2. `nb_stats_bird_groups_koodit.py`
3. `nb_stats_bird_presence_koodit.py`
4. `nb_stats_bird_species_koodit.py`

**Implementation pattern** (minimal, conditional):

Each script loads themes around line ~90-100. Look for:
```python
# Load theme data
outcome_binary = pd.read_csv(themes_file, index_col=0)
```

Add immediately after this:
```python
# Optionally translate theme names to English
if os.environ.get('ENABLE_TRANSLATION', '0') == '1':
    from utils_translate import translate_theme_names
    outcome_binary = translate_theme_names(outcome_binary)
    print(f"✓ Translated {len(outcome_binary.columns)} theme names to English")
```

**Where exactly to add this:**
- **nb_stats_esa_koodit.py**: After line ~95 (after `outcome_binary = pd.read_csv(...)`)
- **nb_stats_bird_groups_koodit.py**: After line ~90 (after `outcome_binary = pd.read_csv(...)`)
- **nb_stats_bird_presence_koodit.py**: After line ~88 (after `outcome_binary = pd.read_csv(...)`)
- **nb_stats_bird_species_koodit.py**: After line ~92 (after `outcome_binary = pd.read_csv(...)`)

**How to test:**
```bash
# Test WITHOUT translation (should work exactly as before)
STATS_MODE=chisquared python3 nb_stats_esa_koodit.py \
  ./output/analyysi_koodit/88d43208/themes_98x452.csv \
  ./output/test/original

# Test WITH translation (theme names should be in English)
ENABLE_TRANSLATION=1 STATS_MODE=chisquared python3 nb_stats_esa_koodit.py \
  ./output/analyysi_koodit/88d43208/themes_98x452.csv \
  ./output/test/translated

# Verify: Check output_table.csv - theme names should differ between the two runs
```

**Note**: All plot labels and titles are already in English (verified from code inspection).
The only Finnish text is the theme names themselves, which come from the input CSV.

#### Task 2.2: Verify English Output
- Check all matplotlib labels, titles, footnotes
- Check summary statistics tables
- Check saved CSV files

### Phase 3: Update Supporting Scripts

#### Task 3.1: Modify `nb_create_comparison_plots.py`
- Add optional `--base-dir` parameter to read from custom location (default: ./output/stats_comparison)
- When ENABLE_TRANSLATION=1, translate theme names in plots
- Verify all labels are in English (already are!)
- Keep default behavior for backwards compatibility

#### Task 3.2: Modify `nb_methodology_summary.py`
- Add `--both-datasets` flag to generate extended version
- When flag present, describe both 452 and 710 analyses
- Add section on statistical methods (chi-squared vs mixed-effects)
- Add section on prevalence filtering (710-based for ossi_all):
  - **Document**: 98 themes extracted initially
  - **Document**: Prevalence filter applied (20%-80%) for statistical analyses
  - **Document**: Number of themes after filtering (varies by dataset)
- **Include theme translations table**: Show all 98 themes with both Finnish and English names
- Verify theme extraction process description:
  - **Confirmed**: `nb_koodit.py` (run_id: 487ef9e9) extracts codes
  - **Confirmed**: `nb_code_consolidation.py` (seed=11) consolidates themes
  - **Not used**: `nb_code_explanation_consolidation.py` (seed=10, has MERGE_THRESHOLD)
  - Evidence: merge_log.txt shows "Random Seed: 11" and no MERGE_THRESHOLD
- Ensure all text is in English
- Add dataset size information (452 vs 710 interviews)
- Keep default behavior for backwards compatibility

#### Task 3.3: Modify `nb_kartat.py`
- Add `--english` flag to enable English output
- Add `--all-themes` flag to process all themes from themes_98x710.csv
- Add `--output` parameter for output directory
- When `--english` is set:
  - Translate theme names using translation utility
  - Change title from "'{code_name}' yleisyys" to "'{code_name}' prevalence"
  - Change colorbar label from 'Osuus' to 'Ratio'
- Update data source to use themes_98x710.csv when `--all-themes` is set
- Keep default behavior for backwards compatibility (paikat/koodit modes)

#### Task 3.4: Modify `nb_interview_comprehensive.py`
- Verify all outputs are in English
- Update output path to accept parameter
- Check that all plots/tables use English labels

### Phase 4: Create Orchestration Files

#### Task 4.1: Create `Makefile.ossi`
**Purpose**: New Makefile for publication materials, keeps Makefile.stats intact

**Key differences from Makefile.stats**:
- Uses 710 prevalence for filtering (not 452)
- Outputs to `./output/ossi_all/` structure
- Enables English translation
- Includes all 6 deliverable groups

**Structure**:
```makefile
# Makefile.ossi - Publication materials generation
# Uses 710 prevalence for filtering, outputs to ossi_all/

THEMES_452 = ./output/analyysi_koodit/88d43208/themes_98x452.csv
THEMES_710 = ./output/analyysi_koodit/7176421e/themes_98x710.csv
OUTPUT_BASE = ./output/ossi_all

# Use 710 dataset prevalence for filtering both analyses
PREVALENCE_FILTER = $(THEMES_710)

# Enable English translation
export ENABLE_TRANSLATION = 1

# ==================== Run Analyses ====================

ossi-run-452:
	@echo "Running 452 dataset analyses (chi-squared)..."
	MPLBACKEND=Agg STATS_MODE=chisquared \
	  PREVALENCE_FILTER_DATASET=$(PREVALENCE_FILTER) \
	  python3 nb_stats_esa_koodit.py $(THEMES_452) $(OUTPUT_BASE)/452/esa_koodit
	# ... repeat for other analyses

ossi-run-710:
	@echo "Running 710 dataset analyses (random effects)..."
	MPLBACKEND=Agg STATS_MODE=random_effects \
	  PREVALENCE_FILTER_DATASET=$(PREVALENCE_FILTER) \
	  python3 nb_stats_esa_koodit.py $(THEMES_710) $(OUTPUT_BASE)/710/esa_koodit
	# ... repeat for other analyses

ossi-run-comparison:
	@echo "Running comparison analysis..."
	bash ./scripts/run_ossi_comparison.sh

ossi-run-maps:
	@echo "Generating theme maps..."
	python3 nb_kartat.py --output $(OUTPUT_BASE)/maps --all-themes --english

ossi-run-interview:
	@echo "Running interview analysis..."
	python3 nb_interview_comprehensive.py --output $(OUTPUT_BASE)/interview

ossi-run-methodology:
	@echo "Generating methodology summary..."
	python3 nb_methodology_summary.py --output $(OUTPUT_BASE)/methodology --both-datasets

# ==================== Generate PDFs ====================

ossi-pdf-452:
	nix shell nixpkgs/25.05#imagemagick --command \
	  ./scripts/create_pdf.sh "$(OUTPUT_BASE)/esa_themes_452.pdf" \
	    "ESA Habitats × Themes (452)" "$(OUTPUT_BASE)/452/esa_koodit" 50
	# ... repeat for other analyses

ossi-pdf-710:
	# Similar to 452 but for 710 outputs

ossi-pdf-all: ossi-pdf-452 ossi-pdf-710 ossi-pdf-comparison \
              ossi-pdf-maps ossi-pdf-interview ossi-pdf-methodology
	@echo "✓ All 12 PDFs generated in $(OUTPUT_BASE)/"

# ==================== Full Pipeline ====================

ossi-all: ossi-run-all ossi-pdf-all
	@echo "✓ Publication materials ready in $(OUTPUT_BASE)/"

ossi-run-all: ossi-run-452 ossi-run-710 ossi-run-comparison \
              ossi-run-maps ossi-run-interview ossi-run-methodology
```

#### Task 4.2: Create `scripts/run_ossi_comparison.sh`
Modified comparison script that uses ossi_all directory structure:
```bash
#!/bin/bash
# Modified to read from and write to ossi_all structure
RESULTS_BASE="./output/ossi_all"
# Run comparison analysis using 452 and 710 results from ossi_all
```

#### Task 4.3: Update Main Makefile
Add include for Makefile.ossi (optional, keep backwards compatible):
```makefile
# In main Makefile, add:
-include Makefile.ossi  # Optional, for publication materials
```

### Phase 5: Theme Translation Execution

#### Task 5.1: Generate Theme Translations
Use LLM to translate all 98 themes from Finnish to English:
- Context: Academic research on nature experience themes
- Maintain conceptual meaning, not literal translation
- Consider: environment, emotions, sensory experiences, activities
- Examples needed for consistency

#### Task 5.2: Review and Refine Translations
Manual quality check:
- Semantic accuracy
- Consistency across related themes
- Appropriateness for academic publication
- Conciseness (avoid overly long translations)

### Phase 6: Testing and Validation

#### Task 6.1: Test Individual Scripts
Run each modified script individually:
```bash
# Test 452 analysis
STATS_MODE=chisquared python3 nb_stats_esa_koodit.py \
  "./output/analyysi_koodit/88d43208/themes_98x452.csv" \
  "./output/test/452/esa_koodit"

# Test 710 analysis
STATS_MODE=random_effects \
PREVALENCE_FILTER_DATASET="./output/analyysi_koodit/88d43208/themes_98x452.csv" \
python3 nb_stats_esa_koodit.py \
  "./output/analyysi_koodit/7176421e/themes_98x710.csv" \
  "./output/test/710/esa_koodit"
```

#### Task 6.2: Verify English Output
Check sample outputs:
- Theme names in plots
- Axis labels
- Titles
- Table headers
- CSV files
- Summary statistics

#### Task 6.3: Test Full Pipeline
Run complete ossi_all generation:
```bash
make ossi-all
```

#### Task 6.4: Validate PDFs
Check each PDF:
- All text in English
- Figures render correctly
- Page layout is appropriate
- File sizes are reasonable
- No missing images

### Phase 7: Quality Assurance

#### Task 7.1: Cross-Check Methodology Summary
Compare with source scripts:
- `nb_koodit.py` - theme extraction process
- `nb_code_consolidation.py` - theme consolidation
- `nb_analyysi_koodit.py` - theme analysis
- Verify LLM parameters, prompts, merge criteria
- Ensure 452 vs 710 differences are documented

#### Task 7.2: Verify Statistical Consistency
- Check that 452 uses chi-squared
- Check that 710 uses mixed-effects with user random effects
- Check that both use same prevalence filter (452 dataset)
- Verify same themes are analyzed in both

#### Task 7.3: Visual Quality Check
- Font sizes are readable
- Colors are distinguishable
- Legends are clear
- No overlapping labels
- Maps are geographically accurate

---

## Technical Considerations

### English Translations
1. **Bird metadata**: Already available in `./inputs/bird-metadata-refined/*_english.csv`
2. **Habitat types**: Already in `habitat_esa_english.csv`
3. **Themes**: Need to create translations (98 themes)

### Script Modifications
- Minimal changes to statistical logic
- Focus on i18n (internationalization) of outputs
- Maintain backward compatibility where possible

### Output Organization
```
./output/ossi_all/
├── 452/
│   ├── esa_koodit/           # Raw output files
│   ├── bird_groups_koodit/
│   ├── bird_presence_koodit/
│   └── bird_species_koodit/
├── 710/
│   ├── esa_koodit/
│   ├── bird_groups_koodit/
│   ├── bird_presence_koodit/
│   └── bird_species_koodit/
├── comparison/
│   └── plots/
├── maps/
│   └── theme_maps/
├── interview/
│   └── analysis/
├── methodology/
│   └── METHODOLOGY_SUMMARY.md
├── esa_themes_452.pdf
├── esa_themes_710.pdf
├── bird_groups_themes_452.pdf
├── bird_groups_themes_710.pdf
├── bird_presence_themes_452.pdf
├── bird_presence_themes_710.pdf
├── bird_species_themes_452.pdf
├── bird_species_themes_710.pdf
├── stats_comparison.pdf
├── theme_maps.pdf
├── interview_analysis.pdf
└── methodology_summary.pdf
```

---

## Dependencies and Requirements

### Software
- Python 3.x with packages: pandas, numpy, matplotlib, seaborn, statsmodels, geopandas
- R with lme4 (for mixed-effects models)
- Nix with: imagemagick, pandoc, texlive
- Make

### Data Files Required
- ✅ `./output/analyysi_koodit/88d43208/themes_98x452.csv`
- ✅ `./output/analyysi_koodit/7176421e/themes_98x710.csv`
- ✅ `./inputs/bird-metadata-refined/*_english.csv`
- ✅ `./inputs/esa_worldcover/`
- ✅ `./geo/` (map data)
- 🔧 `./inputs/theme_translations.csv` (to be created)

---

## Estimated Effort

### Development Tasks
- Theme translation infrastructure: 2-3 hours
- Script modifications: 4-6 hours
- Testing and debugging: 3-4 hours
- PDF generation setup: 2-3 hours
- Quality assurance: 2-3 hours

**Total: ~15-20 hours**

### Translation Task
- Generate initial translations: 1 hour (with LLM)
- Manual review and refinement: 2-3 hours

**Total: ~3-4 hours**

### Overall: ~18-24 hours

---

## Success Criteria

### ✅ Core Requirements Met
1. ✅ All 12 PDFs generated successfully
2. ✅ All text in English (themes, labels, titles, tables)
3. ✅ 452 dataset uses chi-squared tests correctly
4. ✅ 710 dataset uses mixed-effects models correctly
5. ✅ Both datasets use same prevalence filter (710) for comparability
6. ✅ Comparison analysis correctly shows differences
7. ✅ Methodology summary generated (improvements needed - see Phase 7.4)
8. ✅ Theme maps cover all 98 themes
9. ✅ Interview analysis is complete
10. ✅ All outputs organized in `./output/ossi_all/`
11. ✅ Reproducible via `make -f Makefile.ossi ossi-all`
12. ✅ Original functionality preserved - Makefile.stats still works unchanged

### 🔧 Quality Improvements Identified (Phase 7)
- Translation completeness (bird groups, species names)
- Visual consistency (colormap standardization)
- Plot clarity (significance thresholds, label simplification)
- Documentation refinements (methodology summary)
11. ✅ Reproducible via `make -f Makefile.ossi ossi-all`
12. ✅ **Original functionality preserved** - Makefile.stats still works unchanged

---

## Risks and Mitigation

### Risk 1: Theme Translation Quality
- **Mitigation**: Use context-aware LLM, manual review, consistency checks

### Risk 2: Script Compatibility
- **Mitigation**: Test each modification incrementally, maintain version control

### Risk 3: Long Runtime
- **Mitigation**: Mixed-effects models on 710 dataset may take hours; plan accordingly

### Risk 4: Memory Issues
- **Mitigation**: Generate PDFs individually if needed, use appropriate image compression

### Risk 5: PDF Generation Failures
- **Mitigation**: Test PDF scripts early, have fallback to individual image exports

---

## Next Steps

1. **Review this plan** with user for feedback and adjustments
2. **Start with Phase 1**: Create theme translation infrastructure
3. **Proceed incrementally**: Complete one phase before starting next
4. **Test frequently**: Catch issues early
5. **Document changes**: Keep track of modifications for reproducibility

---

## Notes

- The existing infrastructure is quite good - scripts already support English metadata and parameterization
- Main work is: (1) theme translation, (2) ensuring all display text is English, (3) orchestration
- The statistical logic should remain unchanged
- Output directory structure needs to be carefully managed
- PDF generation may need adjustment for new directory structure

## Quick Start for Next Session

If starting fresh with only this PLAN.md and the repository:

### 1. What's been done (Phase 1 complete):
```bash
# Check translation infrastructure exists
ls -la ./inputs/translations/theme_translations.csv  # Should exist
ls -la utils_translate.py                            # Should exist

# Quick test that translations work
python3 -c "from utils_translate import load_translation_dict; print(f'✓ {len(load_translation_dict())} translations loaded')"
```

### 2. What to do next (Phase 2):
```bash
# Modify 4 statistical scripts to add optional translation
# Pattern: After loading themes (~line 90-100 in each script), add:
#   if os.environ.get('ENABLE_TRANSLATION', '0') == '1':
#       from utils_translate import translate_theme_names
#       outcome_binary = translate_theme_names(outcome_binary)

# Test each modification:
ENABLE_TRANSLATION=1 STATS_MODE=chisquared python3 nb_stats_esa_koodit.py \
  ./output/analyysi_koodit/88d43208/themes_98x452.csv ./output/test/esa
```

### 3. Key context to understand:
- **Goal**: Create 12 English PDFs for publication (4 from 452 dataset, 4 from 710, 4 supporting docs)
- **Strategy**: Minimal disruption - add optional parameters, keep Makefile.stats unchanged
- **Translation**: Only active when `ENABLE_TRANSLATION=1` environment variable set
- **Prevalence filtering**: Using 710 dataset prevalence for both analyses (not 452)

### 4. Critical files to know:
- `./inputs/translations/theme_translations.csv` - The 98 theme translations (Finnish → English)
- `utils_translate.py` - Translation utility (cached, tested, ready to use)
- `Makefile.stats` - Original orchestration (DO NOT MODIFY)
- `Makefile.ossi` - To be created for publication materials
- 4 scripts: `nb_stats_{esa,bird_groups,bird_presence,bird_species}_koodit.py` - Next to modify

### 5. How to verify progress:
```bash
# Check Phase 1 complete
[ -f ./inputs/translations/theme_translations.csv ] && echo "✓ Phase 1: Translation file exists"
[ -f utils_translate.py ] && echo "✓ Phase 1: Utility exists"

# Check Phase 2 progress (grep for translation code in stats scripts)
grep -l "ENABLE_TRANSLATION" nb_stats_*.py | wc -l  # Should be 4 when Phase 2 complete
```

---

## Summary of Key Changes from Original Plan

### 1. **Verified Theme Consolidation Process**
- Confirmed that `nb_code_consolidation.py` (seed=11) was used, NOT `nb_code_explanation_consolidation.py`
- Evidence: merge_log.txt shows "Random Seed: 11"
- This uses code names only (not explanations) for similarity matching

### 2. **Changed Prevalence Filter Strategy**
- **Old**: Use 452 prevalence for filtering both datasets
- **New**: Use 710 prevalence for filtering both datasets
- **Rationale**: More comprehensive, represents full dataset better

### 3. **Minimal Disruption Approach**
- **Do NOT modify**: Makefile.stats (keep completely unchanged)
- **Create new**: Makefile.ossi (all publication-specific logic)
- **Scripts**: Add optional parameters only (via environment variables and flags)
- **Backwards compatible**: Original workflows still work unchanged

### 4. **Conditional Translation**
- Translation only activated when `ENABLE_TRANSLATION=1`
- Scripts work with Finnish themes by default (existing behavior)
- Ossi pipeline explicitly sets ENABLE_TRANSLATION=1

### 5. **Parameterization Strategy**
- Use environment variables: `ENABLE_TRANSLATION`, `PREVALENCE_FILTER_DATASET`, `STATS_MODE`
- Use command-line flags: `--english`, `--all-themes`, `--output`, `--both-datasets`
- Keep defaults that maintain original behavior
