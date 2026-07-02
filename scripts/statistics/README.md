# LLM Answer Inconsistency Analysis Scripts

This directory contains a suite of independent analysis scripts designed to identify factors and features that cause LLM (Qwen2.5-7B-Instruct) to produce inconsistent answers between original and mutated questions.

## Overview

The analysis suite examines consistency results from knowledge graph-based metamorphic testing, specifically analyzing data from `data/examples/consistency_results_nli_Qwen2.5-7B-Instruct/`.

## Analysis Scripts

All scripts are independent and can be run individually or together using the master script.

### 1. Rule-Based Analysis (`analyze_inconsistency_by_rule.py`)
**Purpose**: Analyze which KG rules lead to more inconsistencies

**Key Analyses**:
- Inconsistency rate per rule
- Rule complexity (number of relations) vs inconsistency
- Top problematic rules
- Rule pattern analysis

**Outputs**:
- `data/statistics/inconsistency_by_rule/rule_inconsistency_analysis.csv`
- `data/statistics/inconsistency_by_rule/rule_summary.json`
- Visualizations: Top rules, distribution, complexity correlation

### 2. Mutation-Based Analysis (`analyze_inconsistency_by_mutation.py`)
**Purpose**: Compare different mutation types (body_augmentation, body_permutation, entity_rename)

**Key Analyses**:
- Inconsistency rate by mutation type
- NLI score distributions across mutations
- Extraction confidence comparison

**Outputs**:
- `data/statistics/inconsistency_by_mutation/mutation_inconsistency_analysis.csv`
- `data/statistics/inconsistency_by_mutation/mutation_summary.json`
- Visualizations: Mutation comparison, NLI distributions, sample counts

### 3. Entity-Based Analysis (`analyze_inconsistency_by_entity.py`)
**Purpose**: Identify which entity types and specific entities are prone to errors

**Key Analyses**:
- Inconsistency by entity type
- Problematic entities (by ID)
- Entity count correlation
- Entity label complexity

**Outputs**:
- `data/statistics/inconsistency_by_entity/entity_type_inconsistency.csv`
- `data/statistics/inconsistency_by_entity/problematic_entities.csv`
- `data/statistics/inconsistency_by_entity/entity_summary.json`
- Visualizations: Top entity types, entity counts, frequency analysis

### 4. Relation-Based Analysis (`analyze_inconsistency_by_relation.py`)
**Purpose**: Determine which Wikidata relations (properties) cause inconsistencies

**Key Analyses**:
- Inconsistency by individual relations (e.g., P2817, P361)
- Relation pattern combinations
- Rule complexity vs inconsistency

**Outputs**:
- `data/statistics/inconsistency_by_relation/relation_inconsistency.csv`
- `data/statistics/inconsistency_by_relation/relation_pattern_inconsistency.csv`
- `data/statistics/inconsistency_by_relation/relation_summary.json`
- Visualizations: Top relations, patterns, frequency correlation

### 5. NLI Score Analysis (`analyze_inconsistency_by_nli_scores.py`)
**Purpose**: Analyze how NLI (contradiction/entailment/neutral) scores correlate with inconsistencies

**Key Analyses**:
- Score distributions for consistent vs inconsistent
- Dominant score type analysis
- Score gap (confidence) analysis
- Consistency confidence correlation

**Outputs**:
- `data/statistics/inconsistency_by_nli_scores/inconsistency_by_score_type.csv`
- `data/statistics/inconsistency_by_nli_scores/score_comparison.csv`
- `data/statistics/inconsistency_by_nli_scores/nli_scores_summary.json`
- Visualizations: Violin plots, score comparisons, heatmaps

### 6. Complexity-Based Analysis (`analyze_inconsistency_by_complexity.py`)
**Purpose**: Examine how question and answer complexity affects consistency

**Key Analyses**:
- Question length vs inconsistency
- Answer length and structure analysis
- Question type comparison
- Complexity features (commas, parentheses, conjunctions)

**Outputs**:
- `data/statistics/inconsistency_by_complexity/question_type_analysis.csv`
- `data/statistics/inconsistency_by_complexity/complexity_comparison.csv`
- `data/statistics/inconsistency_by_complexity/complexity_summary.json`
- Visualizations: Length analysis, type comparison, scatter plots

## Usage

### Command-Line Arguments

All analysis scripts support the following command-line arguments:

- `--input-dir`: Input directory containing consistency results (default: `data/examples/consistency_results_nli_Qwen2.5-7B-Instruct`)
- `--output-dir`: Output directory for analysis results (default: script-specific subdirectory under `data/statistics/`)

The master script `run_all_analyses.py` supports:

- `--input-dir`: Input directory for all analyses (default: `data/examples/consistency_results_nli_Qwen2.5-7B-Instruct`)
- `--output-base-dir`: Base output directory for all results (default: `data/statistics`)

### Run Individual Analysis

With default paths:
```bash
conda run -n karma python scripts/statistics/analyze_inconsistency_by_rule.py
```

With custom paths:
```bash
conda run -n karma python scripts/statistics/analyze_inconsistency_by_rule.py \
  --input-dir data/examples/my_results \
  --output-dir data/my_statistics/rule_analysis
```

View help for any script:
```bash
conda run -n karma python scripts/statistics/analyze_inconsistency_by_mutation.py --help
```

### Run All Analyses (Recommended)

With default paths:
```bash
conda run -n karma python scripts/statistics/run_all_analyses.py
```

With custom paths:
```bash
conda run -n karma python scripts/statistics/run_all_analyses.py \
  --input-dir data/examples/consistency_results_nli_Qwen2.5-14B-Instruct \
  --output-base-dir data/statistics_qwen14b
```

The master script will:
1. Run all 6 analyses sequentially
2. Generate individual outputs for each analysis
3. Create a master summary (`{output-base-dir}/master_summary.json`)
4. Generate a human-readable report (`{output-base-dir}/ANALYSIS_REPORT.txt`)

## Output Structure

```
data/statistics/
├── master_summary.json                    # Combined results from all analyses
├── ANALYSIS_REPORT.txt                    # Human-readable summary report
├── inconsistency_by_rule/
│   ├── rule_inconsistency_analysis.csv
│   ├── rule_summary.json
│   └── *.png (visualizations)
├── inconsistency_by_mutation/
│   ├── mutation_inconsistency_analysis.csv
│   ├── mutation_summary.json
│   └── *.png (visualizations)
├── inconsistency_by_entity/
│   ├── entity_type_inconsistency.csv
│   ├── problematic_entities.csv
│   ├── entity_summary.json
│   └── *.png (visualizations)
├── inconsistency_by_relation/
│   ├── relation_inconsistency.csv
│   ├── relation_pattern_inconsistency.csv
│   ├── relation_summary.json
│   └── *.png (visualizations)
├── inconsistency_by_nli_scores/
│   ├── inconsistency_by_score_type.csv
│   ├── score_comparison.csv
│   ├── nli_scores_summary.json
│   └── *.png (visualizations)
└── inconsistency_by_complexity/
    ├── question_type_analysis.csv
    ├── complexity_comparison.csv
    ├── complexity_summary.json
    └── *.png (visualizations)
```

## Dependencies

Required packages (install via pip in karma environment):
- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Key Research Questions Answered

1. **Which rules are most problematic?** → Rule-based analysis
2. **Which mutation type causes most inconsistencies?** → Mutation-based analysis
3. **Are certain entity types more error-prone?** → Entity-based analysis
4. **Which relations/properties are problematic?** → Relation-based analysis
5. **How do NLI scores relate to inconsistencies?** → NLI score analysis
6. **Does question complexity affect consistency?** → Complexity-based analysis

## Notes

- All scripts use English for strings, tables, and figures; comments may be in Chinese
- Scripts are independent and do not depend on each other
- All visualizations are saved as high-resolution PNG files (300 DPI)
- Data is read from the source directory without modification
- Statistics are calculated only from samples with sufficient data (configurable thresholds)

## Author & Date

Created: 2026-01-16
Environment: karma (conda)
Target Model: Qwen2.5-7B-Instruct
