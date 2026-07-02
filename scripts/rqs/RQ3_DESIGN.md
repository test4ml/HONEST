# RQ3 Factor Analysis - Design Documentation

## Overview

This document describes the redesigned RQ3 analysis framework for investigating factors correlated with LLM inconsistency in knowledge graph question answering.

## Architecture

```
scripts/rqs/
├── rq3_config.py              # Unified configuration (models, paths, feature definitions)
├── rq3_feature_extractor.py   # Feature extraction (rule, entity, pattern features)
├── rq3_analyzer.py            # Statistical analysis (correlation, ANOVA, group comparison)
├── rq3_visualizer.py          # Visualization (plots, charts, dashboards)
├── run_rq3_analysis.py        # Main entry point
└── RQ3_DESIGN.md              # This document
```

## Design Principles

### 1. Single Responsibility
Each module has a single, well-defined responsibility:
- **Config**: Centralized configuration management
- **FeatureExtractor**: Extract features from raw data
- **Analyzer**: Perform statistical tests
- **Visualizer**: Generate visualizations
- **ReportGenerator**: Create human-readable reports

### 2. Separation of Concerns
```
Data Loading → Feature Extraction → Analysis → Visualization → Reporting
```

### 3. Configuration Centralization
All configuration (models, paths, thresholds) is in `rq3_config.py`:
- No more scattered `MODELS` lists across 6 files
- Single source of truth for data paths
- Easy to modify experiment settings

### 4. Unified Data Source
All scripts use the same sampled data (`rq2_sampled_data`) ensuring consistency.

## Feature Taxonomy

| Category | Features | Hypothesis |
|----------|----------|------------|
| **Rule Structure** | body_atom_count, variable_count, predicate_count | Positive (more complex → higher inconsistency) |
| **Predicate** | predicate_frequency, scarcity_score | Negative (rarer → higher inconsistency) |
| **Pattern** | pattern_instance_count, support_score | Negative (less support → higher inconsistency) |
| **Entity** | entity_count, label_complexity | Positive (more complex → higher inconsistency) |
| **Question** | question_type, question_length | Exploratory |
| **Mutation** | mutation_type | Exploratory |

## Analysis Methods

### 1. Correlation Analysis
- **Pearson correlation**: Linear relationship
- **Spearman correlation**: Monotonic relationship (rank-based)
- **Effect size interpretation**: weak (< 0.3), moderate (0.3-0.5), strong (> 0.5)

### 2. Group Comparison
- **Binomial proportion**: Inconsistency rate per group
- **Wilson confidence interval**: For rate estimation
- **ANOVA**: Multi-group comparison with effect size (η²)

### 3. Quantile Analysis
- Divide continuous features into quartiles
- Compare inconsistency rates across quartiles
- Identify non-linear relationships

## Usage

### Quick Start
```bash
conda activate karma
cd $PROJECT_ROOT
python scripts/rqs/run_rq3_analysis.py
```

### Step-by-Step
```python
# 1. Load configuration
from rq3_config import Config, MODELS

# 2. Extract features
from rq3_feature_extractor import FeatureExtractor, load_all_data

df = load_all_data()
extractor = FeatureExtractor()
features_df = extractor.extract_all_features(df)

# 3. Analyze
from rq3_analyzer import FeatureAnalyzer

analyzer = FeatureAnalyzer(features_df)
results = analyzer.analyze_all()

# 4. Visualize
from rq3_visualizer import RQ3Visualizer

visualizer = RQ3Visualizer(output_dir)
visualizer.plot_correlation_bars(results["correlations"])

# 5. Report
from run_rq3_analysis import ReportGenerator

report_gen = ReportGenerator(output_dir)
report_gen.generate(features_df, results)
```

## Output Files

```
data/examples/rq3_analysis_results/
├── features.csv              # Extracted features (one row per sample)
├── correlations.csv          # Correlation analysis results
├── group_analysis.csv        # Group comparison results
├── rq3_analysis_report.md    # Comprehensive report
└── figures/
    ├── correlation_bars.png
    ├── group_mutation_type.png
    ├── group_question_type.png
    ├── distribution_body_atom_count.png
    ├── correlation_heatmap.png
    └── summary_dashboard.png
```

## Comparison with Original Design

| Aspect | Original (6 scripts) | New Design (5 modules) |
|--------|---------------------|------------------------|
| Lines of code | ~4,000 | ~1,500 (excluding docs) |
| Redundancy | High (rq3_2 and rq3_3a overlap) | None |
| Configuration | Scattered in 6 files | Centralized |
| Data source | Inconsistent | Unified |
| Extensibility | Hard (duplicate code) | Easy (modular) |
| Testing | Difficult | Easy (unit testable) |

## Extending the Framework

### Adding a New Feature
1. Add definition to `FEATURE_DEFINITIONS` in `rq3_config.py`
2. Implement extraction in appropriate extractor class in `rq3_feature_extractor.py`
3. Add to `continuous_features` or `categorical_features` list

### Adding a New Analysis
1. Add method to `FeatureAnalyzer` class in `rq3_analyzer.py`
2. Call from `analyze_all()` method
3. Add visualization in `rq3_visualizer.py`

### Adding a New Visualization
1. Add method to `RQ3Visualizer` class
2. Call from `run_rq3_analysis.py`

## Troubleshooting

### "No data found"
- Check that `RQ2_SAMPLED_PATH` exists
- Verify model directory names match `MODEL_DIR_MAP`

### "pd.qcut failed"
- Occurs when feature has few unique values
- Handled automatically with fallback to `pd.cut`

### Memory issues
- Features are extracted in batches
- Consider sampling if dataset is very large

## Future Improvements

1. **Machine Learning Analysis**: Add feature importance via random forest
2. **Interaction Effects**: Analyze feature interactions (e.g., rule length × predicate scarcity)
3. **Model-Specific Analysis**: Separate analysis per model
4. **Temporal Analysis**: Track consistency over time (if longitudinal data)
