#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ3 Configuration Module

Centralized configuration for all RQ3 analysis scripts.
This ensures consistency across all analysis components.

Usage:
    from rq3_config import Config, MODELS, MUTATION_TYPES, DATA_PATH
"""

from pathlib import Path
from typing import List, Dict


# =============================================================================
# Path Configuration
# =============================================================================

# Base paths
PROJECT_ROOT = Path("$PROJECT_ROOT")
DATA_BASE_PATH = PROJECT_ROOT / "data" / "examples"

# Output directory for RQ3 results
OUTPUT_DIR = DATA_BASE_PATH / "rq3_analysis_results"


# =============================================================================
# Model Configuration
# =============================================================================

# Model display names (standardized)
MODELS: List[str] = [
    "DeepSeek-V3",
    "DeepSeek-V4-Flash",
    "GLM-5-Turbo",
    "GPT-5.5",
    "Qwen2.5-7B",
]

# Mapping from display name to actual directory name under DATA_BASE_PATH
MODEL_DIR_MAP: Dict[str, str] = {
    "DeepSeek-V3": "consistency_results_nli_deepseek-v3",
    "DeepSeek-V4-Flash": "consistency_results_nli_deepseek-v4-flash",
    "GLM-5-Turbo": "consistency_results_nli_GLM-5-Turbo",
    "GPT-5.5": "consistency_results_nli_gpt-5.5",
    "Qwen2.5-7B": "consistency_results_nli_Qwen2.5-7B-Instruct",
}


# =============================================================================
# Mutation Types Configuration
# =============================================================================

# Mutation file names
MUTATION_FILES: List[str] = [
    "body_augmentation_llm_answers.csv",
    "body_permutation_llm_answers.csv",
    "entity_rename_llm_answers.csv",
]

# Mutation display names
MUTATION_DISPLAY_NAMES: Dict[str, str] = {
    "body_augmentation": "Body Augmentation",
    "body_permutation": "Body Permutation",
    "entity_rename": "Entity Rename",
}


# =============================================================================
# Question Types Configuration
# =============================================================================

QUESTION_TYPES: List[str] = ["multiple_choice", "wh_question", "yes_no"]

QUESTION_TYPE_DISPLAY: Dict[str, str] = {
    "multiple_choice": "Multi-Choice",
    "wh_question": "WH-Question",
    "yes_no": "Yes/No",
}

# Alias mapping for yes_no questions
YESNO_ALIASES = ["true_false", "yes_no"]


# =============================================================================
# Feature Definitions
# =============================================================================

# Features to extract and analyze
FEATURE_DEFINITIONS = {
    # Rule Structure Features
    "body_atom_count": {
        "category": "Rule Structure",
        "description": "Number of atoms in rule body",
        "type": "discrete",
        "hypothesis": "positive",  # More atoms → Higher inconsistency
    },
    "total_atom_count": {
        "category": "Rule Structure",
        "description": "Total atoms in rule (body + head)",
        "type": "discrete",
        "hypothesis": "positive",
    },
    "unique_variable_count": {
        "category": "Rule Structure",
        "description": "Number of unique variables",
        "type": "discrete",
        "hypothesis": "positive",
    },
    "unique_predicate_count": {
        "category": "Rule Structure",
        "description": "Number of unique predicates",
        "type": "discrete",
        "hypothesis": "positive",
    },

    # Predicate Features
    "primary_predicate_frequency": {
        "category": "Predicate",
        "description": "Frequency of primary predicate in dataset",
        "type": "continuous",
        "hypothesis": "negative",  # Higher frequency → Lower inconsistency
    },
    "predicate_scarcity_score": {
        "category": "Predicate",
        "description": "Scarcity score (1 - percentile rank)",
        "type": "continuous",
        "hypothesis": "positive",  # Higher scarcity → Higher inconsistency
    },

    # Pattern Features
    "pattern_instance_count": {
        "category": "Pattern",
        "description": "Number of instances with same pattern",
        "type": "discrete",
        "hypothesis": "negative",
    },
    "pattern_support_score": {
        "category": "Pattern",
        "description": "Support score (percentile rank)",
        "type": "continuous",
        "hypothesis": "negative",
    },

    # Entity Features
    "entity_count": {
        "category": "Entity",
        "description": "Number of entities in sample",
        "type": "discrete",
        "hypothesis": "positive",
    },
    "avg_entity_label_length": {
        "category": "Entity",
        "description": "Average length of entity labels",
        "type": "continuous",
        "hypothesis": "positive",
    },

    # Question Features
    "question_type": {
        "category": "Question",
        "description": "Type of question (MC/WH/YN)",
        "type": "categorical",
        "hypothesis": "exploratory",
    },
    "question_length": {
        "category": "Question",
        "description": "Length of question text",
        "type": "continuous",
        "hypothesis": "exploratory",
    },
}


# =============================================================================
# Analysis Configuration
# =============================================================================

class Config:
    """Configuration class for RQ3 analysis."""

    # Statistical significance threshold
    ALPHA = 0.05

    # Minimum samples required for analysis
    MIN_SAMPLES_PER_GROUP = 10

    # Correlation thresholds
    WEAK_CORRELATION_THRESHOLD = 0.1
    MODERATE_CORRELATION_THRESHOLD = 0.3
    STRONG_CORRELATION_THRESHOLD = 0.5

    # Visualization settings (sized for single-column paper, ~3.5in width)
    FIGURE_DPI = 600
    FIGURE_SIZE_SINGLE = (3.5, 2.6)
    FIGURE_SIZE_DOUBLE = (3.5, 3.0)
    FIGURE_SIZE_TRIPLE = (3.5, 4.5)
    FIGURE_SIZE_HEATMAP = (3.5, 3.2)
    FIGURE_SIZE_DASHBOARD = (7.0, 5.5)

    # Color palette
    COLORS = {
        "primary": "steelblue",
        "secondary": "coral",
        "positive": "lightgreen",
        "negative": "salmon",
        "neutral": "gray",
    }

    @classmethod
    def get_model_dir(cls, model_name: str) -> Path:
        """Get the base directory path for a model."""
        dir_name = MODEL_DIR_MAP.get(model_name, f"consistency_results_nli_{model_name}")
        return DATA_BASE_PATH / dir_name

    @classmethod
    def ensure_output_dir(cls, subdir: str = "") -> Path:
        """Ensure output directory exists and return path."""
        path = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path


# =============================================================================
# Utility Functions
# =============================================================================

def get_all_data_files() -> List[Dict[str, Path]]:
    """
    Get all data file paths with metadata.

    Iterates over all kg_rule_N subdirectories for each model.

    Returns:
        List of dicts with keys: path, model, mutation_type
    """
    files = []

    for model in MODELS:
        model_dir = Config.get_model_dir(model)

        if not model_dir.exists():
            print(f"Warning: {model_dir} does not exist")
            continue

        # Iterate over all kg_rule_N subdirectories
        rule_dirs = sorted(model_dir.glob("kg_rule_*"))
        if not rule_dirs:
            print(f"Warning: No kg_rule_* dirs in {model_dir}")
            continue

        for rule_dir in rule_dirs:
            if not rule_dir.is_dir():
                continue

            for mutation_file in MUTATION_FILES:
                file_path = rule_dir / mutation_file

                if file_path.exists():
                    mutation_type = mutation_file.replace("_llm_answers.csv", "")
                    files.append({
                        "path": file_path,
                        "model": model,
                        "mutation_type": mutation_type,
                    })

    return files


if __name__ == "__main__":
    # Test configuration
    print("=" * 60)
    print("RQ3 Configuration Test")
    print("=" * 60)

    print(f"\nData Path: {DATA_BASE_PATH}")
    print(f"Output Path: {OUTPUT_DIR}")

    print(f"\nModels ({len(MODELS)}):")
    for model in MODELS:
        print(f"  - {model}")

    print(f"\nMutation Types ({len(MUTATION_FILES)}):")
    for mutation in MUTATION_FILES:
        print(f"  - {mutation}")

    print(f"\nFeature Categories:")
    categories = set(f["category"] for f in FEATURE_DEFINITIONS.values())
    for cat in sorted(categories):
        features = [k for k, v in FEATURE_DEFINITIONS.items() if v["category"] == cat]
        print(f"  {cat}: {len(features)} features")

    print(f"\nData Files Found:")
    files = get_all_data_files()
    print(f"  Total: {len(files)} files")

    print("\n" + "=" * 60)
    print("Configuration OK!")
    print("=" * 60)
