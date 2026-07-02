#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ3 Analyzer Module

This module provides statistical analysis functions for investigating
the relationship between features and inconsistency rates.

Analysis Types:
1. Correlation Analysis (Pearson, Spearman)
2. Group Comparison with Chi-square test (for categorical vs binary)
3. Effect Size Calculation (Cramér's V for chi-square)

Note: We use Chi-square test instead of ANOVA for categorical variables
because the outcome variable (inconsistent) is binary, not continuous.
ANOVA assumes normally distributed continuous dependent variable,
which is violated for binary outcomes.

Usage:
    from rq3_analyzer import FeatureAnalyzer

    analyzer = FeatureAnalyzer(features_df)
    results = analyzer.analyze_all()
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Import configuration
from rq3_config import (
    Config, FEATURE_DEFINITIONS, QUESTION_TYPE_DISPLAY,
    MUTATION_DISPLAY_NAMES
)


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    feature: str
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    n_samples: int
    is_significant: bool
    effect_size: str  # "weak", "moderate", "strong"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "pearson_r": round(self.pearson_r, 4),
            "pearson_p": round(self.pearson_p, 4),
            "spearman_r": round(self.spearman_r, 4),
            "spearman_p": round(self.spearman_p, 4),
            "n_samples": self.n_samples,
            "is_significant": self.is_significant,
            "effect_size": self.effect_size,
        }


@dataclass
class GroupComparisonResult:
    """Result of group comparison analysis."""
    factor: str
    group_name: str
    n_samples: int
    n_inconsistent: int
    inconsistency_rate: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor,
            "group_name": self.group_name,
            "n_samples": self.n_samples,
            "n_inconsistent": self.n_inconsistent,
            "inconsistency_rate": round(self.inconsistency_rate, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
        }


@dataclass
class ChiSquareResult:
    """Result of Chi-square test for categorical vs binary outcome."""
    factor: str
    chi2_statistic: float
    p_value: float
    dof: int  # degrees of freedom
    cramers_v: float  # effect size for chi-square
    is_significant: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor,
            "chi2_statistic": round(self.chi2_statistic, 4),
            "p_value": round(self.p_value, 4),
            "dof": self.dof,
            "cramers_v": round(self.cramers_v, 4),
            "is_significant": self.is_significant,
        }


# =============================================================================
# Statistical Helper Functions
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def eta_squared(groups: List[np.ndarray]) -> float:
    """Calculate eta-squared for ANOVA effect size."""
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    # Between-group sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)

    # Total sum of squares
    ss_total = sum((x - grand_mean) ** 2 for x in all_data)

    if ss_total == 0:
        return 0.0

    return ss_between / ss_total


def cramers_v(confusion_matrix: np.ndarray) -> float:
    """
    Calculate Cramér's V effect size for chi-square test.

    Cramér's V interpretation:
    - 0.1: small effect
    - 0.3: medium effect
    - 0.5: large effect

    Args:
        confusion_matrix: Contingency table from chi-square test

    Returns:
        Cramér's V statistic (0 to 1)
    """
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1

    if n == 0 or min_dim == 0:
        return 0.0

    return np.sqrt(chi2 / (n * min_dim))


def wilson_confidence_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for a proportion."""
    if trials == 0:
        return 0.0, 0.0

    p = successes / trials
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    denominator = 1 + z ** 2 / trials
    center = (p + z ** 2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * trials)) / trials) / denominator

    return max(0, center - margin), min(1, center + margin)


def interpret_effect_size(r: float) -> str:
    """Interpret correlation coefficient as effect size."""
    abs_r = abs(r)
    if abs_r >= Config.STRONG_CORRELATION_THRESHOLD:
        return "strong"
    elif abs_r >= Config.MODERATE_CORRELATION_THRESHOLD:
        return "moderate"
    elif abs_r >= Config.WEAK_CORRELATION_THRESHOLD:
        return "weak"
    else:
        return "negligible"


# =============================================================================
# Main Analyzer Class
# =============================================================================

class FeatureAnalyzer:
    """
    Main analysis class for investigating feature-inconsistency relationships.
    """

    def __init__(self, features_df: pd.DataFrame):
        """
        Initialize analyzer with features DataFrame.

        Args:
            features_df: DataFrame with extracted features and 'inconsistent' column
        """
        self.df = features_df.copy()
        self.target = "inconsistent"

        # Identify feature types
        self.continuous_features = self._identify_continuous_features()
        self.categorical_features = self._identify_categorical_features()

    def _identify_continuous_features(self) -> List[str]:
        """Identify continuous (numeric) features."""
        exclude = [self.target, "has_special_chars", "has_numbers"]
        return [
            col for col in self.df.columns
            if self.df[col].dtype in [np.int64, np.float64]
            and col not in exclude
            and self.df[col].nunique() > 2
        ]

    def _identify_categorical_features(self) -> List[str]:
        """Identify categorical features."""
        categorical = ["question_type", "mutation_type", "model"]
        return [col for col in categorical if col in self.df.columns]

    # -------------------------------------------------------------------------
    # Correlation Analysis
    # -------------------------------------------------------------------------

    def analyze_correlations(self) -> List[CorrelationResult]:
        """
        Perform correlation analysis for all continuous features.

        Returns:
            List of CorrelationResult objects
        """
        results = []

        for feature in self.continuous_features:
            # Skip if not enough variance
            if self.df[feature].std() == 0:
                continue

            # Remove NaN values
            valid_mask = self.df[feature].notna() & self.df[self.target].notna()
            x = self.df.loc[valid_mask, feature].values
            y = self.df.loc[valid_mask, self.target].values

            if len(x) < Config.MIN_SAMPLES_PER_GROUP:
                continue

            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(x, y)

            # Spearman correlation (rank-based, more robust)
            spearman_r, spearman_p = stats.spearmanr(x, y)

            # Effect size interpretation
            effect_size = interpret_effect_size(pearson_r)

            results.append(CorrelationResult(
                feature=feature,
                pearson_r=pearson_r,
                pearson_p=pearson_p,
                spearman_r=spearman_r,
                spearman_p=spearman_p,
                n_samples=len(x),
                is_significant=pearson_p < Config.ALPHA,
                effect_size=effect_size,
            ))

        # Sort by absolute correlation strength
        results.sort(key=lambda r: abs(r.pearson_r), reverse=True)

        return results

    # -------------------------------------------------------------------------
    # Group Comparison Analysis
    # -------------------------------------------------------------------------

    def analyze_groups(
        self,
        factor: str,
        display_names: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[GroupComparisonResult], Optional[ChiSquareResult]]:
        """
        Analyze inconsistency rates across groups using Chi-square test.

        Chi-square test is appropriate here because:
        - The outcome variable (inconsistent) is binary
        - We're testing independence between categorical factor and binary outcome
        - No normality assumption required

        Args:
            factor: Column name for grouping
            display_names: Optional mapping of group values to display names

        Returns:
            Tuple of (group results, Chi-square result if applicable)
        """
        if factor not in self.df.columns:
            return [], None

        display_names = display_names or {}

        group_results = []

        for group_value in self.df[factor].unique():
            group_df = self.df[self.df[factor] == group_value]
            n_samples = len(group_df)
            n_inconsistent = int(group_df[self.target].sum())
            rate = n_inconsistent / n_samples if n_samples > 0 else 0

            # Confidence interval (Wilson score)
            ci_lower, ci_upper = wilson_confidence_interval(
                n_inconsistent, n_samples
            )

            group_name = display_names.get(group_value, str(group_value))

            group_results.append(GroupComparisonResult(
                factor=factor,
                group_name=group_name,
                n_samples=n_samples,
                n_inconsistent=n_inconsistent,
                inconsistency_rate=rate,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            ))

        # Perform Chi-square test if multiple groups
        chi2_result = None
        if len(group_results) >= 2:
            # Build contingency table
            contingency = pd.crosstab(self.df[factor], self.df[self.target])

            # Check if we have enough data for chi-square
            # Each cell should have expected count >= 5 for valid chi-square
            if contingency.size > 0 and contingency.sum().sum() >= Config.MIN_SAMPLES_PER_GROUP:
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

                    # Calculate effect size (Cramér's V)
                    v = cramers_v(contingency.values)

                    chi2_result = ChiSquareResult(
                        factor=factor,
                        chi2_statistic=chi2,
                        p_value=p_value,
                        dof=dof,
                        cramers_v=v,
                        is_significant=p_value < Config.ALPHA,
                    )
                except ValueError:
                    # Chi-square may fail with insufficient data
                    pass

        return group_results, chi2_result

    def analyze_all_categorical(self) -> Dict[str, Tuple[List[GroupComparisonResult], Optional[ChiSquareResult]]]:
        """Analyze all categorical features."""

        display_name_maps = {
            "question_type": QUESTION_TYPE_DISPLAY,
            "mutation_type": MUTATION_DISPLAY_NAMES,
        }

        results = {}

        for factor in self.categorical_features:
            results[factor] = self.analyze_groups(
                factor,
                display_names=display_name_maps.get(factor, {})
            )

        return results

    # -------------------------------------------------------------------------
    # Binned Analysis
    # -------------------------------------------------------------------------

    def analyze_by_quantile(
        self,
        feature: str,
        n_bins: int = 4,
        labels: Optional[List[str]] = None,
    ) -> List[GroupComparisonResult]:
        """
        Analyze inconsistency rate by feature quantiles.

        Args:
            feature: Continuous feature to bin
            n_bins: Number of quantile bins
            labels: Optional labels for bins

        Returns:
            List of GroupComparisonResult for each bin
        """
        if feature not in self.continuous_features:
            return []

        # Create quantile bins
        try:
            # Handle potential duplicate values
            binned = pd.qcut(
                self.df[feature],
                q=n_bins,
                labels=labels,
                duplicates="drop"
            )
        except ValueError:
            # Fall back to equal-width bins
            n_unique = self.df[feature].nunique()
            if n_unique < n_bins:
                binned = pd.cut(
                    self.df[feature],
                    bins=n_unique,
                    labels=labels[:n_unique] if labels else None
                )
            else:
                return []

        # Temporarily add bin column
        bin_col = f"{feature}_bin"
        self.df[bin_col] = binned

        # Analyze groups
        results, _ = self.analyze_groups(bin_col)

        # Clean up
        self.df.drop(columns=[bin_col], inplace=True)

        return results

    # -------------------------------------------------------------------------
    # Comprehensive Analysis
    # -------------------------------------------------------------------------

    def analyze_all(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of all features.

        Returns:
            Dictionary with all analysis results
        """
        print("Performing comprehensive feature analysis...")

        results = {
            "summary": self._generate_summary(),
            "correlations": self.analyze_correlations(),
            "categorical": self.analyze_all_categorical(),
        }

        # Quantile analysis for key continuous features
        key_features = [
            "body_atom_count",
            "predicate_scarcity_score",
            "pattern_support_score",
            "entity_count",
        ]

        results["quantile_analysis"] = {}
        for feature in key_features:
            if feature in self.continuous_features:
                results["quantile_analysis"][feature] = self.analyze_by_quantile(
                    feature,
                    n_bins=4,
                    labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
                )

        return results

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""

        return {
            "total_samples": len(self.df),
            "total_inconsistent": int(self.df[self.target].sum()),
            "overall_inconsistency_rate": float(self.df[self.target].mean()),
            "n_continuous_features": len(self.continuous_features),
            "n_categorical_features": len(self.categorical_features),
        }

    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------

    def export_correlations_to_dataframe(
        self,
        results: List[CorrelationResult]
    ) -> pd.DataFrame:
        """Export correlation results to DataFrame."""

        return pd.DataFrame([r.to_dict() for r in results])

    def export_groups_to_dataframe(
        self,
        results: List[GroupComparisonResult]
    ) -> pd.DataFrame:
        """Export group comparison results to DataFrame."""

        return pd.DataFrame([r.to_dict() for r in results])


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Test the analyzer module."""

    print("=" * 70)
    print("RQ3 Analyzer Test")
    print("=" * 70)

    # Load features (assuming features.csv exists)
    output_dir = Config.ensure_output_dir()
    features_path = output_dir / "features.csv"

    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        print("Please run rq3_feature_extractor.py first.")
        return

    print(f"\nLoading features from {features_path}...")
    features_df = pd.read_csv(features_path)
    print(f"Loaded {len(features_df)} samples with {len(features_df.columns)} features")

    # Initialize analyzer
    analyzer = FeatureAnalyzer(features_df)

    # Run comprehensive analysis
    results = analyzer.analyze_all()

    # Print summary
    print("\n" + "=" * 70)
    print("Analysis Summary")
    print("=" * 70)

    print(f"\nTotal samples: {results['summary']['total_samples']}")
    print(f"Overall inconsistency rate: {results['summary']['overall_inconsistency_rate']*100:.2f}%")

    # Print correlation results
    print("\n" + "-" * 70)
    print("Correlation Analysis (sorted by |r|)")
    print("-" * 70)

    corr_df = analyzer.export_correlations_to_dataframe(results["correlations"])
    print(corr_df.to_string(index=False))

    # Print categorical analysis
    print("\n" + "-" * 70)
    print("Categorical Analysis")
    print("-" * 70)

    for factor, (group_results, chi2_result) in results["categorical"].items():
        print(f"\n{factor}:")
        groups_df = analyzer.export_groups_to_dataframe(group_results)
        print(groups_df.to_string(index=False))

        if chi2_result:
            sig = "***" if chi2_result.is_significant else ""
            print(f"  Chi-square: χ²={chi2_result.chi2_statistic:.2f}, p={chi2_result.p_value:.4f}, V={chi2_result.cramers_v:.3f} {sig}")

    # Save results
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    # Save correlations
    corr_path = output_dir / "correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"Correlations saved to: {corr_path}")

    # Save comprehensive results
    all_groups = []
    for factor, (group_results, _) in results["categorical"].items():
        for r in group_results:
            all_groups.append(r.to_dict())

    groups_df = pd.DataFrame(all_groups)
    groups_path = output_dir / "group_analysis.csv"
    groups_df.to_csv(groups_path, index=False)
    print(f"Group analysis saved to: {groups_path}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
