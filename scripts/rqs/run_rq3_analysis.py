#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ3 Main Analysis Script

This is the main entry point for running the complete RQ3 factor analysis.
It orchestrates feature extraction, statistical analysis, and visualization.

Usage:
    conda activate karma
    python scripts/rqs/run_rq3_analysis.py

Output:
    - data/examples/rq3_analysis_results/features.csv
    - data/examples/rq3_analysis_results/correlations.csv
    - data/examples/rq3_analysis_results/group_analysis.csv
    - data/examples/rq3_analysis_results/figures/*.png
    - data/examples/rq3_analysis_results/rq3_analysis_report.md
"""

import sys
from pathlib import Path
from typing import List

# Add scripts directory to path for imports
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import numpy as np
import pandas as pd
from datetime import datetime

# Import modules
from rq3_config import Config, MODELS, FEATURE_DEFINITIONS, QUESTION_TYPE_DISPLAY, MUTATION_DISPLAY_NAMES
from rq3_feature_extractor import FeatureExtractor, load_all_data
from rq3_analyzer import FeatureAnalyzer
from rq3_visualizer import RQ3Visualizer


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """Generate comprehensive analysis report in Markdown format."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def generate(
        self,
        features_df: pd.DataFrame,
        analysis_results: dict,
    ) -> Path:
        """Generate complete analysis report."""

        lines = []

        # Header
        lines.extend([
            "# RQ3: Factor Analysis for LLM Inconsistency in KGQA",
            "=" * 80,
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
        ])

        # Executive Summary
        lines.extend(self._generate_executive_summary(analysis_results))

        # Data Overview
        lines.extend(self._generate_data_overview(features_df, analysis_results))

        # Correlation Analysis
        lines.extend(self._generate_correlation_section(analysis_results))

        # Categorical Analysis
        lines.extend(self._generate_categorical_section(analysis_results))

        # Quantile Analysis
        lines.extend(self._generate_quantile_section(analysis_results))

        # Key Findings
        lines.extend(self._generate_key_findings(analysis_results))

        # Footer
        lines.extend([
            "",
            "---",
            "",
            "## Appendix",
            "",
            "### Feature Definitions",
            "",
        ])

        for feature, definition in FEATURE_DEFINITIONS.items():
            if feature in features_df.columns:
                lines.extend([
                    f"**{feature}** ({definition['category']})",
                    f"- Description: {definition['description']}",
                    f"- Type: {definition['type']}",
                    f"- Hypothesis: {definition['hypothesis']}",
                    "",
                ])

        # Write report
        report_path = self.output_dir / "rq3_analysis_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return report_path

    def _generate_executive_summary(self, results: dict) -> List[str]:
        """Generate executive summary section."""

        summary = results["summary"]

        lines = [
            "## Executive Summary",
            "",
            f"This analysis investigates which features of knowledge graph rules and questions",
            f"are correlated with LLM inconsistency in question answering.",
            "",
            "### Key Statistics",
            "",
            f"- **Total samples analyzed**: {summary['total_samples']:,}",
            f"- **Overall inconsistency rate**: {summary['overall_inconsistency_rate']*100:.2f}%",
            f"- **Features analyzed**: {summary['n_continuous_features']} continuous + {summary['n_categorical_features']} categorical",
            "",
        ]

        # Significant correlations
        sig_correlations = [
            r for r in results["correlations"]
            if r["is_significant"]
        ]

        if sig_correlations:
            lines.extend([
                "### Significant Correlations Found",
                "",
            ])

            for corr in sig_correlations[:5]:  # Top 5
                direction = "positive" if corr["pearson_r"] > 0 else "negative"
                strength = corr["effect_size"]
                lines.append(
                    f"- **{corr['feature']}**: r={corr['pearson_r']:.3f} ({direction}, {strength})"
                )

            lines.append("")

        return lines

    def _generate_data_overview(self, df: pd.DataFrame, results: dict) -> List[str]:
        """Generate data overview section."""

        summary = results["summary"]

        lines = [
            "## Data Overview",
            "",
            "### Sample Distribution",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total samples | {summary['total_samples']:,} |",
            f"| Inconsistent samples | {summary['total_inconsistent']:,} |",
            f"| Consistent samples | {summary['total_samples'] - summary['total_inconsistent']:,} |",
            "",
            "### Feature Statistics",
            "",
            "| Feature | Mean | Std | Min | Max |",
            "|---------|------|-----|-----|-----|",
        ]

        # Add feature statistics
        key_features = [
            "body_atom_count",
            "unique_variable_count",
            "predicate_scarcity_score",
            "pattern_support_score",
            "entity_count",
        ]

        for feature in key_features:
            if feature in df.columns:
                lines.append(
                    f"| {feature} | {df[feature].mean():.2f} | {df[feature].std():.2f} | "
                    f"{df[feature].min():.0f} | {df[feature].max():.0f} |"
                )

        lines.append("")
        return lines

    def _generate_correlation_section(self, results: dict) -> List[str]:
        """Generate correlation analysis section."""

        correlations = results["correlations"]

        lines = [
            "## Correlation Analysis",
            "",
            "Analysis of linear relationships between continuous features and inconsistency.",
            "",
            "### Correlation Summary Table",
            "",
            "| Feature | Pearson r | p-value | Significant | Effect Size |",
            "|---------|-----------|---------|-------------|-------------|",
        ]

        for corr in correlations:
            sig = "Yes ***" if corr["pearson_p"] < 0.001 else \
                  "Yes **" if corr["pearson_p"] < 0.01 else \
                  "Yes *" if corr["pearson_p"] < 0.05 else "No"
            lines.append(
                f"| {corr['feature']} | {corr['pearson_r']:.4f} | {corr['pearson_p']:.4f} | "
                f"{sig} | {corr['effect_size']} |"
            )

        lines.append("")
        lines.extend([
            "### Interpretation",
            "",
            "Effect size interpretation:",
            "- **Strong**: |r| >= 0.5",
            "- **Moderate**: 0.3 <= |r| < 0.5",
            "- **Weak**: 0.1 <= |r| < 0.3",
            "- **Negligible**: |r| < 0.1",
            "",
            "Significance levels: * p<0.05, ** p<0.01, *** p<0.001",
            "",
        ])

        return lines

    def _generate_categorical_section(self, results: dict) -> List[str]:
        """Generate categorical analysis section."""

        lines = [
            "## Categorical Analysis",
            "",
            "Chi-square test of independence is used for categorical variables vs binary outcome.",
            "",
        ]

        categorical = results["categorical"]

        for factor, (group_results, chi2_result) in categorical.items():
            lines.extend([
                f"### By {factor.replace('_', ' ').title()}",
                "",
                "| Group | N | Inconsistent | Rate (%) | 95% CI |",
                "|-------|---|--------------|----------|--------|",
            ])

            for gr in group_results:
                ci = f"[{gr['ci_lower']*100:.1f}%, {gr['ci_upper']*100:.1f}%]"
                lines.append(
                    f"| {gr['group_name']} | {gr['n_samples']:,} | {gr['n_inconsistent']:,} | "
                    f"{gr['inconsistency_rate']*100:.2f} | {ci} |"
                )

            if chi2_result:
                sig = "Yes" if chi2_result["is_significant"] else "No"
                # Effect size interpretation for Cramér's V
                v = chi2_result["cramers_v"]
                v_interp = "large" if v >= 0.5 else "medium" if v >= 0.3 else "small" if v >= 0.1 else "negligible"
                lines.extend([
                    "",
                    f"**Chi-square Test**: χ²={chi2_result['chi2_statistic']:.2f}, "
                    f"df={chi2_result['dof']}, "
                    f"p={chi2_result['p_value']:.4f}, "
                    f"V={chi2_result['cramers_v']:.3f} ({v_interp} effect)",
                    "",
                ])

        return lines

    def _generate_quantile_section(self, results: dict) -> List[str]:
        """Generate quantile analysis section."""

        quantile_results = results.get("quantile_analysis", {})

        if not quantile_results:
            return []

        lines = [
            "## Quantile Analysis",
            "",
            "Analysis of inconsistency rates across quartiles of continuous features.",
            "",
        ]

        for feature, qr in quantile_results.items():
            if not qr:
                continue

            lines.extend([
                f"### {feature.replace('_', ' ').title()}",
                "",
                "| Quartile | N | Rate (%) |",
                "|----------|---|----------|",
            ])

            for gr in qr:
                lines.append(
                    f"| {gr['group_name']} | {gr['n_samples']:,} | {gr['inconsistency_rate']*100:.2f}% |"
                )

            lines.append("")

        return lines

    def _generate_key_findings(self, results: dict) -> List[str]:
        """Generate key findings section."""

        lines = [
            "## Key Findings",
            "",
        ]

        correlations = results["correlations"]
        categorical = results["categorical"]

        # Findings from correlations
        sig_positive = [r for r in correlations if r["is_significant"] and r["pearson_r"] > 0]
        sig_negative = [r for r in correlations if r["is_significant"] and r["pearson_r"] < 0]

        if sig_positive:
            lines.extend([
                "### Positive Correlations (Higher feature → Higher inconsistency)",
                "",
            ])
            for r in sig_positive[:3]:
                lines.append(f"- **{r['feature']}** (r={r['pearson_r']:.3f}): {FEATURE_DEFINITIONS.get(r['feature'], {}).get('description', '')}")
            lines.append("")

        if sig_negative:
            lines.extend([
                "### Negative Correlations (Higher feature → Lower inconsistency)",
                "",
            ])
            for r in sig_negative[:3]:
                lines.append(f"- **{r['feature']}** (r={r['pearson_r']:.3f}): {FEATURE_DEFINITIONS.get(r['feature'], {}).get('description', '')}")
            lines.append("")

        # Findings from categorical analysis
        for factor, (group_results, chi2_result) in categorical.items():
            if chi2_result and chi2_result.get("is_significant", False):
                # Find highest and lowest groups
                sorted_groups = sorted(group_results, key=lambda x: x["inconsistency_rate"])
                lowest = sorted_groups[0]
                highest = sorted_groups[-1]

                lines.extend([
                    f"### {factor.replace('_', ' ').title()} Effect",
                    "",
                    f"- **Lowest inconsistency**: {lowest['group_name']} ({lowest['inconsistency_rate']*100:.1f}%)",
                    f"- **Highest inconsistency**: {highest['group_name']} ({highest['inconsistency_rate']*100:.1f}%)",
                    f"- **Difference**: {(highest['inconsistency_rate'] - lowest['inconsistency_rate'])*100:.1f} percentage points",
                    f"- **Effect size**: Cramér's V = {chi2_result['cramers_v']:.3f}",
                    "",
                ])

        return lines


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_analysis():
    """Run the complete RQ3 analysis pipeline."""

    print("=" * 70)
    print("RQ3: Factor Analysis for LLM Inconsistency")
    print("=" * 70)
    print()

    # Setup output directory
    output_dir = Config.ensure_output_dir()
    print(f"Output directory: {output_dir}")

    # Step 1: Load data
    print("\n[Step 1/5] Loading data...")
    df = load_all_data()
    print(f"  Loaded {len(df):,} samples")

    # Step 2: Extract features
    print("\n[Step 2/5] Extracting features...")
    extractor = FeatureExtractor()
    features_df = extractor.extract_all_features(df)

    # Save features
    features_path = output_dir / "features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"  Features saved to: {features_path}")

    # Step 3: Perform analysis
    print("\n[Step 3/5] Performing statistical analysis...")
    analyzer = FeatureAnalyzer(features_df)
    analysis_results = analyzer.analyze_all()

    # Save analysis results
    corr_df = analyzer.export_correlations_to_dataframe(analysis_results["correlations"])
    corr_path = output_dir / "correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"  Correlations saved to: {corr_path}")

    # Save group analysis
    all_groups = []
    for factor, (group_results, _) in analysis_results["categorical"].items():
        for r in group_results:
            all_groups.append(r.to_dict())
    groups_df = pd.DataFrame(all_groups)
    groups_path = output_dir / "group_analysis.csv"
    groups_df.to_csv(groups_path, index=False)
    print(f"  Group analysis saved to: {groups_path}")

    # Step 4: Generate visualizations
    print("\n[Step 4/5] Generating visualizations...")
    visualizer = RQ3Visualizer(output_dir)

    # Correlation bar chart
    visualizer.plot_correlation_bars([r.to_dict() for r in analysis_results["correlations"]])
    print("  - Correlation bars plot created")

    # Group comparison plots
    for factor, (group_results, _) in analysis_results["categorical"].items():
        visualizer.plot_group_comparison(
            [r.to_dict() for r in group_results],
            title=f"Inconsistency Rate by {factor.replace('_', ' ').title()}",
            filename=f"group_{factor}",
        )
    print("  - Group comparison plots created")

    # Distribution plots
    key_features = ["body_atom_count", "predicate_scarcity_score", "entity_count"]
    for feature in key_features:
        if feature in features_df.columns:
            visualizer.plot_feature_distribution(features_df, feature)
    print("  - Distribution plots created")

    # Correlation heatmap
    continuous_features = [
        f for f in analyzer.continuous_features
        if f in features_df.columns
    ][:8]  # Top 8 features
    if continuous_features:
        visualizer.plot_correlation_heatmap(features_df, continuous_features)
        print("  - Correlation heatmap created")

    # Summary dashboard
    # Convert results to dict format for the visualizer
    categorical_dicts = {}
    for factor, (group_results, chi2_result) in analysis_results["categorical"].items():
        chi2_dict = chi2_result.to_dict() if chi2_result else None
        categorical_dicts[factor] = ([r.to_dict() for r in group_results], chi2_dict)

    visualizer.create_summary_dashboard(
        features_df,
        [r.to_dict() for r in analysis_results["correlations"]],
        categorical_dicts,
    )
    print("  - Summary dashboard created")

    # Step 5: Generate report
    print("\n[Step 5/5] Generating analysis report...")
    report_gen = ReportGenerator(output_dir)
    # Convert results to dict format for the report generator
    report_results = {
        "summary": analysis_results["summary"],
        "correlations": [r.to_dict() for r in analysis_results["correlations"]],
        "categorical": categorical_dicts,
        "quantile_analysis": {},
    }
    for feature, qr_list in analysis_results.get("quantile_analysis", {}).items():
        report_results["quantile_analysis"][feature] = [r.to_dict() for r in qr_list] if qr_list else []
    report_path = report_gen.generate(features_df, report_results)
    print(f"  Report saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Analysis Summary")
    print("=" * 70)

    summary = analysis_results["summary"]
    print(f"\nTotal samples: {summary['total_samples']:,}")
    print(f"Overall inconsistency rate: {summary['overall_inconsistency_rate']*100:.2f}%")

    # Top correlations
    print("\nTop Correlations with Inconsistency:")
    for corr in analysis_results["correlations"][:5]:
        sig = "***" if corr.pearson_p < 0.001 else \
              "**" if corr.pearson_p < 0.01 else \
              "*" if corr.pearson_p < 0.05 else ""
        print(f"  {corr.feature}: r={corr.pearson_r:.3f} {sig}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"All results saved to: {output_dir}")
    print("=" * 70)

    return analysis_results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_analysis()
