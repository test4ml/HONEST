#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consistency Analysis for Metamorphic Testing Results

Analyzes consistency results from metamorphic testing of QA models,
computing statistics and generating visualizations.

Usage:
    python scripts/analysis/analyze_consistency.py \
        --data-dir data/examples/consistency_results_Qwen2.5-7B-Instruct \
        --output-dir results/consistency_analysis
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import json


class ConsistencyAnalyzer:
    """Analyze metamorphic testing consistency results"""

    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize consistency analyzer

        Args:
            data_dir: Directory containing consistency results
            output_dir: Directory to save analysis results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create figures subdirectory
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        self.df_all = None
        self.metrics = {}

        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10

    def load_data(self) -> pd.DataFrame:
        """
        Load and combine all CSV files from consistency results

        Returns:
            Combined DataFrame with all test results
        """
        print("Loading consistency data...")

        all_data = []

        # Iterate through rule directories
        for rule_dir in self.data_dir.iterdir():
            if not rule_dir.is_dir():
                continue

            rule_name = rule_dir.name  # e.g., kg_rule_11

            # Load CSV files in this rule directory
            for csv_file in rule_dir.glob("*.csv"):
                mr_type = csv_file.stem.replace("_llm_answers", "")

                df = pd.read_csv(csv_file)
                df['rule'] = rule_name
                df['mr_type'] = mr_type

                all_data.append(df)

        # Combine all data
        self.df_all = pd.concat(all_data, ignore_index=True)

        print(f"Loaded {len(self.df_all)} test cases from {len(all_data)} files")
        print(f"Rules: {self.df_all['rule'].unique()}")
        print(f"MR types: {self.df_all['mr_type'].unique()}")
        print(f"Question types: {self.df_all['original_question_type'].unique()}")

        # Save aggregated data
        aggregated_file = self.output_dir / "consistency_data_aggregated.csv"
        self.df_all.to_csv(aggregated_file, index=False)
        print(f"Saved aggregated data to {aggregated_file}")

        return self.df_all

    def compute_overall_metrics(self) -> Dict:
        """Compute overall consistency metrics"""
        print("\nComputing overall metrics...")

        metrics = {
            'total_tests': len(self.df_all),
            'consistent_tests': self.df_all['answers_consistent'].sum(),
            'inconsistent_tests': (~self.df_all['answers_consistent']).sum(),
            'consistency_rate': self.df_all['answers_consistent'].mean(),
            'inconsistency_rate': 1 - self.df_all['answers_consistent'].mean()
        }

        self.metrics['overall'] = metrics

        print(f"Total tests: {metrics['total_tests']}")
        print(f"Consistent: {metrics['consistent_tests']} ({metrics['consistency_rate']:.2%})")
        print(f"Inconsistent: {metrics['inconsistent_tests']} ({metrics['inconsistency_rate']:.2%})")

        return metrics

    def compute_by_factor(self, factor: str) -> pd.DataFrame:
        """
        Compute metrics by a single factor

        Args:
            factor: Column name to group by (e.g., 'mr_type', 'original_question_type', 'rule')

        Returns:
            DataFrame with metrics for each level of the factor
        """
        print(f"\nComputing metrics by {factor}...")

        results = []

        for level in self.df_all[factor].unique():
            subset = self.df_all[self.df_all[factor] == level]

            n = len(subset)
            n_consistent = subset['answers_consistent'].sum()
            n_inconsistent = n - n_consistent
            consistency_rate = subset['answers_consistent'].mean()
            inconsistency_rate = 1 - consistency_rate

            # Compute 95% confidence interval for inconsistency rate
            # Using Wilson score interval
            p = inconsistency_rate
            z = 1.96  # 95% CI
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denominator
            margin = z * np.sqrt(p * (1-p) / n + z**2 / (4*n**2)) / denominator
            ci_lower = max(0, center - margin)
            ci_upper = min(1, center + margin)

            results.append({
                factor: level,
                'n_tests': n,
                'n_consistent': n_consistent,
                'n_inconsistent': n_inconsistent,
                'consistency_rate': consistency_rate,
                'inconsistency_rate': inconsistency_rate,
                'inconsistency_ci_lower': ci_lower,
                'inconsistency_ci_upper': ci_upper
            })

        df_results = pd.DataFrame(results)
        self.metrics[f'by_{factor}'] = df_results

        print(df_results.to_string(index=False))

        return df_results

    def compute_interaction_matrix(self, factor1: str, factor2: str) -> pd.DataFrame:
        """
        Compute consistency matrix for two factors

        Args:
            factor1: First factor (rows)
            factor2: Second factor (columns)

        Returns:
            Pivot table with inconsistency rates
        """
        print(f"\nComputing interaction: {factor1} × {factor2}...")

        # Group by both factors
        grouped = self.df_all.groupby([factor1, factor2])

        results = []
        for (level1, level2), subset in grouped:
            n = len(subset)
            inconsistency_rate = 1 - subset['answers_consistent'].mean()

            results.append({
                factor1: level1,
                factor2: level2,
                'n_tests': n,
                'inconsistency_rate': inconsistency_rate
            })

        df_results = pd.DataFrame(results)

        # Create pivot table for inconsistency rate
        pivot_rate = df_results.pivot(index=factor1, columns=factor2, values='inconsistency_rate')
        pivot_count = df_results.pivot(index=factor1, columns=factor2, values='n_tests')

        self.metrics[f'{factor1}_x_{factor2}'] = {
            'rates': pivot_rate,
            'counts': pivot_count
        }

        print("\nInconsistency rates:")
        print(pivot_rate)
        print("\nTest counts:")
        print(pivot_count)

        return pivot_rate, pivot_count

    def statistical_tests(self) -> Dict:
        """
        Perform statistical significance tests

        Returns:
            Dictionary with test results
        """
        print("\nPerforming statistical tests...")

        results = {}

        # Chi-square test for each factor
        for factor in ['mr_type', 'original_question_type', 'rule']:
            contingency_table = pd.crosstab(
                self.df_all[factor],
                self.df_all['answers_consistent']
            )

            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            # Cramér's V for effect size
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim))

            results[factor] = {
                'chi2': chi2,
                'p_value': p_value,
                'dof': dof,
                'cramers_v': cramers_v,
                'significant': p_value < 0.05
            }

            print(f"\n{factor}:")
            print(f"  Chi-square = {chi2:.3f}, p = {p_value:.4f}, Cramér's V = {cramers_v:.3f}")
            print(f"  {'Significant' if p_value < 0.05 else 'Not significant'} (α = 0.05)")

        self.metrics['statistical_tests'] = results

        # Save results
        stats_file = self.output_dir / "statistical_tests_results.txt"
        with open(stats_file, 'w') as f:
            f.write("Statistical Significance Tests\n")
            f.write("="*80 + "\n\n")

            for factor, result in results.items():
                f.write(f"{factor}:\n")
                f.write(f"  Chi-square statistic: {result['chi2']:.3f}\n")
                f.write(f"  p-value: {result['p_value']:.6f}\n")
                f.write(f"  Degrees of freedom: {result['dof']}\n")
                f.write(f"  Cramér's V (effect size): {result['cramers_v']:.3f}\n")
                f.write(f"  Result: {'Significant' if result['significant'] else 'Not significant'} (α = 0.05)\n")
                f.write("\n")

        print(f"Statistical test results saved to {stats_file}")

        return results

    def generate_summary_report(self) -> str:
        """
        Generate markdown report with all findings

        Returns:
            Report text
        """
        print("\nGenerating summary report...")

        report = []
        report.append("# Metamorphic Testing Consistency Analysis Report")
        report.append("")
        report.append(f"**Model**: {self.data_dir.name}")
        report.append("")

        # Overall metrics
        report.append("## Overall Consistency Metrics")
        report.append("")
        overall = self.metrics['overall']
        report.append(f"- **Total test cases**: {overall['total_tests']}")
        report.append(f"- **Consistent**: {overall['consistent_tests']} ({overall['consistency_rate']:.2%})")
        report.append(f"- **Inconsistent**: {overall['inconsistent_tests']} ({overall['inconsistency_rate']:.2%})")
        report.append("")

        # By MR type
        report.append("## Inconsistency by Metamorphic Relation Type")
        report.append("")
        df_mr = self.metrics['by_mr_type']
        report.append(df_mr.to_markdown(index=False))
        report.append("")

        # By question type
        report.append("## Inconsistency by Question Type")
        report.append("")
        df_qt = self.metrics['by_original_question_type']
        report.append(df_qt.to_markdown(index=False))
        report.append("")

        # By rule
        report.append("## Inconsistency by Inference Rule")
        report.append("")
        df_rule = self.metrics['by_rule']
        report.append(df_rule.to_markdown(index=False))
        report.append("")

        # Statistical tests
        report.append("## Statistical Significance Tests")
        report.append("")
        for factor, result in self.metrics['statistical_tests'].items():
            report.append(f"### {factor}")
            report.append(f"- Chi-square: {result['chi2']:.3f}")
            report.append(f"- p-value: {result['p_value']:.6f}")
            report.append(f"- Cramér's V: {result['cramers_v']:.3f}")
            report.append(f"- **{'Significant' if result['significant'] else 'Not significant'}** (α = 0.05)")
            report.append("")

        # Key findings
        report.append("## Key Findings")
        report.append("")
        report.append("### Most Problematic Areas")
        report.append("")

        # Find highest inconsistency question type
        max_qt = df_qt.loc[df_qt['inconsistency_rate'].idxmax()]
        report.append(f"- **Question Type**: {max_qt['original_question_type']} ({max_qt['inconsistency_rate']:.2%} inconsistency)")

        # Find highest inconsistency rule
        max_rule = df_rule.loc[df_rule['inconsistency_rate'].idxmax()]
        report.append(f"- **Inference Rule**: {max_rule['rule']} ({max_rule['inconsistency_rate']:.2%} inconsistency)")

        # Find highest inconsistency MR
        max_mr = df_mr.loc[df_mr['inconsistency_rate'].idxmax()]
        report.append(f"- **MR Type**: {max_mr['mr_type']} ({max_mr['inconsistency_rate']:.2%} inconsistency)")
        report.append("")

        report_text = "\n".join(report)

        # Save report
        report_file = self.output_dir / "consistency_analysis_report.md"
        with open(report_file, 'w') as f:
            f.write(report_text)

        print(f"Report saved to {report_file}")

        return report_text

    def plot_all_figures(self):
        """Generate all visualization figures"""
        print("\nGenerating visualizations...")

        self.plot_overall_summary()
        self.plot_by_mr_type()
        self.plot_by_question_type()
        self.plot_by_rule()
        self.plot_heatmap_rule_x_question_type()
        self.plot_heatmap_rule_x_mr_type()
        self.plot_heatmap_question_type_x_mr_type()
        self.plot_grouped_bar()
        self.plot_stacked_bar()

        print(f"All figures saved to {self.figures_dir}")

    def plot_overall_summary(self):
        """Figure 1: Overall summary dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Overall Consistency Summary', fontsize=16, fontweight='bold')

        # Panel A: Pie chart - Overall consistency
        ax = axes[0, 0]
        overall = self.metrics['overall']
        sizes = [overall['consistent_tests'], overall['inconsistent_tests']]
        labels = ['Consistent', 'Inconsistent']
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Overall Consistency Rate', fontweight='bold')

        # Panel B: Bar chart - Tests per rule
        ax = axes[0, 1]
        df_rule = self.metrics['by_rule']
        ax.bar(df_rule['rule'], df_rule['n_tests'], color='#3498db')
        ax.set_xlabel('Inference Rule')
        ax.set_ylabel('Number of Tests')
        ax.set_title('Test Cases per Rule', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Panel C: Bar chart - Tests per question type
        ax = axes[1, 0]
        df_qt = self.metrics['by_original_question_type']
        ax.bar(df_qt['original_question_type'], df_qt['n_tests'], color='#9b59b6')
        ax.set_xlabel('Question Type')
        ax.set_ylabel('Number of Tests')
        ax.set_title('Test Cases per Question Type', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Panel D: Bar chart - Tests per MR type
        ax = axes[1, 1]
        df_mr = self.metrics['by_mr_type']
        ax.bar(df_mr['mr_type'], df_mr['n_tests'], color='#e67e22')
        ax.set_xlabel('Metamorphic Relation Type')
        ax.set_ylabel('Number of Tests')
        ax.set_title('Test Cases per MR Type', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig1_overall_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_by_mr_type(self):
        """Figure 2: Inconsistency by MR type"""
        df = self.metrics['by_mr_type']

        fig, ax = plt.subplots(figsize=(8, 6))

        x = np.arange(len(df))
        bars = ax.bar(x, df['inconsistency_rate'] * 100, color='#e74c3c', alpha=0.7)

        # Error bars (95% CI)
        errors = [(df['inconsistency_rate'] - df['inconsistency_ci_lower']) * 100,
                  (df['inconsistency_ci_upper'] - df['inconsistency_rate']) * 100]
        ax.errorbar(x, df['inconsistency_rate'] * 100, yerr=errors,
                   fmt='none', ecolor='black', capsize=5, capthick=2)

        # Add sample size labels on bars
        for i, (bar, n) in enumerate(zip(bars, df['n_tests'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'n={n}', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Inconsistency Rate (%)', fontweight='bold')
        ax.set_xlabel('Metamorphic Relation Type', fontweight='bold')
        ax.set_title('Inconsistency Rate by Metamorphic Relation Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['mr_type'], rotation=45, ha='right')
        ax.set_ylim(0, max(df['inconsistency_rate'] * 100) * 1.2)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig2_inconsistency_by_mr.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_by_question_type(self):
        """Figure 3: Inconsistency by question type"""
        df = self.metrics['by_original_question_type']

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(df))
        bars = ax.bar(x, df['inconsistency_rate'] * 100, color='#3498db', alpha=0.7)

        # Error bars (95% CI)
        errors = [(df['inconsistency_rate'] - df['inconsistency_ci_lower']) * 100,
                  (df['inconsistency_ci_upper'] - df['inconsistency_rate']) * 100]
        ax.errorbar(x, df['inconsistency_rate'] * 100, yerr=errors,
                   fmt='none', ecolor='black', capsize=5, capthick=2)

        # Add sample size labels
        for i, (bar, n) in enumerate(zip(bars, df['n_tests'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'n={n}', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Inconsistency Rate (%)', fontweight='bold')
        ax.set_xlabel('Question Type', fontweight='bold')
        ax.set_title('Inconsistency Rate by Question Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['original_question_type'], rotation=45, ha='right')
        ax.set_ylim(0, max(df['inconsistency_rate'] * 100) * 1.2)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig3_inconsistency_by_question_type.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_by_rule(self):
        """Figure 4: Inconsistency by rule"""
        df = self.metrics['by_rule']

        fig, ax = plt.subplots(figsize=(8, 6))

        x = np.arange(len(df))
        bars = ax.bar(x, df['inconsistency_rate'] * 100, color='#9b59b6', alpha=0.7)

        # Error bars
        errors = [(df['inconsistency_rate'] - df['inconsistency_ci_lower']) * 100,
                  (df['inconsistency_ci_upper'] - df['inconsistency_rate']) * 100]
        ax.errorbar(x, df['inconsistency_rate'] * 100, yerr=errors,
                   fmt='none', ecolor='black', capsize=5, capthick=2)

        # Add sample size labels
        for i, (bar, n) in enumerate(zip(bars, df['n_tests'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'n={n}', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Inconsistency Rate (%)', fontweight='bold')
        ax.set_xlabel('Inference Rule', fontweight='bold')
        ax.set_title('Inconsistency Rate by Inference Rule', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['rule'], rotation=45, ha='right')
        ax.set_ylim(0, max(df['inconsistency_rate'] * 100) * 1.2)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig4_inconsistency_by_rule.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heatmap_rule_x_question_type(self):
        """Figure 5: Heatmap - Rule × Question Type"""
        rates, counts = self.metrics['rule_x_original_question_type']['rates'], \
                        self.metrics['rule_x_original_question_type']['counts']

        fig, ax = plt.subplots(figsize=(12, 8))

        # Convert to percentage
        rates_pct = rates * 100

        # Create heatmap
        sns.heatmap(rates_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                   cbar_kws={'label': 'Inconsistency Rate (%)'}, ax=ax)

        # Add sample sizes as text
        for i in range(len(rates)):
            for j in range(len(rates.columns)):
                count = counts.iloc[i, j]
                if pd.notna(count):
                    ax.text(j + 0.5, i + 0.7, f'(n={int(count)})',
                           ha='center', va='center', fontsize=8, color='white')

        ax.set_xlabel('Question Type', fontweight='bold', fontsize=12)
        ax.set_ylabel('Inference Rule', fontweight='bold', fontsize=12)
        ax.set_title('Inconsistency Rate: Rule × Question Type', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig5_heatmap_rule_x_question_type.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heatmap_rule_x_mr_type(self):
        """Figure 6: Heatmap - Rule × MR Type"""
        rates, counts = self.metrics['rule_x_mr_type']['rates'], \
                        self.metrics['rule_x_mr_type']['counts']

        fig, ax = plt.subplots(figsize=(10, 6))

        rates_pct = rates * 100

        sns.heatmap(rates_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                   cbar_kws={'label': 'Inconsistency Rate (%)'}, ax=ax)

        # Add sample sizes
        for i in range(len(rates)):
            for j in range(len(rates.columns)):
                count = counts.iloc[i, j]
                if pd.notna(count):
                    ax.text(j + 0.5, i + 0.7, f'(n={int(count)})',
                           ha='center', va='center', fontsize=8, color='white')

        ax.set_xlabel('Metamorphic Relation Type', fontweight='bold', fontsize=12)
        ax.set_ylabel('Inference Rule', fontweight='bold', fontsize=12)
        ax.set_title('Inconsistency Rate: Rule × MR Type', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig6_heatmap_rule_x_mr_type.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heatmap_question_type_x_mr_type(self):
        """Figure 7: Heatmap - Question Type × MR Type"""
        rates, counts = self.metrics['original_question_type_x_mr_type']['rates'], \
                        self.metrics['original_question_type_x_mr_type']['counts']

        fig, ax = plt.subplots(figsize=(10, 8))

        rates_pct = rates * 100

        sns.heatmap(rates_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                   cbar_kws={'label': 'Inconsistency Rate (%)'}, ax=ax)

        # Add sample sizes
        for i in range(len(rates)):
            for j in range(len(rates.columns)):
                count = counts.iloc[i, j]
                if pd.notna(count):
                    ax.text(j + 0.5, i + 0.7, f'(n={int(count)})',
                           ha='center', va='center', fontsize=8, color='white')

        ax.set_xlabel('Metamorphic Relation Type', fontweight='bold', fontsize=12)
        ax.set_ylabel('Question Type', fontweight='bold', fontsize=12)
        ax.set_title('Inconsistency Rate: Question Type × MR Type', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig7_heatmap_question_type_x_mr_type.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_grouped_bar(self):
        """Figure 8: Grouped bar chart - Inconsistency by rule and question type"""
        # Prepare data
        grouped = self.df_all.groupby(['rule', 'original_question_type']).agg({
            'answers_consistent': lambda x: 1 - x.mean()  # Inconsistency rate
        }).reset_index()
        grouped.columns = ['rule', 'question_type', 'inconsistency_rate']

        # Pivot for grouped bar plot
        pivot = grouped.pivot(index='rule', columns='question_type', values='inconsistency_rate')

        fig, ax = plt.subplots(figsize=(12, 6))

        pivot.plot(kind='bar', ax=ax, width=0.8)

        ax.set_ylabel('Inconsistency Rate', fontweight='bold')
        ax.set_xlabel('Inference Rule', fontweight='bold')
        ax.set_title('Inconsistency Rate by Rule and Question Type', fontsize=14, fontweight='bold')
        ax.legend(title='Question Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig8_grouped_bar_rule_question.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_stacked_bar(self):
        """Figure 9: Stacked bar chart - Consistency distribution"""
        # Group by rule and MR type
        grouped = self.df_all.groupby(['rule', 'mr_type']).agg({
            'answers_consistent': ['sum', 'count']
        }).reset_index()
        grouped.columns = ['rule', 'mr_type', 'consistent', 'total']
        grouped['inconsistent'] = grouped['total'] - grouped['consistent']

        # Create combined label
        grouped['label'] = grouped['rule'] + '\n' + grouped['mr_type']

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(grouped))
        width = 0.8

        # Calculate percentages
        consistent_pct = grouped['consistent'] / grouped['total'] * 100
        inconsistent_pct = grouped['inconsistent'] / grouped['total'] * 100

        p1 = ax.bar(x, consistent_pct, width, label='Consistent', color='#2ecc71')
        p2 = ax.bar(x, inconsistent_pct, width, bottom=consistent_pct,
                   label='Inconsistent', color='#e74c3c')

        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title('Consistency Distribution by Rule and MR Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped['label'], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig9_stacked_bar_consistency.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*80)
        print("Metamorphic Testing Consistency Analysis")
        print("="*80)

        # Load data
        self.load_data()

        # Compute metrics
        self.compute_overall_metrics()
        self.compute_by_factor('mr_type')
        self.compute_by_factor('original_question_type')
        self.compute_by_factor('rule')

        # Compute interactions
        self.compute_interaction_matrix('rule', 'original_question_type')
        self.compute_interaction_matrix('rule', 'mr_type')
        self.compute_interaction_matrix('original_question_type', 'mr_type')

        # Statistical tests
        self.statistical_tests()

        # Generate report
        self.generate_summary_report()

        # Plot figures
        self.plot_all_figures()

        # Save metrics as JSON
        metrics_file = self.output_dir / "metrics_summary.json"
        # Convert DataFrames to dicts for JSON serialization
        # Helper function to convert numpy types to Python types
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return obj

        metrics_json = {}
        for key, value in self.metrics.items():
            if isinstance(value, pd.DataFrame):
                metrics_json[key] = convert_numpy_types(value.to_dict())
            elif isinstance(value, dict) and 'rates' in value:
                metrics_json[key] = {
                    'rates': convert_numpy_types(value['rates'].to_dict()),
                    'counts': convert_numpy_types(value['counts'].to_dict())
                }
            else:
                metrics_json[key] = convert_numpy_types(value)

        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)

        print(f"\nMetrics summary saved to {metrics_file}")
        print("\n" + "="*80)
        print("Analysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze metamorphic testing consistency results"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing consistency results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save analysis results'
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = ConsistencyAnalyzer(args.data_dir, args.output_dir)
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()
