#!/usr/bin/env python3
"""
Analyze LLM answer inconsistencies by mutation type.
Analyzes LLM answer inconsistency across different mutation types.
"""

import os
import pandas as pd
import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def analyze_by_mutation(data_dir, output_dir):
    """Analyze inconsistencies grouped by mutation type."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mutation_stats = defaultdict(lambda: {
        'total': 0,
        'inconsistent': 0,
        'samples': [],
        'contradiction_scores': [],
        'entailment_scores': [],
        'neutral_scores': [],
        'extraction_confidence_original': [],
        'extraction_confidence_mutated': []
    })

    # Iterate through all rule directories
    for rule_dir in sorted(data_dir.iterdir()):
        if not rule_dir.is_dir():
            continue

        # Process each mutation type CSV
        for csv_file in rule_dir.glob('*.csv'):
            mutation_type = csv_file.stem.replace('_llm_answers', '')

            try:
                df = pd.read_csv(csv_file)
                if df.empty:
                    continue

                total = len(df)
                inconsistent = (~df['answers_consistent']).sum()

                mutation_stats[mutation_type]['total'] += total
                mutation_stats[mutation_type]['inconsistent'] += inconsistent

                # Collect detailed statistics for inconsistent cases
                inconsistent_df = df[~df['answers_consistent']]
                mutation_stats[mutation_type]['samples'].extend(
                    inconsistent_df.to_dict('records')
                )

                # Collect NLI scores
                mutation_stats[mutation_type]['contradiction_scores'].extend(
                    df['contradiction_score'].tolist()
                )
                mutation_stats[mutation_type]['entailment_scores'].extend(
                    df['entailment_score'].tolist()
                )
                mutation_stats[mutation_type]['neutral_scores'].extend(
                    df['neutral_score'].tolist()
                )

                # Collect extraction confidence
                mutation_stats[mutation_type]['extraction_confidence_original'].extend(
                    df['extraction_confidence_original'].dropna().tolist()
                )
                mutation_stats[mutation_type]['extraction_confidence_mutated'].extend(
                    df['extraction_confidence_mutated'].dropna().tolist()
                )

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

    # Compute summary statistics
    results = []
    for mutation_type, stats in mutation_stats.items():
        if stats['total'] == 0:
            continue

        result = {
            'mutation_type': mutation_type,
            'total_samples': stats['total'],
            'inconsistent_samples': stats['inconsistent'],
            'inconsistency_rate': stats['inconsistent'] / stats['total'],
            'avg_contradiction_score': np.mean(stats['contradiction_scores']),
            'std_contradiction_score': np.std(stats['contradiction_scores']),
            'avg_entailment_score': np.mean(stats['entailment_scores']),
            'std_entailment_score': np.std(stats['entailment_scores']),
            'avg_neutral_score': np.mean(stats['neutral_scores']),
            'std_neutral_score': np.std(stats['neutral_scores']),
            'avg_extraction_conf_original': np.mean(stats['extraction_confidence_original']) if stats['extraction_confidence_original'] else 0,
            'avg_extraction_conf_mutated': np.mean(stats['extraction_confidence_mutated']) if stats['extraction_confidence_mutated'] else 0,
        }
        results.append(result)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('inconsistency_rate', ascending=False)

    # Save results
    df_results.to_csv(output_dir / 'mutation_inconsistency_analysis.csv', index=False)

    # Generate summary
    summary = {
        'total_mutation_types': len(df_results),
        'overall_stats': df_results.to_dict('records'),
        'key_findings': {
            'most_problematic_mutation': df_results.iloc[0]['mutation_type'],
            'most_problematic_rate': df_results.iloc[0]['inconsistency_rate'],
            'least_problematic_mutation': df_results.iloc[-1]['mutation_type'],
            'least_problematic_rate': df_results.iloc[-1]['inconsistency_rate'],
        }
    }

    with open(output_dir / 'mutation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Visualization 1: Inconsistency rate by mutation type
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax.bar(df_results['mutation_type'], df_results['inconsistency_rate'],
                   color=colors[:len(df_results)])
    ax.set_xlabel('Mutation Type')
    ax.set_ylabel('Inconsistency Rate')
    ax.set_title('Inconsistency Rate by Mutation Type')
    ax.set_ylim(0, max(df_results['inconsistency_rate']) * 1.2)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'mutation_inconsistency_rates.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 2: NLI scores distribution by mutation type
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, score_type in enumerate(['contradiction', 'entailment', 'neutral']):
        ax = axes[idx]
        data_to_plot = []
        labels = []

        for mutation_type in df_results['mutation_type']:
            scores = mutation_stats[mutation_type][f'{score_type}_scores']
            data_to_plot.append(scores)
            labels.append(mutation_type)

        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax.set_title(f'{score_type.capitalize()} Score Distribution')
        ax.set_ylabel('Score')
        ax.set_xlabel('Mutation Type')
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'nli_scores_by_mutation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 3: Sample counts
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_results))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_results['total_samples'], width,
                   label='Total Samples', color='skyblue')
    bars2 = ax.bar(x + width/2, df_results['inconsistent_samples'], width,
                   label='Inconsistent Samples', color='coral')

    ax.set_xlabel('Mutation Type')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Distribution by Mutation Type')
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['mutation_type'], rotation=15, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_counts_by_mutation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 4: Extraction confidence comparison
    if any(df_results['avg_extraction_conf_original'] > 0):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_results))
        width = 0.35

        bars1 = ax.bar(x - width/2, df_results['avg_extraction_conf_original'], width,
                       label='Original Question', color='#3498db')
        bars2 = ax.bar(x + width/2, df_results['avg_extraction_conf_mutated'], width,
                       label='Mutated Question', color='#e74c3c')

        ax.set_xlabel('Mutation Type')
        ax.set_ylabel('Average Extraction Confidence')
        ax.set_title('Answer Extraction Confidence Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df_results['mutation_type'], rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(output_dir / 'extraction_confidence_by_mutation.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n{'='*60}")
    print("MUTATION-BASED INCONSISTENCY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total mutation types: {summary['total_mutation_types']}")
    print(f"\nKey findings:")
    print(f"  Most problematic: {summary['key_findings']['most_problematic_mutation']} "
          f"({summary['key_findings']['most_problematic_rate']:.2%})")
    print(f"  Least problematic: {summary['key_findings']['least_problematic_mutation']} "
          f"({summary['key_findings']['least_problematic_rate']:.2%})")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - mutation_inconsistency_analysis.csv")
    print(f"  - mutation_summary.json")
    print(f"  - *.png (visualizations)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze LLM answer inconsistencies by mutation type',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/examples/consistency_results_nli_Qwen2.5-7B-Instruct',
        help='Input directory containing consistency results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/statistics/inconsistency_by_mutation',
        help='Output directory for analysis results'
    )

    args = parser.parse_args()
    analyze_by_mutation(args.input_dir, args.output_dir)
