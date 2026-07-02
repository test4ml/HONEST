#!/usr/bin/env python3
"""
Analyze LLM answer inconsistencies by KG rule type.
Analyzes LLM answer inconsistency across different KG rule types.
"""

import os
import pandas as pd
import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set font support
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_rule(rule_str):
    """Parse rule string to extract relation patterns."""
    # Example: "?h  P2817  ?b  ?a  P361  ?h   => ?a  P2817  ?b"
    if not isinstance(rule_str, str):
        return None, None, None

    parts = rule_str.split('=>')
    if len(parts) != 2:
        return None, None, None

    premise = parts[0].strip()
    conclusion = parts[1].strip()

    # Extract relations from premise
    premise_tokens = premise.split()
    relations = [token for token in premise_tokens if token.startswith('P')]

    return premise, conclusion, relations


def analyze_by_rule(data_dir, output_dir):
    """Analyze inconsistencies grouped by different KG rules."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rule_stats = defaultdict(lambda: {
        'total': 0,
        'inconsistent': 0,
        'by_mutation': defaultdict(lambda: {'total': 0, 'inconsistent': 0}),
        'relations': set(),
        'avg_contradiction_score': [],
        'avg_entailment_score': []
    })

    # Iterate through all rule directories
    for rule_dir in sorted(data_dir.iterdir()):
        if not rule_dir.is_dir():
            continue

        rule_name = rule_dir.name
        print(f"Processing {rule_name}...")

        # Process each CSV file in the rule directory
        for csv_file in rule_dir.glob('*.csv'):
            mutation_type = csv_file.stem.replace('_llm_answers', '')

            try:
                df = pd.read_csv(csv_file)

                if df.empty:
                    continue

                # Get rule pattern from first row
                if 'original_rule' in df.columns and len(df) > 0:
                    rule_pattern = df.iloc[0]['original_rule']
                    premise, conclusion, relations = parse_rule(rule_pattern)
                    if relations:
                        rule_stats[rule_name]['relations'].update(relations)

                # Count statistics
                total = len(df)
                inconsistent = (~df['answers_consistent']).sum()

                rule_stats[rule_name]['total'] += total
                rule_stats[rule_name]['inconsistent'] += inconsistent
                rule_stats[rule_name]['by_mutation'][mutation_type]['total'] = total
                rule_stats[rule_name]['by_mutation'][mutation_type]['inconsistent'] = inconsistent

                # Collect NLI scores for inconsistent cases
                inconsistent_df = df[~df['answers_consistent']]
                if len(inconsistent_df) > 0:
                    rule_stats[rule_name]['avg_contradiction_score'].extend(
                        inconsistent_df['contradiction_score'].tolist()
                    )
                    rule_stats[rule_name]['avg_entailment_score'].extend(
                        inconsistent_df['entailment_score'].tolist()
                    )

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

    # Convert to DataFrame for analysis
    results = []
    for rule_name, stats in rule_stats.items():
        if stats['total'] == 0:
            continue

        inconsistency_rate = stats['inconsistent'] / stats['total']

        result = {
            'rule_name': rule_name,
            'total_samples': stats['total'],
            'inconsistent_samples': stats['inconsistent'],
            'inconsistency_rate': inconsistency_rate,
            'relations': ', '.join(sorted(stats['relations'])),
            'num_relations': len(stats['relations']),
            'avg_contradiction_score': sum(stats['avg_contradiction_score']) / len(stats['avg_contradiction_score']) if stats['avg_contradiction_score'] else 0,
            'avg_entailment_score': sum(stats['avg_entailment_score']) / len(stats['avg_entailment_score']) if stats['avg_entailment_score'] else 0
        }

        # Add mutation-specific stats
        for mutation_type, mutation_stats in stats['by_mutation'].items():
            if mutation_stats['total'] > 0:
                result[f'{mutation_type}_total'] = mutation_stats['total']
                result[f'{mutation_type}_inconsistent'] = mutation_stats['inconsistent']
                result[f'{mutation_type}_rate'] = mutation_stats['inconsistent'] / mutation_stats['total']

        results.append(result)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('inconsistency_rate', ascending=False)

    # Save results
    df_results.to_csv(output_dir / 'rule_inconsistency_analysis.csv', index=False)

    # Generate summary statistics
    summary = {
        'total_rules': len(df_results),
        'avg_inconsistency_rate': df_results['inconsistency_rate'].mean(),
        'median_inconsistency_rate': df_results['inconsistency_rate'].median(),
        'max_inconsistency_rate': df_results['inconsistency_rate'].max(),
        'min_inconsistency_rate': df_results['inconsistency_rate'].min(),
        'top_10_problematic_rules': df_results.nlargest(10, 'inconsistency_rate')[
            ['rule_name', 'inconsistency_rate', 'total_samples', 'relations']
        ].to_dict('records')
    }

    with open(output_dir / 'rule_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Visualization 1: Top 20 rules by inconsistency rate
    fig, ax = plt.subplots(figsize=(12, 8))
    top_20 = df_results.nlargest(20, 'inconsistency_rate')
    ax.barh(range(len(top_20)), top_20['inconsistency_rate'], color='coral')
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['rule_name'])
    ax.set_xlabel('Inconsistency Rate')
    ax.set_title('Top 20 Rules with Highest Inconsistency Rate')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'top_20_rules_inconsistency.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 2: Inconsistency rate distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_results['inconsistency_rate'], bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel('Inconsistency Rate')
    ax.set_ylabel('Number of Rules')
    ax.set_title('Distribution of Inconsistency Rates Across Rules')
    plt.tight_layout()
    plt.savefig(output_dir / 'inconsistency_rate_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 3: Relation count vs inconsistency rate
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_results['num_relations'], df_results['inconsistency_rate'],
               alpha=0.6, s=df_results['total_samples']/2, color='purple')
    ax.set_xlabel('Number of Relations in Rule')
    ax.set_ylabel('Inconsistency Rate')
    ax.set_title('Rule Complexity (Number of Relations) vs Inconsistency Rate')
    plt.tight_layout()
    plt.savefig(output_dir / 'relations_vs_inconsistency.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}")
    print("RULE-BASED INCONSISTENCY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total rules analyzed: {summary['total_rules']}")
    print(f"Average inconsistency rate: {summary['avg_inconsistency_rate']:.2%}")
    print(f"Median inconsistency rate: {summary['median_inconsistency_rate']:.2%}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - rule_inconsistency_analysis.csv")
    print(f"  - rule_summary.json")
    print(f"  - *.png (visualizations)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze LLM answer inconsistencies by KG rule type',
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
        default='data/statistics/inconsistency_by_rule',
        help='Output directory for analysis results'
    )

    args = parser.parse_args()
    analyze_by_rule(args.input_dir, args.output_dir)
