#!/usr/bin/env python3
"""
Analyze LLM answer inconsistencies by relation patterns.
Analyzes how different relation patterns affect LLM answer inconsistency.
"""

import os
import pandas as pd
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def parse_rule_relations(rule_str):
    """Extract relation patterns from rule string."""
    if not isinstance(rule_str, str):
        return None, None, []

    parts = rule_str.split('=>')
    if len(parts) != 2:
        return None, None, []

    premise = parts[0].strip()
    conclusion = parts[1].strip()

    # Extract relations (properties that start with P)
    premise_tokens = premise.split()
    conclusion_tokens = conclusion.split()

    premise_relations = [token for token in premise_tokens if token.startswith('P')]
    conclusion_relations = [token for token in conclusion_tokens if token.startswith('P')]

    return premise_relations, conclusion_relations, premise_relations + conclusion_relations


def analyze_relation_combinations(relations):
    """Analyze relation combinations and patterns."""
    if not relations:
        return []

    # Get all unique combinations
    combinations = []
    n = len(relations)

    # Single relations
    for rel in relations:
        combinations.append(rel)

    # Pairs of relations
    if n >= 2:
        for i in range(n - 1):
            combinations.append(f"{relations[i]}-{relations[i+1]}")

    # Triplets
    if n >= 3:
        for i in range(n - 2):
            combinations.append(f"{relations[i]}-{relations[i+1]}-{relations[i+2]}")

    return combinations


def analyze_by_relation(data_dir, output_dir):
    """Analyze inconsistencies based on relation patterns."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics collectors
    relation_stats = defaultdict(lambda: {'total': 0, 'inconsistent': 0, 'rules': set()})
    relation_pair_stats = defaultdict(lambda: {'total': 0, 'inconsistent': 0})
    rule_pattern_stats = defaultdict(lambda: {
        'total': 0,
        'inconsistent': 0,
        'premise_relations': [],
        'conclusion_relations': [],
        'contradiction_scores': [],
        'entailment_scores': []
    })

    all_samples = []

    # Iterate through all data
    for rule_dir in sorted(data_dir.iterdir()):
        if not rule_dir.is_dir():
            continue

        for csv_file in rule_dir.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                if df.empty:
                    continue

                for idx, row in df.iterrows():
                    is_inconsistent = not row['answers_consistent']
                    rule_str = row.get('original_rule', '')

                    # Parse rule relations
                    premise_rels, conclusion_rels, all_rels = parse_rule_relations(rule_str)

                    if not all_rels:
                        continue

                    # Statistics for individual relations
                    for rel in all_rels:
                        relation_stats[rel]['total'] += 1
                        relation_stats[rel]['rules'].add(rule_dir.name)
                        if is_inconsistent:
                            relation_stats[rel]['inconsistent'] += 1

                    # Statistics for relation combinations
                    combinations = analyze_relation_combinations(all_rels)
                    for combo in combinations:
                        if '-' in combo:  # Only pairs and triplets
                            relation_pair_stats[combo]['total'] += 1
                            if is_inconsistent:
                                relation_pair_stats[combo]['inconsistent'] += 1

                    # Statistics for rule patterns
                    pattern_key = f"{' '.join(premise_rels)} => {' '.join(conclusion_rels)}"
                    rule_pattern_stats[pattern_key]['total'] += 1
                    if is_inconsistent:
                        rule_pattern_stats[pattern_key]['inconsistent'] += 1
                    rule_pattern_stats[pattern_key]['premise_relations'] = premise_rels
                    rule_pattern_stats[pattern_key]['conclusion_relations'] = conclusion_rels
                    rule_pattern_stats[pattern_key]['contradiction_scores'].append(
                        row.get('contradiction_score', 0)
                    )
                    rule_pattern_stats[pattern_key]['entailment_scores'].append(
                        row.get('entailment_score', 0)
                    )

                    # Collect sample data
                    all_samples.append({
                        'rule': rule_dir.name,
                        'mutation_type': csv_file.stem.replace('_llm_answers', ''),
                        'inconsistent': is_inconsistent,
                        'premise_relations': ','.join(premise_rels) if premise_rels else '',
                        'conclusion_relations': ','.join(conclusion_rels) if conclusion_rels else '',
                        'num_relations': len(all_rels),
                        'contradiction_score': row.get('contradiction_score', 0),
                        'entailment_score': row.get('entailment_score', 0),
                    })

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

    # Analyze individual relation statistics
    relation_results = []
    for relation, stats in relation_stats.items():
        if stats['total'] >= 5:  # Only consider relations with sufficient samples
            relation_results.append({
                'relation': relation,
                'total_occurrences': stats['total'],
                'inconsistent_occurrences': stats['inconsistent'],
                'inconsistency_rate': stats['inconsistent'] / stats['total'],
                'num_rules_involved': len(stats['rules'])
            })

    df_relations = pd.DataFrame(relation_results)
    df_relations = df_relations.sort_values('inconsistency_rate', ascending=False)
    df_relations.to_csv(output_dir / 'relation_inconsistency.csv', index=False)

    # Analyze relation pair statistics
    relation_pair_results = []
    for pair, stats in relation_pair_stats.items():
        if stats['total'] >= 5:  # Only consider pairs with sufficient samples
            relation_pair_results.append({
                'relation_pattern': pair,
                'total_occurrences': stats['total'],
                'inconsistent_occurrences': stats['inconsistent'],
                'inconsistency_rate': stats['inconsistent'] / stats['total']
            })

    df_relation_pairs = pd.DataFrame(relation_pair_results)
    df_relation_pairs = df_relation_pairs.sort_values('inconsistency_rate', ascending=False)
    df_relation_pairs.to_csv(output_dir / 'relation_pattern_inconsistency.csv', index=False)

    # Analyze rule pattern statistics
    rule_pattern_results = []
    for pattern, stats in rule_pattern_stats.items():
        if stats['total'] > 0:
            rule_pattern_results.append({
                'rule_pattern': pattern,
                'total_samples': stats['total'],
                'inconsistent_samples': stats['inconsistent'],
                'inconsistency_rate': stats['inconsistent'] / stats['total'],
                'num_premise_relations': len(stats['premise_relations']),
                'num_conclusion_relations': len(stats['conclusion_relations']),
                'avg_contradiction_score': np.mean(stats['contradiction_scores']),
                'avg_entailment_score': np.mean(stats['entailment_scores'])
            })

    df_rule_patterns = pd.DataFrame(rule_pattern_results)
    df_rule_patterns = df_rule_patterns.sort_values('inconsistency_rate', ascending=False)
    df_rule_patterns.to_csv(output_dir / 'rule_pattern_analysis.csv', index=False)

    # Analyze number of relations vs inconsistency
    df_samples = pd.DataFrame(all_samples)
    relation_count_stats = df_samples.groupby('num_relations').agg({
        'inconsistent': ['count', 'sum', 'mean']
    }).reset_index()
    relation_count_stats.columns = ['num_relations', 'total_samples', 'inconsistent_samples', 'inconsistency_rate']
    relation_count_stats.to_csv(output_dir / 'relation_count_analysis.csv', index=False)

    # Generate summary
    summary = {
        'total_unique_relations': len(relation_stats),
        'total_unique_patterns': len(relation_pair_stats),
        'total_rule_patterns': len(rule_pattern_stats),
        'top_10_problematic_relations': df_relations.nlargest(10, 'inconsistency_rate').to_dict('records') if not df_relations.empty else [],
        'top_10_problematic_patterns': df_relation_pairs.nlargest(10, 'inconsistency_rate').to_dict('records') if not df_relation_pairs.empty else [],
        'most_common_relations': df_relations.nlargest(10, 'total_occurrences')[['relation', 'total_occurrences', 'inconsistency_rate']].to_dict('records') if not df_relations.empty else []
    }

    with open(output_dir / 'relation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Visualization 1: Top problematic relations
    if not df_relations.empty and len(df_relations) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        top_relations = df_relations.nlargest(20, 'inconsistency_rate')

        y_pos = np.arange(len(top_relations))
        ax.barh(y_pos, top_relations['inconsistency_rate'], color='#e74c3c')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_relations['relation'], fontsize=9)
        ax.set_xlabel('Inconsistency Rate')
        ax.set_title('Top 20 Relations with Highest Inconsistency Rate')
        ax.invert_yaxis()

        # Add occurrence counts
        for i, (idx, row) in enumerate(top_relations.iterrows()):
            ax.text(row['inconsistency_rate'], i, f" n={row['total_occurrences']}",
                   va='center', fontsize=7)

        plt.tight_layout()
        plt.savefig(output_dir / 'top_problematic_relations.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Visualization 2: Relation frequency vs inconsistency rate
    if not df_relations.empty and len(df_relations) >= 10:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_relations['total_occurrences'],
                   df_relations['inconsistency_rate'],
                   alpha=0.6, s=100, color='#3498db')
        ax.set_xlabel('Total Occurrences (log scale)')
        ax.set_ylabel('Inconsistency Rate')
        ax.set_title('Relation Frequency vs Inconsistency Rate')
        ax.set_xscale('log')
        plt.tight_layout()
        plt.savefig(output_dir / 'relation_frequency_vs_inconsistency.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Visualization 3: Number of relations vs inconsistency rate
    if not relation_count_stats.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(relation_count_stats['num_relations'],
                relation_count_stats['inconsistency_rate'],
                marker='o', linewidth=2, markersize=8, color='#2ecc71')
        ax.set_xlabel('Number of Relations in Rule')
        ax.set_ylabel('Inconsistency Rate')
        ax.set_title('Rule Complexity (Relation Count) vs Inconsistency')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'relation_count_vs_inconsistency.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Visualization 4: Top relation patterns
    if not df_relation_pairs.empty and len(df_relation_pairs) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        top_patterns = df_relation_pairs.nlargest(20, 'inconsistency_rate')

        y_pos = np.arange(len(top_patterns))
        ax.barh(y_pos, top_patterns['inconsistency_rate'], color='#9b59b6')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_patterns['relation_pattern'], fontsize=8)
        ax.set_xlabel('Inconsistency Rate')
        ax.set_title('Top 20 Relation Patterns with Highest Inconsistency Rate')
        ax.invert_yaxis()

        # Add occurrence counts
        for i, (idx, row) in enumerate(top_patterns.iterrows()):
            ax.text(row['inconsistency_rate'], i, f" n={row['total_occurrences']}",
                   va='center', fontsize=7)

        plt.tight_layout()
        plt.savefig(output_dir / 'top_relation_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Visualization 5: Distribution of inconsistency rates
    if not df_relations.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_relations['inconsistency_rate'], bins=20, color='skyblue', edgecolor='black')
        ax.set_xlabel('Inconsistency Rate')
        ax.set_ylabel('Number of Relations')
        ax.set_title('Distribution of Inconsistency Rates Across Relations')
        ax.axvline(df_relations['inconsistency_rate'].mean(), color='red',
                   linestyle='--', label=f'Mean: {df_relations["inconsistency_rate"].mean():.2%}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'inconsistency_rate_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n{'='*60}")
    print("RELATION-BASED INCONSISTENCY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total unique relations: {summary['total_unique_relations']}")
    print(f"Total unique patterns: {summary['total_unique_patterns']}")
    print(f"Total rule patterns: {summary['total_rule_patterns']}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - relation_inconsistency.csv")
    print(f"  - relation_pattern_inconsistency.csv")
    print(f"  - rule_pattern_analysis.csv")
    print(f"  - relation_count_analysis.csv")
    print(f"  - relation_summary.json")
    print(f"  - *.png (visualizations)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze LLM answer inconsistencies by relation patterns',
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
        default='data/statistics/inconsistency_by_relation',
        help='Output directory for analysis results'
    )

    args = parser.parse_args()
    analyze_by_relation(args.input_dir, args.output_dir)
