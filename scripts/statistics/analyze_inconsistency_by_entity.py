#!/usr/bin/env python3
"""
Analyze LLM answer inconsistencies by entity characteristics.
Analyzes how entity characteristics affect LLM answer inconsistency.
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
import re

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def extract_entity_info(entity_labels_str):
    """Extract entity IDs and types from entity_labels string."""
    if not isinstance(entity_labels_str, str):
        return []

    entities = []
    # Parse the JSON-like string
    try:
        import ast
        entity_list = ast.literal_eval(entity_labels_str)
        for entity in entity_list:
            # Format: "Q1833956: Cultural Heritage Monuments in Altstadt | Wikimedia list article"
            parts = entity.split(':', 1)
            if len(parts) == 2:
                entity_id = parts[0].strip()
                rest = parts[1].strip()

                # Split by | to get name and type
                if '|' in rest:
                    name, entity_type = rest.split('|', 1)
                    entities.append({
                        'id': entity_id,
                        'name': name.strip(),
                        'type': entity_type.strip()
                    })
    except:
        pass

    return entities


def analyze_entity_label_length(label):
    """Analyze entity label complexity by character and word count."""
    if not isinstance(label, str):
        return 0, 0
    return len(label), len(label.split())


def analyze_by_entity(data_dir, output_dir):
    """Analyze inconsistencies based on entity characteristics."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics collectors
    entity_type_stats = defaultdict(lambda: {'total': 0, 'inconsistent': 0})
    entity_id_stats = defaultdict(lambda: {'total': 0, 'inconsistent': 0})
    entity_count_stats = defaultdict(lambda: {'total': 0, 'inconsistent': 0})

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

                    # Extract entity information
                    entities = extract_entity_info(row.get('entity_labels', ''))
                    num_entities = len(entities)

                    # Collect entity count statistics
                    entity_count_stats[num_entities]['total'] += 1
                    if is_inconsistent:
                        entity_count_stats[num_entities]['inconsistent'] += 1

                    # Collect entity type and ID statistics
                    for entity in entities:
                        entity_type = entity['type']
                        entity_id = entity['id']

                        entity_type_stats[entity_type]['total'] += 1
                        entity_id_stats[entity_id]['total'] += 1

                        if is_inconsistent:
                            entity_type_stats[entity_type]['inconsistent'] += 1
                            entity_id_stats[entity_id]['inconsistent'] += 1

                    # Collect sample data
                    sample_data = {
                        'rule': rule_dir.name,
                        'mutation_type': csv_file.stem.replace('_llm_answers', ''),
                        'inconsistent': is_inconsistent,
                        'num_entities': num_entities,
                        'entities': entities,
                        'contradiction_score': row.get('contradiction_score', 0),
                        'entailment_score': row.get('entailment_score', 0),
                    }

                    # Add entity label lengths
                    for i, entity in enumerate(entities):
                        char_len, word_len = analyze_entity_label_length(entity['name'])
                        sample_data[f'entity_{i}_char_len'] = char_len
                        sample_data[f'entity_{i}_word_len'] = word_len

                    all_samples.append(sample_data)

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

    # Analyze entity type statistics
    entity_type_results = []
    for entity_type, stats in entity_type_stats.items():
        if stats['total'] >= 5:  # Only consider types with sufficient samples
            entity_type_results.append({
                'entity_type': entity_type,
                'total_occurrences': stats['total'],
                'inconsistent_occurrences': stats['inconsistent'],
                'inconsistency_rate': stats['inconsistent'] / stats['total']
            })

    df_entity_types = pd.DataFrame(entity_type_results)
    df_entity_types = df_entity_types.sort_values('inconsistency_rate', ascending=False)
    df_entity_types.to_csv(output_dir / 'entity_type_inconsistency.csv', index=False)

    # Analyze entity count statistics
    entity_count_results = []
    for count, stats in sorted(entity_count_stats.items()):
        if stats['total'] > 0:
            entity_count_results.append({
                'num_entities': count,
                'total_samples': stats['total'],
                'inconsistent_samples': stats['inconsistent'],
                'inconsistency_rate': stats['inconsistent'] / stats['total']
            })

    df_entity_counts = pd.DataFrame(entity_count_results)
    df_entity_counts.to_csv(output_dir / 'entity_count_inconsistency.csv', index=False)

    # Analyze most problematic entities
    entity_id_results = []
    for entity_id, stats in entity_id_stats.items():
        if stats['total'] >= 10:  # Only consider entities with sufficient samples
            entity_id_results.append({
                'entity_id': entity_id,
                'total_occurrences': stats['total'],
                'inconsistent_occurrences': stats['inconsistent'],
                'inconsistency_rate': stats['inconsistent'] / stats['total']
            })

    df_entity_ids = pd.DataFrame(entity_id_results)
    df_entity_ids = df_entity_ids.sort_values('inconsistency_rate', ascending=False)
    df_entity_ids.to_csv(output_dir / 'problematic_entities.csv', index=False)

    # Analyze entity label complexity
    df_samples = pd.DataFrame(all_samples)

    # Calculate average entity name lengths
    entity_length_stats = []
    for col in df_samples.columns:
        if '_char_len' in col:
            entity_lengths = df_samples.groupby('inconsistent')[col].mean()
            if len(entity_lengths) == 2:
                entity_length_stats.append({
                    'entity_position': col.replace('_char_len', ''),
                    'avg_char_len_consistent': entity_lengths[False] if False in entity_lengths else 0,
                    'avg_char_len_inconsistent': entity_lengths[True] if True in entity_lengths else 0,
                })

    df_entity_lengths = pd.DataFrame(entity_length_stats)
    if not df_entity_lengths.empty:
        df_entity_lengths.to_csv(output_dir / 'entity_length_analysis.csv', index=False)

    # Generate summary
    summary = {
        'total_unique_entity_types': len(entity_type_stats),
        'total_unique_entities': len(entity_id_stats),
        'top_10_problematic_entity_types': df_entity_types.nlargest(10, 'inconsistency_rate').to_dict('records') if not df_entity_types.empty else [],
        'top_10_problematic_entities': df_entity_ids.nlargest(10, 'inconsistency_rate').to_dict('records') if not df_entity_ids.empty else [],
        'entity_count_correlation': df_entity_counts.to_dict('records')
    }

    with open(output_dir / 'entity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Visualization 1: Top problematic entity types
    if not df_entity_types.empty and len(df_entity_types) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        top_types = df_entity_types.nlargest(20, 'inconsistency_rate')

        y_pos = np.arange(len(top_types))
        ax.barh(y_pos, top_types['inconsistency_rate'], color='coral')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_types['entity_type'], fontsize=8)
        ax.set_xlabel('Inconsistency Rate')
        ax.set_title('Top 20 Entity Types with Highest Inconsistency Rate')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / 'top_entity_types.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Visualization 2: Number of entities vs inconsistency rate
    if not df_entity_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_entity_counts['num_entities'], df_entity_counts['inconsistency_rate'],
                marker='o', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Number of Entities in Question')
        ax.set_ylabel('Inconsistency Rate')
        ax.set_title('Relationship Between Entity Count and Inconsistency')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'entity_count_vs_inconsistency.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Visualization 3: Entity type distribution
    if not df_entity_types.empty and len(df_entity_types) >= 10:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_entity_types['total_occurrences'],
                   df_entity_types['inconsistency_rate'],
                   alpha=0.6, s=100, color='purple')
        ax.set_xlabel('Total Occurrences')
        ax.set_ylabel('Inconsistency Rate')
        ax.set_title('Entity Type: Frequency vs Inconsistency Rate')
        ax.set_xscale('log')
        plt.tight_layout()
        plt.savefig(output_dir / 'entity_type_frequency_vs_inconsistency.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Visualization 4: Top problematic entities bar chart
    if not df_entity_ids.empty and len(df_entity_ids) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        top_entities = df_entity_ids.nlargest(15, 'inconsistency_rate')

        y_pos = np.arange(len(top_entities))
        ax.barh(y_pos, top_entities['inconsistency_rate'], color='#3498db')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_entities['entity_id'], fontsize=9)
        ax.set_xlabel('Inconsistency Rate')
        ax.set_title('Top 15 Most Problematic Entities (by ID)')
        ax.invert_yaxis()

        # Add occurrence counts as text
        for i, (idx, row) in enumerate(top_entities.iterrows()):
            ax.text(row['inconsistency_rate'], i, f" n={row['total_occurrences']}",
                   va='center', fontsize=7)

        plt.tight_layout()
        plt.savefig(output_dir / 'top_problematic_entities.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n{'='*60}")
    print("ENTITY-BASED INCONSISTENCY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total unique entity types: {summary['total_unique_entity_types']}")
    print(f"Total unique entities: {summary['total_unique_entities']}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - entity_type_inconsistency.csv")
    print(f"  - entity_count_inconsistency.csv")
    print(f"  - problematic_entities.csv")
    print(f"  - entity_summary.json")
    print(f"  - *.png (visualizations)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze LLM answer inconsistencies by entity characteristics',
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
        default='data/statistics/inconsistency_by_entity',
        help='Output directory for analysis results'
    )

    args = parser.parse_args()
    analyze_by_entity(args.input_dir, args.output_dir)
