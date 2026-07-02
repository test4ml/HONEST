#!/usr/bin/env python3
"""
Analyze LLM answer inconsistencies by NLI score distributions.
Analyzes how NLI score distributions affect inconsistency judgements.
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
from scipy import stats

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def categorize_by_score_distribution(row):
    """Categorize sample by NLI score distribution pattern."""
    contradiction = row['contradiction_score']
    entailment = row['entailment_score']
    neutral = row['neutral_score']

    scores = {
        'contradiction': contradiction,
        'entailment': entailment,
        'neutral': neutral
    }

    max_score_type = max(scores, key=scores.get)
    max_score = scores[max_score_type]

    # Calculate score gap (difference between highest and second highest)
    sorted_scores = sorted(scores.values(), reverse=True)
    score_gap = sorted_scores[0] - sorted_scores[1]

    return max_score_type, max_score, score_gap


def analyze_by_nli_scores(data_dir, output_dir):
    """Analyze inconsistencies based on NLI score distributions."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Collect all data
    for rule_dir in sorted(data_dir.iterdir()):
        if not rule_dir.is_dir():
            continue

        for csv_file in rule_dir.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                if df.empty:
                    continue

                for idx, row in df.iterrows():
                    max_score_type, max_score, score_gap = categorize_by_score_distribution(row)

                    sample = {
                        'rule': rule_dir.name,
                        'mutation_type': csv_file.stem.replace('_llm_answers', ''),
                        'inconsistent': not row['answers_consistent'],
                        'contradiction_score': row['contradiction_score'],
                        'entailment_score': row['entailment_score'],
                        'neutral_score': row['neutral_score'],
                        'max_score_type': max_score_type,
                        'max_score_value': max_score,
                        'score_gap': score_gap,
                        'consistency_confidence': row.get('consistency_confidence', 0),
                        'extraction_confidence_original': row.get('extraction_confidence_original', 0),
                        'extraction_confidence_mutated': row.get('extraction_confidence_mutated', 0),
                    }
                    all_samples.append(sample)

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

    df_samples = pd.DataFrame(all_samples)

    # Analysis 1: Inconsistency by dominant score type
    score_type_analysis = df_samples.groupby('max_score_type').agg({
        'inconsistent': ['count', 'sum', 'mean']
    }).reset_index()
    score_type_analysis.columns = ['max_score_type', 'total_samples', 'inconsistent_samples', 'inconsistency_rate']
    score_type_analysis.to_csv(output_dir / 'inconsistency_by_score_type.csv', index=False)

    # Analysis 2: Score distribution for consistent vs inconsistent
    consistent_samples = df_samples[~df_samples['inconsistent']]
    inconsistent_samples = df_samples[df_samples['inconsistent']]

    score_comparison = pd.DataFrame({
        'score_type': ['contradiction', 'entailment', 'neutral'],
        'consistent_mean': [
            consistent_samples['contradiction_score'].mean(),
            consistent_samples['entailment_score'].mean(),
            consistent_samples['neutral_score'].mean()
        ],
        'consistent_std': [
            consistent_samples['contradiction_score'].std(),
            consistent_samples['entailment_score'].std(),
            consistent_samples['neutral_score'].std()
        ],
        'inconsistent_mean': [
            inconsistent_samples['contradiction_score'].mean(),
            inconsistent_samples['entailment_score'].mean(),
            inconsistent_samples['neutral_score'].mean()
        ],
        'inconsistent_std': [
            inconsistent_samples['contradiction_score'].std(),
            inconsistent_samples['entailment_score'].std(),
            inconsistent_samples['neutral_score'].std()
        ]
    })
    score_comparison.to_csv(output_dir / 'score_comparison.csv', index=False)

    # Analysis 3: Score gap analysis
    gap_bins = [0, 1, 2, 3, 5, 10, 100]
    df_samples['score_gap_bin'] = pd.cut(df_samples['score_gap'], bins=gap_bins)

    gap_analysis = df_samples.groupby('score_gap_bin', observed=True).agg({
        'inconsistent': ['count', 'sum', 'mean']
    }).reset_index()
    gap_analysis.columns = ['score_gap_bin', 'total_samples', 'inconsistent_samples', 'inconsistency_rate']
    gap_analysis.to_csv(output_dir / 'score_gap_analysis.csv', index=False)

    # Analysis 4: Confidence correlation
    confidence_corr = df_samples[['consistency_confidence', 'score_gap', 'max_score_value']].corr()
    confidence_corr.to_csv(output_dir / 'confidence_correlation.csv')

    # Analysis 5: Detailed score ranges for inconsistent cases
    inconsistent_score_ranges = pd.DataFrame({
        'score_type': ['contradiction', 'entailment', 'neutral'],
        'min': [
            inconsistent_samples['contradiction_score'].min(),
            inconsistent_samples['entailment_score'].min(),
            inconsistent_samples['neutral_score'].min()
        ],
        'max': [
            inconsistent_samples['contradiction_score'].max(),
            inconsistent_samples['entailment_score'].max(),
            inconsistent_samples['neutral_score'].max()
        ],
        'mean': [
            inconsistent_samples['contradiction_score'].mean(),
            inconsistent_samples['entailment_score'].mean(),
            inconsistent_samples['neutral_score'].mean()
        ],
        'median': [
            inconsistent_samples['contradiction_score'].median(),
            inconsistent_samples['entailment_score'].median(),
            inconsistent_samples['neutral_score'].median()
        ],
        'q25': [
            inconsistent_samples['contradiction_score'].quantile(0.25),
            inconsistent_samples['entailment_score'].quantile(0.25),
            inconsistent_samples['neutral_score'].quantile(0.25)
        ],
        'q75': [
            inconsistent_samples['contradiction_score'].quantile(0.75),
            inconsistent_samples['entailment_score'].quantile(0.75),
            inconsistent_samples['neutral_score'].quantile(0.75)
        ]
    })
    inconsistent_score_ranges.to_csv(output_dir / 'inconsistent_score_ranges.csv', index=False)

    # Generate summary
    summary = {
        'total_samples': len(df_samples),
        'inconsistent_samples': len(inconsistent_samples),
        'overall_inconsistency_rate': len(inconsistent_samples) / len(df_samples),
        'score_type_distribution': score_type_analysis.to_dict('records'),
        'average_scores': {
            'consistent': {
                'contradiction': consistent_samples['contradiction_score'].mean(),
                'entailment': consistent_samples['entailment_score'].mean(),
                'neutral': consistent_samples['neutral_score'].mean()
            },
            'inconsistent': {
                'contradiction': inconsistent_samples['contradiction_score'].mean(),
                'entailment': inconsistent_samples['entailment_score'].mean(),
                'neutral': inconsistent_samples['neutral_score'].mean()
            }
        },
        'score_gap_stats': {
            'consistent_avg_gap': consistent_samples['score_gap'].mean(),
            'inconsistent_avg_gap': inconsistent_samples['score_gap'].mean()
        }
    }

    with open(output_dir / 'nli_scores_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Visualization 1: Score distributions (violin plot)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    score_types = ['contradiction_score', 'entailment_score', 'neutral_score']
    titles = ['Contradiction Score', 'Entailment Score', 'Neutral Score']

    for idx, (score_type, title) in enumerate(zip(score_types, titles)):
        ax = axes[idx]

        data_consistent = consistent_samples[score_type]
        data_inconsistent = inconsistent_samples[score_type]

        parts = ax.violinplot([data_consistent, data_inconsistent],
                              positions=[1, 2],
                              showmeans=True,
                              showmedians=True)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Consistent', 'Inconsistent'])
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'score_distributions_violin.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 2: Score comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(score_comparison))
    width = 0.35

    bars1 = ax.bar(x - width/2, score_comparison['consistent_mean'], width,
                   label='Consistent', color='#2ecc71', yerr=score_comparison['consistent_std'])
    bars2 = ax.bar(x + width/2, score_comparison['inconsistent_mean'], width,
                   label='Inconsistent', color='#e74c3c', yerr=score_comparison['inconsistent_std'])

    ax.set_xlabel('Score Type')
    ax.set_ylabel('Average Score')
    ax.set_title('Average NLI Scores: Consistent vs Inconsistent Answers')
    ax.set_xticks(x)
    ax.set_xticklabels(score_comparison['score_type'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'score_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 3: Inconsistency rate by dominant score type
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c', '#3498db', '#95a5a6']
    bars = ax.bar(score_type_analysis['max_score_type'],
                  score_type_analysis['inconsistency_rate'],
                  color=colors)

    ax.set_xlabel('Dominant Score Type')
    ax.set_ylabel('Inconsistency Rate')
    ax.set_title('Inconsistency Rate by Dominant NLI Score Type')
    ax.set_ylim(0, max(score_type_analysis['inconsistency_rate']) * 1.2)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')

    # Add sample counts
    for i, row in score_type_analysis.iterrows():
        ax.text(i, 0.01, f"n={row['total_samples']}",
               ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'inconsistency_by_score_type.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 4: Score gap vs inconsistency
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(consistent_samples['score_gap'],
              consistent_samples['max_score_value'],
              alpha=0.3, s=20, color='green', label='Consistent')
    ax.scatter(inconsistent_samples['score_gap'],
              inconsistent_samples['max_score_value'],
              alpha=0.3, s=20, color='red', label='Inconsistent')

    ax.set_xlabel('Score Gap (Difference between highest and second highest)')
    ax.set_ylabel('Maximum Score Value')
    ax.set_title('Score Gap vs Maximum Score: Consistent vs Inconsistent')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'score_gap_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 5: 2D histogram heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    score_pairs = [
        ('contradiction_score', 'entailment_score', 'Contradiction vs Entailment'),
        ('contradiction_score', 'neutral_score', 'Contradiction vs Neutral'),
        ('entailment_score', 'neutral_score', 'Entailment vs Neutral')
    ]

    for idx, (score1, score2, title) in enumerate(score_pairs):
        ax = axes[idx]

        # Plot inconsistent samples
        h = ax.hist2d(inconsistent_samples[score1],
                     inconsistent_samples[score2],
                     bins=30, cmap='Reds', alpha=0.7)
        ax.set_xlabel(score1.replace('_', ' ').title())
        ax.set_ylabel(score2.replace('_', ' ').title())
        ax.set_title(f'{title}\n(Inconsistent Samples)')
        plt.colorbar(h[3], ax=ax, label='Count')

    plt.tight_layout()
    plt.savefig(output_dir / 'score_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 6: Consistency confidence distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist([consistent_samples['consistency_confidence'],
             inconsistent_samples['consistency_confidence']],
            bins=30, label=['Consistent', 'Inconsistent'],
            color=['green', 'red'], alpha=0.6)
    ax.set_xlabel('Consistency Confidence Score')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Distribution of Consistency Confidence Scores')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'consistency_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}")
    print("NLI SCORE-BASED INCONSISTENCY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Inconsistent samples: {summary['inconsistent_samples']}")
    print(f"Overall inconsistency rate: {summary['overall_inconsistency_rate']:.2%}")
    print(f"\nAverage score gaps:")
    print(f"  Consistent: {summary['score_gap_stats']['consistent_avg_gap']:.2f}")
    print(f"  Inconsistent: {summary['score_gap_stats']['inconsistent_avg_gap']:.2f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - inconsistency_by_score_type.csv")
    print(f"  - score_comparison.csv")
    print(f"  - score_gap_analysis.csv")
    print(f"  - nli_scores_summary.json")
    print(f"  - *.png (visualizations)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze LLM answer inconsistencies by NLI score distributions',
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
        default='data/statistics/inconsistency_by_nli_scores',
        help='Output directory for analysis results'
    )

    args = parser.parse_args()
    analyze_by_nli_scores(args.input_dir, args.output_dir)
