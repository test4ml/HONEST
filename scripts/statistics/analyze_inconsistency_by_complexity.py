#!/usr/bin/env python3
"""
Analyze LLM answer inconsistencies by question complexity features.
Analyzes how question complexity features affect LLM answer inconsistency.
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


def analyze_question_complexity(question_text):
    """Extract complexity features from question text."""
    if not isinstance(question_text, str):
        return {}

    features = {
        'length_chars': len(question_text),
        'length_words': len(question_text.split()),
        'num_commas': question_text.count(','),
        'num_parentheses': question_text.count('(') + question_text.count(')'),
        'num_question_marks': question_text.count('?'),
        'has_and': ' and ' in question_text.lower(),
        'has_or': ' or ' in question_text.lower(),
        'has_given_that': 'given that' in question_text.lower(),
        'num_sentences': question_text.count('.') + question_text.count('?'),
    }

    return features


def analyze_answer_complexity(answer_text):
    """Extract complexity features from answer text."""
    if not isinstance(answer_text, str):
        return {}

    features = {
        'answer_length_chars': len(answer_text),
        'answer_length_words': len(answer_text.split()),
        'answer_num_sentences': max(1, answer_text.count('.') + answer_text.count('?')),
        'answer_num_paragraphs': max(1, answer_text.count('\n\n') + 1),
    }

    # Calculate average sentence length
    features['answer_avg_sentence_length'] = features['answer_length_words'] / features['answer_num_sentences']

    return features


def analyze_by_complexity(data_dir, output_dir):
    """Analyze inconsistencies based on question and answer complexity."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Collect all data with complexity features
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

                    # Analyze original question
                    original_q = row.get('original_question', '')
                    original_features = analyze_question_complexity(original_q)

                    # Analyze mutated question
                    mutated_q = row.get('mutated_question', '')
                    mutated_features = analyze_question_complexity(mutated_q)

                    # Analyze original answer
                    original_answer = row.get('original_llm_answer', '')
                    original_answer_features = analyze_answer_complexity(original_answer)

                    # Analyze mutated answer
                    mutated_answer = row.get('mutated_llm_answer', '')
                    mutated_answer_features = analyze_answer_complexity(mutated_answer)

                    # Compute complexity changes
                    length_change = mutated_features['length_chars'] - original_features['length_chars']
                    word_change = mutated_features['length_words'] - original_features['length_words']

                    sample = {
                        'rule': rule_dir.name,
                        'mutation_type': csv_file.stem.replace('_llm_answers', ''),
                        'inconsistent': is_inconsistent,
                        'question_type': row.get('original_question_type', ''),

                        # Original question features
                        'orig_q_length_chars': original_features['length_chars'],
                        'orig_q_length_words': original_features['length_words'],
                        'orig_q_num_commas': original_features['num_commas'],
                        'orig_q_num_parentheses': original_features['num_parentheses'],
                        'orig_q_has_and': original_features['has_and'],
                        'orig_q_has_given_that': original_features['has_given_that'],

                        # Mutated question features
                        'mut_q_length_chars': mutated_features['length_chars'],
                        'mut_q_length_words': mutated_features['length_words'],
                        'mut_q_num_commas': mutated_features['num_commas'],

                        # Change features
                        'q_length_change_chars': length_change,
                        'q_length_change_words': word_change,
                        'q_length_change_ratio': length_change / max(1, original_features['length_chars']),

                        # Answer features
                        'orig_answer_length': original_answer_features['answer_length_words'],
                        'orig_answer_sentences': original_answer_features['answer_num_sentences'],
                        'orig_answer_paragraphs': original_answer_features['answer_num_paragraphs'],
                        'orig_answer_avg_sent_len': original_answer_features['answer_avg_sentence_length'],

                        'mut_answer_length': mutated_answer_features['answer_length_words'],
                        'mut_answer_sentences': mutated_answer_features['answer_num_sentences'],
                        'mut_answer_paragraphs': mutated_answer_features['answer_num_paragraphs'],
                        'mut_answer_avg_sent_len': mutated_answer_features['answer_avg_sentence_length'],

                        # NLI scores
                        'contradiction_score': row.get('contradiction_score', 0),
                        'entailment_score': row.get('entailment_score', 0),
                    }
                    all_samples.append(sample)

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

    df_samples = pd.DataFrame(all_samples)

    # Analysis 1: Complexity by question type
    question_type_analysis = df_samples.groupby('question_type').agg({
        'inconsistent': ['count', 'sum', 'mean'],
        'orig_q_length_words': 'mean',
        'orig_answer_length': 'mean'
    }).reset_index()
    question_type_analysis.columns = [
        'question_type', 'total_samples', 'inconsistent_samples',
        'inconsistency_rate', 'avg_question_length', 'avg_answer_length'
    ]
    question_type_analysis.to_csv(output_dir / 'question_type_analysis.csv', index=False)

    # Analysis 2: Compare complexity features
    consistent_samples = df_samples[~df_samples['inconsistent']]
    inconsistent_samples = df_samples[df_samples['inconsistent']]

    complexity_comparison = pd.DataFrame({
        'feature': [
            'Original Question Length (words)',
            'Original Question Commas',
            'Original Question Parentheses',
            'Original Answer Length (words)',
            'Original Answer Sentences',
            'Question Length Change (words)',
            'Question Length Change Ratio'
        ],
        'consistent_mean': [
            consistent_samples['orig_q_length_words'].mean(),
            consistent_samples['orig_q_num_commas'].mean(),
            consistent_samples['orig_q_num_parentheses'].mean(),
            consistent_samples['orig_answer_length'].mean(),
            consistent_samples['orig_answer_sentences'].mean(),
            consistent_samples['q_length_change_words'].mean(),
            consistent_samples['q_length_change_ratio'].mean(),
        ],
        'inconsistent_mean': [
            inconsistent_samples['orig_q_length_words'].mean(),
            inconsistent_samples['orig_q_num_commas'].mean(),
            inconsistent_samples['orig_q_num_parentheses'].mean(),
            inconsistent_samples['orig_answer_length'].mean(),
            inconsistent_samples['orig_answer_sentences'].mean(),
            inconsistent_samples['q_length_change_words'].mean(),
            inconsistent_samples['q_length_change_ratio'].mean(),
        ]
    })
    complexity_comparison.to_csv(output_dir / 'complexity_comparison.csv', index=False)

    # Analysis 3: Question length bins
    length_bins = [0, 50, 100, 150, 200, 300, 1000]
    df_samples['orig_q_length_bin'] = pd.cut(df_samples['orig_q_length_words'], bins=length_bins)

    length_analysis = df_samples.groupby('orig_q_length_bin', observed=True).agg({
        'inconsistent': ['count', 'sum', 'mean']
    }).reset_index()
    length_analysis.columns = ['question_length_bin', 'total_samples', 'inconsistent_samples', 'inconsistency_rate']
    length_analysis.to_csv(output_dir / 'question_length_analysis.csv', index=False)

    # Analysis 4: Answer length correlation
    answer_length_bins = [0, 50, 100, 200, 300, 500, 10000]
    df_samples['orig_answer_length_bin'] = pd.cut(df_samples['orig_answer_length'], bins=answer_length_bins)

    answer_length_analysis = df_samples.groupby('orig_answer_length_bin', observed=True).agg({
        'inconsistent': ['count', 'sum', 'mean']
    }).reset_index()
    answer_length_analysis.columns = ['answer_length_bin', 'total_samples', 'inconsistent_samples', 'inconsistency_rate']
    answer_length_analysis.to_csv(output_dir / 'answer_length_analysis.csv', index=False)

    # Analysis 5: Features with "given that" or "and"
    given_that_analysis = df_samples.groupby('orig_q_has_given_that').agg({
        'inconsistent': ['count', 'sum', 'mean']
    }).reset_index()
    given_that_analysis.columns = ['has_given_that', 'total_samples', 'inconsistent_samples', 'inconsistency_rate']
    given_that_analysis.to_csv(output_dir / 'given_that_analysis.csv', index=False)

    # Generate summary
    summary = {
        'total_samples': len(df_samples),
        'inconsistent_samples': len(inconsistent_samples),
        'overall_inconsistency_rate': len(inconsistent_samples) / len(df_samples),
        'question_type_stats': question_type_analysis.to_dict('records'),
        'complexity_comparison': complexity_comparison.to_dict('records'),
        'key_findings': {
            'avg_question_length_consistent': consistent_samples['orig_q_length_words'].mean(),
            'avg_question_length_inconsistent': inconsistent_samples['orig_q_length_words'].mean(),
            'avg_answer_length_consistent': consistent_samples['orig_answer_length'].mean(),
            'avg_answer_length_inconsistent': inconsistent_samples['orig_answer_length'].mean(),
        }
    }

    with open(output_dir / 'complexity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Visualization 1: Complexity comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(complexity_comparison))
    width = 0.35

    bars1 = ax.bar(x - width/2, complexity_comparison['consistent_mean'], width,
                   label='Consistent', color='#2ecc71')
    bars2 = ax.bar(x + width/2, complexity_comparison['inconsistent_mean'], width,
                   label='Inconsistent', color='#e74c3c')

    ax.set_ylabel('Average Value')
    ax.set_title('Complexity Features: Consistent vs Inconsistent Answers')
    ax.set_xticks(x)
    ax.set_xticklabels(complexity_comparison['feature'], rotation=45, ha='right', fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'complexity_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 2: Question length vs inconsistency rate
    if not length_analysis.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        x_labels = [str(interval) for interval in length_analysis['question_length_bin']]
        x_pos = np.arange(len(x_labels))

        bars = ax.bar(x_pos, length_analysis['inconsistency_rate'], color='coral')
        ax.set_xlabel('Question Length (words)')
        ax.set_ylabel('Inconsistency Rate')
        ax.set_title('Question Length vs Inconsistency Rate')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        # Add sample counts
        for i, row in length_analysis.iterrows():
            ax.text(i, row['inconsistency_rate'], f"n={row['total_samples']}",
                   ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'question_length_vs_inconsistency.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Visualization 3: Answer length vs inconsistency rate
    if not answer_length_analysis.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        x_labels = [str(interval) for interval in answer_length_analysis['answer_length_bin']]
        x_pos = np.arange(len(x_labels))

        bars = ax.bar(x_pos, answer_length_analysis['inconsistency_rate'], color='skyblue')
        ax.set_xlabel('Answer Length (words)')
        ax.set_ylabel('Inconsistency Rate')
        ax.set_title('Answer Length vs Inconsistency Rate')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        # Add sample counts
        for i, row in answer_length_analysis.iterrows():
            ax.text(i, row['inconsistency_rate'], f"n={row['total_samples']}",
                   ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'answer_length_vs_inconsistency.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Visualization 4: Question type analysis
    if not question_type_analysis.empty and len(question_type_analysis) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(question_type_analysis['question_type'],
                     question_type_analysis['inconsistency_rate'],
                     color='#9b59b6')
        ax.set_xlabel('Question Type')
        ax.set_ylabel('Inconsistency Rate')
        ax.set_title('Inconsistency Rate by Question Type')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}',
                   ha='center', va='bottom')

        # Add sample counts
        for i, row in question_type_analysis.iterrows():
            ax.text(i, 0.01, f"n={row['total_samples']}",
                   ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'question_type_inconsistency.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Visualization 5: Scatter plots - complexity vs inconsistency
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Question length scatter
    axes[0, 0].scatter(consistent_samples['orig_q_length_words'],
                      consistent_samples['orig_answer_length'],
                      alpha=0.3, s=10, color='green', label='Consistent')
    axes[0, 0].scatter(inconsistent_samples['orig_q_length_words'],
                      inconsistent_samples['orig_answer_length'],
                      alpha=0.3, s=10, color='red', label='Inconsistent')
    axes[0, 0].set_xlabel('Question Length (words)')
    axes[0, 0].set_ylabel('Answer Length (words)')
    axes[0, 0].set_title('Question vs Answer Length')
    axes[0, 0].legend()

    # Question change scatter
    axes[0, 1].scatter(consistent_samples['q_length_change_words'],
                      consistent_samples['contradiction_score'],
                      alpha=0.3, s=10, color='green', label='Consistent')
    axes[0, 1].scatter(inconsistent_samples['q_length_change_words'],
                      inconsistent_samples['contradiction_score'],
                      alpha=0.3, s=10, color='red', label='Inconsistent')
    axes[0, 1].set_xlabel('Question Length Change (words)')
    axes[0, 1].set_ylabel('Contradiction Score')
    axes[0, 1].set_title('Length Change vs Contradiction Score')
    axes[0, 1].legend()

    # Commas vs inconsistency
    axes[1, 0].scatter(consistent_samples['orig_q_num_commas'],
                      consistent_samples['orig_answer_sentences'],
                      alpha=0.3, s=10, color='green', label='Consistent')
    axes[1, 0].scatter(inconsistent_samples['orig_q_num_commas'],
                      inconsistent_samples['orig_answer_sentences'],
                      alpha=0.3, s=10, color='red', label='Inconsistent')
    axes[1, 0].set_xlabel('Number of Commas in Question')
    axes[1, 0].set_ylabel('Number of Sentences in Answer')
    axes[1, 0].set_title('Question Commas vs Answer Sentences')
    axes[1, 0].legend()

    # Answer sentences distribution
    axes[1, 1].hist([consistent_samples['orig_answer_sentences'],
                     inconsistent_samples['orig_answer_sentences']],
                    bins=20, label=['Consistent', 'Inconsistent'],
                    color=['green', 'red'], alpha=0.6)
    axes[1, 1].set_xlabel('Number of Sentences in Answer')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of Answer Sentence Counts')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'complexity_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}")
    print("COMPLEXITY-BASED INCONSISTENCY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Inconsistent samples: {summary['inconsistent_samples']}")
    print(f"Overall inconsistency rate: {summary['overall_inconsistency_rate']:.2%}")
    print(f"\nKey findings:")
    print(f"  Avg question length (consistent): {summary['key_findings']['avg_question_length_consistent']:.1f} words")
    print(f"  Avg question length (inconsistent): {summary['key_findings']['avg_question_length_inconsistent']:.1f} words")
    print(f"  Avg answer length (consistent): {summary['key_findings']['avg_answer_length_consistent']:.1f} words")
    print(f"  Avg answer length (inconsistent): {summary['key_findings']['avg_answer_length_inconsistent']:.1f} words")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - question_type_analysis.csv")
    print(f"  - complexity_comparison.csv")
    print(f"  - question_length_analysis.csv")
    print(f"  - answer_length_analysis.csv")
    print(f"  - complexity_summary.json")
    print(f"  - *.png (visualizations)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze LLM answer inconsistencies by question complexity features',
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
        default='data/statistics/inconsistency_by_complexity',
        help='Output directory for analysis results'
    )

    args = parser.parse_args()
    analyze_by_complexity(args.input_dir, args.output_dir)
