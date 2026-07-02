#!/usr/bin/env python3
"""
Master script to run all inconsistency analyses.
Master script that runs all inconsistency analyses.

This script orchestrates all individual analysis scripts and generates
a comprehensive summary report of LLM answer inconsistencies.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
import time
from datetime import datetime


def run_analysis(script_name, description, input_dir, output_subdir):
    """Run a single analysis script."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*70}\n")

    script_path = Path('scripts/statistics') / script_name
    output_dir = Path('data/statistics') / output_subdir
    start_time = time.time()

    try:
        cmd = [
            'conda', 'run', '-n', 'karma', 'python', str(script_path),
            '--input-dir', str(input_dir),
            '--output-dir', str(output_dir)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        elapsed_time = time.time() - start_time

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        print(f"\n✓ Completed in {elapsed_time:.2f} seconds")
        return True, elapsed_time

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Failed after {elapsed_time:.2f} seconds")
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False, elapsed_time


def generate_master_summary(output_base_dir):
    """Generate a master summary combining all analysis results."""
    print(f"\n{'='*70}")
    print("Generating Master Summary Report")
    print(f"{'='*70}\n")

    output_dir = Path(output_base_dir)
    master_summary = {
        'generated_at': datetime.now().isoformat(),
        'analyses': {}
    }

    # Collect summaries from each analysis
    analysis_dirs = [
        ('inconsistency_by_rule', 'Rule-Based Analysis'),
        ('inconsistency_by_mutation', 'Mutation-Based Analysis'),
        ('inconsistency_by_entity', 'Entity-Based Analysis'),
        ('inconsistency_by_relation', 'Relation-Based Analysis'),
        ('inconsistency_by_nli_scores', 'NLI Score-Based Analysis'),
        ('inconsistency_by_complexity', 'Complexity-Based Analysis')
    ]

    for dir_name, analysis_name in analysis_dirs:
        summary_file = output_dir / dir_name / f"{dir_name.replace('inconsistency_by_', '')}_summary.json"

        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    master_summary['analyses'][analysis_name] = json.load(f)
                print(f"✓ Loaded: {analysis_name}")
            except Exception as e:
                print(f"✗ Failed to load {analysis_name}: {e}")
        else:
            print(f"⚠ Summary not found: {summary_file}")

    # Save master summary
    master_summary_file = output_dir / 'master_summary.json'
    with open(master_summary_file, 'w') as f:
        json.dump(master_summary, f, indent=2)

    print(f"\n✓ Master summary saved to: {master_summary_file}")

    # Generate text report
    generate_text_report(master_summary, output_dir / 'ANALYSIS_REPORT.txt')


def generate_text_report(master_summary, output_file):
    """Generate a human-readable text report."""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LLM ANSWER INCONSISTENCY ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {master_summary['generated_at']}\n")
        f.write(f"Model: Qwen2.5-7B-Instruct\n")
        f.write("="*80 + "\n\n")

        # Summary of each analysis
        for analysis_name, data in master_summary['analyses'].items():
            f.write(f"\n{'-'*80}\n")
            f.write(f"{analysis_name}\n")
            f.write(f"{'-'*80}\n\n")

            # Extract key metrics based on analysis type
            if 'Rule-Based' in analysis_name:
                f.write(f"Total Rules Analyzed: {data.get('total_rules', 'N/A')}\n")
                f.write(f"Average Inconsistency Rate: {data.get('avg_inconsistency_rate', 0):.2%}\n")
                f.write(f"Median Inconsistency Rate: {data.get('median_inconsistency_rate', 0):.2%}\n\n")

                if 'top_10_problematic_rules' in data:
                    f.write("Top 5 Most Problematic Rules:\n")
                    for i, rule in enumerate(data['top_10_problematic_rules'][:5], 1):
                        f.write(f"  {i}. {rule['rule_name']}: {rule['inconsistency_rate']:.2%} "
                               f"({rule['total_samples']} samples)\n")

            elif 'Mutation-Based' in analysis_name:
                if 'key_findings' in data:
                    findings = data['key_findings']
                    f.write(f"Most Problematic Mutation: {findings.get('most_problematic_mutation', 'N/A')}\n")
                    f.write(f"  Inconsistency Rate: {findings.get('most_problematic_rate', 0):.2%}\n")
                    f.write(f"Least Problematic Mutation: {findings.get('least_problematic_mutation', 'N/A')}\n")
                    f.write(f"  Inconsistency Rate: {findings.get('least_problematic_rate', 0):.2%}\n")

            elif 'Entity-Based' in analysis_name:
                f.write(f"Total Unique Entity Types: {data.get('total_unique_entity_types', 'N/A')}\n")
                f.write(f"Total Unique Entities: {data.get('total_unique_entities', 'N/A')}\n\n")

                if 'top_10_problematic_entity_types' in data and data['top_10_problematic_entity_types']:
                    f.write("Top 5 Most Problematic Entity Types:\n")
                    for i, entity_type in enumerate(data['top_10_problematic_entity_types'][:5], 1):
                        f.write(f"  {i}. {entity_type['entity_type']}: {entity_type['inconsistency_rate']:.2%} "
                               f"({entity_type['total_occurrences']} occurrences)\n")

            elif 'Relation-Based' in analysis_name:
                f.write(f"Total Unique Relations: {data.get('total_unique_relations', 'N/A')}\n")
                f.write(f"Total Unique Patterns: {data.get('total_unique_patterns', 'N/A')}\n\n")

                if 'top_10_problematic_relations' in data and data['top_10_problematic_relations']:
                    f.write("Top 5 Most Problematic Relations:\n")
                    for i, relation in enumerate(data['top_10_problematic_relations'][:5], 1):
                        f.write(f"  {i}. {relation['relation']}: {relation['inconsistency_rate']:.2%} "
                               f"({relation['total_occurrences']} occurrences)\n")

            elif 'NLI' in analysis_name:
                f.write(f"Total Samples: {data.get('total_samples', 'N/A')}\n")
                f.write(f"Inconsistent Samples: {data.get('inconsistent_samples', 'N/A')}\n")
                f.write(f"Overall Inconsistency Rate: {data.get('overall_inconsistency_rate', 0):.2%}\n\n")

                if 'average_scores' in data:
                    avg_scores = data['average_scores']
                    f.write("Average NLI Scores:\n")
                    f.write("  Consistent Samples:\n")
                    if 'consistent' in avg_scores:
                        for score_type, value in avg_scores['consistent'].items():
                            f.write(f"    {score_type}: {value:.3f}\n")
                    f.write("  Inconsistent Samples:\n")
                    if 'inconsistent' in avg_scores:
                        for score_type, value in avg_scores['inconsistent'].items():
                            f.write(f"    {score_type}: {value:.3f}\n")

            elif 'Complexity-Based' in analysis_name:
                f.write(f"Total Samples: {data.get('total_samples', 'N/A')}\n")
                f.write(f"Inconsistent Samples: {data.get('inconsistent_samples', 'N/A')}\n")
                f.write(f"Overall Inconsistency Rate: {data.get('overall_inconsistency_rate', 0):.2%}\n\n")

                if 'key_findings' in data:
                    findings = data['key_findings']
                    f.write("Question Length (words):\n")
                    f.write(f"  Consistent: {findings.get('avg_question_length_consistent', 0):.1f}\n")
                    f.write(f"  Inconsistent: {findings.get('avg_question_length_inconsistent', 0):.1f}\n")
                    f.write("Answer Length (words):\n")
                    f.write(f"  Consistent: {findings.get('avg_answer_length_consistent', 0):.1f}\n")
                    f.write(f"  Inconsistent: {findings.get('avg_answer_length_inconsistent', 0):.1f}\n")

            f.write("\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"✓ Text report saved to: {output_file}")


def main():
    """Main function to run all analyses."""
    parser = argparse.ArgumentParser(
        description='Run all LLM answer inconsistency analyses',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/examples/consistency_results_nli_Qwen2.5-7B-Instruct',
        help='Input directory containing consistency results'
    )
    parser.add_argument(
        '--output-base-dir',
        type=str,
        default='data/statistics',
        help='Base output directory for all analysis results'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("LLM ANSWER INCONSISTENCY ANALYSIS SUITE")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Base Directory: {args.output_base_dir}")
    print("="*70)

    analyses = [
        ('analyze_inconsistency_by_rule.py', 'Rule-Based Inconsistency Analysis', 'inconsistency_by_rule'),
        ('analyze_inconsistency_by_mutation.py', 'Mutation-Based Inconsistency Analysis', 'inconsistency_by_mutation'),
        ('analyze_inconsistency_by_entity.py', 'Entity-Based Inconsistency Analysis', 'inconsistency_by_entity'),
        ('analyze_inconsistency_by_relation.py', 'Relation-Based Inconsistency Analysis', 'inconsistency_by_relation'),
        ('analyze_inconsistency_by_nli_scores.py', 'NLI Score-Based Inconsistency Analysis', 'inconsistency_by_nli_scores'),
        ('analyze_inconsistency_by_complexity.py', 'Complexity-Based Inconsistency Analysis', 'inconsistency_by_complexity'),
    ]

    results = {}
    total_start_time = time.time()

    for script_name, description, output_subdir in analyses:
        success, elapsed_time = run_analysis(script_name, description, args.input_dir, output_subdir)
        results[description] = {
            'success': success,
            'elapsed_time': elapsed_time
        }

    # Generate master summary
    generate_master_summary(args.output_base_dir)

    # Print final summary
    total_elapsed_time = time.time() - total_start_time

    print(f"\n{'='*70}")
    print("ANALYSIS SUITE COMPLETE")
    print(f"{'='*70}")
    print(f"Total Time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)\n")

    print("Results Summary:")
    for description, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"  {status:12} {description:45} ({result['elapsed_time']:.2f}s)")

    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    print(f"\nSuccess Rate: {successful}/{total} ({successful/total:.0%})")

    print(f"\nAll results saved to: {args.output_base_dir}/")
    print(f"Master summary: {args.output_base_dir}/master_summary.json")
    print(f"Text report: {args.output_base_dir}/ANALYSIS_REPORT.txt")

    return 0 if all(r['success'] for r in results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
