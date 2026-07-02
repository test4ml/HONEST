#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply mutation operators to all positive_examples files to generate mutation_examples
Each original file corresponds to a folder, each mutation operator generates a separate CSV file
"""

import os
import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import json
import argparse
from typing import List
from honest.mutation import (
    MutationEngine, FactInstance,
    BodyPermutation, BodyAugmentation, BodyReduction,
    EntityRename, RuleMerging
)
from honest.kg import MemgraphKnowledgeGraph


def load_rule_instances_from_csv(csv_file: str, sep=',') -> List[FactInstance]:
    """Load rule instances from CSV file"""
    df = pd.read_csv(csv_file, sep=sep)
    rule_instances = []

    for _, row in df.iterrows():
        try:
            # Parse instantiation mapping
            instantiation_mapping = json.loads(row['InstantiationMapping'])

            # Parse entity label descriptions
            entity_labels = json.loads(row['EntityLabelsDescriptions'])

            rule_instance = FactInstance(
                original_rule_str=row['OriginalRule'],
                matched_instance_str=row['RuleMatchedInstance'],
                instantiation_mapping=instantiation_mapping,
                entity_labels=entity_labels,
                natural_language=row['NaturalLanguage']
            )
            rule_instances.append(rule_instance)
        except Exception as e:
            print(f"Error parsing row in {csv_file}: {e}")
            continue

    return rule_instances


def apply_mutations_to_file(input_file: str, output_base_dir: str, kg: MemgraphKnowledgeGraph, sep=','):
    """Apply all mutation operators to a single file"""
    print(f"Processing file: {input_file}")

    # Load rule instances
    rule_instances = load_rule_instances_from_csv(input_file, sep=sep)
    if not rule_instances:
        print(f"No valid rule instances found in {input_file}")
        return

    # Create mutation engine
    operators = [
        BodyPermutation(),
        BodyAugmentation(),
        # BodyReduction(),  # Temporarily disabled due to issues
        EntityRename(),
        # RuleMerging()
    ]
    mutation_engine = MutationEngine(operators)

    # Create separate folder for each original file
    base_name = os.path.basename(input_file).replace('_examples.csv', '')
    file_output_dir = os.path.join(output_base_dir, base_name)
    os.makedirs(file_output_dir, exist_ok=True)

    # Create separate data collectors for each mutation operator
    operator_mutations = {operator.name: [] for operator in operators}

    # Process rule instances
    for i, rule_instance in enumerate(rule_instances[:10]):  # Limit to first 10 instances per file
        print(f"  Mutating rule instance {i+1}/{min(10, len(rule_instances))}")

        try:
            mutations = mutation_engine.mutate_rule(rule_instance, kg=kg)

            for operator_name, operator_mutations_list in mutations.items():
                for mutated_body, expected_head in operator_mutations_list:
                    # Build complete mutated instance format: mutated_body => expected_head
                    mutated_instance = f"{mutated_body} => {expected_head}"

                    mutation_record = {
                        'original_rule': rule_instance.original_rule_str,
                        'original_instance': rule_instance.matched_instance_str,
                        'mutated_instance': mutated_instance,
                        'instantiation_mapping': json.dumps(rule_instance.instantiation_mapping),
                        'entity_labels': json.dumps(rule_instance.entity_labels),
                        'natural_language': rule_instance.natural_language
                    }
                    operator_mutations[operator_name].append(mutation_record)
        except Exception as e:
            print(f"Error mutating rule instance {i+1}: {e}")
            continue

    # Save separate CSV file for each mutation operator
    total_mutations = 0
    for operator_name, mutations in operator_mutations.items():
        if mutations:
            # Clean operator name for filename
            safe_operator_name = operator_name.replace(' ', '_').lower()
            output_file = os.path.join(file_output_dir, f"{safe_operator_name}.csv")

            df_mutations = pd.DataFrame(mutations)
            df_mutations.to_csv(output_file, index=False, sep=sep)

            print(f"    {operator_name}: {len(mutations)} mutations saved to {safe_operator_name}.csv")
            total_mutations += len(mutations)

    print(f"  Total mutations for {base_name}: {total_mutations}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Apply mutation operators to positive examples files')
    parser.add_argument('--input_dir', default='data/examples/positive',
                       help='Input directory containing positive examples (default: data/examples/positive)')
    parser.add_argument('--output_dir', default='data/examples/mutated',
                       help='Output directory for mutated examples (default: data/examples/mutated)')
    parser.add_argument('--sep', default=',',
                       help='CSV file separator (default: comma)')
    parser.add_argument('--memgraph_uri', default='bolt://localhost:7687',
                       help='Memgraph connection URI (default: bolt://localhost:7687)')
    parser.add_argument('--memgraph_user', default='',
                       help='Memgraph username (default: empty)')
    parser.add_argument('--memgraph_password', default='',
                       help='Memgraph password (default: empty)')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_base_dir = args.output_dir

    # Initialize knowledge graph connection
    print("Initializing knowledge graph connection...")
    try:
        kg = MemgraphKnowledgeGraph(
            uri=args.memgraph_uri,
            user=args.memgraph_user,
            password=args.memgraph_password,
            enable_metadata=True
        )
        print("✅ Knowledge graph connection established")
    except Exception as e:
        print(f"❌ Failed to connect to knowledge graph: {e}")
        print("Please ensure Memgraph is running and accessible.")
        return

    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Get all CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('_examples.csv')]
    print(f"Found {len(csv_files)} CSV files to process")

    # Process each file
    for i, csv_file in enumerate(csv_files):
        input_file = os.path.join(input_dir, csv_file)
        print(f"\n[{i+1}/{len(csv_files)}] ")
        apply_mutations_to_file(input_file, output_base_dir, kg, sep=args.sep)

    print(f"\nMutation completed! Results saved to {output_base_dir}/")

    # Overall statistics
    print("\nOverall statistics:")
    total_folders = len([d for d in os.listdir(output_base_dir)
                        if os.path.isdir(os.path.join(output_base_dir, d))])
    print(f"Total folders created: {total_folders}")

    # Close knowledge graph connection
    kg.close()
    print("Knowledge graph connection closed.")


if __name__ == "__main__":
    main()