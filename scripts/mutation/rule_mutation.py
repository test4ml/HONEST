#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule Mutation Script

Reads inference rule instances from the positive_examples directory, connects to the memgraph knowledge graph,
applies metamorphic rules to mutate each instance, and outputs CSV files to the mutation_examples directory.
"""

import os
import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import pandas as pd
from typing import List, Dict, Any
import logging
from tqdm import tqdm

# Import HONEST-related modules
from honest.kg.memgraph_knowledge_graph import MemgraphKnowledgeGraph
from honest.mutation.base import FactInstance, MutationEngine
from honest.mutation.body_permutation import BodyPermutation
from honest.mutation.body_augmentation import BodyAugmentation
from honest.mutation.body_reduction import BodyReduction
from honest.mutation.entity_rename import EntityRename
from honest.mutation.rule_merging import RuleMerging

# Import configuration management
from configs import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RuleMutationProcessor:
    """Rule mutation processor."""

    def __init__(self, kg_uri: str = None, kg_user: str = None, kg_password: str = None,
                 mutation_operator: str = None):
        """
        Initialize the processor.

        Args:
            kg_uri: Memgraph connection URI (uses config when None)
            kg_user: Username (uses config when None)
            kg_password: Password (uses config when None)
            mutation_operator: Name of the mutation operator to use; None means use all operators
        """
        # Use config or parameter values
        memgraph_config = get_config('knowledge_graph.memgraph', {})
        kg_uri = kg_uri or memgraph_config.get('uri', 'bolt://localhost:7687')
        kg_user = kg_user or memgraph_config.get('user', '')
        kg_password = kg_password or memgraph_config.get('password', '')

        self.kg = MemgraphKnowledgeGraph(uri=kg_uri, user=kg_user, password=kg_password)

        # All available mutation operators
        all_operators = {
            'BodyPermutation': BodyPermutation(),
            'BodyAugmentation': BodyAugmentation(),
            'BodyReduction': BodyReduction(),
            'EntityRename': EntityRename(),
            'RuleMerging': RuleMerging()
        }

        # Select operators based on the parameter
        self.selected_operator = mutation_operator
        if mutation_operator:
            if mutation_operator not in all_operators:
                raise ValueError(f"Unknown mutation operator: {mutation_operator}. "
                               f"Available operators: {list(all_operators.keys())}")
            self.mutation_operators = [all_operators[mutation_operator]]
            self.operator_names = [mutation_operator]
        else:
            self.mutation_operators = list(all_operators.values())
            self.operator_names = list(all_operators.keys())

        # Create the mutation engine
        self.mutation_engine = MutationEngine(self.mutation_operators)

        # Ensure the output directory exists
        self.output_dir = get_config('paths.examples.mutated', './data/examples/mutated')
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a subdirectory for each operator
        if mutation_operator:
            # Single-operator mode: create the corresponding subdirectory
            operator_dir = os.path.join(self.output_dir, mutation_operator)
            os.makedirs(operator_dir, exist_ok=True)
        else:
            # Multi-operator mode: create subdirectories for all operators
            for op_name in all_operators.keys():
                operator_dir = os.path.join(self.output_dir, op_name)
                os.makedirs(operator_dir, exist_ok=True)

    def load_rule_instances_from_file(self, filepath: str) -> List[FactInstance]:
        """
        Load rule instances from a single CSV file.

        Args:
            filepath: CSV file path

        Returns:
            List[FactInstance]: list of rule instances
        """
        rule_instances = []
        filename = os.path.basename(filepath)

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loading {len(df)} rules from {filename}")

            for idx, row in df.iterrows():
                try:
                    # Parse the instantiation mapping
                    instantiation_mapping = {}
                    if 'InstantiationMapping' in row and pd.notna(row['InstantiationMapping']):
                        try:
                            instantiation_mapping = json.loads(row['InstantiationMapping'])
                        except json.JSONDecodeError:
                            instantiation_mapping = {}

                    # Parse entity labels
                    entity_labels = []
                    if 'EntityLabelsDescriptions' in row and pd.notna(row['EntityLabelsDescriptions']):
                        try:
                            entity_labels = json.loads(row['EntityLabelsDescriptions'])
                        except json.JSONDecodeError:
                            entity_labels = []

                    # Create the FactInstance object
                    rule_instance = FactInstance(
                        original_rule_str=row.get('OriginalRule', ''),
                        matched_instance_str=row.get('RuleMatchedInstance', ''),
                        instantiation_mapping=instantiation_mapping,
                        entity_labels=entity_labels,
                        natural_language=row.get('NaturalLanguage', '')
                    )

                    rule_instances.append(rule_instance)

                except Exception as e:
                    logger.warning(f"Error processing row {idx} in {filename}: {e}")
                    continue

            logger.info(f"Successfully loaded {len(rule_instances)} rule instances from {filename}")

        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")

        return rule_instances

    def mutate_rule_instance(self, rule_instance: FactInstance,
                           rule_id: int, source_file: str) -> List[Dict[str, Any]]:
        """
        Mutate a single rule instance.

        Args:
            rule_instance: rule instance
            rule_id: rule ID
            source_file: source filename

        Returns:
            List[Dict]: list of mutation results
        """
        mutations = []

        try:
            # Apply all mutation operators
            mutation_results = self.mutation_engine.mutate_rule(rule_instance, self.kg)

            # Process the mutation results
            for operator_name, operator_mutations in mutation_results.items():
                for mutation_idx, (mutated_body, expected_head) in enumerate(operator_mutations):
                    # Create a mutation record
                    mutation_record = {
                        'rule_id': rule_id,
                        'source_file': source_file,
                        'original_rule': rule_instance.original_rule,
                        'original_matched_instance': rule_instance.matched_instance,
                        'original_natural_language': rule_instance.natural_language,
                        'mutation_operator': operator_name,
                        'mutation_index': mutation_idx,
                        'mutated_body': mutated_body,
                        'expected_head': expected_head,
                        'mutated_rule': f"{mutated_body} => {expected_head}",
                        'instantiation_mapping': json.dumps(rule_instance.instantiation_mapping),
                        'entity_labels': json.dumps(rule_instance.entity_labels),
                        'mutation_type': self._classify_mutation_type(operator_name,
                                                                    rule_instance.original_rule,
                                                                    f"{mutated_body} => {expected_head}")
                    }
                    mutations.append(mutation_record)

        except Exception as e:
            logger.warning(f"Error mutating rule {rule_id}: {e}")

        return mutations

    def _classify_mutation_type(self, operator_name: str, original_rule: str, mutated_rule: str) -> str:
        """
        Classify the mutation type.

        Args:
            operator_name: operator name
            original_rule: original rule
            mutated_rule: mutated rule

        Returns:
            str: mutation type
        """
        if "P31" in mutated_rule and "P31" not in original_rule:
            return "type_addition"
        elif len(mutated_rule.split()) > len(original_rule.split()) * 1.3:
            return "complexity_increase"
        elif len(mutated_rule.split()) < len(original_rule.split()) * 0.8:
            return "complexity_decrease"
        elif any(pred in mutated_rule for pred in ["P580", "P582", "P571"]):
            return "temporal_constraint"
        elif "P279" in mutated_rule:
            return "hierarchy_relation"
        else:
            return "semantic_variation"

    def process_file(self, input_filepath: str, max_rules: int = None) -> bool:
        """
        Process a single file.

        Args:
            input_filepath: input file path
            max_rules: maximum number of rules to process (None means process all)

        Returns:
            bool: whether processing succeeded
        """
        filename = os.path.basename(input_filepath)
        base_name = os.path.splitext(filename)[0]

        logger.info(f"Processing file: {filename}")

        # Load rule instances
        rule_instances = self.load_rule_instances_from_file(input_filepath)
        if not rule_instances:
            logger.warning(f"No rule instances loaded from {filename}")
            return False

        # Limit the number processed
        if max_rules:
            rule_instances = rule_instances[:max_rules]
            logger.info(f"Limited to processing {len(rule_instances)} rules")

        # Test the knowledge graph connection
        try:
            if not self.kg.test_connection():
                logger.error("Failed to connect to knowledge graph")
                return False
            logger.info("Connected to Memgraph knowledge graph")
        except Exception as e:
            logger.error(f"Failed to connect to knowledge graph: {e}")
            return False

        # Process all rule instances
        all_mutations = []

        for rule_id, rule_instance in enumerate(tqdm(rule_instances, desc=f"Mutating {filename}")):
            mutations = self.mutate_rule_instance(rule_instance, rule_id, filename)
            all_mutations.extend(mutations)

        # Close the connection after processing
        try:
            self.kg.close()
            logger.info("Disconnected from knowledge graph")
        except:
            pass

        # Save results - grouped by operator
        if all_mutations:
            self._save_mutations_by_operator(all_mutations, base_name)
            logger.info(f"Generated {len(all_mutations)} mutations for {filename}")
            return True
        else:
            logger.warning(f"No mutations generated for {filename}")
            return False

    def _save_mutations_by_operator(self, mutations: List[Dict[str, Any]], base_name: str):
        """
        Save mutation results grouped by mutation operator.

        Args:
            mutations: list of mutation results
            base_name: base filename (without extension)
        """
        # Group by operator
        operator_mutations = {}
        for mutation in mutations:
            operator_display_name = mutation['mutation_operator']
            # Map display name to class name
            class_name_mapping = {
                'Body Permutation': 'BodyPermutation',
                'Body Augmentation': 'BodyAugmentation',
                'Body Reduction': 'BodyReduction',
                'Entity Rename': 'EntityRename',
                'Rule Merging': 'RuleMerging'
            }
            operator_class_name = class_name_mapping.get(operator_display_name, operator_display_name)

            if operator_class_name not in operator_mutations:
                operator_mutations[operator_class_name] = []
            operator_mutations[operator_class_name].append(mutation)

        # Save a separate file for each operator
        for operator, op_mutations in operator_mutations.items():
            operator_dir = os.path.join(self.output_dir, operator)
            os.makedirs(operator_dir, exist_ok=True)  # Ensure the directory exists
            output_filename = f"{base_name}_mutations.csv"
            output_filepath = os.path.join(operator_dir, output_filename)

            self._save_mutations_to_csv(op_mutations, output_filepath)

            # Generate an operator-specific statistics report
            stats_filepath = output_filepath.replace('.csv', '_stats.txt')
            self._generate_mutation_stats(op_mutations, stats_filepath)

    def _save_mutations_to_csv(self, mutations: List[Dict[str, Any]], output_filepath: str):
        """
        Save mutation results to a CSV file.

        Args:
            mutations: list of mutation results
            output_filepath: output file path
        """
        # Define the column order
        column_order = [
            'rule_id',
            'source_file',
            'original_rule',
            'original_matched_instance',
            'original_natural_language',
            'mutation_operator',
            'mutation_index',
            'mutated_body',
            'expected_head',
            'mutated_rule',
            'mutation_type',
            'instantiation_mapping',
            'entity_labels'
        ]

        try:
            # Create the DataFrame
            df = pd.DataFrame(mutations)

            # Reorder columns (keep only existing ones)
            existing_columns = [col for col in column_order if col in df.columns]
            other_columns = [col for col in df.columns if col not in column_order]
            final_columns = existing_columns + other_columns

            df = df[final_columns]

            # Save to CSV
            df.to_csv(output_filepath, index=False, encoding='utf-8')

            logger.info(f"Mutations saved to {output_filepath}")

            # Generate the statistics report
            self._generate_mutation_stats(mutations, output_filepath.replace('.csv', '_stats.txt'))

        except Exception as e:
            logger.error(f"Error saving mutations to CSV: {e}")

    def _generate_mutation_stats(self, mutations: List[Dict[str, Any]], stats_filepath: str):
        """
        Generate a mutation statistics report.

        Args:
            mutations: list of mutation results
            stats_filepath: statistics file path
        """
        try:
            # Aggregate various information
            operator_counts = {}
            mutation_type_counts = {}

            for mutation in mutations:
                operator = mutation['mutation_operator']
                mutation_type = mutation['mutation_type']

                operator_counts[operator] = operator_counts.get(operator, 0) + 1
                mutation_type_counts[mutation_type] = mutation_type_counts.get(mutation_type, 0) + 1

            # Write the statistics report
            with open(stats_filepath, 'w', encoding='utf-8') as f:
                f.write("Rule Mutation Statistics Report\n")
                f.write("=" * 50 + "\n\n")

                f.write("Mutation operator statistics:\n")
                for operator, count in sorted(operator_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {operator}: {count} mutations\n")

                f.write("\nMutation type statistics:\n")
                for mutation_type, count in sorted(mutation_type_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {mutation_type}: {count} mutations\n")

                f.write(f"\nTotal mutations: {len(mutations)}\n")
                f.write(f"Number of original rules: {len(set(m['rule_id'] for m in mutations))}\n")
                f.write(f"Average mutations per rule: {len(mutations) / len(set(m['rule_id'] for m in mutations)):.2f}\n")

            logger.info(f"Statistics saved to {stats_filepath}")

        except Exception as e:
            logger.error(f"Error generating statistics: {e}")

    def process_all_files(self, positive_examples_dir: str = None,
                         max_files: int = None, max_rules_per_file: int = None):
        """
        Process all files.

        Args:
            positive_examples_dir: path to the positive_examples directory (uses config when None)
            max_files: maximum number of files to process (uses config when None)
            max_rules_per_file: maximum number of rules to process per file (uses config when None)
        """
        # Use config or parameter values
        positive_examples_dir = positive_examples_dir or get_config('paths.examples.positive', './data/examples/positive')
        max_files = max_files or get_config('mutation.limits.max_files')
        max_rules_per_file = max_rules_per_file or get_config('mutation.limits.max_rules_per_file', 1000)

        if not os.path.exists(positive_examples_dir):
            logger.error(f"Directory {positive_examples_dir} not found")
            return

        # Get all CSV files
        csv_files = [f for f in os.listdir(positive_examples_dir) if f.endswith('.csv')]

        if max_files:
            csv_files = csv_files[:max_files]

        logger.info(f"Found {len(csv_files)} CSV files to process")
        if self.selected_operator:
            logger.info(f"Using mutation operator: {self.selected_operator}")
        else:
            logger.info("Using all mutation operators")

        success_count = 0
        total_mutations = 0

        for filename in csv_files:
            filepath = os.path.join(positive_examples_dir, filename)

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {filename}")
            logger.info(f"{'='*60}")

            success = self.process_file(filepath, max_rules_per_file)
            if success:
                success_count += 1

                # Count the mutations for this file
                base_name = os.path.splitext(filename)[0]
                if self.selected_operator:
                    # Single-operator mode
                    operator_dir = os.path.join(self.output_dir, self.selected_operator)
                    output_filepath = os.path.join(operator_dir, f"{base_name}_mutations.csv")
                    if os.path.exists(output_filepath):
                        try:
                            df = pd.read_csv(output_filepath)
                            file_mutations = len(df)
                            total_mutations += file_mutations
                            logger.info(f"File {filename}: {file_mutations} mutations generated with {self.selected_operator}")
                        except:
                            pass
                else:
                    # Multi-operator mode - count mutations across all operators
                    for op_name in ['BodyPermutation', 'BodyAugmentation', 'BodyReduction', 'EntityRename', 'RuleMerging']:
                        operator_dir = os.path.join(self.output_dir, op_name)
                        output_filepath = os.path.join(operator_dir, f"{base_name}_mutations.csv")
                        if os.path.exists(output_filepath):
                            try:
                                df = pd.read_csv(output_filepath)
                                file_mutations = len(df)
                                total_mutations += file_mutations
                            except:
                                pass

        # Final statistics
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Successfully processed: {success_count}/{len(csv_files)} files")
        logger.info(f"Total mutations generated: {total_mutations}")
        logger.info(f"Output directory: {self.output_dir}")
        if self.selected_operator:
            logger.info(f"Mutation operator used: {self.selected_operator}")


def main():
    """Main function."""
    import argparse

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Rule Mutation Processing Script')
    parser.add_argument('--operator', '-o', type=str, default=None,
                       choices=['BodyPermutation', 'BodyAugmentation', 'BodyReduction',
                               'EntityRename', 'RuleMerging'],
                       help='Specify which mutation operator to use (default: use all operators)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process')
    parser.add_argument('--max-rules', type=int, default=1000,
                       help='Maximum number of rules per file to process')
    parser.add_argument('--input-dir', type=str, default='./positive_examples',
                       help='Input directory containing CSV files')

    args = parser.parse_args()

    # Configuration parameters
    positive_examples_dir = args.input_dir
    max_files = args.max_files
    max_rules_per_file = args.max_rules
    mutation_operator = args.operator

    # Memgraph connection configuration - use the configuration system
    memgraph_config = get_config('knowledge_graph.memgraph', {})
    kg_uri = memgraph_config.get('uri', 'bolt://localhost:7687')
    kg_user = memgraph_config.get('user', '')
    kg_password = memgraph_config.get('password', '')

    logger.info("Starting Rule Mutation Processing...")
    logger.info(f"Input directory: {positive_examples_dir}")
    logger.info(f"Max files to process: {max_files}")
    logger.info(f"Max rules per file: {max_rules_per_file}")
    if mutation_operator:
        logger.info(f"Selected mutation operator: {mutation_operator}")
    else:
        logger.info("Using all available mutation operators")

    # Create the processor
    processor = RuleMutationProcessor(
        kg_uri=kg_uri,
        kg_user=kg_user,
        kg_password=kg_password,
        mutation_operator=mutation_operator
    )

    # Process all files
    processor.process_all_files(
        positive_examples_dir=positive_examples_dir,
        max_files=max_files,
        max_rules_per_file=max_rules_per_file
    )

    logger.info("Rule mutation processing completed!")


if __name__ == "__main__":
    main()