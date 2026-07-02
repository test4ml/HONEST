#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Match positive examples of inference rules from Wikidata knowledge graph in Memgraph - Memgraph version

Using high-performance query engine based on Memgraph graph database:
- Graph pattern matching using Cypher query language
- High-performance computation based on in-memory analytics
- Native graph database index optimization
- Fast SPO triple queries

Specifically designed for Wikidata data in Memgraph, supporting:
- Graph structure of Entity nodes and HAS_PROPERTY relationships
- Readable labels and descriptions for entities and properties
- Efficient rule pattern matching
- Real-time query support for large-scale knowledge graphs
"""

import pandas as pd
import os
import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import time
from tqdm import tqdm
import argparse
import gc

# Use modern line_profiler method
from honest.utils.profiling import profile

# Import new Memgraph knowledge graph modules
from honest.kg import MemgraphKnowledgeGraph
from honest import PositiveExampleMatcher
from honest.rule_parser import RuleParser

@profile
def get_entity_descriptions(kg, qids):
    """Get entity labels and descriptions.

    Use MemgraphKnowledgeGraph's metadata interface, supporting complete metadata queries.
    """
    entity_info = []
    for qid in qids:
        # Use knowledge graph's metadata interface to get labels and descriptions
        label = kg.get_entity_label(qid) or "<<NO_LABEL>>"
        description = kg.get_entity_description(qid) or "<<NO_DESCRIPTION>>"
        entity_info.append(f"{qid}: {label} | {description}")
    return entity_info


@profile
def format_rule_instance(variables, body_triples, head_triple):
    """Format rule instance by replacing variables with actual values"""
    # Build instantiated rule string
    instance_parts = []

    # Process body part
    for body_triple in body_triples:
        s, p, o = body_triple
        instance_parts.append(f"{s} {p} {o}")

    # Add head part
    s, p, o = head_triple
    head_part = f"{s} {p} {o}"

    # Combine into complete instance
    if instance_parts:
        return "  ".join(instance_parts) + " => " + head_part
    else:
        return head_part


class MemgraphPositiveExampleMatcher(PositiveExampleMatcher):
    """Rule positive example matcher optimized for Memgraph

    Inherits from PositiveExampleMatcher to maintain interface compatibility,
    but overrides core matching methods to leverage Memgraph's native graph query capabilities.
    """

    def __init__(self, kg, max_examples=1000):
        # Call parent constructor
        super().__init__(kg)
        self.max_examples = max_examples

    def match_rule(self, rule_str):
        """Use Memgraph native capabilities for rule matching, better performance"""
        try:
            # Parse rule
            if '=>' in rule_str:
                rule = RuleParser.parse_rule(rule_str)
                body_patterns = [fact.to_tuple() for fact in rule.body]
                head_pattern = rule.head.to_tuple()
            else:
                # Single triple rule
                body_patterns = []
                head_pattern = RuleParser._parse_pattern(rule_str.strip())

            # Use MemgraphKnowledgeGraph's native pattern matching functionality
            if hasattr(self.kg, 'match_rule_patterns'):
                # If KG supports native pattern matching, use it (faster)
                return self.kg.match_rule_patterns(body_patterns, head_pattern, max_examples=self.max_examples)
            else:
                # Fallback to parent method
                examples = super().match_rule(rule_str)
                # Limit the number of returned positive examples
                if len(examples) > self.max_examples:
                    examples = examples[:self.max_examples]
                return examples

        except Exception as e:
            print(f"Rule matching failed '{rule_str}': {e}")
            # Fallback to parent method
            try:
                examples = super().match_rule(rule_str)
                if len(examples) > self.max_examples:
                    examples = examples[:self.max_examples]
                return examples
            except Exception as fallback_error:
                print(f"Fallback matching also failed: {fallback_error}")
                return []


@profile
def main():
    parser = argparse.ArgumentParser(description='Match positive examples of inference rules from Wikidata knowledge graph in Memgraph - Memgraph version')

    # Memgraph connection parameters
    parser.add_argument('--memgraph_uri', default='bolt://localhost:7687',
                        help='Memgraph connection URI (default: bolt://localhost:7687)')
    parser.add_argument('--memgraph_user', default='',
                        help='Memgraph username (default: empty)')
    parser.add_argument('--memgraph_password', default='',
                        help='Memgraph password (default: empty)')

    # Rules and output parameters
    parser.add_argument('--rules_file', default='data/processed/rules/rules_logic_y_filtered.csv',
                        help='Inference rules file path (default: data/processed/rules/rules_logic_y_filtered.csv)')
    parser.add_argument('--output_dir', default='data/examples/positive',
                        help='Output directory (default: data/examples/positive)')
    parser.add_argument('--sep', default=',',
                        help='CSV file separator (default: comma)')

    # Instance matching parameters
    parser.add_argument('--max_examples', type=int, default=1000,
                        help='Maximum number of positive examples per rule (default: 1000)')

    # Performance and debugging parameters
    parser.add_argument('--test_rules', type=int, default=None,
                        help='Only test first N rules (for debugging) (default: None)')
    parser.add_argument('--disable_query_optimization', action='store_true',
                        help='Disable intelligent query optimization (for comparison testing)')
    parser.add_argument('--enable_metadata_cache', action='store_true', default=True,
                        help='Enable metadata cache (improves performance) (default: True)')
    parser.add_argument('--skip_stats', action='store_true',
                        help='Skip database statistics retrieval (significantly speeds up startup)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Using Memgraph Knowledge Graph for Rule Matching ===\n")
    print(f"Memgraph URI: {args.memgraph_uri}")
    print(f"Output directory: {args.output_dir}")

    # Initialize Memgraph knowledge graph
    print("\nConnecting to Memgraph database...")
    start_time = time.time()

    kg = MemgraphKnowledgeGraph(
        uri=args.memgraph_uri,
        user=args.memgraph_user,
        password=args.memgraph_password,
        enable_metadata=args.enable_metadata_cache
    )

    # Test connection
    if not kg.test_connection():
        print("❌ Unable to connect to Memgraph database, program terminated")
        return

    end_time = time.time()
    print(f"✅ Memgraph connection successful, time taken: {end_time - start_time:.2f} seconds")

    # Display database statistics (optional)
    if not args.skip_stats:
        print("\nRetrieving Memgraph database statistics...")
        stats = kg.get_stats()
        print("\nMemgraph Knowledge Graph Statistics:")
        for key, value in stats.items():
            if isinstance(value, int):
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value}")
    else:
        print("⏭️  Skipping statistics retrieval (using --skip_stats)")

    # Create matcher
    matcher = MemgraphPositiveExampleMatcher(kg, max_examples=args.max_examples)

    # Configure query optimization
    if args.disable_query_optimization:
        matcher.enable_query_optimization = False
        optimization_status = "Disabled"
    else:
        matcher.enable_query_optimization = True
        optimization_status = "Enabled"

    print(f"\nMatcher type: Memgraph Graph Database Mode - Intelligent Query Optimization {optimization_status}")

    # Load inference rules
    print(f"\nLoading inference rules: {args.rules_file}")
    if not os.path.exists(args.rules_file):
        print(f"Error: Rules file does not exist: {args.rules_file}")
        kg.close()
        return

    rules_df = pd.read_csv(args.rules_file, sep=args.sep)
    print(f"Loaded {len(rules_df)} rules")

    # If test mode is set, only process first N rules
    if args.test_rules:
        rules_df = rules_df.head(args.test_rules)
        print(f"Test mode: Only processing first {len(rules_df)} rules")

    # Statistics for processing each rule
    total_examples = 0
    processed_rules = 0
    total_query_time = 0
    failed_rules = 0

    print(f"\nStarting rule processing (using Memgraph engine)...\n")

    # Process each rule
    for idx, row in tqdm(rules_df.iterrows(), total=len(rules_df), desc="Processing inference rules"):
        rule_str = row['Rule']
        rule_id = idx + 1

        print(f"\nProcessing rule {rule_id}: {rule_str}")

        # Display optimization information for complex rules
        if '=>' in rule_str and not args.disable_query_optimization:
            try:
                rule = RuleParser.parse_rule(rule_str)
                body_patterns = [fact.to_tuple() for fact in rule.body]
                if len(body_patterns) > 1:  # Consider rules with more than 1 body clause as complex
                    print(f"  Detected complex rule ({len(body_patterns)} body clauses), enabling Cypher query optimization...")
            except Exception as e:
                # If parsing fails, fall back to simple length-based judgment
                if len(rule_str.split()) > 6:
                    print(f"  Detected complex rule (parsing failed, using length judgment), enabling query optimization...")

        # Time the complete rule processing process
        rule_start_time = time.time()

        try:
            # Time query and post-processing separately
            query_start_time = time.time()
            positive_examples = matcher.match_rule(rule_str)
            query_end_time = time.time()

            query_time = query_end_time - query_start_time
            total_query_time += query_time
            metadata_time = 0  # Initialize metadata time

            if positive_examples:
                # Limit the number of positive examples
                if len(positive_examples) > args.max_examples:
                    positive_examples = positive_examples[:args.max_examples]

                # Prepare output data for current rule
                rule_output_data = []

                # Create output row for each instance
                for example in positive_examples:
                    body_triples = example['body_triples']
                    head_triple = example['head_triple']
                    variables = example['variables']

                    # Collect all involved entity QIDs
                    all_entities = set()
                    for triple in body_triples + [head_triple]:
                        s, p, o = triple
                        if s.startswith('Q'):
                            all_entities.add(s)
                        if o.startswith('Q'):
                            all_entities.add(o)

                    # Get entity information (using Memgraph's metadata functionality)
                    metadata_start_time = time.time()
                    entity_descriptions = get_entity_descriptions(kg, list(all_entities))
                    metadata_end_time = time.time()
                    metadata_time = metadata_end_time - metadata_start_time

                    # Create output row, including all fields from original CSV
                    output_row = {
                        # New fields
                        'OriginalRule': rule_str,
                        'RuleMatchedInstance': format_rule_instance(variables, body_triples, head_triple),
                        'InstantiationMapping': json.dumps(variables, ensure_ascii=False),
                        'EntityLabelsDescriptions': json.dumps(entity_descriptions, ensure_ascii=False),
                        'MatchQueryTimeSec': round(query_time, 3),
                        'EngineType': 'MemgraphKnowledgeGraph',

                        # Fields from original CSV (unified to camelCase naming)
                        'Rule': row['Rule'],
                        'HeadCoverage': row.get('Head Coverage', ''),
                        'StdConfidence': row.get('Std Confidence', ''),
                        'PCAConfidence': row.get('PCA Confidence', ''),
                        'PositiveExamples': row.get('Positive Examples', ''),
                        'BodySize': row.get('Body size', ''),
                        'PCABodySize': row.get('PCA Body size', ''),
                        'FunctionalVariable': row.get('Functional variable', ''),
                        'NaturalLanguage': row.get('Natural_Language', ''),
                        'LogicJudgment': row.get('Logic_Judgment', ''),
                        'JudgmentReason': row.get('Judgment_Reason', ''),
                    }

                    rule_output_data.append(output_row)

                # Save current rule's results to separate CSV file
                save_start_time = time.time()
                output_file = os.path.join(args.output_dir, f'kg_rule_{rule_id}_examples.csv')
                rule_df = pd.DataFrame(rule_output_data)
                rule_df.to_csv(output_file, index=False, encoding='utf-8')
                save_end_time = time.time()
                save_time = save_end_time - save_start_time

                rule_end_time = time.time()
                total_rule_time = rule_end_time - rule_start_time

                print(f"Rule {rule_id} found {len(positive_examples)} positive examples")
                print(f"  Query time: {query_time:.2f}s")
                print(f"  Metadata retrieval: {metadata_time:.2f}s")
                print(f"  Save time: {save_time:.2f}s")
                print(f"  Total time: {total_rule_time:.2f}s")
                print(f"  Saved to: {output_file}")
                total_examples += len(positive_examples)
                processed_rules += 1
            else:
                rule_end_time = time.time()
                total_rule_time = rule_end_time - rule_start_time
                print(f"Rule {rule_id} found no positive examples")
                print(f"  Query time: {query_time:.2f}s")
                print(f"  Total time: {total_rule_time:.2f}s")

        except Exception as e:
            failed_rules += 1
            rule_end_time = time.time()
            error_time = rule_end_time - rule_start_time
            total_query_time += error_time
            print(f"Rule {rule_id} processing failed: {e}")
            print(f"  Error time: {error_time:.2f}s")

        # Force garbage collection
        gc.collect()

    # Close Memgraph connection
    kg.close()

    # Output final statistics
    print(f"\n=== Memgraph Rule Matching Completed ===\n")
    print(f"Total rules processed: {len(rules_df)}")
    print(f"Successfully matched rules: {processed_rules}")
    print(f"Failed rules: {failed_rules}")
    print(f"Total positive examples: {total_examples}")
    print(f"Total query time: {total_query_time:.2f} seconds")
    print(f"Average query time per rule: {total_query_time/len(rules_df):.2f}s")
    print(f"Output directory: {args.output_dir}")

    if processed_rules > 0:
        print(f"Average positive examples per rule: {total_examples/processed_rules:.1f}")


if __name__ == "__main__":
    main()