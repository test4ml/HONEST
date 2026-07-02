#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate natural language reasoning questions from mutated rule instances (Enhanced version)

Using honest/question_generator module, supporting multiple question types:
- Yes/No questions
- Wh-questions
- True/False questions
- Multiple choice questions

Based on rule instances in mutation_examples, query Q and P labels from knowledge graph,
generate English natural language reasoning questions and expected answers.

Output to test_questions, one source file outputs one folder,
each mutation operator generates one csv file.
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
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple
import argparse

# Import knowledge graph query related modules
from honest.kg import MemgraphKnowledgeGraph
from honest.qgen import QuestionGenerator, QuestionType
from honest.rule_parser import Rule, Fact, LogicExpression
from honest.horn_rule_parser import HornRuleParser
from honest.functionality_cache_loader import load_functionality_cache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedQuestionGenerator:
    """Enhanced question generator using honest/question_generator module"""

    def __init__(self, knowledge_graph: MemgraphKnowledgeGraph):
        self.kg = knowledge_graph
        # Create honest's QuestionGenerator instance
        self.honest_generator = QuestionGenerator(knowledge_graph)

    def _parse_mutated_body(self, mutated_body: str, instantiation: Dict) -> List[Tuple[str, str, str]]:
        """Parse mutated_body format, e.g.: Q106961095 P2673 Q106594316 Q106961095 P17 Q30"""
        # Use HornRuleParser's internal method to parse body part (simple AND-only format)
        return HornRuleParser._parse_body_patterns(mutated_body)

    def generate_questions_for_instance(self, instance: str, original_rule: str,
                                       instantiation: Dict, question_types: List[QuestionType] = None,
                                       focus_entities: List[str] = None, focus_relations: List[str] = None,
                                       num_choices: int = 4, answer_position_seed: Optional[int] = None) -> List[Dict[str, str]]:
        """Generate questions for a single instance

        Args:
            answer_position_seed: Optional seed for consistent answer positions (passed to QuestionGenerator).
        """
        try:
            # Default to generate all question types
            if question_types is None:
                question_types = [QuestionType.YES_NO, QuestionType.WH_QUESTION,
                                 QuestionType.TRUE_FALSE, QuestionType.MULTIPLE_CHOICE]

            # Check if instance contains => separator
            if instance and " => " in instance:
                # Use HornRuleParser to parse simple Horn rule format (no OR/AND/NOT keywords)
                rule = HornRuleParser.parse_rule(instance)

                # Extract body and head from Rule object
                # Handle both List[Fact] and LogicExpression
                if isinstance(rule.body, list):
                    # Traditional format: List[Fact]
                    body_patterns = [fact.to_tuple() for fact in rule.body]
                else:
                    # New format: LogicExpression
                    # For now, try to convert to fact list if possible (AND-only)
                    try:
                        body_facts = rule.body.to_fact_list()
                        body_patterns = [fact.to_tuple() for fact in body_facts]
                    except ValueError:
                        # Cannot convert OR/NOT to simple list - use LogicExpression directly
                        body_patterns = rule.body

                head_pattern = rule.head.to_tuple()

                # Call honest's question generator - pass the original instance string
                # instead of parsed format to avoid double parsing
                questions = self.honest_generator.generate_questions(
                    rule=instance,  # Pass the original string, not parsed format
                    instantiation=instantiation,
                    question_types=question_types,
                    focus_entities=focus_entities,
                    focus_relations=focus_relations,
                    num_choices=num_choices,
                    answer_position_seed=answer_position_seed  # Pass seed for consistent positions
                )
            else:
                # Use original_rule
                questions = self.honest_generator.generate_questions(
                    rule=original_rule,
                    instantiation=instantiation,
                    question_types=question_types,
                    focus_entities=focus_entities,
                    focus_relations=focus_relations,
                    num_choices=num_choices,
                    answer_position_seed=answer_position_seed  # Pass seed for consistent positions
                )

            # Format output
            formatted_questions = []
            for q in questions:
                formatted_q = {
                    'question': q['question'],
                    'answer': q['answer'],
                    'question_type': q['type'],
                    'reasoning': q.get('reasoning', ''),
                    'correct_answer': q.get('correct_answer', ''),
                    'options': json.dumps(q.get('options', []), ensure_ascii=False)
                }
                formatted_questions.append(formatted_q)

            return formatted_questions

        except Exception as e:
            logger.error(f"Error generating questions for instance: {e}")
            return [{
                'question': 'Failed to generate question',
                'answer': 'Failed to generate answer',
                'question_type': 'error',
                'reasoning': '',
                'correct_answer': '',
                'options': '[]'
            }]

    def generate_questions(self, rule_data: Dict, question_types: List[QuestionType] = None,
                          focus_entities: List[str] = None, focus_relations: List[str] = None,
                          num_choices: int = 4, answer_position_seed: Optional[int] = None) -> List[Dict[str, str]]:
        """Generate question pairs for original and mutated instances

        Args:
            answer_position_seed: Optional seed for consistent answer positions.
                                 If provided, ensures original and mutated questions
                                 have the same answer position (for metamorphic testing).
        """
        try:
            # Parse rule data
            original_rule = rule_data['original_rule']
            original_instance = rule_data.get('original_instance', '')
            mutated_instance = rule_data.get('mutated_instance', '')
            instantiation = json.loads(rule_data['instantiation_mapping'])

            # Generate questions for original instance (with seed)
            original_questions = self.generate_questions_for_instance(
                original_instance, original_rule, instantiation,
                question_types, focus_entities, focus_relations, num_choices,
                answer_position_seed=answer_position_seed  # Pass SAME seed
            )

            # Generate questions for mutated instance (with SAME seed)
            mutated_questions = self.generate_questions_for_instance(
                mutated_instance, original_rule, instantiation,
                question_types, focus_entities, focus_relations, num_choices,
                answer_position_seed=answer_position_seed  # Pass SAME seed for consistency
            )

            # Pair original and mutated questions
            paired_questions = []
            max_questions = max(len(original_questions), len(mutated_questions))

            for i in range(max_questions):
                original_q = original_questions[i] if i < len(original_questions) else {
                    'question': '', 'correct_answer': '', 'question_type': '', 'options': '[]'
                }
                mutated_q = mutated_questions[i] if i < len(mutated_questions) else {
                    'question': '', 'correct_answer': '', 'question_type': '', 'options': '[]'
                }

                paired_q = {
                    'original_question': original_q['question'],
                    'original_correct_answer': original_q['correct_answer'],
                    'original_question_type': original_q['question_type'],
                    'original_options': original_q['options'],
                    'mutated_question': mutated_q['question'],
                    'mutated_correct_answer': mutated_q['correct_answer'],
                    'mutated_question_type': mutated_q['question_type'],
                    'mutated_options': mutated_q['options']
                }
                paired_questions.append(paired_q)

            return paired_questions

        except Exception as e:
            logger.error(f"Error generating question pairs: {e}")
            return [{
                'original_question': 'Failed to generate original question',
                'original_correct_answer': '',
                'original_question_type': 'error',
                'original_options': '[]',
                'mutated_question': 'Failed to generate mutated question',
                'mutated_correct_answer': '',
                'mutated_question_type': 'error',
                'mutated_options': '[]'
            }]

    def generate_single_question(self, rule_data: Dict, question_type: QuestionType = QuestionType.YES_NO) -> Dict[str, str]:
        """Generate single question (backward compatibility)"""
        questions = self.generate_questions(rule_data, question_types=[question_type])
        if questions:
            # Return mutated question from first question pair (maintain backward compatibility)
            q = questions[0]
            return {
                'question': q['mutated_question'],
                'correct_answer': q['mutated_correct_answer'],
                'question_type': q['mutated_question_type'],
                'options': q['mutated_options']
            }
        else:
            return {
                'question': 'Failed to generate question',
                'correct_answer': '',
                'question_type': 'error',
                'options': '[]'
            }


def _is_rule_template(instance_str: str) -> bool:
    """Check if an instance string is a rule template (contains variables)

    A rule template contains variables like ?a, ?b, ?c, etc.
    A valid instance should contain entity IDs (Q123...) or renamed entities (Alice, Bob, etc.)

    Args:
        instance_str: Rule instance string

    Returns:
        True if it's a rule template (invalid), False if it's a valid instance

    Examples:
        "?b P2743 ?a => ?a P2743 ?b" -> True (template, invalid)
        "Q123 P2743 Q456 => Q456 P2743 Q123" -> False (valid instance)
        "Alice P2743 Bob => Bob P2743 Alice" -> False (valid renamed instance)
    """
    # A rule template contains variable placeholders starting with '?'
    # If the instance contains '?' followed by alphanumeric characters, it's a template
    import re
    return bool(re.search(r'\?[a-zA-Z0-9_]+', instance_str))


def _validate_mutation_row(row: pd.Series) -> tuple:
    """Validate that a mutation row contains valid data for question generation

    Returns:
        (is_valid: bool, reason: str)

    Validation checks:
    1. mutated_instance should not be a rule template
    2. mutated_instance should be different from original_instance (for meaningful mutation)
    3. mutated_instance should be different from original_rule
    """
    original_rule = row.get('original_rule', '')
    original_instance = row.get('original_instance', '')
    mutated_instance = row.get('mutated_instance', '')

    # Check 1: mutated_instance should not be a rule template
    if _is_rule_template(mutated_instance):
        return False, f"mutated_instance is a rule template (contains variables): {mutated_instance[:80]}"

    # Check 2: mutated_instance should differ from original_instance
    # For entity_rename, this check is actually OK to be same if renaming to similar virtual names
    # So we make this a warning, not an error
    if mutated_instance == original_instance:
        # This is actually OK for some mutation types (e.g., body permutation with same order)
        pass

    # Check 3: mutated_instance should not be identical to original_rule
    # This indicates the mutation operator failed to produce a proper instance
    if mutated_instance == original_rule:
        return False, f"mutated_instance equals original_rule (mutation failed): {mutated_instance[:80]}"

    return True, ""


def process_mutation_file(csv_file_path: str, knowledge_graph: MemgraphKnowledgeGraph,
                         question_generator: EnhancedQuestionGenerator,
                         question_types: List[QuestionType] = None,
                         focus_entities: List[str] = None,
                         focus_relations: List[str] = None,
                         sep=',') -> List[Dict]:
    """Process a single mutation file"""
    import random  # Import here to avoid polluting global namespace

    logger.info(f"Processing file: {csv_file_path}")

    try:
        df = pd.read_csv(csv_file_path, sep=sep)
        results = []
        skipped_count = 0
        skipped_reasons = {}

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {Path(csv_file_path).name}"):
            try:
                # VALIDATION: Check if mutation row is valid
                is_valid, reason = _validate_mutation_row(row)
                if not is_valid:
                    skipped_count += 1
                    skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                    logger.warning(f"Skipping row {idx}: {reason}")
                    continue

                # CRITICAL FIX: Generate a random seed for this sample
                # This seed ensures original and mutated questions have the SAME answer position
                # Required for proper metamorphic testing
                answer_position_seed = random.randint(0, 2**31 - 1)

                # Generate multiple types of questions (with seed)
                questions_data = question_generator.generate_questions(
                    row.to_dict(),
                    question_types=question_types,
                    focus_entities=focus_entities,
                    focus_relations=focus_relations,
                    answer_position_seed=answer_position_seed  # Pass seed for consistent positions
                )

                for q_data in questions_data:
                    result = {
                        'original_rule': row.get('original_rule', ''),
                        'original_instance': row.get('original_instance', ''),
                        'mutated_instance': row.get('mutated_instance', ''),
                        'answer_position_seed': answer_position_seed,  # NEW: Store seed for reproducibility

                        # Original instance questions and answers (keep concise format)
                        'original_question': q_data['original_question'],
                        'original_correct_answer': q_data['original_correct_answer'],
                        'original_question_type': q_data['original_question_type'],
                        'original_options': q_data['original_options'],

                        # Mutated instance questions and answers (keep concise format)
                        'mutated_question': q_data['mutated_question'],
                        'mutated_correct_answer': q_data['mutated_correct_answer'],
                        'mutated_question_type': q_data['mutated_question_type'],
                        'mutated_options': q_data['mutated_options'],

                        'instantiation_mapping': row.get('instantiation_mapping', ''),
                        'entity_labels': row.get('entity_labels', ''),
                        'natural_language': row.get('natural_language', '')
                    }
                    results.append(result)

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue

        # Log summary of skipped rows
        if skipped_count > 0:
            logger.warning(f"\n{'='*80}")
            logger.warning(f"VALIDATION SUMMARY for {Path(csv_file_path).name}")
            logger.warning(f"{'='*80}")
            logger.warning(f"Total rows: {len(df)}")
            logger.warning(f"Skipped rows: {skipped_count}")
            logger.warning(f"Processed rows: {len(df) - skipped_count}")
            logger.warning(f"\nSkip reasons:")
            for reason, count in sorted(skipped_reasons.items(), key=lambda x: x[1], reverse=True):
                logger.warning(f"  - {count}x: {reason}")
            logger.warning(f"{'='*80}\n")

        return results

    except Exception as e:
        logger.error(f"Error reading file {csv_file_path}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Generate natural language reasoning questions from mutated rule instances (Enhanced version)')

    # Memgraph connection parameters
    parser.add_argument('--memgraph_uri', default='bolt://localhost:7687',
                        help='Memgraph connection URI (default: bolt://localhost:7687)')
    parser.add_argument('--memgraph_user', default='',
                        help='Memgraph username (default: empty)')
    parser.add_argument('--memgraph_password', default='',
                        help='Memgraph password (default: empty)')

    # Input/output parameters
    parser.add_argument('--mutation_examples_dir', default='data/examples/mutated',
                        help='Mutation examples directory (default: data/examples/mutated)')
    parser.add_argument('--output_dir', default='data/examples/questions',
                        help='Output directory (default: data/examples/questions)')
    parser.add_argument('--sep', default=',',
                        help='CSV file separator (default: comma)')
    parser.add_argument('--test_limit', type=int, default=None,
                        help='Test limit (only process first N folders)')

    # Question type parameters
    parser.add_argument('--question_types', nargs='+', default=['yes_no', 'wh_question', 'true_false', 'multiple_choice'],
                        choices=['yes_no', 'wh_question', 'true_false', 'multiple_choice'],
                        help='Question types to generate')
    parser.add_argument('--focus_entities', nargs='+', default=None,
                        help='List of entities to focus on')
    parser.add_argument('--focus_relations', nargs='+', default=None,
                        help='List of relations to focus on')
    parser.add_argument('--num_choices', type=int, default=4,
                        help='Number of choices for multiple choice questions')

    # Functionality cache parameters
    parser.add_argument('--functionality_cache_file', default='data/processed/properties/property_functionality.jsonl',
                        help='Precomputed functionality cache file path')
    parser.add_argument('--disable_functionality_cache', action='store_true',
                        help='Disable precomputed functionality cache')

    args = parser.parse_args()

    # Convert question type parameters
    question_type_map = {
        'yes_no': QuestionType.YES_NO,
        'wh_question': QuestionType.WH_QUESTION,
        'true_false': QuestionType.TRUE_FALSE,
        'multiple_choice': QuestionType.MULTIPLE_CHOICE
    }
    question_types = [question_type_map[qt] for qt in args.question_types]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=== Starting natural language reasoning question generation (Enhanced version) ===")
    logger.info(f"Memgraph URI: {args.memgraph_uri}")
    logger.info(f"Input directory: {args.mutation_examples_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Question types: {args.question_types}")
    logger.info(f"Focus entities: {args.focus_entities}")
    logger.info(f"Focus relations: {args.focus_relations}")

    # Load precomputed functionality cache
    functionality_cache = {}
    inv_functionality_cache = {}

    if not args.disable_functionality_cache:
        logger.info(f"Loading precomputed functionality cache: {args.functionality_cache_file}")
        try:
            functionality_cache, inv_functionality_cache = load_functionality_cache(args.functionality_cache_file)
            cache_count = len(functionality_cache) + len(inv_functionality_cache)
            if cache_count > 0:
                logger.info(f"Successfully loaded functionality cache: functionality={len(functionality_cache)}, inverse functionality={len(inv_functionality_cache)}")
            else:
                logger.warning("No functionality cache data loaded")
        except Exception as e:
            logger.warning(f"Failed to load functionality cache: {e}")
    else:
        logger.info("Precomputed functionality cache disabled")

    # Initialize knowledge graph connection
    logger.info("Connecting to knowledge graph...")
    kg = MemgraphKnowledgeGraph(
        uri=args.memgraph_uri,
        user=args.memgraph_user,
        password=args.memgraph_password,
        enable_metadata=True,
        functionality_cache=functionality_cache if functionality_cache else None,
        inv_functionality_cache=inv_functionality_cache if inv_functionality_cache else None
    )

    if not kg.test_connection():
        logger.error("Unable to connect to knowledge graph, program terminated")
        return

    logger.info("Knowledge graph connection successful")

    # Initialize question generator
    question_generator = EnhancedQuestionGenerator(kg)

    # Process all mutation example folders
    mutation_dirs = sorted([d for d in os.listdir(args.mutation_examples_dir)
                           if os.path.isdir(os.path.join(args.mutation_examples_dir, d))])

    if args.test_limit:
        mutation_dirs = mutation_dirs[:args.test_limit]
        logger.info(f"Limited to processing first {args.test_limit} folders")

    total_questions = 0

    for i, dir_name in enumerate(mutation_dirs):
        print(f"Processing [{i+1}/{len(mutation_dirs)}] {dir_name}")
        dir_path = os.path.join(args.mutation_examples_dir, dir_name)
        output_subdir = os.path.join(args.output_dir, dir_name)
        os.makedirs(output_subdir, exist_ok=True)

        # Process all CSV files in this folder
        csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

        for csv_file in csv_files:
            csv_path = os.path.join(dir_path, csv_file)
            output_path = os.path.join(output_subdir, csv_file)

            # Skip if output file already exists
            if os.path.exists(output_path):
                logger.info(f"Skipping {csv_file}, output file already exists: {output_path}")
                continue

            # Generate questions
            results = process_mutation_file(csv_path, kg, question_generator,
                                          question_types, args.focus_entities, args.focus_relations, sep=args.sep)

            if results:
                # Create DataFrame and save results
                df = pd.DataFrame(results)

                # Define column order
                column_order = [
                    'original_rule', 'original_instance', 'mutated_instance',
                    'answer_position_seed',  # NEW: Seed for consistent answer positions
                    # Original instance questions and answers
                    'original_question', 'original_correct_answer', 'original_question_type', 'original_options',
                    # Mutated instance questions and answers
                    'mutated_question', 'mutated_correct_answer', 'mutated_question_type', 'mutated_options',
                    # Other information
                    'instantiation_mapping', 'entity_labels', 'natural_language'
                ]

                # Reorder columns (only keep existing columns)
                existing_columns = [col for col in column_order if col in df.columns]
                other_columns = [col for col in df.columns if col not in column_order]
                final_columns = existing_columns + other_columns

                df = df[final_columns]

                # Save to CSV
                df.to_csv(output_path, index=False, encoding='utf-8', sep=args.sep)

                logger.info(f"Generated {len(results)} questions for {csv_file}")
                total_questions += len(results)
            else:
                logger.warning(f"No results generated for {csv_file}")

    # Close knowledge graph connection
    kg.close()
    logger.info(f"=== Question generation completed, total {total_questions} questions generated ===")


if __name__ == '__main__':
    main()