#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Property formatting rule generation script.
Read all properties from wiki_properties.jsonl and use a local LLM to generate formatting rules.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from honest.llm import SyncLLMClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PropertyRuleGenerator:
    """Property formatting rule generator."""

    def __init__(self, llm_client: SyncLLMClient):
        self.llm_client = llm_client

        # Few-shot examples, based on existing property formatting rules
        self.few_shot_examples = [
            {
                "property_id": "P31",
                "label": "instance of",
                "pattern": "{subject} is {object}",
                "needs_article_subject": True,
                "needs_article_object": True,
                "description": "instance of - 实例"
            },
            {
                "property_id": "P569",
                "label": "date of birth",
                "pattern": "{subject} was born on {object}",
                "needs_article_subject": True,
                "needs_article_object": False,
                "description": "date of birth - 出生日期"
            },
            {
                "property_id": "P22",
                "label": "father",
                "pattern": "{subject} is the father of {object}",
                "needs_article_subject": True,
                "needs_article_object": False,
                "description": "father - 父亲"
            },
            {
                "property_id": "P17",
                "label": "country",
                "pattern": "{subject} is located in {object}",
                "needs_article_subject": True,
                "needs_article_object": True,
                "description": "country - 国家"
            },
            {
                "property_id": "P279",
                "label": "subclass of",
                "pattern": "{subject} is a subclass of {object}",
                "needs_article_subject": True,
                "needs_article_object": False,
                "description": "subclass of - 子类"
            }
        ]

    def create_few_shot_prompt(self, property_id: str, label: str) -> str:
        """Create a prompt containing few-shot examples."""

        # Build the few-shot examples section
        examples_text = "Here are some examples of how to generate formatting rules for Wikidata properties:\n\n"

        for i, example in enumerate(self.few_shot_examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Property ID: {example['property_id']}\n"
            examples_text += f"Label: {example['label']}\n"
            examples_text += f"Output:\n"
            examples_text += "{\n"
            examples_text += f'  "property_id": "{example["property_id"]}",\n'
            examples_text += f'  "pattern": "{example["pattern"]}",\n'
            examples_text += f'  "needs_article_subject": {str(example["needs_article_subject"]).lower()},\n'
            examples_text += f'  "needs_article_object": {str(example["needs_article_object"]).lower()},\n'
            examples_text += f'  "description": "{example["description"]}"\n'
            examples_text += "}\n\n"

        # Main task description
        task_description = """Your task is to generate a formatting rule for a Wikidata property. The rule should specify:

1. pattern: A natural language template using {subject} and {object} placeholders
2. needs_article_subject: Whether the subject needs an article (a/an/the)
3. needs_article_object: Whether the object needs an article (a/an/the)
4. description: A brief description including English and Chinese translations

Guidelines:
- Use natural, grammatically correct English patterns
- Consider whether entities typically need articles (proper nouns usually don't, common nouns do)
- For dates, numbers, and measurements: usually no articles needed
- For people's names, countries, organizations: usually no articles needed
- For common nouns, roles, types: usually articles needed
- Make the pattern sound natural when read aloud

Now generate a formatting rule for this property:

Property ID: """ + property_id + """
Label: """ + label + """

Return ONLY the JSON object with the required fields:"""

        full_prompt = examples_text + task_description

        return full_prompt

    def generate_rule(self, property_id: str, label: str) -> Optional[Dict]:
        """Generate a formatting rule for a single property."""
        try:
            prompt = self.create_few_shot_prompt(property_id, label)
            response = self.llm_client.generate_answer(
                prompt,
                max_retries=3,
                user_prompt=prompt,
                system_prompt="You are a helpful assistant that generates JSON output.",
                max_tokens=512,
                temperature=0.1
            )

            if response.startswith("ERROR:"):
                logger.error(f"Failed to generate response for {property_id}")
                return None

            logger.debug(f"Raw LLM response for {property_id}: {repr(response)}")

            # Try to parse the JSON response
            try:
                # Clean up the response and extract the JSON part
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.endswith("```"):
                    response = response[:-3]
                response = response.strip()

                # Parse the JSON
                rule_data = json.loads(response)

                # Validate the required fields
                required_fields = ["property_id", "pattern", "needs_article_subject", "needs_article_object", "description"]
                for field in required_fields:
                    if field not in rule_data:
                        logger.error(f"Missing field '{field}' in response for {property_id}")
                        return None

                return rule_data

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for {property_id}: {e}")
                logger.error(f"Raw response was: {repr(response)}")
                return None

        except Exception as e:
            logger.error(f"Error generating rule for {property_id}: {e}")
            return None

    def generate_fallback_rule(self, property_id: str, label: str) -> Dict:
        """Generate a fallback rule (used when LLM generation fails)."""
        # Simple heuristic rule
        needs_article_object = True

        # Some common cases that do not need an article
        no_article_patterns = [
            'name', 'date', 'time', 'year', 'number', 'id', 'code',
            'country', 'place', 'location', 'person', 'people'
        ]

        label_lower = label.lower()
        if any(pattern in label_lower for pattern in no_article_patterns):
            needs_article_object = False

        return {
            "property_id": property_id,
            "pattern": "{subject} {predicate} {object}",  # Use the default pattern
            "needs_article_subject": True,
            "needs_article_object": needs_article_object,
            "description": f"{label} - {label}"  # Simple description
        }


def load_existing_rules(rules_file: str) -> Dict[str, Dict]:
    """Load existing rules."""
    existing_rules = {}
    if os.path.exists(rules_file):
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rule = json.loads(line)
                        existing_rules[rule['property_id']] = rule
        except Exception as e:
            logger.warning(f"Error loading existing rules: {e}")
    return existing_rules


def save_rules(rules: List[Dict], output_file: str, append: bool = False):
    """Save rules to a JSONL file."""
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create it when there is a directory path
            os.makedirs(output_dir, exist_ok=True)

        mode = 'a' if append else 'w'
        with open(output_file, mode, encoding='utf-8') as f:
            for rule in rules:
                f.write(json.dumps(rule, ensure_ascii=False) + '\n')

        action = "Appended" if append else "Saved"
        logger.info(f"{action} {len(rules)} rules to {output_file}")

    except Exception as e:
        logger.error(f"Error saving rules: {e}")


def save_batch_rules(rules: List[Dict], output_file: str, batch_size: int = 50):
    """Save rules in batches, persisting to disk every batch_size rules."""
    if not rules:
        return

    try:
        # The first batch overwrites; subsequent batches append
        first_batch = True

        for i in range(0, len(rules), batch_size):
            batch = rules[i:i + batch_size]
            save_rules(batch, output_file, append=not first_batch)
            first_batch = False

        logger.info(f"Completed batch saving of {len(rules)} rules")

    except Exception as e:
        logger.error(f"Error in batch saving: {e}")


def main():
    parser = argparse.ArgumentParser(description='Property formatting rule generation script')

    # LLM connection parameters
    parser.add_argument('--base_url', default='http://localhost:8000/v1',
                        help='OpenAI API base URL')
    parser.add_argument('--api_key', default='your-api-key-here',
                        help='OpenAI API key')
    parser.add_argument('--model', default='Qwen2.5-7B-Instruct',
                        help='Model name')

    # Input/output parameters
    parser.add_argument('--wiki_properties_file', default='data/processed/properties/wiki_properties.jsonl',
                        help='Input wiki properties file')
    parser.add_argument('--output_file', default='honest/qgen/property_format_rules_generated.jsonl',
                        help='Output rules file')
    parser.add_argument('--existing_rules_file', default='honest/qgen/property_format_rules.jsonl',
                        help='Existing rules file (used to skip already-processed properties)')

    # Processing parameters
    parser.add_argument('--max_properties', type=int, default=None,
                        help='Maximum number of properties to process (for testing)')
    parser.add_argument('--start_from', type=int, default=0,
                        help='Start processing from the Nth property')
    parser.add_argument('--use_fallback', action='store_true',
                        help='Use fallback rules for properties where LLM generation failed')
    parser.add_argument('--batch_save_size', type=int, default=50,
                        help='Persist to disk every N properties (to prevent data loss)')

    args = parser.parse_args()

    logger.info("=== Property formatting rule generation started ===")
    logger.info(f"Base URL: {args.base_url}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Wiki properties file: {args.wiki_properties_file}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Max properties: {args.max_properties}")
    logger.info(f"Start from: {args.start_from}")

    # Check the input file
    if not os.path.exists(args.wiki_properties_file):
        logger.error(f"Wiki properties file not found: {args.wiki_properties_file}")
        return

    # Initialize the LLM client
    logger.info("Initializing LLM client...")
    llm_client = SyncLLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model
    )

    # Test the connection
    test_response = llm_client.generate_answer("Hello, please respond with 'OK' if you can hear me.")
    if test_response.startswith("ERROR:"):
        logger.error(f"Cannot connect to LLM: {test_response}")
        return
    logger.info(f"LLM connected successfully, test answer: {test_response}")

    # Initialize the rule generator
    generator = PropertyRuleGenerator(llm_client)

    # Load existing rules
    logger.info("Loading existing rules...")
    existing_rules = load_existing_rules(args.existing_rules_file)
    logger.info(f"Already have {len(existing_rules)} existing rules")

    # Load all properties
    logger.info("Loading property list...")
    properties = []
    try:
        with open(args.wiki_properties_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    prop = json.loads(line)
                    properties.append(prop)
    except Exception as e:
        logger.error(f"Error loading properties: {e}")
        return

    logger.info(f"Total of {len(properties)} properties")

    # Filter and slice the processing range
    if args.start_from > 0:
        properties = properties[args.start_from:]
        logger.info(f"Starting from the {args.start_from}th property")

    if args.max_properties:
        properties = properties[:args.max_properties]
        logger.info(f"Limited to processing the first {args.max_properties} properties")

    # Generate rules
    generated_rules = []
    failed_properties = []
    processed_count = 0
    saved_file_initialized = False

    for prop in tqdm(properties, desc="Generating rules"):
        property_id = prop['property_id']
        label = prop['label']

        # Skip properties that already have a rule
        if property_id in existing_rules:
            logger.debug(f"Skipping property with an existing rule: {property_id}")
            generated_rules.append(existing_rules[property_id])
        else:
            # Generate a new rule
            logger.debug(f"Generating rule for property {property_id} ({label})...")
            rule = generator.generate_rule(property_id, label)

            if rule:
                generated_rules.append(rule)
                logger.debug(f"Rule generated successfully: {property_id}")
            else:
                failed_properties.append((property_id, label))
                logger.warning(f"Rule generation failed: {property_id} ({label})")

                # If fallback rules are enabled
                if args.use_fallback:
                    fallback_rule = generator.generate_fallback_rule(property_id, label)
                    generated_rules.append(fallback_rule)
                    logger.info(f"Using fallback rule: {property_id}")

        processed_count += 1

        # Batch-save check
        if processed_count % args.batch_save_size == 0:
            if generated_rules:
                # Overwrite on the first save; append afterwards
                save_rules(generated_rules, args.output_file, append=saved_file_initialized)
                if not saved_file_initialized:
                    saved_file_initialized = True
                logger.info(f"Processed {processed_count}/{len(properties)} properties; batch save complete")
                generated_rules = []  # Clear the saved rules

    # Save the remaining rules (if any)
    if generated_rules:
        save_rules(generated_rules, args.output_file, append=saved_file_initialized)
        logger.info(f"Saved the last {len(generated_rules)} rules")

    # Statistics
    total_processed = processed_count
    logger.info("=== Generation complete ===")
    logger.info(f"Total properties processed: {total_processed}")
    logger.info(f"Generation failures: {len(failed_properties)}")
    logger.info(f"Total LLM requests: {llm_client.request_count}")
    logger.info(f"Batch save setting: save every {args.batch_save_size} properties")

    if failed_properties:
        logger.info("Failed properties:")
        for prop_id, label in failed_properties[:10]:  # Show only the first 10
            logger.info(f"  {prop_id}: {label}")
        if len(failed_properties) > 10:
            logger.info(f"  ... and {len(failed_properties) - 10} more")


if __name__ == '__main__':
    main()