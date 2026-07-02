#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grammar check script - test version
Traverse the CSV files under the test_questions directory and check original_question and mutated_question for grammar errors.
Only processes the first few files for testing.
"""

import os
import csv
import pandas as pd
import time
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GrammarChecker:
    """Grammar checker"""

    def __init__(self, base_url: str = 'http://localhost:8000/v1',
                 api_key: str = 'your-api-key-here',
                 model_name: str = 'Qwen2.5-7B-Instruct'):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = model_name
        self.request_count = 0

    def check_grammar(self, text: str) -> Dict[str, str]:
        """Check the text for grammar errors."""
        try:
            self.request_count += 1

            prompt = f"""Please check the grammar of the following text and identify any grammatical errors.

Text: "{text}"

Instructions:
1. If there are NO grammatical errors, respond with exactly: "NO_ERRORS"
2. If there ARE grammatical errors, provide:
   - ERROR_REASON: Brief explanation of the grammatical errors found
   - CORRECTED_TEXT: The corrected version of the text
3. 如果句子有逻辑错误，有不合理的推理，请不要将其视为语法错误，你仍然返回 "NO_ERRORS"

Format your response as:
ERROR_REASON: [your explanation]
CORRECTED_TEXT: [corrected text]

OR simply:
NO_ERRORS

Text to check: {text}"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.1,
                stop=None
            )

            # Add an interval between requests
            time.sleep(0.1)

            result_text = response.choices[0].message.content.strip()

            if result_text == "NO_ERRORS":
                return {
                    "has_error": False,
                    "error_reason": "",
                    "corrected_text": ""
                }
            else:
                # Parse the error message
                lines = result_text.split('\n')
                error_reason = ""
                corrected_text = ""

                for line in lines:
                    if line.startswith("ERROR_REASON:"):
                        error_reason = line.replace("ERROR_REASON:", "").strip()
                    elif line.startswith("CORRECTED_TEXT:"):
                        corrected_text = line.replace("CORRECTED_TEXT:", "").strip()
                
                if "logical" in error_reason.lower() or "logic" in error_reason.lower() or "redundant" in error_reason.lower() or "repetitive" in error_reason.lower():
                    return {
                        "has_error": False,
                        "error_reason": error_reason if error_reason else result_text,
                        "corrected_text": corrected_text if corrected_text else "N/A"
                    }

                return {
                    "has_error": True,
                    "error_reason": error_reason if error_reason else result_text,
                    "corrected_text": corrected_text if corrected_text else "N/A"
                }

        except Exception as e:
            logger.error(f"Error checking grammar: {e}")
            return {
                "has_error": False,
                "error_reason": f"API Error: {str(e)}",
                "corrected_text": ""
            }


def process_csv_file(csv_file_path: str, grammar_checker: GrammarChecker, output_file: str, error_count_ref: List[int], max_rows: int = 5) -> List[Dict]:
    """Process a single CSV file: check for grammar errors and write results in real time."""
    logger.info(f"Processing file: {csv_file_path} (max {max_rows} rows)")

    errors_found = []

    try:
        df = pd.read_csv(csv_file_path)
        # Only process the first few rows
        df = df.head(max_rows)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Checking {os.path.basename(csv_file_path)}"):
            # Check original_question
            original_question = row.get('original_question', '').strip()
            if original_question:
                original_result = grammar_checker.check_grammar(original_question)
                if original_result["has_error"]:
                    error_count_ref[0] += 1
                    error = {
                        "file_path": csv_file_path,
                        "row_index": idx,
                        "question_type": "original_question",
                        "question_text": original_question,
                        "error_reason": original_result["error_reason"],
                        "corrected_text": original_result["corrected_text"]
                    }
                    errors_found.append(error)
                    # Append to file in real time
                    append_grammar_error(error, output_file, error_count_ref[0])

            # Check mutated_question
            mutated_question = row.get('mutated_question', '').strip()
            if mutated_question:
                mutated_result = grammar_checker.check_grammar(mutated_question)
                if mutated_result["has_error"]:
                    error_count_ref[0] += 1
                    error = {
                        "file_path": csv_file_path,
                        "row_index": idx,
                        "question_type": "mutated_question",
                        "question_text": mutated_question,
                        "error_reason": mutated_result["error_reason"],
                        "corrected_text": mutated_result["corrected_text"]
                    }
                    errors_found.append(error)
                    # Append to file in real time
                    append_grammar_error(error, output_file, error_count_ref[0])

    except Exception as e:
        logger.error(f"Error processing file {csv_file_path}: {e}")

    return errors_found


def init_output_file(output_file: str):
    """Initialize the output file."""
    try:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Grammar Errors Report\n")
            f.write("=" * 50 + "\n\n")
        logger.info(f"Initialized output file: {output_file}")
    except Exception as e:
        logger.error(f"Error initializing output file: {e}")


def append_grammar_error(error: Dict, output_file: str, error_count: int):
    """Append a grammar error to the file in real time."""
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"Error #{error_count}\n")
            f.write(f"File: {error['file_path']}\n")
            f.write(f"Row Index: {error['row_index']}\n")
            f.write(f"Question Type: {error['question_type']}\n")
            f.write(f"Original Text: {error['question_text']}\n")
            f.write(f"Error Reason: {error['error_reason']}\n")
            f.write(f"Corrected Text: {error['corrected_text']}\n")
            f.write("-" * 40 + "\n\n")
            f.flush()  # Force flush to disk
        logger.info(f"Appended error #{error_count} to {output_file}")
    except Exception as e:
        logger.error(f"Error appending to output file: {e}")


def save_grammar_errors(errors: List[Dict], output_file: str):
    """Save grammar errors to a TXT file (kept for future use)."""
    try:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Grammar Errors Report\n")
            f.write("=" * 50 + "\n\n")

            if not errors:
                f.write("No grammar errors found in any questions.\n")
                return

            for i, error in enumerate(errors, 1):
                f.write(f"Error #{i}\n")
                f.write(f"File: {error['file_path']}\n")
                f.write(f"Row Index: {error['row_index']}\n")
                f.write(f"Question Type: {error['question_type']}\n")
                f.write(f"Original Text: {error['question_text']}\n")
                f.write(f"Error Reason: {error['error_reason']}\n")
                f.write(f"Corrected Text: {error['corrected_text']}\n")
                f.write("-" * 40 + "\n\n")

        logger.info(f"Grammar errors saved to {output_file}")

    except Exception as e:
        logger.error(f"Error saving grammar errors: {e}")


def main():
    """Main function."""
    # Use the default configuration from qa_test.py
    base_url = 'http://localhost:8000/v1'
    api_key = 'your-api-key-here'
    model_name = 'Qwen2.5-7B-Instruct'
    test_questions_dir = 'data/examples/questions'
    output_file = 'grammar_errors_report_test.txt'
    max_files = 1000  # Number of files to process
    max_rows_per_file = 10  # Number of rows to process per file

    logger.info("=== Grammar Check Test Started ===")
    logger.info(f"Base URL: {base_url}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Test questions dir: {test_questions_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Max files: {max_files}")
    logger.info(f"Max rows per file: {max_rows_per_file}")

    # Initialize the grammar checker
    grammar_checker = GrammarChecker(base_url, api_key, model_name)

    # Test the LLM connection
    logger.info("Testing LLM connection...")
    test_result = grammar_checker.check_grammar("This is a test sentence.")
    if test_result.get("error_reason", "").startswith("API Error"):
        logger.error("Cannot connect to LLM, please check configuration")
        return
    logger.info("LLM connection successful")

    # Collect all CSV files and sort them by the numeric order of directory names
    csv_files = []
    for root, _, files in os.walk(test_questions_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    # Sort by the numeric order of rule IDs
    def extract_rule_number(path):
        """Extract the rule number from the path for sorting."""
        try:
            # Extract the part like memgraph_rule_88 from the path
            parts = path.split('/')
            for part in parts:
                if part.startswith('memgraph_rule_'):
                    rule_num = int(part.replace('memgraph_rule_', ''))
                    return rule_num
            return 99999  # If no rule number is found, put it at the end
        except:
            return 99999

    csv_files.sort(key=extract_rule_number)

    logger.info(f"Found {len(csv_files)} CSV files, processing first {max_files}")
    csv_files = csv_files[:max_files]

    # Initialize the output file
    init_output_file(output_file)

    # Process all files, writing errors in real time
    all_errors = []
    error_count = [0]  # Pass-by-reference via a list to track the global error count

    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            errors = process_csv_file(csv_file, grammar_checker, output_file, error_count, max_rows_per_file)
            all_errors.extend(errors)
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            continue

    # If there are no errors, append a "no errors" message
    if not all_errors:
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write("No grammar errors found in any questions.\n")
        except Exception as e:
            logger.error(f"Error appending no-error message: {e}")

    # Statistics
    logger.info("=== Grammar Check Test Completed ===")
    logger.info(f"Total files processed: {len(csv_files)}")
    logger.info(f"Total grammar errors found: {len(all_errors)}")
    logger.info(f"Total LLM requests: {grammar_checker.request_count}")

    if all_errors:
        # Count errors per file
        file_error_count = {}
        for error in all_errors:
            file_path = error['file_path']
            if file_path not in file_error_count:
                file_error_count[file_path] = 0
            file_error_count[file_path] += 1

        logger.info("Errors by file:")
        for file_path, count in sorted(file_error_count.items()):
            logger.info(f"  {file_path}: {count} errors")
    else:
        logger.info("No grammar errors found in the test sample")


if __name__ == '__main__':
    main()