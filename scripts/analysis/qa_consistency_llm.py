#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-based QA consistency checking script.
Use LLM-as-judge to check the consistency of LLM answers to original and mutated questions.

Workflow:
1. Extract the conclusion from the original answer
2. Extract the conclusion from the mutated answer
3. Use an LLM to judge whether the two conclusions are consistent
"""

import os
import sys
import argparse
import pandas as pd
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from honest.llm import SyncLLMClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QAConsistencyChecker:
    """QA consistency checker, wrapping SyncLLMClient."""

    def __init__(self, llm_client: SyncLLMClient):
        self.llm_client = llm_client

    @property
    def request_count(self) -> int:
        """Return the total number of requests (kept for backward compatibility with the old interface)."""
        return self.llm_client.request_count

    def _safe_json_parse(self, json_str: str) -> Dict:
        """Safely parse JSON, handling control character issues."""
        try:
            # Try normal parsing first
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # If it fails, try fixing control characters
            def escape_control_chars(match):
                string_content = match.group(1)
                # Escape control characters
                escaped = string_content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                escaped = escaped.replace('\b', '\\b').replace('\f', '\\f')
                return f'"{escaped}"'

            # Match JSON string values
            fixed_json = re.sub(r'"([^"]*)"', escape_control_chars, json_str)

            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError as e2:
                # If it still fails, try a simpler approach: manual extraction
                match = re.search(r'"extracted_conclusion"\s*:\s*"([^"]*)"', json_str)
                if match:
                    conclusion_text = match.group(1)
                    # Escape control characters
                    conclusion_text = conclusion_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                    return {"extracted_conclusion": conclusion_text}
                else:
                    # If it still fails, return a dict containing the raw string
                    logger.warning(f"JSON parsing failed even after fixing control characters: {e2}")
                    return {"extracted_conclusion": json_str}

    def _generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> Optional[str]:
        """Generate a response (kept for backward compatibility with the old interface)."""
        result = self.llm_client.generate_answer(
            prompt,
            max_retries=3,
            user_prompt=prompt,
            system_prompt="You are a helpful assistant that follows instructions precisely.",
            max_tokens=max_tokens,
            temperature=temperature
        )
        if result.startswith("ERROR:"):
            return None
        return result

    def extract_conclusion(self, answer: str) -> str:
        """Extract the conclusion paragraph from the answer; the original text must not be modified."""
        try:
            extraction_prompt = f"""Extract ONLY the conclusion/final answer from the following text. Do NOT modify any words, do NOT paraphrase, do NOT add explanations. Just extract the exact conclusion text.

Text: {answer}

IMPORTANT RULES:
1. Extract ONLY the conclusion/final answer part
2. Do NOT change any words, keep the exact original text
3. Do NOT add any explanations or reasoning
4. If the text contains reasoning steps, extract only the final conclusion
5. Return the extracted text as-is without any modifications
6. **CRITICAL: Ensure the JSON is valid - escape all control characters like \\n, \\r, \\t in the string values**

Return the extracted conclusion in JSON format:
{{
  "extracted_conclusion": "[exact conclusion text here]"
}}

Make sure to wrap the JSON in a markdown code block and ensure it's valid JSON."""

            response = self._generate_response(extraction_prompt, max_tokens=512, temperature=0.0)

            if response is None:
                return answer  # If extraction fails, return the original answer

            # Extract JSON from the markdown code block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    extracted_data = self._safe_json_parse(json_match.group(1))
                    conclusion = extracted_data.get('extracted_conclusion', answer)
                    return conclusion
                except Exception:
                    logger.warning(f"Failed to parse JSON from extraction response: {response}")
                    return answer
            else:
                # If no JSON is found, try direct extraction
                if '"extracted_conclusion"' in response:
                    # Try to extract from unformatted JSON
                    match = re.search(r'"extracted_conclusion"\s*:\s*"([^"]*)"', response)
                    if match:
                        return match.group(1)
                return answer

        except Exception as e:
            logger.error(f"Error extracting conclusion: {e}")
            return answer

    def check_consistency(self, answer1: str, answer2: str, question_type: str = "unknown") -> Tuple[bool, str, str, str]:
        """Use an LLM to check the consistency of two answers (two-stage method).

        Returns:
            Tuple[bool, str, str, str]: (is_consistent, explanation, extracted_conclusion_1, extracted_conclusion_2)
        """
        try:
            # Stage 1: extract conclusions
            extracted_answer1 = self.extract_conclusion(answer1)
            extracted_answer2 = self.extract_conclusion(answer2)

            # Stage 2: judge consistency
            consistency_prompt = f"""Compare these two extracted conclusions to determine if they are consistent:

Extracted Conclusion 1: {extracted_answer1}

Extracted Conclusion 2: {extracted_answer2}

Question Type: {question_type}

IMPORTANT RULES:
- For YES/NO or True/False questions: Check if both conclusions have the same yes/no judgment
- For multiple-choice questions: Check if the selected options are identical
- For other question types: Check if the core meaning and entities are consistent
- Minor wording differences don't affect consistency
- Focus on the final conclusion, not the reasoning process

Return your analysis in JSON format:
{{
  "consistent": true/false,
  "explanation": "Brief explanation of why they are consistent or inconsistent",
  "question_type": "{question_type}",
  "extracted_answer1": "{extracted_answer1}",
  "extracted_answer2": "{extracted_answer2}"
}}

Make sure to wrap the JSON in a markdown code block."""

            response = self._generate_response(consistency_prompt, max_tokens=512, temperature=0.0)

            if response is None:
                return False, "Failed to check consistency", extracted_answer1, extracted_answer2

            # Extract JSON from the markdown code block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    consistency_data = self._safe_json_parse(json_match.group(1))
                    consistent = consistency_data.get('consistent', False)
                    explanation = consistency_data.get('explanation', 'No explanation provided')
                    return consistent, explanation, extracted_answer1, extracted_answer2
                except Exception:
                    logger.warning(f"Failed to parse JSON from consistency response: {response}")
                    # Fall back to the old method
                    fallback_consistent, fallback_explanation = self._fallback_consistency_check(response)
                    return fallback_consistent, fallback_explanation, extracted_answer1, extracted_answer2
            else:
                # Fall back to the old method
                fallback_consistent, fallback_explanation = self._fallback_consistency_check(response)
                return fallback_consistent, fallback_explanation, extracted_answer1, extracted_answer2

        except Exception as e:
            logger.error(f"Error checking consistency: {e}")
            return False, f"Error: {str(e)}", answer1, answer2

    def _fallback_consistency_check(self, response: str) -> Tuple[bool, str]:
        """Fallback consistency-checking method."""
        response = response.strip()
        if response.startswith("YES"):
            return True, response[4:].strip(": ")
        elif response.startswith("NO"):
            return False, response[3:].strip(": ")
        else:
            # If the format is non-standard, make a simple judgment
            if "YES" in response.upper():
                return True, response
            else:
                return False, response


def process_csv_file(csv_file_path: str, llm_checker: QAConsistencyChecker) -> List[Dict]:
    """Process a single CSV file."""
    logger.info(f"Processing file: {csv_file_path}")

    try:
        df = pd.read_csv(csv_file_path)
        results = []

        total_rows = len(df)
        logger.info(f"Checking consistency for {total_rows} answer pairs")

        for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Checking consistency"):
            try:
                # Get the answers to the original and mutated questions
                original_answer = row.get('original_llm_answer', '').strip()
                mutated_answer = row.get('mutated_llm_answer', '').strip()

                # Try to get the question type
                question_type = row.get('question_type', 'unknown')
                if not question_type or question_type == 'unknown':
                    # Infer the type from the question content
                    question_text = row.get('question', '')
                    if any(word in question_text.lower() for word in ['yes', 'no', 'true', 'false']):
                        question_type = 'yes_no'
                    elif any(word in question_text.lower() for word in ['multiple', 'choice', 'option']):
                        question_type = 'multiple_choice'
                    else:
                        question_type = 'other'

                if not original_answer or not mutated_answer:
                    # If an answer is empty, skip the consistency check
                    result = {
                        **row.to_dict(),
                        'answers_consistent': False,
                        'consistency_explanation': 'Empty answer(s)',
                        'consistency_method': 'llm',
                        'question_type': question_type,
                        'extracted_original_conclusion': '',
                        'extracted_mutated_conclusion': ''
                    }
                    results.append(result)
                    continue

                # Check consistency
                consistent, explanation, extracted_original, extracted_mutated = llm_checker.check_consistency(
                    original_answer, mutated_answer, question_type
                )

                # Build the result record
                result = {
                    **row.to_dict(),
                    'answers_consistent': consistent,
                    'consistency_explanation': explanation,
                    'consistency_method': 'llm',
                    'question_type': question_type,
                    'extracted_original_conclusion': extracted_original,
                    'extracted_mutated_conclusion': extracted_mutated
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                # Append an error record
                error_result = {
                    **row.to_dict(),
                    'answers_consistent': False,
                    'consistency_explanation': f'Processing error: {str(e)}',
                    'consistency_method': 'error',
                    'extracted_original_conclusion': '',
                    'extracted_mutated_conclusion': ''
                }
                results.append(error_result)
                continue

        return results

    except Exception as e:
        logger.error(f"Error reading CSV file {csv_file_path}: {e}")
        return []


def save_results(results: List[Dict], output_file: str):
    """Save the consistency-check results."""
    if not results:
        logger.warning("No results to save")
        return

    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Create the DataFrame
        df = pd.DataFrame(results)

        # Reorder columns: put the consistency-check related fields first
        consistency_fields = ['answers_consistent', 'consistency_explanation', 'consistency_method',
                             'extracted_original_conclusion', 'extracted_mutated_conclusion',
                             'original_llm_answer', 'mutated_llm_answer']

        # Get the fields that actually exist
        existing_consistency_fields = [field for field in consistency_fields if field in df.columns]
        other_fields = sorted([col for col in df.columns if col not in consistency_fields])

        # Reorder columns
        ordered_columns = existing_consistency_fields + other_fields
        df = df[ordered_columns]

        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')

        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description='LLM-based QA consistency checking script')

    # LLM connection parameters
    parser.add_argument('--base_url', default='http://localhost:8001/v1',
                        help='OpenAI API base URL')
    parser.add_argument('--api_key', default='your-api-key-here',
                        help='OpenAI API key')
    parser.add_argument('--model', default='Qwen3-8B',
                        help='Model name')

    # Input/output parameters
    parser.add_argument('--answers_dir', default='qa_answers',
                        help='Directory containing LLM answers')
    parser.add_argument('--output_dir', default='consistency_results_llm',
                        help='Output results directory')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')

    args = parser.parse_args()

    logger.info("=== LLM-based QA consistency check started ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Base URL: {args.base_url}")
    logger.info(f"Answers dir: {args.answers_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the LLM client
    logger.info("Initializing LLM client...")
    llm_client = SyncLLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model
    )
    llm_checker = QAConsistencyChecker(llm_client)

    # Test the connection
    test_response = llm_client.generate_answer("Hello, are you working?")
    if test_response.startswith("ERROR:"):
        logger.error(f"Cannot connect to LLM: {test_response}")
        return
    else:
        logger.info(f"LLM connected successfully, test answer: {test_response[:100]}...")

    # Process all CSV files
    answer_files = []
    for root, _, files in os.walk(args.answers_dir):
        for file in files:
            if file.endswith('.csv'):
                answer_files.append(os.path.join(root, file))

    if args.max_files:
        answer_files = answer_files[:args.max_files]
        logger.info(f"Limited to processing the first {args.max_files} files")

    logger.info(f"Found {len(answer_files)} answer files")

    total_results = []
    processed_files = 0

    for file_path in tqdm(answer_files, desc="Processing files"):
        try:
            # Determine the output file path
            relative_path = os.path.relpath(file_path, args.answers_dir)
            output_file = os.path.join(args.output_dir, relative_path)

            # Check whether the output file already exists
            if os.path.exists(output_file):
                logger.info(f"Output file already exists, skipping: {output_file}")
                # Read the existing results and append them to the overall results
                try:
                    existing_df = pd.read_csv(output_file)
                    existing_results = existing_df.to_dict('records')
                    total_results.extend(existing_results)
                    processed_files += 1
                except Exception as e:
                    logger.warning(f"Failed to read existing file {output_file}: {e}, will reprocess")
                    # If reading fails, reprocess it
                    results = process_csv_file(file_path, llm_checker)
                    total_results.extend(results)
                    processed_files += 1
                    save_results(results, output_file)
                continue

            # Process the file
            results = process_csv_file(file_path, llm_checker)
            total_results.extend(results)
            processed_files += 1

            # Save results per file
            save_results(results, output_file)

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue

    # Save the aggregated results
    if total_results:
        summary_file = os.path.join(args.output_dir, 'all_consistency_summary.csv')
        save_results(total_results, summary_file)

        # Statistics
        total_checks = len(total_results)
        consistent_checks = sum(1 for r in total_results if r.get('answers_consistent', False))
        consistency_rate = consistent_checks / total_checks if total_checks > 0 else 0

        logger.info("=== Consistency check completed ===")
        logger.info(f"Files processed: {processed_files}")
        logger.info(f"Total checks: {total_checks}")
        logger.info(f"Consistent cases: {consistent_checks}")
        logger.info(f"Consistency rate: {consistency_rate:.2%}")
        logger.info(f"Total LLM requests: {llm_client.request_count}")

    else:
        logger.warning("No consistency-check results were generated")


if __name__ == '__main__':
    main()
