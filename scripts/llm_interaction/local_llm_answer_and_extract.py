#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local LLM Answer Generation Script

This script:
1. Reads questions (both original and mutated) from CSV files
2. Generates answers using a local LLM model (HuggingFace or checkpoint)
3. Outputs CSV files with LLM raw answers for downstream analysis

Unlike llm_answer_and_extract.py which uses API calls, this script:
- Loads the model directly onto GPU
- Supports HuggingFace model names or local checkpoint paths
- Supports LoRA adapters
- Uses transformers library for inference

Error Handling:
- Errors are logged but allow processing to continue
- Use --resume flag to continue from where it stopped
"""

import os
import sys
import argparse
import traceback

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import torch
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
from pathlib import Path

# Import configuration management
from configs import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Local LLM Answer Generation Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration arguments
    parser.add_argument('--model', type=str, required=True,
                       help='HuggingFace model name or local checkpoint path')
    parser.add_argument('--lora-adapter', type=str, default=None,
                       help='Path to LoRA adapter (optional)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--load-in-8bit', action='store_true',
                       help='Load model in 8-bit mode (requires bitsandbytes)')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='Load model in 4-bit mode (requires bitsandbytes)')
    parser.add_argument('--trust-remote-code', action='store_true',
                       help='Trust remote code when loading model')

    # Processing arguments
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Input directory containing CSV files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of CSV files to process')
    parser.add_argument('--max-rows', type=int, default=None,
                       help='Maximum number of rows to process per file')
    parser.add_argument('--rules', type=str, default=None,
                       help='Comma-separated list of rule numbers to process')
    parser.add_argument('--summary-file', type=str, default='conclusion_extraction_summary.txt',
                       help='Summary report file name')

    # LLM parameters
    parser.add_argument('--max-tokens', type=int, default=2048,
                       help='Maximum tokens for LLM response')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for LLM generation')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p sampling parameter')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference (for processing multiple questions at once)')

    # Checkpoint/Resume
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous interrupted run')
    parser.add_argument('--retry-errors', action='store_true', default=True,
                       help='Retry rows with ERROR answers when resuming')

    return parser.parse_args()


class LocalLLMAnswerGenerator:
    """Local LLM Answer Generator - Direct model loading and inference"""

    def __init__(
        self,
        model_name_or_path: str,
        lora_adapter_path: Optional[str] = None,
        device: str = 'auto',
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ):
        """
        Initialize local LLM answer generator

        Args:
            model_name_or_path: HuggingFace model name or local checkpoint path
            lora_adapter_path: Path to LoRA adapter (optional)
            device: Device to run on ('auto', 'cuda', 'cpu')
            load_in_8bit: Load in 8-bit mode
            load_in_4bit: Load in 4-bit mode
            trust_remote_code: Trust remote code
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        self.model_name_or_path = model_name_or_path
        self.lora_adapter_path = lora_adapter_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.request_count = 0

        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self._load_model(load_in_8bit, load_in_4bit, trust_remote_code)

    def _load_model(self, load_in_8bit: bool, load_in_4bit: bool, trust_remote_code: bool):
        """Load model and tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model from: {self.model_name_or_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=trust_remote_code
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = "[PAD]"

        # Prepare model loading arguments
        model_kwargs = {
            'trust_remote_code': trust_remote_code,
            'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32,
        }

        if self.device == 'cuda':
            model_kwargs['device_map'] = 'auto'

        if load_in_8bit:
            model_kwargs['load_in_8bit'] = True
        elif load_in_4bit:
            model_kwargs['load_in_4bit'] = True

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **model_kwargs
        )

        # Load LoRA adapter if provided
        if self.lora_adapter_path:
            logger.info(f"Loading LoRA adapter from: {self.lora_adapter_path}")
            from peft import PeftModel

            # Check if it's a LoRA checkpoint or base model with adapter
            if os.path.exists(os.path.join(self.lora_adapter_path, 'adapter_config.json')):
                # It's a pure LoRA adapter
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_adapter_path,
                    is_trainable=False
                )
            else:
                # It might be a merged checkpoint or base model path
                # Try to load as a fine-tuned model
                logger.info("Attempting to load as fine-tuned model...")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.lora_adapter_path,
                        **model_kwargs
                    )
                except Exception as e:
                    logger.warning(f"Could not load as fine-tuned model: {e}")
                    logger.info("Trying as LoRA adapter...")
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        self.lora_adapter_path,
                        is_trainable=False
                    )

        # Set model to eval mode
        self.model.eval()

        logger.info("Model loaded successfully")

    def _format_messages(self, question: str) -> str:
        """Format question into prompt for the model"""
        system_prompt = """You are a helpful assistant that answers questions thoughtfully and thoroughly.
Think step by step and explain your reasoning before providing your final answer.
You can organize your response in any way that makes sense to you."""

        user_prompt = f"""{question}

Please think through this question carefully. You can:
- Break down the problem into steps
- Consider different aspects or perspectives
- Explain your reasoning process
- Then provide your conclusion

Take your time and think it through."""

        # Use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to simple format
            prompt = f"{system_prompt}\n\n{user_prompt}"

        return prompt

    @torch.no_grad()
    def generate_answer(self, question: str) -> str:
        """
        Generate answer for the given question

        Args:
            question: The question to answer

        Returns:
            The answer or error message (starts with "ERROR:")
        """
        try:
            self.request_count += 1

            # Format prompt
            prompt = self._format_messages(question)

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=4096
            ).to(self.device)

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Decode response
            # Only decode the generated part (not the prompt)
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            traceback.print_exc()
            return f"ERROR: {str(e)}"

    def generate_batch(self, questions: List[str]) -> List[str]:
        """
        Generate answers for multiple questions in batch

        Args:
            questions: List of questions

        Returns:
            List of answers
        """
        answers = []
        for question in questions:
            answer = self.generate_answer(question)
            answers.append(answer)
        return answers


def process_single_row(row_idx: int, row: pd.Series,
                      answer_generator: LocalLLMAnswerGenerator,
                      stats: Dict) -> Optional[Dict]:
    """Process a single row

    Returns None if questions are empty, otherwise returns result dict
    """
    # Get original question - safely handle NaN and None values
    original_question_raw = row.get('original_question', '')
    if pd.isna(original_question_raw) or original_question_raw is None:
        original_question = ''
    else:
        original_question = str(original_question_raw).strip()

    # Get mutated question - safely handle NaN and None values
    mutated_question_raw = row.get('mutated_question', '')
    if pd.isna(mutated_question_raw) or mutated_question_raw is None:
        mutated_question = ''
    else:
        mutated_question = str(mutated_question_raw).strip()

    if not original_question or not mutated_question:
        logger.warning(f"Empty question at row {row_idx}")
        return None

    stats["total"] += 1

    # Generate ORIGINAL and MUTATED answers
    original_llm_answer = answer_generator.generate_answer(original_question)
    mutated_llm_answer = answer_generator.generate_answer(mutated_question)

    # Check for ERROR responses
    if isinstance(original_llm_answer, str) and original_llm_answer.startswith("ERROR:"):
        stats["errors"] += 1
    if isinstance(mutated_llm_answer, str) and mutated_llm_answer.startswith("ERROR:"):
        stats["errors"] += 1

    # Build result record
    result = {
        **row.to_dict(),
        'original_llm_answer': original_llm_answer,
        'mutated_llm_answer': mutated_llm_answer,
    }

    return result


def process_csv_file(csv_file_path: str, answer_generator: LocalLLMAnswerGenerator,
                    input_dir: str, output_dir: str,
                    max_rows: int = None, resume: bool = False,
                    retry_errors: bool = True) -> Dict:
    """Process a single CSV file and generate LLM answers"""
    logger.info(f"Processing file: {csv_file_path}")

    # Calculate relative path from input directory
    relative_path = os.path.relpath(csv_file_path, input_dir)
    relative_dir = os.path.dirname(relative_path)
    filename = os.path.basename(csv_file_path).replace('.csv', '_llm_answers.csv')

    # Create output directory structure
    output_subdir = os.path.join(output_dir, relative_dir)
    os.makedirs(output_subdir, exist_ok=True)

    output_file = os.path.join(output_subdir, filename)

    # Check for existing results
    processed_rows = set()
    if os.path.exists(output_file):
        if not resume:
            logger.info(f"Results already exist for {csv_file_path}, skipping...")
            return {"total": 0, "errors": 0}
        else:
            try:
                existing_df = pd.read_csv(output_file)
                retry_count = 0
                for _, row in existing_df.iterrows():
                    key = f"{row.get('original_question', '')}||{row.get('mutated_question', '')}"

                    if retry_errors:
                        original_ans = str(row.get('original_llm_answer', ''))
                        mutated_ans = str(row.get('mutated_llm_answer', ''))
                        has_error = original_ans.startswith('ERROR:') or mutated_ans.startswith('ERROR:')
                        if has_error:
                            retry_count += 1
                            continue

                    processed_rows.add(key)

                logger.info(f"Resuming from checkpoint: {len(processed_rows)} rows processed")
                if retry_count > 0:
                    logger.info(f"Will retry {retry_count} rows with ERROR answers")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
                processed_rows = set()

    stats = {"total": 0, "errors": 0}

    try:
        df = pd.read_csv(csv_file_path)

        if max_rows:
            df = df.head(max_rows)

        # Filter out already processed rows
        rows_to_process = []
        for idx, row in df.iterrows():
            key = f"{row.get('original_question', '')}||{row.get('mutated_question', '')}"
            if key not in processed_rows:
                rows_to_process.append((idx, row))

        if not rows_to_process:
            logger.info(f"All rows already processed for {csv_file_path}")
            return {"total": 0, "errors": 0}

        logger.info(f"Processing {len(rows_to_process)} rows...")

        results = []
        for idx, row in tqdm(rows_to_process, desc=f"Processing {os.path.basename(csv_file_path)}"):
            result = process_single_row(idx, row, answer_generator, stats)
            if result is not None:
                results.append(result)

                # Incremental save every 10 rows
                if len(results) % 10 == 0:
                    save_results_checkpoint(results, output_file, processed_rows, resume)

        # Final save
        if results:
            save_results_checkpoint(results, output_file, processed_rows, resume)

        logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error processing file {csv_file_path}: {e}")
        traceback.print_exc()

    return stats


def save_results_checkpoint(results: List[Dict], output_file: str,
                          processed_rows: set, resume: bool):
    """Save results checkpoint incrementally"""
    try:
        df_new = pd.DataFrame(results)

        if resume and os.path.exists(output_file):
            df_existing = pd.read_csv(output_file)

            new_data_dict = {}
            for _, row in df_new.iterrows():
                key = str(row.get('original_question', '')) + '||' + str(row.get('mutated_question', ''))
                new_data_dict[key] = row.to_dict()

            updated_rows = []
            existing_keys = set()
            for _, row in df_existing.iterrows():
                key = str(row.get('original_question', '')) + '||' + str(row.get('mutated_question', ''))
                existing_keys.add(key)
                if key in new_data_dict:
                    updated_rows.append(new_data_dict[key])
                else:
                    updated_rows.append(row.to_dict())

            for key, row_dict in new_data_dict.items():
                if key not in existing_keys:
                    updated_rows.append(row_dict)

            df_combined = pd.DataFrame(updated_rows)
        else:
            df_combined = df_new

        # Reorder columns
        llm_columns = ['original_llm_answer', 'mutated_llm_answer']
        other_columns = [col for col in df_combined.columns if col not in llm_columns]
        ordered_columns = llm_columns + other_columns
        df_combined = df_combined[ordered_columns]

        df_combined.to_csv(output_file, index=False, encoding='utf-8')

    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def save_summary_report(all_stats: List[Dict], output_file: str):
    """Save summary report"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Local LLM Answer Generation Report\n")
            f.write("=" * 80 + "\n\n")

            total_questions = sum(s["total"] for s in all_stats)
            total_errors = sum(s["errors"] for s in all_stats)

            f.write(f"Total Question Pairs Processed: {total_questions}\n")
            f.write(f"  - Original questions: {total_questions}\n")
            f.write(f"  - Mutated questions: {total_questions}\n")
            f.write(f"  - Total LLM requests: {total_questions * 2}\n\n")

            f.write(f"Errors: {total_errors}\n")
            if total_questions > 0:
                error_rate = (total_errors / (total_questions * 2)) * 100
                f.write(f"Error Rate: {error_rate:.2f}%\n")
            f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("Per-File Statistics:\n")
            f.write("-" * 80 + "\n\n")

            for i, stats in enumerate(all_stats, 1):
                f.write(f"File {i}:\n")
                f.write(f"  Question Pairs: {stats['total']}\n")
                f.write(f"  LLM Requests: {stats['total'] * 2}\n")
                f.write(f"  Errors: {stats['errors']}\n")
                f.write("\n")

        logger.info(f"Summary report saved to {output_file}")

    except Exception as e:
        logger.error(f"Error saving summary report: {e}")


def main():
    """Main function"""
    args = parse_arguments()

    # Path configuration
    test_questions_dir = args.input_dir or get_config('paths.examples.questions', 'data/examples/questions')
    output_dir = args.output_dir or 'data/examples/local_llm_answers'
    summary_report_file = args.summary_file
    max_files = args.max_files
    max_rows_per_file = args.max_rows

    logger.info("=" * 80)
    logger.info("Local LLM Answer Generation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"LoRA adapter: {args.lora_adapter or 'None'}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Test questions dir: {test_questions_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max files: {max_files}")
    logger.info(f"Max rows per file: {max_rows_per_file}")
    logger.info(f"Rules filter: {args.rules if args.rules else 'All rules'}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Load in 8-bit: {args.load_in_8bit}")
    logger.info(f"Load in 4-bit: {args.load_in_4bit}")
    logger.info(f"Resume mode: {args.resume}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize local LLM answer generator
    answer_generator = LocalLLMAnswerGenerator(
        model_name_or_path=args.model,
        lora_adapter_path=args.lora_adapter,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    try:
        # Test model
        logger.info("Testing model...")
        test_answer = answer_generator.generate_answer("What is 2+2?")
        if test_answer.startswith("ERROR:"):
            logger.error(f"Model test failed: {test_answer}")
            return
        logger.info(f"Model test successful, test answer: {test_answer[:100]}...")

        # Collect CSV files
        csv_files = []
        for root, _, files in os.walk(test_questions_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))

        # Sort by rule number
        def extract_rule_number(path):
            try:
                parts = path.split('/')
                for part in parts:
                    if part.startswith('kg_rule_'):
                        rule_num = int(part.replace('kg_rule_', ''))
                        return rule_num
                    elif part.startswith('memgraph_rule_'):
                        rule_num = int(part.replace('memgraph_rule_', ''))
                        return rule_num
                return 99999
            except:
                return 99999

        csv_files.sort(key=extract_rule_number)

        # Filter by specified rules
        if args.rules:
            try:
                specified_rules = set(int(r.strip()) for r in args.rules.split(','))
                logger.info(f"Filtering by specified rules: {sorted(specified_rules)}")

                filtered_files = []
                for csv_file in csv_files:
                    rule_num = extract_rule_number(csv_file)
                    if rule_num in specified_rules:
                        filtered_files.append(csv_file)

                csv_files = filtered_files
                logger.info(f"After filtering: {len(csv_files)} CSV files match")
            except ValueError as e:
                logger.error(f"Invalid --rules parameter: {args.rules}")
                return

        if max_files is not None:
            logger.info(f"Found {len(csv_files)} CSV files, processing first {max_files}")
            csv_files = csv_files[:max_files]
        else:
            logger.info(f"Found {len(csv_files)} CSV files, processing all files")

        # Process all files
        all_stats = []
        for csv_file in csv_files:
            stats = process_csv_file(csv_file, answer_generator,
                                   test_questions_dir, output_dir, max_rows_per_file,
                                   resume=args.resume, retry_errors=args.retry_errors)
            all_stats.append(stats)

        # Save summary report
        save_summary_report(all_stats, summary_report_file)

        # Final statistics
        logger.info("=" * 80)
        logger.info("Processing Completed")
        logger.info("=" * 80)
        total_questions = sum(s["total"] for s in all_stats)
        total_errors = sum(s["errors"] for s in all_stats)

        logger.info(f"Total question pairs processed: {total_questions}")
        logger.info(f"Total LLM requests: {answer_generator.request_count}")
        logger.info(f"Total errors: {total_errors}")

        if answer_generator.request_count > 0:
            error_rate = (total_errors / answer_generator.request_count) * 100
            logger.info(f"Error rate: {error_rate:.2f}%")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
