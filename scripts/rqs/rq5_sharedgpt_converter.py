#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ5 SharedGPT Format Converter

Convert the training dataset to SharedGPT format for LLaMA-Factory fine-tuning.

SharedGPT format is a JSONL format where each line is a conversation:
{
  "conversations": [
    {"from": "human", "value": "question"},
    {"from": "gpt", "value": "correct answer with reasoning"}
  ]
}

For fine-tuning to fix logical errors, we use the correct answers and
add system prompts that emphasize logical reasoning.

Usage:
    # Convert training set to SharedGPT format
    python scripts/rqs/rq5_sharedgpt_converter.py --input data/examples/rq5_results/splits/train/train.csv

    # Convert with custom output path
    python scripts/rqs/rq5_sharedgpt_converter.py --input train.csv --output train_sharedgpt.jsonl
"""

import os
import argparse
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any


def create_system_prompt(question_type: str, rule_text: str = "") -> str:
    """
    Create system prompt for the question type.

    The system prompt emphasizes logical reasoning and consistency.
    """
    base_prompt = """You are a careful and logical assistant. When answering questions:

1. Think step by step through the reasoning
2. Pay attention to all the information given in the question
3. Ensure your conclusion follows logically from the premises
4. Be consistent in your answers - equivalent questions should have equivalent answers
5. For yes/no questions, give a clear yes or no answer with reasoning
6. For multiple choice questions, select the option that is logically valid"""

    if rule_text:
        base_prompt += f"\n\nRule to follow: {rule_text}"

    return base_prompt


def format_question_with_options(row: pd.Series) -> str:
    """Format question with options if it's a multiple choice question"""
    question = row.get('original_question', '')
    question_type = row.get('original_question_type', '')
    options = row.get('original_options', '')

    if question_type == 'multiple_choice' and options:
        # Parse options from string representation
        try:
            if isinstance(options, str):
                # Try to evaluate the options string
                import ast
                options_list = ast.literal_eval(options)
                if isinstance(options_list, list) and len(options_list) > 0:
                    formatted_options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options_list)])
                    return f"{question}\n\n{formatted_options}"
        except:
            pass

    return question


def format_answer_with_reasoning(row: pd.Series) -> str:
    """
    Format the correct answer with reasoning.

    For fine-tuning, we want the model to learn to give correct answers
    with proper reasoning.
    """
    question_type = row.get('original_question_type', '')
    correct_answer = row.get('original_correct_answer', '')
    rule_text = row.get('natural_language', '')

    # Build answer with reasoning
    answer_parts = []

    # Add reasoning based on the rule
    if rule_text and isinstance(rule_text, str) and rule_text.strip():
        answer_parts.append(f"Reasoning: Based on the given rule '{rule_text}', I can determine the answer.")

    # Add the answer
    if question_type == 'yes_no':
        if isinstance(correct_answer, bool):
            answer_text = "Yes" if correct_answer else "No"
        else:
            answer_text = str(correct_answer)
        answer_parts.append(f"Answer: {answer_text}")
    elif question_type == 'true_false':
        if isinstance(correct_answer, bool):
            answer_text = "True" if correct_answer else "False"
        else:
            answer_text = str(correct_answer)
        answer_parts.append(f"Answer: {answer_text}")
    elif question_type == 'multiple_choice':
        # Find the correct option letter
        options = row.get('original_options', '')
        correct_answer = str(correct_answer)

        try:
            if isinstance(options, str):
                import ast
                options_list = ast.literal_eval(options)
                if isinstance(options_list, list):
                    for i, opt in enumerate(options_list):
                        if opt == correct_answer or (isinstance(opt, str) and str(opt).strip() == correct_answer.strip()):
                            answer_parts.append(f"Answer: {chr(65+i)}")
                            break
                    else:
                        answer_parts.append(f"Answer: {correct_answer}")
                else:
                    answer_parts.append(f"Answer: {correct_answer}")
        except:
            answer_parts.append(f"Answer: {correct_answer}")
    else:  # wh_question
        answer_parts.append(f"Answer: {correct_answer}")

    return "\n\n".join(answer_parts)


def convert_to_sharedgpt(df: pd.DataFrame, add_system_prompt: bool = True) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to SharedGPT format.

    Args:
        df: Input dataframe with question data
        add_system_prompt: Whether to add system prompt to conversations

    Returns:
        List of conversation dictionaries in SharedGPT format
    """
    conversations = []

    for idx, row in df.iterrows():
        # Format question
        question = format_question_with_options(row)

        # Format answer with reasoning
        answer = format_answer_with_reasoning(row)

        # Build conversation
        conversation_dict = {
            "conversations": []
        }

        # Add system prompt if requested
        if add_system_prompt:
            rule_text = row.get('natural_language', '')
            system_prompt = create_system_prompt(
                row.get('original_question_type', ''),
                rule_text
            )
            conversation_dict["conversations"].append({
                "from": "system",
                "value": system_prompt
            })

        # Add user question
        conversation_dict["conversations"].append({
            "from": "human",
            "value": question
        })

        # Add assistant answer
        conversation_dict["conversations"].append({
            "from": "gpt",
            "value": answer
        })

        # Add metadata (optional, for debugging)
        conversation_dict["_metadata"] = {
            "rule_dir": row.get('rule_dir', ''),
            "method_name": row.get('method_name', ''),
            "question_type": row.get('original_question_type', ''),
            "index": idx
        }

        conversations.append(conversation_dict)

    return conversations


def save_sharedgpt(conversations: List[Dict], output_file: str):
    """Save conversations to JSONL file in SharedGPT format"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            # Remove metadata before saving (not needed for training)
            conv_to_save = {k: v for k, v in conv.items() if k != '_metadata'}
            f.write(json.dumps(conv_to_save, ensure_ascii=False) + '\n')

    print(f"Saved {len(conversations)} conversations to {output_file}")


def generate_statistics(conversations: List[Dict], df: pd.DataFrame) -> Dict:
    """Generate statistics about the converted dataset"""
    stats = {
        "total_conversations": len(conversations),
        "question_types": {},
        "total_tokens_estimate": 0
    }

    # Count question types
    for conv in conversations:
        metadata = conv.get('_metadata', {})
        qtype = metadata.get('question_type', 'unknown')
        stats["question_types"][qtype] = stats["question_types"].get(qtype, 0) + 1

    # Estimate token count (rough approximation: 1 token ≈ 4 characters)
    for conv in conversations:
        for turn in conv.get("conversations", []):
            stats["total_tokens_estimate"] += len(turn.get("value", "")) // 4

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="RQ5: Convert dataset to SharedGPT format for LLaMA-Factory"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input CSV file (e.g., data/examples/rq5_results/splits/train/train.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL file (default: auto-generated from input name)"
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Do not add system prompt to conversations"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to convert (for testing)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("RQ5 SharedGPT Format Converter")
    print("=" * 70)
    print(f"Input: {args.input}")

    # Load dataset
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} samples")

    # Check required columns
    required_cols = ['original_question', 'original_correct_answer', 'original_question_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        print(f"Available columns: {list(df.columns)}")

    # Convert to SharedGPT format
    print("\nConverting to SharedGPT format...")
    conversations = convert_to_sharedgpt(
        df,
        add_system_prompt=not args.no_system_prompt
    )

    # Generate output path
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input)
        output_file = str(input_path.parent / f"{input_path.stem}_sharedgpt.jsonl")

    # Save to file
    save_sharedgpt(conversations, output_file)

    # Print statistics
    stats = generate_statistics(conversations, df)
    print("\nStatistics:")
    print(f"  Total conversations: {stats['total_conversations']}")
    print(f"  Question types:")
    for qtype, count in sorted(stats['question_types'].items()):
        print(f"    {qtype}: {count}")
    print(f"  Estimated total tokens: {stats['total_tokens_estimate']:,}")

    # Show example
    print("\nExample conversation:")
    if conversations:
        print(json.dumps(conversations[0], ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("Conversion completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
