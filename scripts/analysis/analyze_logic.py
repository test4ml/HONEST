import pandas as pd
import os
import time
from openai import OpenAI
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogicAnalyzer:
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        
    def analyze_logic(self, natural_language: str) -> tuple[str, str]:
        """Analyze the logical validity of natural language descriptions"""

        prompt = f"""Please analyze whether the logical reasoning in the following natural language description is valid. Judge from the perspective of logical reasoning, considering causality, common sense validity, and whether counterexamples exist.

Natural language description: "{natural_language}"

Please reply in the following format:
Validity judgment: [Y/N] (Y means valid, N means invalid)
Reason: [Brief explanation]

Note: Only judge the logical reasoning validity, do not consider the accuracy of specific domain knowledge."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in logical reasoning analysis, specializing in analyzing the logical validity of natural language descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            analysis = response.choices[0].message.content

            # Parse results
            lines = analysis.strip().split('\n')
            judgment = ""
            reason = ""

            for line in lines:
                if line.startswith('Validity judgment:'):
                    judgment = line.replace('Validity judgment:', '').strip()
                elif line.startswith('Reason:'):
                    reason = line.replace('Reason:', '').strip()

            # Standardize judgment results
            if judgment.upper() in ['Y', 'YES', '是', '合理', 'VALID']:
                judgment = 'Y'
            elif judgment.upper() in ['N', 'NO', '否', '不合理', 'INVALID']:
                judgment = 'N'
            else:
                judgment = 'U'  # Unknown

            return judgment, reason

        except Exception as e:
            error_msg = str(e)
            logger.error(f"API call failed: {error_msg}")
            # If it is a connection error, raise an exception to exit the program
            if "Connection" in error_msg or "connection" in error_msg:
                logger.error("Connection error detected, exiting program...")
                raise Exception(f"API connection error: {error_msg}")
            return "E", f"API error: {error_msg}"
    
    def process_file(self, input_file: str, output_file: str, sep=',', row_delay=1.0):
        """Process a single CSV file with checkpoint resume capability"""

        logger.info(f"Starting to process file: {input_file}")

        try:
            df = pd.read_csv(input_file, sep=sep)

            # Check if output file already exists (resume from checkpoint)
            if os.path.exists(output_file):
                logger.info(f"Found existing output file, resuming from checkpoint: {output_file}")
                processed_df = pd.read_csv(output_file, sep=sep)

                # Assume files already written are complete; skip this file
                logger.info(f"Skipping already completed file: {input_file}")
                return

            # Add new columns
            df['Logic_Judgment'] = ''
            df['Judgment_Reason'] = ''

            total_rows = len(df)

            for idx, row in df.iterrows():
                natural_language = row['Natural_Language']

                if pd.isna(natural_language) or natural_language.strip() == '':
                    df.at[idx, 'Logic_Judgment'] = 'N'
                    df.at[idx, 'Judgment_Reason'] = 'Natural language description is empty'
                    continue

                logger.info(f"Processing row {idx+1}/{total_rows}: {natural_language[:50]}...")

                judgment, reason = self.analyze_logic(natural_language)
                df.at[idx, 'Logic_Judgment'] = judgment
                df.at[idx, 'Judgment_Reason'] = reason

                # Add delay to avoid API rate limits
                time.sleep(row_delay)

            # Save results
            df.to_csv(output_file, index=False, sep=sep)
            logger.info(f"File processing completed: {output_file}")

        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}")
            raise

def batch_process_files(input_dir: str, output_dir: str, api_key: str,
                        model="deepseek-chat", base_url="https://api.deepseek.com/v1",
                        sep=',', row_delay=1.0, file_delay=2.0):
    """Batch process all split files with checkpoint resume"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    analyzer = LogicAnalyzer(api_key, model, base_url)

    # Get all split files
    input_files = sorted([f for f in os.listdir(input_dir) if f.startswith('chunk_') and f.endswith('.csv')])

    logger.info(f"Found {len(input_files)} files to process")

    processed_count = 0
    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, f"analyzed_{input_file}")

        # Check whether the output file already exists; if so, skip it
        if os.path.exists(output_path):
            logger.info(f"Skipping already processed file: {input_file}")
            processed_count += 1
            continue

        logger.info(f"Processing file: {input_file}")
        try:
            analyzer.process_file(input_path, output_path, sep=sep, row_delay=row_delay)
            processed_count += 1
            logger.info(f"Successfully processed file: {input_file} ({processed_count}/{len(input_files)})")
        except Exception as e:
            logger.error(f"Failed to process file {input_file}: {e}")
            # If it is a connection error, re-raise it
            if "connection" in str(e).lower():
                raise
            # For other errors, continue to the next file
            continue

        # Delay between files
        time.sleep(file_delay)

    logger.info(f"Batch processing completed. Processed {processed_count}/{len(input_files)} files")

def merge_results(output_dir: str, final_output: str, sep=','):
    """Merge all processing results"""

    analyzed_files = sorted([f for f in os.listdir(output_dir) if f.startswith('analyzed_chunk_') and f.endswith('.csv')])

    if not analyzed_files:
        logger.warning("No analyzed files found")
        return

    dfs = []
    for file in analyzed_files:
        file_path = os.path.join(output_dir, file)
        df = pd.read_csv(file_path, sep=sep)
        dfs.append(df)

    # Merge all DataFrames
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(final_output, index=False, sep=sep)

    logger.info(f"Merge completed! Total rows: {len(final_df)}")
    logger.info(f"Final result saved to: {final_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze logical validity of natural language descriptions")
    parser.add_argument('--api-key', default="sk-xxxx",
                       help='DeepSeek API key')
    parser.add_argument('--input-dir', default="data/processed/split_chunks",
                       help='Input directory containing split CSV files')
    parser.add_argument('--output-dir', default="data/processed/analyzed_chunks",
                       help='Output directory for analyzed files')
    parser.add_argument('--final-output', default="data/processed/rules/rules_with_nl_analyzed.csv",
                       help='Final merged output file path')
    parser.add_argument('--model', default="deepseek-chat",
                       help='Model to use (default: deepseek-chat)')
    parser.add_argument('--base-url', default="https://api.deepseek.com/v1",
                       help='API base URL (default: https://api.deepseek.com/v1)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Model temperature parameter (default: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=500,
                       help='Maximum output tokens (default: 500)')
    parser.add_argument('--row-delay', type=float, default=1.0,
                       help='Delay between row processing in seconds (default: 1.0)')
    parser.add_argument('--file-delay', type=float, default=2.0,
                       help='Delay between file processing in seconds (default: 2.0)')
    parser.add_argument('--sep', default=',',
                       help='CSV file separator (default: tab)')

    args = parser.parse_args()

    print("Please ensure you have set the correct DeepSeek API key")
    print("Processing workflow:")
    print("1. Batch process all split files")
    print("2. Merge processing results")

    try:
        # Batch process
        batch_process_files(
            args.input_dir,
            args.output_dir,
            args.api_key,
            model=args.model,
            base_url=args.base_url,
            sep=args.sep,
            row_delay=args.row_delay,
            file_delay=args.file_delay
        )

        # Merge results
        merge_results(args.output_dir, args.final_output, sep=args.sep)

    except Exception as e:
        logger.error(f"Program terminated due to error: {e}")
        logger.error("Please check your API connection and try again.")
        exit(1)