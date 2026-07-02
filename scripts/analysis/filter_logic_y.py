#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter CSV files to keep only rows where Logic_Judgment is 'Y'
"""

import pandas as pd
import sys
import argparse

def filter_logic_y(input_file: str, output_file: str, sep=','):
    """
    Read CSV file and filter rows where Logic_Judgment is 'Y'

    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path
        sep: CSV file separator (default: tab)
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_file, sep=sep)

        print(f"Original data rows: {len(df)}")

        # Filter rows where Logic_Judgment is 'Y'
        filtered_df = df[df['Logic_Judgment'] == 'Y']

        print(f"Filtered data rows: {len(filtered_df)}")

        # Save to new file
        filtered_df.to_csv(output_file, sep=sep, index=False)

        print(f"Saved to: {output_file}")

    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter CSV files to keep only rows where Logic_Judgment is 'Y'")
    parser.add_argument('--input', '-i', default="rules_with_nl_analyzed.csv",
                       help='Input CSV file path (default: rules_with_nl_analyzed.csv)')
    parser.add_argument('--output', '-o', default="rules_logic_y_filtered.csv",
                       help='Output CSV file path (default: rules_logic_y_filtered.csv)')
    parser.add_argument('--sep', default=',',
                       help='CSV file separator')

    args = parser.parse_args()

    filter_logic_y(args.input, args.output, args.sep)