import pandas as pd
import os
import argparse

def split_csv(input_file, output_dir, chunk_size=50, input_sep='\t', output_sep=','):
    """Split CSV file into chunks of specified size"""

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read CSV file
    df = pd.read_csv(input_file, sep=input_sep)
    total_rows = len(df)

    print(f"Total rows: {total_rows}")
    print(f"Column names: {list(df.columns)}")

    # Split files
    num_chunks = (total_rows + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)

        chunk_df = df.iloc[start_idx:end_idx]
        output_file = os.path.join(output_dir, f"chunk_{i+1:03d}.csv")

        chunk_df.to_csv(output_file, sep=output_sep, index=False)
        print(f"Saved: {output_file} (rows {start_idx+1}-{end_idx})")

    return num_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CSV file into chunks")
    parser.add_argument('input_file', nargs='?',
                       default='./data/processed/rules/rules_with_nl_mid2.csv',
                       help='Input CSV file path (default: ./data/processed/rules/rules_with_nl_mid2.csv)')
    parser.add_argument('output_dir', nargs='?',
                       default='./data/processed/split_chunks',
                       help='Output directory for chunks (default: ./data/processed/split_chunks)')
    parser.add_argument('--chunk-size', type=int, default=50,
                       help='Number of rows per chunk (default: 50)')
    parser.add_argument('--input-sep', default='\t',
                       help='Input CSV separator (default: \\t for tab)')
    parser.add_argument('--output-sep', default=',',
                       help='Output CSV separator (default: , for comma)')

    args = parser.parse_args()

    num_files = split_csv(args.input_file, args.output_dir,
                         chunk_size=args.chunk_size,
                         input_sep=args.input_sep,
                         output_sep=args.output_sep)
    print(f"\nSplit completed! Generated {num_files} files in total")