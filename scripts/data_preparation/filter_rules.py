#!/usr/bin/env python3
"""
Filter association rules based on quality metrics.
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import configuration management
from configs import get_config

def filter_rules(input_file, output_file, 
                head_coverage_threshold=0.05,
                std_confidence_threshold=0.5, 
                pca_confidence_threshold=0.9,
                positive_examples_threshold=10):
    """
    Filter association rules based on quality metrics.
    
    Args:
        input_file: Path to input CSV file with rules
        output_file: Path to output CSV file for filtered rules
        head_coverage_threshold: Minimum head coverage (default: 0.05)
        std_confidence_threshold: Minimum standard confidence (default: 0.5)
        pca_confidence_threshold: Minimum PCA confidence (default: 0.9)
        positive_examples_threshold: Minimum positive examples (default: 10)
    """
    
    try:
        # Load the rules data
        print(f"Loading rules from {input_file}...")
        df = pd.read_csv(input_file, sep='\t')
        print(f"Loaded {len(df)} rules")
        
        # Apply filters
        filtered_df = df[
            (df['Head Coverage'] >= head_coverage_threshold) &
            (df['Std Confidence'] >= std_confidence_threshold) &
            (df['PCA Confidence'] >= pca_confidence_threshold) &
            (df['Positive Examples'] >= positive_examples_threshold)
        ]
        
        print(f"\nFiltering results:")
        print(f"Original rules: {len(df)}")
        print(f"After Head Coverage >= {head_coverage_threshold}: {len(df[df['Head Coverage'] >= head_coverage_threshold])}")
        print(f"After Std Confidence >= {std_confidence_threshold}: {len(df[df['Std Confidence'] >= std_confidence_threshold])}")
        print(f"After PCA Confidence >= {pca_confidence_threshold}: {len(df[df['PCA Confidence'] >= pca_confidence_threshold])}")
        print(f"After Positive Examples >= {positive_examples_threshold}: {len(df[df['Positive Examples'] >= positive_examples_threshold])}")
        print(f"Final filtered rules: {len(filtered_df)}")
        print(f"Retention rate: {len(filtered_df)/len(df)*100:.2f}%")
        
        if len(filtered_df) == 0:
            print("Warning: No rules pass all filters. Consider lowering thresholds.")
            return
        
        # Sort by quality metrics (highest quality first)
        filtered_df = filtered_df.sort_values(
            ['PCA Confidence', 'Std Confidence', 'Positive Examples', 'Head Coverage'], 
            ascending=[False, False, False, False]
        )
        
        # Save filtered rules
        filtered_df.to_csv(output_file, index=False)
        print(f"\nFiltered rules saved to {output_file}")
        
        # Show top 10 filtered rules
        print("\nTop 10 highest quality rules:")
        print("=" * 80)
        display_cols = ['Rule', 'Head Coverage', 'Std Confidence', 'PCA Confidence', 'Positive Examples']
        for i, (idx, row) in enumerate(filtered_df[display_cols].head(10).iterrows(), 1):
            print(f"{i:2d}. Rule: {row['Rule']}")
            print(f"    Head Coverage: {row['Head Coverage']:.6f}")
            print(f"    Std Confidence: {row['Std Confidence']:.6f}")
            print(f"    PCA Confidence: {row['PCA Confidence']:.6f}")
            print(f"    Positive Examples: {row['Positive Examples']}")
            print()
        
        # Statistical summary of filtered data
        print("Quality metrics summary for filtered rules:")
        print("=" * 50)
        for metric in ['Head Coverage', 'Std Confidence', 'PCA Confidence', 'Positive Examples']:
            values = filtered_df[metric]
            print(f"{metric}:")
            print(f"  Min: {values.min():.6f}, Max: {values.max():.6f}")
            print(f"  Mean: {values.mean():.6f}, Median: {values.median():.6f}")
    
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Filter high-quality association rules based on confidence metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Optimal default thresholds (based on data analysis):
- Head Coverage >= 0.05 (above median, ensures reasonable coverage)
- Std Confidence >= 0.5 (high confidence, above 75th percentile)
- PCA Confidence >= 0.9 (very high confidence, near 95th percentile)  
- Positive Examples >= 10 (sufficient evidence, above 25th percentile)

These thresholds select rules that are highly confident, well-supported,
and have no exceptions (PCA Confidence near 1.0).
        """
    )
    
    # Use the configuration system to get the default thresholds
    rule_filtering_config = get_config('data_processing.rule_filtering', {})
    default_head_coverage = rule_filtering_config.get('head_coverage', 0.05)
    default_std_confidence = rule_filtering_config.get('std_confidence', 0.5)
    default_pca_confidence = rule_filtering_config.get('pca_confidence', 0.9)
    default_positive_examples = rule_filtering_config.get('positive_examples', 10)

    parser.add_argument('input_file', help='Input CSV file with association rules')
    parser.add_argument('output_file', help='Output CSV file for filtered rules')

    parser.add_argument('--head-coverage', type=float, default=default_head_coverage,
                       help=f'Minimum head coverage threshold (default: {default_head_coverage})')

    parser.add_argument('--std-confidence', type=float, default=default_std_confidence,
                       help=f'Minimum standard confidence threshold (default: {default_std_confidence})')

    parser.add_argument('--pca-confidence', type=float, default=default_pca_confidence,
                       help=f'Minimum PCA confidence threshold (default: {default_pca_confidence})')

    parser.add_argument('--positive-examples', type=int, default=default_positive_examples,
                       help=f'Minimum positive examples threshold (default: {default_positive_examples})')
    
    args = parser.parse_args()
    
    print("High-Quality Association Rules Filter")
    print("=" * 40)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Thresholds:")
    print(f"  Head Coverage: >= {args.head_coverage}")
    print(f"  Std Confidence: >= {args.std_confidence}")
    print(f"  PCA Confidence: >= {args.pca_confidence}")
    print(f"  Positive Examples: >= {args.positive_examples}")
    print()
    
    filter_rules(
        args.input_file,
        args.output_file,
        args.head_coverage,
        args.std_confidence,
        args.pca_confidence,
        args.positive_examples
    )

if __name__ == "__main__":
    main()