#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove duplicate entries from the precomputed functionality file.

Because concurrent writes from multiple processes may produce duplicate entries,
this script deduplicates them and keeps the latest result.
"""

import json
import os
import logging
from typing import Dict, Set
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def deduplicate_functionality_file(input_file: str, output_file: str = None):
    """Deduplicate the functionality file, keeping the latest entry for each property."""

    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return False

    if output_file is None:
        output_file = input_file.replace('.jsonl', '_deduplicated.jsonl')

    logger.info(f"Starting deduplication: {input_file}")

    # Store the latest entry for each property
    property_data: Dict[str, Dict] = {}
    total_lines = 0

    try:
        # Read all entries
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        property_id = data.get('property_id')
                        timestamp = data.get('timestamp', 0)

                        if property_id:
                            total_lines += 1

                            # If this property has no record yet, or the timestamp is newer, update the record
                            if (property_id not in property_data or
                                timestamp > property_data[property_id].get('timestamp', 0)):
                                property_data[property_id] = data

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue

        # Statistics
        unique_properties = len(property_data)
        duplicates_removed = total_lines - unique_properties

        logger.info(f"Processing complete:")
        logger.info(f"  Total entries: {total_lines}")
        logger.info(f"  Unique properties: {unique_properties}")
        logger.info(f"  Duplicate entries removed: {duplicates_removed}")

        # Sort by property ID and write to the output file
        sorted_properties = sorted(property_data.items())

        with open(output_file, 'w', encoding='utf-8') as f:
            for property_id, data in sorted_properties:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        logger.info(f"Deduplication results saved to: {output_file}")

        # Show some statistics
        successful_count = sum(1 for _, data in sorted_properties if not data.get('error'))
        error_count = unique_properties - successful_count

        logger.info(f"Result statistics:")
        logger.info(f"  Successfully computed: {successful_count}")
        logger.info(f"  Computation errors: {error_count}")
        logger.info(f"  Success rate: {successful_count/unique_properties*100:.1f}%")

        return True

    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        return False


def analyze_duplicates(input_file: str):
    """Analyze detailed information about duplicate entries."""

    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return

    logger.info(f"Analyzing duplicate entries: {input_file}")

    property_entries: Dict[str, list] = {}

    try:
        # Collect all entries
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        property_id = data.get('property_id')

                        if property_id:
                            if property_id not in property_entries:
                                property_entries[property_id] = []
                            property_entries[property_id].append(data)

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue

        # Find duplicated properties
        duplicated_properties = {prop_id: entries for prop_id, entries in property_entries.items() if len(entries) > 1}

        logger.info(f"Duplicate analysis results:")
        logger.info(f"  Total properties: {len(property_entries)}")
        logger.info(f"  Duplicated properties: {len(duplicated_properties)}")

        if duplicated_properties:
            logger.info(f"Duplicated property details:")
            for prop_id, entries in list(duplicated_properties.items())[:10]:  # Show only the first 10
                logger.info(f"  {prop_id}: {len(entries)} entries")
                for i, entry in enumerate(entries):
                    timestamp = entry.get('timestamp', 0)
                    comp_time = entry.get('computation_time', 0)
                    logger.info(f"    [{i+1}] timestamp: {timestamp:.0f}, computation time: {comp_time:.2f}s")

        return duplicated_properties

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Remove duplicate entries from the precomputed functionality file')
    parser.add_argument('--input_file', default='data/processed/properties/property_functionality.jsonl',
                       help='Input functionality file path')
    parser.add_argument('--output_file', default=None,
                       help='Output deduplicated file path (defaults to appending _deduplicated to the input filename)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze duplicates without performing deduplication')
    parser.add_argument('--replace_original', action='store_true',
                       help='Replace the original file with the deduplicated file')

    args = parser.parse_args()

    if args.analyze_only:
        analyze_duplicates(args.input_file)
    else:
        # Analyze first
        duplicates = analyze_duplicates(args.input_file)

        if duplicates:
            # Perform deduplication
            output_file = args.output_file
            if args.replace_original:
                output_file = args.input_file + '.tmp'

            success = deduplicate_functionality_file(args.input_file, output_file)

            if success and args.replace_original:
                # Replace the original file
                import shutil
                shutil.move(output_file, args.input_file)
                logger.info(f"Original file replaced: {args.input_file}")
        else:
            logger.info("No duplicate entries found; no deduplication needed")


if __name__ == "__main__":
    main()