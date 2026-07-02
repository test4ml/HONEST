#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute the functionality and inverse functionality of all properties in wiki_properties.jsonl.
Save to a JSONL file in real time, with resume support.
"""

import json
import time
import os
import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
from pathlib import Path
from typing import Dict, Set
import logging

# Add the project path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from honest.kg import MemgraphKnowledgeGraph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('precompute_functionality.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FunctionalityPrecomputer:
    """Functionality precomputer."""

    def __init__(self, kg: MemgraphKnowledgeGraph,
                 properties_file: str = "data/processed/properties/wiki_properties.jsonl",
                 output_file: str = "data/processed/properties/property_functionality.jsonl"):
        self.kg = kg
        self.properties_file = properties_file
        self.output_file = output_file
        self.processed_properties: Set[str] = set()

        # Load already-processed properties
        self._load_existing_results()

    def _load_existing_results(self):
        """Load existing results, supporting resume from checkpoint."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            self.processed_properties.add(data['property_id'])
                logger.info(f"Loaded {len(self.processed_properties)} already-processed properties")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")

    def _save_result(self, property_id: str, label: str,
                    functionality: float, inverse_functionality: float,
                    computation_time: float, error: str = None):
        """Save a single result to the JSONL file."""
        result = {
            "property_id": property_id,
            "label": label,
            "functionality": functionality,
            "inverse_functionality": inverse_functionality,
            "computation_time": computation_time,
            "timestamp": time.time(),
            "error": error
        }

        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()  # Force flush to disk
            logger.info(f"Saved {property_id} ({label}): func={functionality:.4f}, inv_func={inverse_functionality:.4f}, time={computation_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to save result for {property_id}: {e}")

    def _load_properties(self):
        """Load all properties."""
        properties = []
        try:
            with open(self.properties_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            prop = json.loads(line.strip())
                            properties.append(prop)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse a property line: {e}")
            logger.info(f"Successfully loaded {len(properties)} properties")
            return properties
        except Exception as e:
            logger.error(f"Failed to load the properties file: {e}")
            return []

    def compute_property_functionality(self, property_id: str, label: str) -> Dict:
        """Compute the functionality of a single property."""
        start_time = time.time()

        try:
            # Compute functionality and inverse functionality
            functionality = self.kg.functionality(property_id)
            inverse_functionality = self.kg.inverse_functionality(property_id)

            computation_time = time.time() - start_time

            return {
                "functionality": functionality,
                "inverse_functionality": inverse_functionality,
                "computation_time": computation_time,
                "error": None
            }

        except Exception as e:
            computation_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"Failed to compute {property_id}: {error_msg}")

            return {
                "functionality": 0.0,
                "inverse_functionality": 0.0,
                "computation_time": computation_time,
                "error": error_msg
            }

    def precompute_all(self, batch_size: int = 10):
        """Precompute functionality for all properties."""
        logger.info("Starting precomputation of functionality for all properties...")

        # Load all properties
        properties = self._load_properties()
        if not properties:
            logger.error("No properties found")
            return

        # Filter out already-processed properties
        remaining_properties = [
            prop for prop in properties
            if prop['property_id'] not in self.processed_properties
        ]

        total_properties = len(properties)
        remaining_count = len(remaining_properties)
        processed_count = len(self.processed_properties)

        logger.info(f"Total properties: {total_properties}")
        logger.info(f"Already processed: {processed_count}")
        logger.info(f"Pending: {remaining_count}")

        if remaining_count == 0:
            logger.info("All properties have been processed!")
            return

        # Start processing
        failed_count = 0
        success_count = 0

        for i, prop in enumerate(remaining_properties):
            property_id = prop['property_id']
            label = prop.get('label', 'Unknown')

            current_progress = processed_count + i + 1
            progress_percent = (current_progress / total_properties) * 100

            logger.info(f"[{current_progress}/{total_properties}] ({progress_percent:.1f}%) Processing property: {property_id} ({label})")

            # Compute functionality
            result = self.compute_property_functionality(property_id, label)

            # Save the result
            self._save_result(
                property_id=property_id,
                label=label,
                functionality=result["functionality"],
                inverse_functionality=result["inverse_functionality"],
                computation_time=result["computation_time"],
                error=result["error"]
            )

            if result["error"]:
                failed_count += 1
            else:
                success_count += 1

            # Add to the processed set
            self.processed_properties.add(property_id)

            # Output statistics per batch
            if (i + 1) % batch_size == 0:
                logger.info(f"Batch complete: succeeded {success_count}, failed {failed_count}")

                # Output cache statistics
                cache_stats = self.kg.get_functionality_cache_stats()
                logger.info(f"Cache statistics: {cache_stats}")

        # Final statistics
        logger.info("=" * 60)
        logger.info("Precomputation complete!")
        logger.info(f"Total processed: {remaining_count} properties")
        logger.info(f"Succeeded: {success_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Success rate: {success_count/(success_count+failed_count)*100:.1f}%")
        logger.info(f"Results saved to: {self.output_file}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Precompute property functionality')
    parser.add_argument('--memgraph_uri', default='bolt://localhost:7687',
                        help='Memgraph connection URI')
    parser.add_argument('--memgraph_user', default='',
                        help='Memgraph username')
    parser.add_argument('--memgraph_password', default='',
                        help='Memgraph password')
    parser.add_argument('--properties_file', default='data/processed/properties/wiki_properties.jsonl',
                        help='Properties file path')
    parser.add_argument('--output_file', default='data/processed/properties/property_functionality.jsonl',
                        help='Output file path')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size (for progress reporting)')

    args = parser.parse_args()

    logger.info("=== Property functionality precomputation tool ===")
    logger.info(f"Memgraph URI: {args.memgraph_uri}")
    logger.info(f"Properties file: {args.properties_file}")
    logger.info(f"Output file: {args.output_file}")

    # Check the input file
    if not os.path.exists(args.properties_file):
        logger.error(f"Properties file does not exist: {args.properties_file}")
        return

    # Connect to the knowledge graph
    logger.info("Connecting to the knowledge graph...")
    kg = MemgraphKnowledgeGraph(
        uri=args.memgraph_uri,
        user=args.memgraph_user,
        password=args.memgraph_password,
        enable_metadata=True
    )

    if not kg.test_connection():
        logger.error("Cannot connect to the knowledge graph; program terminated")
        return

    logger.info("Knowledge graph connected successfully")

    try:
        # Create the precomputer and start processing
        precomputer = FunctionalityPrecomputer(
            kg=kg,
            properties_file=args.properties_file,
            output_file=args.output_file
        )

        precomputer.precompute_all(batch_size=args.batch_size)

    except KeyboardInterrupt:
        logger.info("Interrupted by user; program stopped")
    except Exception as e:
        logger.error(f"Program execution error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Close the connection
        kg.close()
        logger.info("Knowledge graph connection closed")


if __name__ == "__main__":
    main()