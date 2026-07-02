#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functionality cache loader utility

Provides the ability to load functionality and inverse functionality cache data from precomputed files.
"""

import json
import os
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def load_functionality_cache(file_path: str = "property_functionality.jsonl") -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load functionality cache data from a precomputed file

    Args:
        file_path: Path to the precomputed functionality file

    Returns:
        A tuple of the functionality cache dict and the inverse functionality cache dict
    """
    functionality_cache = {}
    inv_functionality_cache = {}

    if not os.path.exists(file_path):
        logger.warning(f"Precomputed functionality file does not exist: {file_path}")
        return functionality_cache, inv_functionality_cache

    try:
        loaded_count = 0
        error_count = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        property_id = data.get('property_id')
                        functionality = data.get('functionality', 0.0)
                        inverse_functionality = data.get('inverse_functionality', 0.0)
                        error = data.get('error')

                        # Only load results without errors
                        if property_id and error is None:
                            functionality_cache[property_id] = functionality
                            inv_functionality_cache[property_id] = inverse_functionality
                            loaded_count += 1
                        elif error:
                            error_count += 1

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} of the precomputed data: {e}")
                        continue

        logger.info(f"Loaded functionality data for {loaded_count} properties from {file_path} (skipped {error_count} erroneous entries)")

        # Deduplicated statistics
        unique_properties = set(functionality_cache.keys()) | set(inv_functionality_cache.keys())
        logger.info(f"Cache stats: functionality={len(functionality_cache)}, inverse functionality={len(inv_functionality_cache)}, unique properties={len(unique_properties)}")

    except (OSError, UnicodeDecodeError, ValueError) as e:
        logger.error(f"Failed to load precomputed functionality file: {e}")

    return functionality_cache, inv_functionality_cache


def save_functionality_cache(functionality_cache: Dict[str, float],
                           inv_functionality_cache: Dict[str, float],
                           file_path: str = "property_functionality_cache.json") -> bool:
    """Save the functionality cache to a JSON file

    Args:
        functionality_cache: Functionality cache dict
        inv_functionality_cache: Inverse functionality cache dict
        file_path: Output file path

    Returns:
        Whether the save succeeded
    """
    try:
        cache_data = {
            "functionality_cache": functionality_cache,
            "inv_functionality_cache": inv_functionality_cache,
            "metadata": {
                "functionality_count": len(functionality_cache),
                "inverse_functionality_count": len(inv_functionality_cache),
                "total_properties": len(set(functionality_cache.keys()) | set(inv_functionality_cache.keys()))
            }
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Functionality cache saved to {file_path}")
        return True

    except (OSError, TypeError, ValueError) as e:
        logger.error(f"Failed to save functionality cache: {e}")
        return False


def load_functionality_cache_from_json(file_path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load the functionality cache from a JSON file

    Args:
        file_path: JSON cache file path

    Returns:
        A tuple of the functionality cache dict and the inverse functionality cache dict
    """
    functionality_cache = {}
    inv_functionality_cache = {}

    if not os.path.exists(file_path):
        logger.warning(f"Cache file does not exist: {file_path}")
        return functionality_cache, inv_functionality_cache

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        functionality_cache = cache_data.get("functionality_cache", {})
        inv_functionality_cache = cache_data.get("inv_functionality_cache", {})
        metadata = cache_data.get("metadata", {})

        logger.info(f"Loaded cache from {file_path}: {metadata}")

    except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Failed to load JSON cache file: {e}")

    return functionality_cache, inv_functionality_cache


def get_cache_info(functionality_cache: Dict[str, float],
                  inv_functionality_cache: Dict[str, float]) -> Dict:
    """Get cache info statistics

    Args:
        functionality_cache: Functionality cache dict
        inv_functionality_cache: Inverse functionality cache dict

    Returns:
        A dict of cache statistics
    """
    unique_properties = set(functionality_cache.keys()) | set(inv_functionality_cache.keys())
    both_cached = set(functionality_cache.keys()) & set(inv_functionality_cache.keys())

    return {
        "functionality_count": len(functionality_cache),
        "inverse_functionality_count": len(inv_functionality_cache),
        "unique_properties": len(unique_properties),
        "both_cached": len(both_cached),
        "only_functionality": len(functionality_cache) - len(both_cached),
        "only_inverse_functionality": len(inv_functionality_cache) - len(both_cached)
    }


if __name__ == "__main__":
    # Test the cache loading functionality
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("=== Functionality cache loading test ===")

    # Test loading from a JSONL file
    func_cache, inv_func_cache = load_functionality_cache("property_functionality.jsonl")

    if func_cache or inv_func_cache:
        # Show cache info
        info = get_cache_info(func_cache, inv_func_cache)
        print(f"Cache info: {info}")

        # Show a few examples
        print("\nSample cache data:")
        for i, (prop_id, func_val) in enumerate(list(func_cache.items())[:5]):
            inv_val = inv_func_cache.get(prop_id, "N/A")
            print(f"  {prop_id}: func={func_val:.4f}, inv_func={inv_val}")

        # Test saving to JSON format
        if save_functionality_cache(func_cache, inv_func_cache, "test_cache.json"):
            print("\n✅ Cache saved to test_cache.json")

            # Test reloading from JSON
            func_cache2, inv_func_cache2 = load_functionality_cache_from_json("test_cache.json")
            print(f"Reload verification: functionality={len(func_cache2)}, inverse functionality={len(inv_func_cache2)}")
    else:
        print("❌ Failed to load any cache data")
