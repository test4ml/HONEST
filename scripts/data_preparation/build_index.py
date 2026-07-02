#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script for building the knowledge graph index.
"""

from honest import KGIndexBuilder
import argparse


def main():
    parser = argparse.ArgumentParser(description='Build the knowledge graph index')
    parser.add_argument('--kg_file',
                        default='$WIKIDATA_KG_ROOT/data/amie/wikidata_en_amie.tsv',
                        help='Path to the knowledge graph TSV file')
    parser.add_argument('--db_path',
                        default='kg_index.db',
                        help='Path to the index database file')
    parser.add_argument('--force_rebuild', action='store_true',
                        help='Force rebuilding the index')

    args = parser.parse_args()

    # Build the index
    builder = KGIndexBuilder(args.db_path)
    builder.build_index_from_tsv(args.kg_file, args.force_rebuild)


if __name__ == "__main__":
    main()