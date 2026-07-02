#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Property manager: loads property information from wiki_properties.jsonl."""

import json
import os


class PropertyManager:
    """Property manager: loads property information from wiki_properties.jsonl."""

    def __init__(self, properties_file="data/processed/properties/wiki_properties.jsonl"):
        self.properties_file = properties_file
        self.properties_cache = {}
        self._load_properties()

    def _load_properties(self):
        """Load property data."""
        if not os.path.exists(self.properties_file):
            print(f"Warning: Properties file {self.properties_file} not found")
            return

        try:
            with open(self.properties_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    property_id = data.get('property_id')
                    label = data.get('label', '')
                    if property_id and label:
                        self.properties_cache[property_id] = label
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error loading properties file: {e}")

    def get_property_label(self, property_id):
        """Get a property label."""
        return self.properties_cache.get(property_id, property_id)

    def get_all_properties(self):
        """Get all properties."""
        return self.properties_cache
