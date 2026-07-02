#!/usr/bin/env python3
"""
Path setup utility for HONEST scripts.
Adds project root to Python path so scripts can import from honest package.
"""

import os
import sys

def setup_project_path():
    """Add project root directory to Python path."""
    # Get the absolute path of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up to project root (scripts -> research_honest)
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # Add project root to Python path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    return project_root

# Auto-setup when imported
setup_project_path()