#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profiling decorator utility - modern line_profiler approach
"""

import os

# Use the modern line_profiler approach
try:
    import line_profiler
    profile = line_profiler.profile
except ImportError:
    # Check whether profiling was explicitly enabled
    if os.environ.get('LINE_PROFILE') == '1':
        raise ImportError(
            "The environment variable LINE_PROFILE=1 is set, but the line_profiler library is not installed.\n"
            "Please install it: pip install line_profiler"
        )

    # If line_profiler is not installed and profiling is not enabled, use a no-op decorator
    def profile(func):
        """No-op profile decorator, used when line_profiler is unavailable"""
        return func

__all__ = ['profile']
