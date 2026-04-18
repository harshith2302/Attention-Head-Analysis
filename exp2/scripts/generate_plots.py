#!/usr/bin/env python3
"""Generate all plots from experiment results."""
import os
import sys
sys.path.insert(0, "/home/cccp/25m0834/RND5")
sys.path.insert(0, "/home/cccp/25m0834/RND5/plotting")
from plotting.generate_plots import generate_all_plots
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

results_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/cccp/25m0834/RND5/results"
generate_all_plots(results_dir)
