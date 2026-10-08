"""Compatibility entry point for the selected VAR1 inference pipeline.

Use train_final.py for the full nested-CV search. This script only regenerates
the final submission using the selected configuration in inference.py.
"""
from inference import run_variant


if __name__ == "__main__":
    run_variant("var1")
