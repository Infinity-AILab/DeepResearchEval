import argparse
import yaml
from types import SimpleNamespace

def load_config(yaml_file: str):
    """Load YAML config into a SimpleNamespace"""
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)
