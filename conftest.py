"""
Pytest configuration — adds the project root to sys.path so that
`from src.data.chexpert_dataset import ...` works without installing the package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
