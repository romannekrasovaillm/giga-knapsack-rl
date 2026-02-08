"""Pytest configuration: ensure src/ is on sys.path."""
import sys
from pathlib import Path

# Add project root to sys.path so 'from src.xxx import ...' works
# even without pip install -e .
ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
