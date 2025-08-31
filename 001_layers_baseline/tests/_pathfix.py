import os
import sys

# Ensure the parent directory (experiment root) is on sys.path so tests
# can import the sibling `layers_core` package and `run.py` relative to here.
PARENT = os.path.dirname(os.path.dirname(__file__))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

