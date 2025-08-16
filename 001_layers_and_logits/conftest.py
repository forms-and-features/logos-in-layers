"""Ensure tests can import the subpackage when run from repo root.

Pytest auto-discovers this file and updates sys.path for the session.
"""

import os
import sys

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

