"""Allows running the package as a script: ``python -m laser_slice ...``."""
from __future__ import annotations

import sys

from laser_slice.cli import main

if __name__ == "__main__":
    sys.exit(main())
