"""PyInstaller entry point. The bench worker re-invokes this same
executable with --bench-worker (see core/bench.py), so keep argv untouched."""

import multiprocessing
import sys

from ecoutemoi.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
