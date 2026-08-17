#!/usr/bin/env python3
import sys

from capacity_probes.cli import main


if __name__ == "__main__":
    sys.argv.append("--verify-only")
    main()

