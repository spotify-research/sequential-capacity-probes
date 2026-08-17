#!/usr/bin/env python3
import sys

from capacity_probes.cli import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--models",
        "mc,seqrules,pctm",
        "--datasets",
        "all",
        "--device",
        "cpu",
    ]
    main()
