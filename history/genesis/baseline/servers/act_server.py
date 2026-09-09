"""Back-compat shim: ACT now runs through ``baseline/lerobot_server.py``.

This file exists so any caller that still spawns ``baseline/act_server.py``
(e.g. older copies of ``scripts/run_vla_eval.sh``) keeps working. It just
prepends ``--policy-type act`` and re-enters the generic entry point.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lerobot_server import main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--policy-type", "act", *sys.argv[1:]]
    main()
