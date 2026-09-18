"""Validate the source-derived signature and optionally run numerical examples."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-examples", action="store_true")
    options = parser.parse_args()
    config = runpy.run_path(str(ROOT / "docs/conf.py"))
    print("SIGNATURE PASS:", config["public_signature"](), flush=True)
    if options.run_examples:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(ROOT / "src"), environment.get("PYTHONPATH"),
        )))
        environment["PYTHONIOENCODING"] = "utf-8"
        for name in ("quickstart", "compare_modes", "envelope", "batching"):
            script = ROOT / "examples" / f"docs_{name}.py"
            print(f"Running {script.name}", flush=True)
            subprocess.run([sys.executable, str(script)], cwd=ROOT, env=environment, check=True)
        print("ALL DOCUMENTED EXAMPLES PASS", flush=True)


if __name__ == "__main__":
    main()
