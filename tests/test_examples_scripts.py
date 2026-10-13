"""Every script in `examples/` runs to the end on the real models, in a sandbox folder that holds `demo/`.

Runs only with `ANON_DOCS=1` (`scripts/docs.py check` sets it). A script that is not meant to be run alone says so on its
first line with `# docs-test: skip - the reason`; one that only makes sense on the real models says
`# docs-test: real-only - the reason` there (the fast gate in tests/test_docs_blocks.py leaves it out).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from tests import docs_support as ds

SCRIPTS = sorted((ds.ROOT / "examples").glob("*.py")) if (ds.ROOT / "examples").is_dir() else []


@unittest.skipUnless(ds.enabled(), "real-model examples: set ANON_DOCS=1 or run scripts/docs.py check")
class ExampleScriptsRunOnTheRealModels(unittest.TestCase):
    """One test per `examples/*.py` (added below)."""


def _script_test(script: Path):
    def test(self):
        first = script.read_text(encoding="utf-8").splitlines()[:1]
        skip = re.match(r"#\s*docs-test:\s*skip\s*-\s*(.+)", first[0]) if first else None
        if skip:
            self.skipTest(skip.group(1))
        done = subprocess.run(
            [sys.executable, str(ds.RUNNER), "script", str(script), "real"],
            cwd=ds.ROOT,
            env=ds.child_env(),
            capture_output=True,
            text=True,
            timeout=ds.REAL_MODEL_TIMEOUT,
        )
        report = os.environ.get("ANON_DOCS_REPORT")
        if report:
            entry = {
                "page": f"examples/{script.name}",
                "results": [
                    {
                        "number": 1,
                        "line": 1,
                        "status": "ok" if done.returncode == 0 else "fail",
                        "error": done.stderr[-300:] or None,
                        "actual_output": done.stdout,
                    }
                ],
            }
            with open(report, "a", encoding="utf-8") as out:
                out.write(json.dumps(entry) + "\n")
        self.assertEqual(done.returncode, 0, f"examples/{script.name}: exit {done.returncode}\n{done.stderr[-1500:]}")

    return test


for _script in SCRIPTS:
    setattr(ExampleScriptsRunOnTheRealModels, "test_script_" + _script.stem, _script_test(_script))


if __name__ == "__main__":
    unittest.main()
