"""The hand-written parts of the reference stay true to the code.

* Every public name of `anonymizer.__all__` is named in the reference pages (checked once no reference page is a placeholder).
* Every `--flag` that `docs/reference/command-line.md` names under a command's heading is one that command's own `--help`
  prints. The heading of a command's section is the command as typed: `python -m anonymizer.ui`, `scripts/fetch_models.py`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import anonymizer

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference"
PLACEHOLDER = "This page is a placeholder."


class EveryPublicNameIsDocumentedTest(unittest.TestCase):
    def test_every_name_in_all_is_named_in_the_reference_pages(self):
        pages = sorted(REFERENCE.glob("*.md")) if REFERENCE.is_dir() else []
        stubs = [p.name for p in pages if PLACEHOLDER in p.read_text(encoding="utf-8")]
        if stubs or not pages:
            self.skipTest("placeholder reference pages: " + ", ".join(stubs))
        text = "\n".join(p.read_text(encoding="utf-8") for p in pages)
        missing = [n for n in anonymizer.__all__ if not re.search(rf"(?<![\w.]){re.escape(n)}(?![\w])", text)]
        self.assertEqual(missing, [], "public names the reference never mentions")


class CommandLineFlagsTest(unittest.TestCase):
    def help_flags(self, command: list[str]) -> set[str]:
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        done = subprocess.run(command + ["--help"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=180)
        self.assertEqual(done.returncode, 0, done.stderr[-300:])
        return set(re.findall(r"--[a-z][a-z-]*", done.stdout))

    def sections(self) -> dict[str, list[str]]:
        """The lines of the page under each command's heading, by the command's name (`anonymizer.ui` or a script's)."""
        found: dict[str, list[str]] = {}
        current = None
        page = REFERENCE / "command-line.md"
        if not page.is_file():
            return found
        for line in page.read_text(encoding="utf-8").splitlines():
            heading = re.match(r"#{2,3} (.*)", line)
            if heading:
                title = heading.group(1).strip("` ")
                if title in ("python -m anonymizer.ui", "python3 -m anonymizer.ui", "anonymizer.ui"):
                    current = "anonymizer.ui"
                elif re.match(r"(?:python3? )?scripts/[a-z_]+\.py$", title):
                    current = title.rsplit("/", 1)[1][: -len(".py")]
                else:
                    current = None
            if current:
                found.setdefault(current, []).append(line)
        return found

    def test_every_flag_on_the_page_is_one_the_command_prints_in_its_help(self):
        for name, lines in self.sections().items():
            flags = set(re.findall(r"`(--[a-z][a-z-]*)", "\n".join(lines)))
            if not flags:
                continue
            command = (
                [sys.executable, "-m", "anonymizer.ui"]
                if name == "anonymizer.ui"
                else [sys.executable, f"scripts/{name}.py"]
            )
            with self.subTest(command=name):
                known = self.help_flags(command) | {"--help"}
                self.assertEqual(flags - known, set(), f"{name}: flags on the page that its --help does not print")


if __name__ == "__main__":
    unittest.main()
