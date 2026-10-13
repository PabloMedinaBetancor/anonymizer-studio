"""Every ```python block of every page runs on the real models, and the output a page shows is what the run printed.

The blocks run in a new process per page, in a sandbox folder that holds the repository's own `demo/` files, in order and
in one namespace as a reader would paste them. A ```text block that starts with `# output` right after a python block is
the output the page shows: it must equal what the block printed (a line may hold `...` for any text, for times). A failure
names the block as `docs/how-to/x.md block 3 (line 41)`. What every block printed is kept by `scripts/docs.py check` in
`work/docs-check/outputs/`, so a writer can copy it.

Runs only with `ANON_DOCS=1` (`scripts/docs.py check` sets it): real models take real time (minutes, longer on a busy
machine). The same blocks run on the fakes in tests/test_docs_blocks.py in the default suite.
"""

from __future__ import annotations

import unittest

from tests import docs_support as ds


@unittest.skipUnless(ds.enabled(), "real-model documentation examples: set ANON_DOCS=1 or run scripts/docs.py check")
class PagesRunOnTheRealModels(unittest.TestCase):
    """One test per page that has python blocks (added below)."""


def _page_test(path):
    def test(self):
        name = ds.rel(path)
        results = ds.run_blocks(name, ds.python_blocks(path), "real")
        failures = ds.failure_lines(name, results)
        self.assertFalse(failures, "\n".join(failures))

    return test


for _path in ds.python_pages():
    setattr(
        PagesRunOnTheRealModels,
        "test_blocks_of_" + ds.rel(_path).replace("/", "_").replace(".", "_").replace("-", "_"),
        _page_test(_path),
    )


if __name__ == "__main__":
    unittest.main()
