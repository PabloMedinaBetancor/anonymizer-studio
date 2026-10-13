import pathlib
import shutil
import subprocess
import unittest

HERE = pathlib.Path(__file__).resolve().parent


class UiLogicTests(unittest.TestCase):
    def test_S_ui_logic_modules(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        done = subprocess.run(
            [node, str(HERE / "run.mjs")],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
