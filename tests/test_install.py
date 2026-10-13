"""Review 2, N6: after a plain `pip install .` the package sits in site-packages, where the models folder of the checkout is not next to
it. The install is made into a temporary folder (--target, nothing is installed in any Python) and started from an empty working
directory: a missing model says where it looked and how to say where the models are, and ANONYMIZER_MODELS makes it work.

Needs setuptools 64 or newer to build from pyproject.toml; set ANONYMIZER_BUILD_PATH to a folder that has one if the system's is older
(the test is skipped otherwise)."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROBE = """
import anonymizer, anonymizer.classes as classes
print(anonymizer.__file__)
try:
    classes.load(["face"], 0, None)
    print("LOADED")
except Exception as e:
    print("ERROR", e)
"""


def dependency_folders() -> list:
    folders = []
    for name in ("numpy", "cv2", "onnxruntime", "av", "PIL"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        folder = os.path.dirname(os.path.dirname(module.__file__))
        if folder not in folders and not folder.startswith(str(ROOT)):
            folders.append(folder)
    return folders


class InstalledPackage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "pyproject.toml").is_file():
            raise unittest.SkipTest("no pyproject.toml beside the tests")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        tree = pathlib.Path(cls.tmp.name) / "tree"
        tree.mkdir()
        shutil.copy(ROOT / "pyproject.toml", tree)
        shutil.copy(ROOT / "README.md", tree)
        shutil.copytree(ROOT / "src", tree / "src", ignore=shutil.ignore_patterns("__pycache__"))
        cls.site = pathlib.Path(cls.tmp.name) / "site"
        # `-s` here is on purpose: the build and the probe see only the folders named below, never the user site
        build_path = os.environ.get("ANONYMIZER_BUILD_PATH", "")
        env = {**os.environ, "PYTHONPATH": build_path}
        done = subprocess.run(
            [
                sys.executable,
                "-s",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--quiet",
                "--target",
                str(cls.site),
                str(tree),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            check=False,
        )
        if done.returncode != 0 or not (cls.site / "anonymizer").is_dir():
            raise unittest.SkipTest(f"cannot build the package here (setuptools older than 64?): {done.stderr[-200:]}")

    def run_from_empty_folder(self, **env) -> list:
        with tempfile.TemporaryDirectory() as cwd:
            path = os.pathsep.join([str(self.site), *dependency_folders()])
            base = {k: v for k, v in os.environ.items() if k not in ("ANONYMIZER_MODELS", "PYTHONPATH")}
            done = subprocess.run(
                [sys.executable, "-s", "-c", PROBE],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=cwd,
                check=False,
                env={**base, "PYTHONPATH": path, **env},
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            return done.stdout.strip().splitlines() + [cwd]

    def test_N6_the_installed_package_says_where_it_looked_for_the_models(self):
        where, answer, cwd = self.run_from_empty_folder()
        self.assertTrue(where.startswith(str(self.site)), where)
        self.assertTrue(answer.startswith("ERROR face: model file not found"), answer)
        for part in ("Looked in", os.path.join(os.path.realpath(cwd), "models"), "ANONYMIZER_MODELS", "--models"):
            self.assertIn(part, answer)

    def test_N6_the_environment_variable_makes_the_installed_package_find_them(self):
        from anonymizer import contract

        folder = contract.models_dir()
        if not os.path.isfile(os.path.join(folder, "face_detection_yunet_2023mar.onnx")):
            self.skipTest("the models are not here")
        _, answer, _ = self.run_from_empty_folder(ANONYMIZER_MODELS=folder)
        self.assertEqual(answer, "LOADED")


class WhereTheModelsAre(unittest.TestCase):
    def test_N6_a_models_folder_in_the_working_directory_is_found_before_the_source_tree(self):
        from anonymizer import contract

        old = os.getcwd()
        with tempfile.TemporaryDirectory() as cwd:
            os.makedirs(os.path.join(cwd, "models"))
            os.chdir(cwd)
            try:
                saved = os.environ.pop("ANONYMIZER_MODELS", None)
                self.assertEqual(contract.models_dir(), os.path.join(os.path.realpath(cwd), "models"))
                self.assertEqual(contract.models_dir("/x"), "/x")
            finally:
                os.chdir(old)
                if saved is not None:
                    os.environ["ANONYMIZER_MODELS"] = saved


if __name__ == "__main__":
    unittest.main()
