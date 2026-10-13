"""scripts/fetch_models.py: the way for a person to get the models that are neither in the repository nor in the package
(QA pass of 04/10/2026: a fresh install had no way to obtain them; the README said only where they come from)."""

from __future__ import annotations

import hashlib
import io
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_models  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Sources(unittest.TestCase):
    def test_every_model_of_the_checksum_list_has_a_source_and_none_is_extra(self):
        listed = set(fetch_models.expected_sums(str(ROOT / "models")))
        self.assertEqual(listed, set(fetch_models.SOURCES))
        for name, url in fetch_models.SOURCES.items():
            self.assertTrue(url.startswith("https://") and url.endswith(name), (name, url))

    def test_the_sources_are_the_ones_models_md_names(self):
        text = (ROOT / "MODELS.md").read_text(encoding="utf-8")
        for name in ("retinaface_r34.onnx", "rf-detr-nano-384-coco.onnx"):
            self.assertIn(fetch_models.SOURCES[name], text)


class AFolderOfYourOwn(unittest.TestCase):
    """The product checks every model against the SHA256SUMS in the folder it loads it from: models fetched into another folder
    (--models DIR) without that list are refused at load ("has no entry in SHA256SUMS"); found when a pip-installed copy was tried."""

    def test_the_folder_gets_the_checksum_list_of_the_checkout_and_an_existing_one_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "mine")
            sums = fetch_models.ensure_sums(folder)
            self.assertEqual(sums, fetch_models.expected_sums(str(ROOT / "models")))
            self.assertEqual(
                (pathlib.Path(folder) / "SHA256SUMS").read_bytes(), (ROOT / "models" / "SHA256SUMS").read_bytes()
            )
            (pathlib.Path(folder) / "SHA256SUMS").write_text("0" * 64 + "  other.onnx\n")
            self.assertEqual(fetch_models.ensure_sums(folder), {"other.onnx": "0" * 64})


class Fetching(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        self.good = b"model bytes"
        (self.dir / "models").mkdir()
        self.src = self.dir / "src.bin"
        self.src.write_bytes(self.good)
        self.sources = {"m.onnx": self.src.as_uri()}
        self.sums = {"m.onnx": sha(self.good)}
        self.out = io.StringIO()
        self.calls = 0

    def opener(self, url, timeout=0):
        self.calls += 1
        return __import__("urllib.request").request.urlopen(url, timeout=timeout)

    def run_fetch(self):
        return fetch_models.fetch(
            str(self.dir / "models"), sources=self.sources, sums=self.sums, opener=self.opener, out=self.out
        )

    def test_a_missing_model_is_downloaded_checked_and_a_second_run_downloads_nothing(self):
        self.assertEqual(self.run_fetch(), 0)
        self.assertEqual((self.dir / "models" / "m.onnx").read_bytes(), self.good)
        self.assertEqual(self.run_fetch(), 0)
        self.assertEqual(self.calls, 1)
        self.assertEqual(os.listdir(self.dir / "models"), ["m.onnx"])

    def test_a_download_that_does_not_match_is_refused_and_leaves_nothing_under_the_name(self):
        self.src.write_bytes(b"something else")
        self.assertEqual(self.run_fetch(), 1)
        self.assertEqual(os.listdir(self.dir / "models"), [])
        self.assertIn("does not match", self.out.getvalue())

    def test_a_damaged_file_in_place_is_replaced_by_a_good_one(self):
        (self.dir / "models" / "m.onnx").write_bytes(b"half a model")
        self.assertEqual(self.run_fetch(), 0)
        self.assertEqual((self.dir / "models" / "m.onnx").read_bytes(), self.good)

    def test_a_failed_download_keeps_the_old_file_and_no_partial(self):
        (self.dir / "models" / "m.onnx").write_bytes(b"half a model")
        self.sources["m.onnx"] = (self.dir / "nothing.bin").as_uri()
        self.assertEqual(self.run_fetch(), 1)
        self.assertEqual((self.dir / "models" / "m.onnx").read_bytes(), b"half a model")
        self.assertEqual(os.listdir(self.dir / "models"), ["m.onnx"])


if __name__ == "__main__":
    unittest.main()
