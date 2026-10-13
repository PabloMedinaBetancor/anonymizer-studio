"""The licence and provenance rules: models are hash-checked (CL3), `src/` never reaches into `bench/` or a reference model (CL5, RM2),
and the third-party list says what the text path uses (PT8)."""

from __future__ import annotations

import ast
import os
import re
import shutil
import sys
import tempfile
import unittest

from anonymizer import contract
from anonymizer.classes import heavy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "anonymizer")
REFERENCE_NAMES = re.compile(r"egoblur|deface|centerface", re.I)
ALLOWED_LICENCES = re.compile(r"^(MIT|BSD-[23]-Clause|Apache-2\.0|ISC|Zlib|HPND)\b")
ALLOWED_NAMES = re.compile(r"^(MIT|BSD-[23]-Clause|Apache-2\.0|ISC|Zlib|HPND)$")


def source_files():
    for folder, _, names in os.walk(SRC):
        for name in names:
            if name.endswith(".py"):
                yield os.path.join(folder, name)


def read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


class ModelHashes(unittest.TestCase):
    def test_cl3_every_listed_model_that_is_present_matches_its_hash(self):
        folder = contract.models_dir()
        listed = heavy.expected_hashes(folder)
        self.assertGreaterEqual(len(listed), 4, "models/SHA256SUMS lists the models of MODELS.md")
        present = [n for n in listed if os.path.isfile(os.path.join(folder, n))]
        self.assertIn(heavy.RETINAFACE_FILE, present, f"{heavy.RETINAFACE_FILE} is not in {folder}")
        for name in present:
            self.assertEqual(heavy.verified_model("face", folder, name)[1], listed[name], name)

    def test_cl3_a_model_with_one_changed_byte_is_a_model_load_error_naming_the_file(self):
        folder = tempfile.mkdtemp()
        try:
            source = contract.models_dir()
            shutil.copy(os.path.join(source, "SHA256SUMS"), folder)
            with open(os.path.join(source, heavy.YUNET_FILE), "rb") as f:
                data = bytearray(f.read())
            data[100] ^= 0xFF
            with open(os.path.join(folder, heavy.YUNET_FILE), "wb") as f:
                f.write(bytes(data))
            with self.assertRaises(contract.ModelLoadError) as raised:
                heavy.ZoomedYuNet(models=folder)
            self.assertIn(heavy.YUNET_FILE, str(raised.exception))
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_cl3_models_md_lists_every_model_with_an_allowed_licence(self):
        text = read("MODELS.md")
        for name in heavy.expected_hashes(os.path.join(ROOT, "models")):
            row = next((line for line in text.splitlines() if f"`{name}`" in line), None)
            self.assertIsNotNone(row, f"{name} is not in MODELS.md")
            licence = row.split("|")[3].strip()
            self.assertRegex(licence, ALLOWED_LICENCES, name)


class SourceTree(unittest.TestCase):
    def test_cl5_src_never_imports_from_bench(self):
        for path in source_files():
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    self.assertNotEqual(name.split(".")[0], "bench", f"{path} imports {name}")

    def test_rm2_no_reference_model_is_named_or_loaded_in_src(self):
        for path in source_files():
            with open(path, encoding="utf-8") as f:
                hit = REFERENCE_NAMES.search(f.read())
            self.assertIsNone(hit, f"{path} mentions the reference model {hit and hit.group(0)}")


class ThirdPartyList(unittest.TestCase):
    """CL1: no licence outside the allow-list is inside the product, and THIRD-PARTY.md is generated from data, not written by hand."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import third_party

        cls.module = third_party
        cls.data = third_party.load()

    def test_cl1_third_party_md_is_what_the_script_generates_from_the_data(self):
        self.assertEqual(read("THIRD-PARTY.md"), self.module.render(self.data))

    def test_cl1_every_licence_name_is_on_the_allow_list(self):
        for entry in [*self.data["models"], *self.data["libraries"]]:
            self.assertRegex(entry["licence"], ALLOWED_NAMES, entry["name"])
            self.assertNotRegex(entry["licence"], r"(?i)gpl|non-?commercial|research|\bNC\b", entry["name"])

    def test_cl2_no_model_that_ships_is_uncleared_and_the_v1_plate_model_is_only_a_named_reference(self):
        uncleared = {m["name"] for m in self.data["models"] if not m["cleared"]}
        self.assertEqual(
            uncleared, set()
        )  # PT4 passed: the YOLOv9-t file left models/; a new uncleared model needs a decision
        self.assertNotIn("yolo-v9-t", read(os.path.join("models", "SHA256SUMS")))
        self.assertIn("not shipped and not cleared", read("THIRD-PARTY.md"))
        for m in self.data["models"]:
            self.assertEqual(m["note"].startswith("**Not cleared.**"), not m["cleared"], m["name"])

    def test_cl1_the_rf_detr_row_names_its_variant_weights_licence_and_hash(self):
        rows = [m for m in self.data["models"] if "RF-DETR" in m["name"]]
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.module.variant_problems(rows[0]), [])
        file = "rf-detr-nano-384-coco.onnx"
        self.assertEqual(rows[0]["sha256"], heavy.expected_hashes(os.path.join(ROOT, "models"))[file])
        self.assertTrue(rows[0]["weights_url"].endswith("/" + file))
        self.assertIn("RF-DETR-N", read("THIRD-PARTY.md"))

    def test_cl1_a_row_that_does_not_name_its_rf_detr_variant_is_refused(self):
        good = next(m for m in self.data["models"] if "RF-DETR" in m["name"])
        for change in (
            {"variant": ""},
            {"variant": "RF-DETR"},
            {"variant": "RF-DETR-XL", "name": "RF-DETR-XL"},
            {"variant": "RF-DETR-2XL", "name": "RF-DETR-2XL"},
            {"name": "RF-DETR nano"},
            {"licence_quote": ""},
            {"licence_url": ""},
            {"weights_url": ""},
            {"sha256": ""},
            {"licence": "PML-1.0"},
        ):
            self.assertTrue(self.module.variant_problems({**good, **change}), change)

    def test_cl1_every_model_file_of_the_checksum_list_has_a_row(self):
        names = " | ".join(m["name"] for m in self.data["models"]).lower()
        keywords = {
            "face_detection_yunet_2023mar.onnx": "yunet",
            "retinaface_r34.onnx": "retinaface",
            "yolox_nano.onnx": "yolox",
            "rf-detr-nano-384-coco.onnx": "rf-detr",
            "PP-OCRv6_det_small.onnx": "pp-ocrv6",
            "latin_PP-OCRv5_rec_mobile.onnx": "pp-ocrv5",
        }
        for file in heavy.expected_hashes(os.path.join(ROOT, "models")):
            self.assertIn(
                file, keywords, f"{file} is a new model: give it a keyword here and a row in scripts/third_party.json"
            )
            self.assertIn(keywords[file], names, file)


class ThirdParty(unittest.TestCase):
    def test_pt8_the_text_detector_and_its_runtime_are_listed_with_licence_and_training_data(self):
        text = read("THIRD-PARTY.md")
        row = next((line for line in text.splitlines() if "PP-OCRv6" in line and "|" in line), None)
        self.assertIsNotNone(row, "THIRD-PARTY.md has no row for the PP-OCRv6 text detector")
        self.assertIn("Apache-2.0", row)
        self.assertIn("RapidOCR", row)
        self.assertRegex(row.lower(), r"trained on|training")
        self.assertIn("classes/text.py", row)  # the wrapper that reads the text detector

    def test_pt8_every_model_row_of_the_third_party_list_names_its_training_data(self):
        section = read("THIRD-PARTY.md").split("## Models")[1].split("## Libraries")[0]
        rows = [line for line in section.splitlines() if line.startswith("|") and not set(line) <= set("|- ")][1:]
        self.assertGreaterEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(len(row.split("|")) - 2, 4, row[:40])
            self.assertTrue(row.split("|")[3].strip(), f"no training data in: {row[:40]}")


if __name__ == "__main__":
    unittest.main()
