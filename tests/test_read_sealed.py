"""M2: the sealed part of truth v2 is opened once, by `bench.read_sealed`, for a frozen product, and every read is logged."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from bench import read_sealed as rs


class SealedCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        for rel, text in (
            ("src/anonymizer/engine.py", "x = 1\n"),
            ("models/SHA256SUMS", "abc  m.onnx\n"),
            ("bench/truth/faces/a.json", "{}\n"),
        ):
            path = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        self.split = os.path.join(self.root, "split.json")
        items = {"set-a/1.jpg": "dev", "set-a/2.jpg": "sealed", "blind/1.jpg": "sealed", "wider/1.jpg": "dev"}
        with open(self.split, "w", encoding="utf-8") as f:
            json.dump({"items": {k: {"part": v} for k, v in items.items()}}, f)
        self.paths = dict(
            root=self.root,
            freeze_path=os.path.join(self.root, "results", "freeze.json"),
            split_path=self.split,
        )
        self.reads = os.path.join(self.root, "results", "reads.json")
        self.result = os.path.join(self.root, "results", "sealed.json")
        self.seen = []
        patcher = mock.patch.dict(
            rs.MEASUREMENTS, {"count": lambda keys: self.seen.append(list(keys)) or {"n": len(keys)}}, clear=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def freeze(self):
        return rs.freeze(**self.paths)

    def read(self, reason=None):
        return rs.read(reason, reads_path=self.reads, result_path=self.result, **self.paths)

    def edit(self, rel, text="changed\n"):
        with open(os.path.join(self.root, rel), "w", encoding="utf-8") as f:
            f.write(text)


class Refusals(SealedCase):
    def test_m2_the_dev_keys_never_hold_a_sealed_photo(self):
        self.assertEqual(rs.dev_keys(self.split), ["set-a/1.jpg", "wider/1.jpg"])
        self.assertEqual(rs.sealed_keys(self.split), ["blind/1.jpg", "set-a/2.jpg"])
        self.assertFalse(set(rs.dev_keys(self.split)) & set(rs.sealed_keys(self.split)))

    def test_m2_refuse_sealed_stops_a_sealed_or_unknown_key_and_lets_dev_through(self):
        rs.refuse_sealed(["set-a/1.jpg", "wider/1.jpg"], self.split)
        with self.assertRaises(rs.SealedError):
            rs.refuse_sealed(["set-a/1.jpg", "blind/1.jpg"], self.split)
        with self.assertRaises(rs.SealedError):
            rs.refuse_sealed(["set-a/never-seen.jpg"], self.split)

    def test_m2_a_read_before_the_freeze_is_refused_and_opens_nothing(self):
        with self.assertRaises(rs.SealedError):
            self.read()
        self.assertEqual(self.seen, [])
        self.assertFalse(os.path.exists(self.reads))

    def test_m2_the_product_cannot_be_frozen_twice(self):
        self.freeze()
        with self.assertRaises(rs.SealedError):
            self.freeze()

    def test_m2_a_read_with_no_registered_measurement_is_refused_before_it_is_logged(self):
        self.freeze()
        with mock.patch.dict(rs.MEASUREMENTS, clear=True), self.assertRaises(rs.SealedError):
            self.read()
        self.assertFalse(os.path.exists(self.reads))


class TheFreezeHolds(SealedCase):
    def test_m2_a_change_to_the_product_a_model_list_a_truth_file_or_the_split_voids_the_freeze(self):
        for rel in ("src/anonymizer/engine.py", "models/SHA256SUMS", "bench/truth/faces/a.json", "split.json"):
            shutil.rmtree(os.path.join(self.root, "results"), ignore_errors=True)
            with open(os.path.join(self.root, rel), encoding="utf-8") as f:
                before = f.read()
            self.freeze()
            self.edit(rel, before + " ")
            with self.assertRaises(rs.SealedError, msg=rel):
                self.read()
            self.assertEqual(self.seen, [], rel)
            self.edit(rel, before)

    def test_m2_a_new_or_a_removed_frozen_file_voids_the_freeze_too(self):
        self.freeze()
        self.edit("src/anonymizer/new.py", "y = 2\n")
        with self.assertRaises(rs.SealedError):
            self.read()
        os.remove(os.path.join(self.root, "src/anonymizer/new.py"))
        os.remove(os.path.join(self.root, "src/anonymizer/engine.py"))
        with self.assertRaises(rs.SealedError):
            self.read()


class OneRead(SealedCase):
    def test_m2_the_first_read_runs_every_measurement_on_the_sealed_keys_only_and_is_logged(self):
        self.freeze()
        result = self.read()
        self.assertEqual(self.seen, [["blind/1.jpg", "set-a/2.jpg"]])
        self.assertEqual((result["read"], result["results"]), (1, {"count": {"n": 2}}))
        with open(self.result, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["results"], {"count": {"n": 2}})
        with open(self.reads, encoding="utf-8") as f:
            log = json.load(f)
        self.assertEqual((len(log), log[0]["reason"], log[0]["keys"]), (1, None, 2))

    def test_m2_a_second_read_needs_a_reason_and_the_reason_is_logged(self):
        self.freeze()
        self.read()
        with self.assertRaises(rs.SealedError):
            self.read()
        self.assertEqual(len(self.seen), 1)
        again = self.read("DECISIONS-V2-B 12: a second read was allowed after the fix of the deep tiles")
        self.assertEqual(again["read"], 2)
        with open(self.reads, encoding="utf-8") as f:
            self.assertEqual(
                [r["reason"] for r in json.load(f)],
                [None, "DECISIONS-V2-B 12: a second read was allowed after the fix of the deep tiles"],
            )

    def test_m2_a_read_that_crashes_still_counts_as_the_read(self):
        self.freeze()
        with mock.patch.dict(rs.MEASUREMENTS, {"boom": lambda keys: 1 / 0}):
            result = self.read()
        self.assertEqual(result["failed"], ["boom"])
        self.assertIn("ZeroDivisionError", result["results"]["boom"]["error"])
        self.assertEqual(result["results"]["count"], {"n": 2})  # the others are kept
        with open(self.result, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["failed"], ["boom"])
        with self.assertRaises(rs.SealedError):
            self.read()  # the sealed photos were opened once already

    def test_m2_a_measurement_that_exits_or_is_interrupted_leaves_the_partial_result_on_disk(self):
        self.freeze()

        def hard_exit(keys):
            raise SystemExit("the image is not the one the split was fixed on")

        def interrupt(keys):
            raise KeyboardInterrupt

        with mock.patch.dict(rs.MEASUREMENTS, {"aaa": lambda keys: {"n": 1}, "exit": hard_exit, "zzz": interrupt}):
            with self.assertRaises(KeyboardInterrupt):
                self.read()
        with open(self.result, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["results"]["aaa"], {"n": 1})
        self.assertIn("SystemExit", saved["results"]["exit"]["error"])
        self.assertEqual(saved["failed"], ["exit", "zzz"])

    def test_m2_the_command_line_refuses_with_status_2_and_prints_why(self):
        with (
            mock.patch.object(rs, "FREEZE", os.path.join(self.root, "none.json")),
            mock.patch.object(rs, "READS", self.reads),
        ):
            self.assertEqual(rs.main(["read"]), 2)


if __name__ == "__main__":
    unittest.main()
