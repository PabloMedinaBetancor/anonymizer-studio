"""The heavy face wrappers (M1): geometry and failure behaviour against fake runs, and the real models on a public photo.

The real-model class is off unless ANONYMIZER_HEAVY=1 (see tests/test_plates_models.py); on the development laptop it runs with it on.
"""

from __future__ import annotations

import glob
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from anonymizer import contract
from anonymizer.classes import context, heavy
from anonymizer.classes.heavy import HeavyRunError, RetinaFaceDetector, ZoomedYuNet
from tests.heavy_fakes import Scripted

HEAVY = os.environ.get("ANONYMIZER_HEAVY") == "1"
SET_A = os.path.join(os.environ.get("ANONYMIZER_BENCH_DATA", ""), "faces", "set-a")


class ZoneGeometry(unittest.TestCase):
    def test_a_box_found_in_the_enlarged_crop_maps_back_to_the_picture(self):
        bgr = np.zeros((500, 800, 3), np.uint8)
        d = Scripted([(120.0, 80.0, 200.0, 160.0, 0.9)])
        (found,) = d.detect(bgr, (100, 50, 400, 250), 2.0, 0.5)
        self.assertEqual(d.pieces, [(400, 600)])
        self.assertAlmostEqual(found.x0, 100 + 120 / 2, delta=0.5)
        self.assertAlmostEqual(found.y0, 50 + 80 / 2, delta=0.5)
        self.assertAlmostEqual(found.x1, 100 + 200 / 2, delta=0.5)
        self.assertAlmostEqual(found.y1, 50 + 160 / 2, delta=0.5)
        self.assertEqual((found.kind, found.source), (contract.FACE, contract.SECOND))

    def test_a_shrunk_region_maps_back_too(self):
        bgr = np.zeros((1000, 1000, 3), np.uint8)
        d = Scripted([(10.0, 20.0, 50.0, 60.0, 0.9)])
        (found,) = d.detect(bgr, (0, 0, 1000, 1000), 0.5, 0.5)
        self.assertEqual(d.pieces, [(500, 500)])
        self.assertAlmostEqual(found.x1, 100.0, delta=0.5)
        self.assertAlmostEqual(found.y1, 120.0, delta=0.5)

    def test_a_region_past_the_picture_is_clipped(self):
        bgr = np.zeros((100, 200, 3), np.uint8)
        d = Scripted([(0.0, 0.0, 400.0, 400.0, 0.9)])
        (found,) = d.detect(bgr, (-50, -50, 900, 900), 1.0, 0.5)
        self.assertEqual(d.pieces, [(100, 200)])
        self.assertLessEqual(found.x1, 200)
        self.assertLessEqual(found.y1, 100)

    def test_a_crop_too_big_for_one_call_is_split_and_a_face_on_the_seam_is_found_once(self):
        bgr = np.zeros((400, 600, 3), np.uint8)
        bgr[25:35, 250:260] = (
            255  # a "face" that lies inside the overlap of two pieces once the crop is enlarged 4 times
        )

        def rows(_, img):
            ys, xs = np.where(img[:, :, 0] > 200)
            return [(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9)] if len(xs) >= 100 else []

        d = Scripted(rows, max_side=640)
        found = d.detect(bgr, (0, 0, 600, 400), 4.0, 0.5)
        self.assertGreater(len(d.pieces), 1)
        self.assertTrue(all(max(p) <= 640 for p in d.pieces))
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0].x0, 250.0, delta=1.5)
        self.assertAlmostEqual(found[0].y1, 35.0, delta=1.5)

    def test_the_slowest_model_run_is_kept_so_that_the_limit_of_one_and_a_half_seconds_can_be_checked(self):
        import time

        def rows(i, img):
            time.sleep(0.03 if i == 1 else 0.0)
            return []

        d = Scripted(rows, max_side=100)
        d.detect(np.zeros((200, 200, 3), np.uint8), (0, 0, 200, 200), 1.0, 0.5)
        self.assertGreater(len(d.pieces), 2)
        self.assertGreaterEqual(d.slowest_piece, 0.03)
        self.assertLess(d.slowest_piece, 0.5)

    def test_scores_below_the_cut_and_empty_regions_give_nothing(self):
        bgr = np.zeros((100, 100, 3), np.uint8)
        self.assertEqual(Scripted([(1.0, 1.0, 20.0, 20.0, 0.4)]).detect(bgr, (0, 0, 100, 100), 1.0, 0.5), [])
        self.assertEqual(Scripted([(1.0, 1.0, 20.0, 20.0, 0.9)]).detect(bgr, (10, 10, 12, 12), 1.0, 0.5), [])


class ModelFiles(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        with open(os.path.join(self.folder, "m.onnx"), "wb") as f:
            f.write(b"not a model")
        self.digest = heavy.sha256_of(os.path.join(self.folder, "m.onnx"))

    def _sums(self, text):
        with open(os.path.join(self.folder, "SHA256SUMS"), "w", encoding="utf-8") as f:
            f.write(text)

    def test_a_listed_file_is_accepted_and_its_hash_returned(self):
        self._sums(f"{self.digest}  m.onnx\n")
        path, digest = heavy.verified_model("face", self.folder, "m.onnx")
        self.assertEqual((path, digest), (os.path.join(self.folder, "m.onnx"), self.digest))

    def test_a_changed_file_a_missing_file_and_an_unlisted_file_are_refused(self):
        self._sums(f"{'0' * 64}  m.onnx\n")
        with self.assertRaises(contract.ModelLoadError) as changed:
            heavy.verified_model("face", self.folder, "m.onnx")
        self.assertIn("does not match", str(changed.exception))
        with self.assertRaises(contract.ModelLoadError) as missing:
            heavy.verified_model("face", self.folder, "absent.onnx")
        self.assertIn("ANONYMIZER_MODELS", str(missing.exception))
        self._sums("")
        with self.assertRaises(contract.ModelLoadError) as unlisted:
            heavy.verified_model("face", self.folder, "m.onnx")
        self.assertIn("no entry", str(unlisted.exception))

    def test_the_wrappers_check_the_hash_when_built_not_when_first_used(self):
        self._sums("")
        for build in (RetinaFaceDetector, ZoomedYuNet):
            with self.assertRaises(contract.ModelLoadError):
                build(models=self.folder)

    def test_an_unknown_heavy_model_is_a_value_error(self):
        with self.assertRaises(ValueError):
            heavy.load_face_heavy("nope")


class Failures(unittest.TestCase):
    class Broken:
        def run(self, *_):
            raise RuntimeError("boom")

    class NotANumber:
        def run(self, *_):
            return np.full((1, 42, 4), np.nan, np.float32), np.full((1, 42, 2), np.nan, np.float32), None

    class BrokenNet:
        """What `cv2.FaceDetectorYN` is to the control: it takes the settings and fails when asked to detect."""

        def __init__(self, rows=None):
            self.rows = rows

        def setScoreThreshold(self, _):
            pass

        def setInputSize(self, _):
            pass

        def detect(self, _):
            if self.rows is None:
                raise cv2.error("boom")
            return None, self.rows

    def test_a_run_that_raises_is_a_heavy_run_error_never_an_empty_result(self):
        d = RetinaFaceDetector()
        d._session = self.Broken()
        with self.assertRaises(HeavyRunError):
            d.detect(np.zeros((200, 200, 3), np.uint8), (0, 0, 200, 200), 1.0, 0.5)

    def test_the_control_that_fails_is_a_heavy_run_error_too(self):
        d = ZoomedYuNet()
        d._net = self.BrokenNet()
        with self.assertRaises(HeavyRunError):
            d.detect(np.zeros((200, 200, 3), np.uint8), (0, 0, 200, 200), 1.0, 0.5)

    def test_scores_that_are_not_numbers_are_a_heavy_run_error_not_nothing_found(self):
        d = RetinaFaceDetector()
        d._session = self.NotANumber()
        with self.assertRaises(HeavyRunError):
            d.detect(np.zeros((32, 32, 3), np.uint8), (0, 0, 32, 32), 1.0, 0.5)
        c = ZoomedYuNet()
        c._net = self.BrokenNet(np.full((1, 15), np.nan, np.float32))
        with self.assertRaises(HeavyRunError):
            c.detect(np.zeros((32, 32, 3), np.uint8), (0, 0, 32, 32), 1.0, 0.5)

    def test_a_closed_detector_refuses_to_run(self):
        for d in (RetinaFaceDetector(), ZoomedYuNet()):
            d.close()
            with self.assertRaises(HeavyRunError):
                d.detect(np.zeros((200, 200, 3), np.uint8), (0, 0, 200, 200), 1.0, 0.5)


class FileSwapped(unittest.TestCase):
    """CL3 says every load checks the hash: the session is built from the bytes that were checked, not from whatever the file is by then."""

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder, True)
        source = contract.models_dir()
        shutil.copy(os.path.join(source, "SHA256SUMS"), self.folder)
        self.model = os.path.join(self.folder, heavy.YUNET_FILE)
        shutil.copy(os.path.join(source, heavy.YUNET_FILE), self.model)

    def test_a_file_replaced_after_the_wrapper_was_built_is_refused_at_the_first_call(self):
        d = ZoomedYuNet(models=self.folder)
        with open(self.model, "wb") as f:
            f.write(b"not the model any more")
        with self.assertRaises(contract.ModelLoadError) as raised:
            d.detect(np.zeros((200, 200, 3), np.uint8), (0, 0, 200, 200), 1.0, 0.5)
        self.assertIn("does not match", str(raised.exception))

    def test_retinaface_refuses_a_file_replaced_by_another_valid_model_after_it_was_built(self):
        """The stand-in is a small valid ONNX listed under the RetinaFace name; the swap is another valid ONNX, which a session built from the path would accept."""
        source = contract.models_dir()
        stand_in = os.path.join(self.folder, heavy.RETINAFACE_FILE)
        shutil.copy(os.path.join(source, heavy.YUNET_FILE), stand_in)
        with open(os.path.join(self.folder, "SHA256SUMS"), "a", encoding="utf-8") as f:
            f.write(f"{heavy.sha256_of(stand_in)}  {heavy.RETINAFACE_FILE}\n")
        d = RetinaFaceDetector(models=self.folder)
        shutil.copy(os.path.join(source, context.PERSON_MODEL), stand_in)  # a different valid model
        with self.assertRaises(contract.ModelLoadError) as raised:
            d.detect(np.zeros((64, 64, 3), np.uint8), (0, 0, 64, 64), 1.0, 0.5)
        self.assertIn("does not match", str(raised.exception))

    def test_a_file_that_changes_between_two_verifications_in_one_process_is_not_taken_from_the_cache(self):
        heavy.verified_model("face", self.folder, heavy.YUNET_FILE)
        with open(self.model, "ab") as f:
            f.write(b"\0")  # another size and mtime, so the remembered hash must not be used
        with self.assertRaises(contract.ModelLoadError):
            heavy.verified_model("face", self.folder, heavy.YUNET_FILE)

    def test_the_person_detector_refuses_a_file_replaced_after_the_finder_was_built(self):
        source = contract.models_dir()
        shutil.copy(os.path.join(source, context.PERSON_MODEL), self.folder)
        with open(os.path.join(self.folder, "SHA256SUMS"), "a", encoding="utf-8") as f:
            f.write(f"{heavy.sha256_of(os.path.join(self.folder, context.PERSON_MODEL))}  {context.PERSON_MODEL}\n")
        finder = context.PersonFinder(models=self.folder)
        with open(os.path.join(self.folder, context.PERSON_MODEL), "wb") as f:
            f.write(b"not the model any more")
        with self.assertRaises(contract.ModelLoadError):
            finder.persons(np.zeros((200, 200, 3), np.uint8))

    def test_the_bytes_that_are_given_to_the_session_are_the_ones_that_matched(self):
        data = heavy.read_verified("face", self.model, heavy.expected_hashes(self.folder)[heavy.YUNET_FILE])
        self.assertEqual(len(data), os.path.getsize(self.model))
        with self.assertRaises(contract.ModelLoadError):
            heavy.read_verified("face", self.model, "0" * 64)


class Context(unittest.TestCase):
    def test_a_person_finder_opens_nothing_until_it_is_asked_and_checks_the_hash_when_built(self):
        finder = context.PersonFinder()
        self.assertIsNone(finder._detector)
        finder.close()
        folder = tempfile.mkdtemp()
        with self.assertRaises(contract.ModelLoadError):
            context.PersonFinder(models=folder)

    def test_a_head_zone_is_the_widened_top_third_of_a_person(self):
        x0, y0, x1, y1 = context.head_zone((100.0, 200.0, 200.0, 600.0))
        self.assertEqual((x0, y0, x1), (90.0, 200.0, 210.0))
        self.assertAlmostEqual(y1, 200 + 0.35 * 400)


@unittest.skipUnless(HEAVY, "set ANONYMIZER_HEAVY=1 to run the tests that load the models")
class WithModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = sorted(glob.glob(os.path.join(SET_A, "*.jpg")))
        if not files:
            raise AssertionError(f"no photos in {SET_A}: run scripts/fetch_models.py")
        cls.bgr = cv2.imread(files[0])
        cls.h, cls.w = cls.bgr.shape[:2]

    def _check(self, detector):
        whole = detector.detect(self.bgr, (0, 0, self.w, self.h), 1.0, 0.7)
        self.assertGreater(len(whole), 3)
        self.assertTrue(all(0 <= d.x0 < d.x1 <= self.w and 0 <= d.y0 < d.y1 <= self.h for d in whole))
        best = max(whole, key=lambda d: d.w * d.h)
        pad = max(best.w, best.h) * 0.25
        zone = (best.x0 - pad, best.y0 - pad, best.x1 + pad, best.y1 + pad)
        again = detector.detect(self.bgr, zone, 256 / (2.5 * max(best.w, best.h)), 0.5)
        from anonymizer.geometry import iou

        self.assertTrue(
            any(iou(a.box(), best.box()) > 0.5 for a in again), "the face is not found again in its own zone"
        )
        detector.close()

    def test_retinaface_finds_the_faces_of_a_crowd_photo_and_again_in_its_zone(self):
        self._check(heavy.load_face_heavy("retinaface-r34"))

    def test_the_control_finds_them_too(self):
        self._check(heavy.load_face_heavy("yunet-zoomed"))

    def test_persons_are_found_with_kind_person_and_the_vehicle_pass_geometry(self):
        detector = context.load_person_detector()
        persons = context.find_persons(detector, self.bgr)
        self.assertGreater(len(persons), 3)
        self.assertEqual({p.kind for p in persons}, {context.PERSON})


if __name__ == "__main__":
    unittest.main()
