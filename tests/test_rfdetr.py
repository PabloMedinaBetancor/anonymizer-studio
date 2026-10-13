"""PT1b: the RF-DETR nano wrapper (vehicles and persons), tested with a fake session and a temporary models folder; the real file is tested
under ANONYMIZER_HEAVY=1 (tests/test_plates_models.py and tests/test_heavy.py style)."""

from __future__ import annotations

import glob
import hashlib
import os
import tempfile
import unittest
from unittest import mock

try:
    import cv2
    import numpy as np

    from anonymizer import classes
    from anonymizer import contract as c
    from anonymizer.classes import context, rfdetr
    from anonymizer.classes.plates import PlateModule, TextPlateModule
    from tests.fakes import ColourTextDetector
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

HEAVY = os.environ.get("ANONYMIZER_HEAVY") == "1"
DATA = os.environ.get("ANONYMIZER_BENCH_DATA", "")
EU = os.path.join(DATA, "plates", "openalpr-eu")
SET_A = os.path.join(DATA, "faces", "set-a")
CAR, MOTORCYCLE, BUS, TRUCK, PERSON, DOG = 3, 4, 6, 8, 1, 18  # columns of the 91-column export


class _Named:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Answers every run with `queries`: (cx, cy, w, h, {column: probability}); records the inputs."""

    def __init__(self, queries=(), fail=False, nan_in=None):
        self.queries, self.fail, self.inputs = list(queries), fail, []
        self.nan_in = nan_in  # "logits" or "boxes": that output comes back full of NaN

    def get_inputs(self):
        return [_Named("input")]

    def get_outputs(self):
        return [_Named("dets"), _Named("labels")]

    def run(self, _outputs, feed):
        self.inputs.append(feed["input"])
        if self.fail:
            raise RuntimeError("fake failure")
        n = max(len(self.queries), 1)
        dets = np.zeros((1, n, 4), np.float32)
        logits = np.full((1, n, 91), -12.0, np.float32)
        for i, (cx, cy, w, h, probs) in enumerate(self.queries):
            dets[0, i] = (cx, cy, w, h)
            for column, p in probs.items():
                logits[0, i, column] = np.log(p / (1 - p))
        if self.nan_in == "logits":
            logits[:] = np.nan
        elif self.nan_in == "boxes":
            dets[:] = np.nan
        return [dets, logits]


class Folder:
    """A temporary models folder with a stand-in for the RF-DETR file and a SHA256SUMS that lists it."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, rfdetr.RFDETR_MODEL)
        self.write(b"not the real model")
        return self

    def write(self, data: bytes, listed: bool = True):
        with open(self.path, "wb") as f:
            f.write(data)
        with open(os.path.join(self.tmp.name, "SHA256SUMS"), "w") as f:
            if listed:
                f.write(f"{hashlib.sha256(data).hexdigest()}  {rfdetr.RFDETR_MODEL}\n")

    def __exit__(self, *exc):
        for key in [k for k in rfdetr._SESSIONS if k[0].startswith(self.tmp.name)]:
            del rfdetr._SESSIONS[key]  # a test that did not close its detectors leaves nothing behind
        self.tmp.cleanup()


def detector(folder, session, **kw):
    """A detector whose session opening is replaced by `session` (the open still reads and checks the bytes)."""
    det = rfdetr.RFDetrDetector(models=folder.tmp.name, **kw)
    return det, mock.patch.object(rfdetr, "open_session", return_value=session)


def picture(w=800, h=600):
    return np.random.RandomState(1).randint(0, 255, (h, w, 3), dtype=np.uint8)


class Decoding(unittest.TestCase):
    def test_a_car_query_becomes_a_vehicle_box_in_the_pixels_of_the_picture(self):
        with Folder() as folder:
            fake = FakeSession([(0.5, 0.5, 0.25, 0.5, {CAR: 0.9})])
            det, patch = detector(folder, fake)
            with patch:
                found = det.detect(picture())
        self.assertEqual(len(found), 1)
        d = found[0]
        self.assertEqual(d.kind, c.VEHICLE)
        self.assertEqual([round(v, 3) for v in d.box()], [300.0, 150.0, 500.0, 450.0])
        self.assertAlmostEqual(d.score, 0.9, places=3)

    def test_a_region_gives_boxes_in_the_pixels_of_the_whole_picture_and_a_squashed_384_input(self):
        with Folder() as folder:
            fake = FakeSession([(0.5, 0.5, 0.5, 0.5, {TRUCK: 0.8})])
            det, patch = detector(folder, fake)
            with patch:
                found = det.detect(picture(), (100, 200, 500, 400))  # 400 x 200 cut
        self.assertEqual([round(v, 3) for v in found[0].box()], [200.0, 250.0, 400.0, 350.0])
        self.assertEqual(fake.inputs[0].shape, (1, 3, 384, 384))
        self.assertEqual(fake.inputs[0].dtype, np.float32)

    def test_the_input_is_rgb_scaled_with_the_imagenet_mean_and_deviation(self):
        bgr = np.zeros((100, 100, 3), np.uint8)
        bgr[:, :] = (0, 128, 255)  # B, G, R
        with Folder() as folder:
            fake = FakeSession()
            det, patch = detector(folder, fake)
            with patch:
                det.detect(bgr)
        blob = fake.inputs[0][0]
        for channel, value in enumerate((255, 128, 0)):  # R, G, B planes
            expected = (value / 255.0 - rfdetr.MEAN[channel]) / rfdetr.STD[channel]
            self.assertAlmostEqual(float(blob[channel].mean()), float(expected), places=4)

    def test_the_four_vehicle_classes_count_and_a_person_or_a_dog_does_not(self):
        queries = [
            (0.1 + 0.1 * i, 0.5, 0.05, 0.05, {col: 0.9})
            for i, col in enumerate((CAR, MOTORCYCLE, BUS, TRUCK, PERSON, DOG))
        ]
        with Folder() as folder:
            det, patch = detector(folder, FakeSession(queries))
            with patch:
                self.assertEqual(len(det.detect(picture())), 4)

    def test_a_bicycle_is_not_a_vehicle(self):
        queries = [(0.3, 0.5, 0.1, 0.1, {2: 0.95}), (0.7, 0.5, 0.1, 0.1, {CAR: 0.95})]  # column 2 is the bicycle
        with Folder() as folder:
            det, patch = detector(folder, FakeSession(queries))
            with patch:
                self.assertEqual(len(det.detect(picture())), 1)

    def test_the_person_detector_keeps_only_persons_with_kind_person(self):
        queries = [(0.3, 0.5, 0.1, 0.4, {PERSON: 0.9}), (0.7, 0.5, 0.2, 0.2, {CAR: 0.95})]
        with Folder() as folder:
            det = rfdetr.RFDetrDetector(models=folder.tmp.name, columns=rfdetr.PERSON_COLUMNS, kind=context.PERSON)
            with mock.patch.object(rfdetr, "open_session", return_value=FakeSession(queries)):
                found = det.detect(picture())
        self.assertEqual([d.kind for d in found], [context.PERSON])

    def test_the_threshold_is_0_30_by_default_and_a_call_can_change_it(self):
        queries = [(0.5, 0.5, 0.2, 0.2, {CAR: 0.29}), (0.2, 0.2, 0.1, 0.1, {CAR: 0.31})]
        with Folder() as folder:
            det, patch = detector(folder, FakeSession(queries))
            with patch:
                self.assertEqual(len(det.detect(picture())), 1)
                self.assertEqual(len(det.detect(picture(), None, 0.2)), 2)
                self.assertEqual(len(det.detect(picture(), None, 0.5)), 0)

    def test_a_box_beyond_the_picture_is_clipped_and_an_empty_one_is_dropped(self):
        queries = [(0.95, 0.5, 0.3, 0.2, {CAR: 0.9}), (1.2, 0.5, 0.1, 0.1, {CAR: 0.9})]
        with Folder() as folder:
            det, patch = detector(folder, FakeSession(queries))
            with patch:
                found = det.detect(picture())
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0].x1, 800.0)

    def test_a_cut_under_16_px_is_not_looked_at(self):
        with Folder() as folder:
            fake = FakeSession([(0.5, 0.5, 0.5, 0.5, {CAR: 0.9})])
            det, patch = detector(folder, fake)
            with patch:
                self.assertEqual(det.detect(picture(), (10, 10, 20, 400)), [])
        self.assertEqual(fake.inputs, [])


class Safety(unittest.TestCase):
    def test_a_run_that_raises_is_counted_and_gives_nothing(self):
        with Folder() as folder:
            det, patch = detector(folder, FakeSession(fail=True))
            with patch:
                self.assertEqual(det.detect(picture()), [])
                self.assertEqual(det.detect(picture()), [])
        self.assertEqual(det.run_failures, 2)

    def test_a_non_finite_output_is_a_counted_failure_not_nothing_found(self):
        # RV-RFDETR F2: all-NaN logits gave [] with no count, NaN boxes gave one box over the whole cut
        for where in ("logits", "boxes"):
            with self.subTest(where):
                with Folder() as folder:
                    queries = [(0.5, 0.5, 0.25, 0.5, {CAR: 0.9})]
                    det, patch = detector(folder, FakeSession(queries, nan_in=where))
                    with patch:
                        self.assertEqual(det.detect(picture()), [])
                        self.assertEqual(det.run_failures, 1)
                        self.assertEqual(det.detect(picture()), [])
                        self.assertEqual(det.run_failures, 2)

    def test_the_file_is_checked_when_the_wrapper_is_built(self):
        with Folder() as folder:
            os.remove(folder.path)
            with self.assertRaises(c.ModelLoadError):
                rfdetr.RFDetrDetector(models=folder.tmp.name)
        with Folder() as folder:
            folder.write(b"unlisted", listed=False)
            with self.assertRaises(c.ModelLoadError):
                rfdetr.RFDetrDetector(models=folder.tmp.name)
        with Folder() as folder:
            hashes = os.path.join(folder.tmp.name, "SHA256SUMS")
            with open(hashes, "w") as f:
                f.write(f"{'0' * 64}  {rfdetr.RFDETR_MODEL}\n")
            with self.assertRaises(c.ModelLoadError):
                rfdetr.RFDetrDetector(models=folder.tmp.name)

    def test_nothing_opens_before_the_first_picture(self):
        with Folder() as folder:
            det, patch = detector(folder, FakeSession())
            with patch as opened:
                self.assertIsNone(det.session)
                opened.assert_not_called()
                det.detect(picture())
                det.detect(picture())
                self.assertEqual(opened.call_count, 1)
                self.assertIsNotNone(det.session)

    def test_a_file_replaced_after_the_build_is_refused_at_the_first_picture(self):
        with Folder() as folder:
            det, patch = detector(folder, FakeSession())
            folder.write(b"something else", listed=False)
            with patch as opened, self.assertRaises(c.ModelLoadError):
                det.detect(picture())
            opened.assert_not_called()  # the session is never built from bytes that were not checked

    def test_a_session_that_is_not_a_detector_is_refused_at_open(self):
        class Wrong(FakeSession):
            def get_outputs(self):
                return [_Named("only")]

        with Folder() as folder:
            det, patch = detector(folder, Wrong())
            with patch, self.assertRaises(c.ModelLoadError):
                det.detect(picture())
            self.assertIsNone(det.session)

    def test_the_vehicle_and_the_person_detector_share_one_session_and_the_last_to_close_frees_it(self):
        with Folder() as folder:
            cars = rfdetr.RFDetrDetector(models=folder.tmp.name)
            people = rfdetr.RFDetrDetector(models=folder.tmp.name, columns=rfdetr.PERSON_COLUMNS, kind=context.PERSON)
            with mock.patch.object(rfdetr, "open_session", return_value=FakeSession()) as opened:
                cars.detect(picture())
                people.detect(picture())
                self.assertEqual(opened.call_count, 1)
                self.assertIs(cars.session, people.session)
                cars.close()
                people.detect(picture())  # still works: the other holds the session
                self.assertEqual(len(rfdetr._SESSIONS), 1)
                people.close()
                self.assertEqual(len(rfdetr._SESSIONS), 0)
                again = rfdetr.RFDetrDetector(models=folder.tmp.name)
                again.detect(picture())
                self.assertEqual(opened.call_count, 2)  # nothing was kept alive
                again.close()

    def test_assigning_none_closes_it_for_good_as_the_plate_modules_do(self):
        with Folder() as folder:
            det, patch = detector(folder, FakeSession())
            with patch:
                det.detect(picture())
                det.session = None
                self.assertIsNone(det.session)
                with self.assertRaises(RuntimeError):
                    det.detect(picture())


class Wiring(unittest.TestCase):
    def test_load_refuses_an_unknown_vehicle_model(self):
        with self.assertRaises(ValueError):
            classes.load([c.PLATE], vehicle_model="detr")

    def test_the_text_plate_module_counts_the_vehicle_failures_with_the_text_ones(self):
        with Folder() as folder:
            det, patch = detector(folder, FakeSession(fail=True))
            module = TextPlateModule(vehicles=det, text=ColourTextDetector(()))
            with patch:
                module.vehicles.detect(picture())
            self.assertEqual(module.model_failures, 1)
            module.close()
            self.assertIsNone(det.session)

    def test_the_yolo_plate_module_counts_the_vehicle_failures_with_the_plate_ones(self):
        with Folder() as folder:
            det, patch = detector(folder, FakeSession(fail=True))
            module = PlateModule.__new__(PlateModule)
            module.plates, module.vehicles = mock.Mock(run_failures=2), det
            with patch:
                det.detect(picture())
            self.assertEqual(module.model_failures, 3)
            det.close()

    def test_the_yolox_choice_is_untouched(self):
        with mock.patch.object(PlateModule, "__init__", return_value=None) as built:
            with mock.patch("anonymizer.classes.rfdetr.load_vehicle_detector") as load:
                classes.load([c.PLATE], plate_path="yolo", vehicle_model="yolox")
                load.assert_not_called()
        self.assertIsNone(built.call_args.kwargs["vehicles"])

    def test_the_rfdetr_default_hands_the_detector_to_the_plate_module(self):
        sentinel = object()
        for kw in (dict(), dict(vehicle_model="rfdetr")):
            with mock.patch.object(TextPlateModule, "__init__", return_value=None) as built:
                with mock.patch("anonymizer.classes.rfdetr.load_vehicle_detector", return_value=sentinel):
                    classes.load([c.PLATE], **kw)
            self.assertIs(built.call_args.kwargs["vehicles"], sentinel)
        with mock.patch.object(PlateModule, "__init__", return_value=None) as built:
            with mock.patch("anonymizer.classes.rfdetr.load_vehicle_detector", return_value=sentinel):
                classes.load([c.PLATE], plate_path="yolo", vehicle_model="rfdetr")
        self.assertIs(built.call_args.kwargs["vehicles"], sentinel)


class Persons(unittest.TestCase):
    def test_the_finder_with_rfdetr_checks_the_rfdetr_file_at_build_and_finds_persons(self):
        with Folder() as folder:
            finder = context.PersonFinder(models=folder.tmp.name, person_model="rfdetr")
            fake = FakeSession([(0.5, 0.5, 0.1, 0.4, {PERSON: 0.9})])
            with mock.patch.object(rfdetr, "open_session", return_value=fake):
                persons = finder.persons(picture())
            self.assertEqual([p.kind for p in persons], [context.PERSON])
            finder.close()

    def test_the_finder_counts_the_failures_and_keeps_the_count_after_close(self):
        with Folder() as folder:
            finder = context.PersonFinder(models=folder.tmp.name, person_model="rfdetr")
            with mock.patch.object(rfdetr, "open_session", return_value=FakeSession(fail=True)):
                self.assertEqual(finder.persons(picture()), [])
            self.assertEqual(finder.run_failures, 1)
            finder.close()
            self.assertEqual(finder.run_failures, 1)

    def test_an_unknown_person_model_is_refused(self):
        with self.assertRaises(ValueError):
            context.PersonFinder(person_model="detr")
        with self.assertRaises(ValueError):
            context.load_person_detector(person_model="detr")

    def test_the_finder_without_the_rfdetr_file_fails_at_build_not_at_the_first_picture(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(c.ModelLoadError):
                context.PersonFinder(models=empty, person_model="rfdetr")


@unittest.skipUnless(HEAVY, "set ANONYMIZER_HEAVY=1 to run the tests that load the models")
class WithTheRealFile(unittest.TestCase):
    def test_a_street_photo_gives_vehicles_and_a_crowd_photo_gives_persons(self):
        eu = [p for p in sorted(glob.glob(os.path.join(EU, "*.jpg")))[:12]]
        crowd = sorted(glob.glob(os.path.join(SET_A, "*.jpg")))
        if not eu or not crowd:
            raise AssertionError(f"no photos in {EU} or {SET_A}: run scripts/fetch_models.py")
        cars = rfdetr.load_vehicle_detector(threads=2)
        people = context.load_person_detector(threads=2, person_model="rfdetr")
        try:
            vehicles = sum(len(cars.detect(cv2.imread(p))) for p in eu)
            self.assertGreater(vehicles, len(eu) // 2)
            found = context.find_persons(people, cv2.imread(crowd[0]))
            self.assertGreater(len(found), 3)
            self.assertEqual({d.kind for d in found}, {context.PERSON})
            self.assertEqual(cars.run_failures + people.run_failures, 0)
        finally:
            cars.close()
            people.close()


if __name__ == "__main__":
    unittest.main()
