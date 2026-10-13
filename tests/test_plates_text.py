"""PT2, PT3, PT6, PT9: plates by text on vehicles, tested with a fake vehicle detector and a fake text detector; the
real text model is tested in tests/test_text_models.py."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
    from PIL import Image

    import anonymizer
    from anonymizer import contract as c
    from anonymizer.classes import plates as plates_module
    from anonymizer.classes import text as text_det
    from anonymizer.classes.plates import PlateModule, TextPlateModule
    from anonymizer.engine import photos
    from tests import test_api as api_tests
    from tests.fakes import ColourTextDetector, FixedVehicles
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

WHITE, YELLOW = (250, 250, 250), (0, 200, 255)
VEHICLE_BOX = (40, 60, 360, 260)
PLATE_BOX = (120, 180, 180, 196)  # white, 60 x 16: shaped like a plate
LOGO_BOX = (250, 100, 280, 130)  # yellow, 30 x 30: not shaped like one
SIZE = (400, 300)  # width, height


def scene(*regions) -> np.ndarray:
    """A noisy picture with a solid block of colour for every (box, colour) region."""
    width, height = SIZE
    bgr = (np.random.RandomState(5).rand(height, width, 3) * 90 + 20).astype(np.uint8)
    for (x0, y0, x1, y1), colour in regions:
        bgr[y0:y1, x0:x1] = colour
    return bgr


def text_module(vehicles=(VEHICLE_BOX,), colours=((WHITE, 0.9), (YELLOW, 0.45)), **kw) -> TextPlateModule:
    return TextPlateModule(vehicles=FixedVehicles(vehicles), text=ColourTextDetector(colours, **kw))


def plan(**kw) -> c.SweepPlan:
    return photos.photo_plan("careful") if not kw else c.SweepPlan(1600, 640, vehicle_tiles=True, **kw)


def covers(found, box, slack=4.0) -> bool:
    return any(
        abs(d.x0 - box[0]) <= slack
        and abs(d.y0 - box[1]) <= slack
        and abs(d.x1 - box[2]) <= slack
        and abs(d.y1 - box[3]) <= slack
        for d in found
    )


class PT2TextPath(unittest.TestCase):
    def test_pt2_a_vehicle_with_two_text_regions_gives_two_plate_detections_on_it(self):
        bgr = scene((PLATE_BOX, WHITE), (LOGO_BOX, YELLOW))
        module = text_module()
        found = module.sweep(bgr, plan())
        self.assertEqual(len(found), 2)
        self.assertTrue(covers(found, PLATE_BOX) and covers(found, LOGO_BOX), [d.box() for d in found])
        self.assertEqual({(d.kind, d.on_vehicle) for d in found}, {(c.PLATE, True)})

    def test_pt2_the_decision_layer_lists_both_as_plate_items_with_the_text_reason(self):
        zones = photos.decide_photo(scene((PLATE_BOX, WHITE), (LOGO_BOX, YELLOW)), [text_module()])
        self.assertEqual(
            sorted((z.kind, z.judgement.reason) for z in zones), [(c.PLATE, "plate.reason.vehicle_text")] * 2
        )

    def test_pt2_text_off_every_vehicle_gives_nothing_and_the_text_detector_is_not_even_asked(self):
        bgr = scene(((10, 10, 70, 26), WHITE))  # a sign: no vehicle anywhere
        module = text_module(vehicles=())
        self.assertEqual(module.sweep(bgr, plan()), [])
        self.assertEqual(module.text.crops, [])
        beside = text_module()  # a vehicle exists, the writing is outside its crop
        self.assertEqual(beside.sweep(scene(((10, 10, 70, 26), WHITE)), plan()), [])

    def test_pt2_the_crop_is_enlarged_so_its_longest_side_measures_the_vehicle_side(self):
        module = text_module()
        module.sweep(scene((PLATE_BOX, WHITE)), plan())
        [(height, width, _)] = module.text.crops
        self.assertEqual(max(height, width), module.text_path.vehicle_side)
        # the crop is the vehicle's box 1.15 times larger: 368 x 230 pixels of the picture
        self.assertAlmostEqual(width / height, 368 / 230, delta=0.01)

    def test_pt2_the_vehicle_stage_is_the_one_of_the_yolo_path(self):
        self.assertIs(TextPlateModule._find_vehicles, PlateModule._find_vehicles)
        big = np.zeros((1200, 1600, 3), np.uint8)
        boxes = [(100, 100, 400, 300), (900, 700, 1300, 1000)]
        text, yolo = text_module(vehicles=boxes), PlateModule.__new__(PlateModule)
        plates_module._VehicleStage.__init__(yolo)  # only the vehicle stage of the yolo path is under test
        yolo.vehicles, yolo.thresholds = FixedVehicles(boxes), plates_module.DEFAULT_THRESHOLDS
        self.assertEqual(
            [d.box() for d in text._find_vehicles(big, True)], [d.box() for d in yolo._find_vehicles(big, True)]
        )
        self.assertEqual(text.vehicles.calls, yolo.vehicles.calls)  # the same whole-picture look and the same tiles
        self.assertGreater(len(text.vehicles.calls), 1)

    def test_pt2_a_small_vehicle_inside_a_big_box_is_kept_by_the_vehicle_merge(self):
        # B's finding (03/10/2026): the face rule of `nms` (a box 80 % inside another is a repeat) dropped a car inside a big
        # box that merged several cars; the merge is by IoU alone, as in the pilot (PT4: 67 -> 76 plates at 12 vehicles).
        big_box, small_box = (100, 100, 900, 700), (300, 300, 420, 380)
        boxes = [big_box, small_box, (1000, 800, 1300, 1000)]
        module = PlateModule.__new__(PlateModule)
        plates_module._VehicleStage.__init__(module)
        module.vehicles, module.thresholds = FixedVehicles(boxes), plates_module.DEFAULT_THRESHOLDS
        found = [d.box() for d in module._find_vehicles(np.zeros((1200, 1600, 3), np.uint8), True)]
        self.assertEqual(sorted(found), sorted(boxes))

    def test_pt2_vehicles_too_small_are_not_searched(self):
        module = text_module(vehicles=[(40, 60, 90, 90)])  # 50 x 30: under the 60 x 40 minimum
        self.assertEqual(module.sweep(scene(((50, 70, 70, 80), WHITE)), plan()), [])
        self.assertEqual((module.vehicles_visited, module.vehicles_not_visited), (0, 0))
        self.assertEqual(module.vehicles_too_small, 1)  # counted, as the ones beyond the budget are (PT6)

    def test_pt2_a_vehicle_is_too_small_when_it_is_under_the_minimum_in_width_or_in_height(self):
        for box, searched in (((40, 60, 140, 90), False), ((40, 60, 80, 160), False), ((40, 60, 100, 100), True)):
            module = text_module(vehicles=[box])
            module.sweep(scene(), plan())
            self.assertEqual(
                (module.vehicles_visited, module.vehicles_too_small), (int(searched), int(not searched)), box
            )

    def test_pt2_text_regions_that_overlap_become_one_region_covering_both(self):
        overlapping = (
            (120, 180, 180, 196),
            (130, 182, 190, 198),
        )  # a white and a yellow block sharing most of their area
        module = text_module()
        found = module.sweep(scene((overlapping[0], WHITE), (overlapping[1], YELLOW)), plan())
        self.assertEqual(len(found), 1)
        self.assertTrue(
            found[0].x0 <= 121 and found[0].y0 <= 181 and found[0].x1 >= 189 and found[0].y1 >= 197, found[0].box()
        )
        self.assertAlmostEqual(found[0].score, 0.9)  # the higher score
        apart = module.sweep(scene((PLATE_BOX, WHITE), (LOGO_BOX, YELLOW)), plan())
        self.assertEqual(len(apart), 2)  # regions that do not overlap stay two

    def test_pt2_a_text_region_cut_by_one_vehicles_crop_is_united_with_the_whole_one_seen_from_the_next(self):
        left, right = (40, 60, 200, 200), (180, 60, 340, 200)
        block = (190, 120, 250, 136)  # straddles the crop of the left vehicle, which cuts it at x = 212
        module = text_module(vehicles=[left, right], colours=((WHITE, 0.9),))
        found = module.sweep(scene((block, WHITE)), plan())
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].x0 <= 191 and found[0].x1 >= 249, found[0].box())

    def test_pt2_a_look_at_one_vehicle_unites_its_overlapping_regions_and_counts_a_vehicle_too_small(self):
        module = text_module()
        found = module.look_at_vehicle(
            scene(((120, 180, 180, 196), WHITE), ((130, 182, 190, 198), YELLOW)),
            c.Detection(*VEHICLE_BOX, 0.9, c.VEHICLE),
        )
        self.assertEqual(len(found), 1)
        small = c.Detection(10, 10, 50, 30, 0.9, c.VEHICLE)
        self.assertEqual(module.look_at_vehicle(scene(), small), [])
        self.assertEqual((module.vehicles_visited, module.vehicles_too_small), (1, 1))

    def test_pt2_uniting_regions_never_loses_coverage_and_chains_them(self):
        def region(x0, y0, x1, y1, score, source="sweep"):
            return c.Detection(x0, y0, x1, y1, score, c.PLATE, source, True)

        regions = [
            region(0, 0, 40, 10, 0.5),
            region(10, 0, 50, 10, 0.9),
            region(20, 0, 60, 10, 0.4),
            region(0, 50, 40, 60, 0.7),
        ]
        united = plates_module.unite_overlapping(regions)
        self.assertEqual(sorted(u.box() for u in united), [(0, 0, 60, 10), (0, 50, 40, 60)])
        self.assertEqual(sorted(u.score for u in united), [0.7, 0.9])
        for r in regions:  # nothing is left uncovered
            self.assertTrue(any(u.x0 <= r.x0 and u.y0 <= r.y0 and u.x1 >= r.x1 and u.y1 >= r.y1 for u in united))
        inside = plates_module.unite_overlapping([region(0, 0, 100, 40, 0.8), region(10, 10, 30, 30, 0.4)])
        self.assertEqual([u.box() for u in inside], [(0, 0, 100, 40)])  # one lying inside the other
        self.assertEqual(len(regions), 4)  # the input is not touched
        self.assertEqual(regions[0].box(), (0, 0, 40, 10))

    def test_pt2_plate_like_needs_the_aspect_and_a_minimum_height(self):
        p = plates_module.DEFAULT_TEXT_PATH
        for box, expected in (
            ((0, 0, 60, 16), True),  # 3.75: a plate
            ((0, 0, 30, 30), False),  # too square
            ((0, 0, 200, 16), False),  # too long
            ((0, 0, 20, 5), False),  # the aspect of a plate but under the minimum height
            ((0, 0, 30, 6), True),  # exactly the minimum height
        ):
            self.assertIs(plates_module.plate_like(c.Detection(*box, 0.9, c.TEXT), p), expected, box)

    def test_pt2_load_picks_the_path_and_refuses_an_unknown_one(self):
        from anonymizer import classes

        with (
            mock.patch.object(plates_module, "TextPlateModule") as text,
            mock.patch.object(plates_module, "PlateModule") as yolo,
        ):
            classes.load([c.PLATE], plate_path="yolo", vehicle_model="yolox")
            classes.load([c.PLATE], plate_path="text", vehicle_model="yolox")
            classes.load([c.PLATE], vehicle_model="yolox")  # the default is the text path
        self.assertEqual((text.call_count, yolo.call_count), (2, 1))
        with self.assertRaises(ValueError):
            classes.load([c.PLATE], plate_path="ocr")


class TextModel(unittest.TestCase):
    """classes/text.py without the real model: the hash check, the lazy open and the decoding of a probability map."""

    def folder(self, content=b"not a model", listed=True, tampered=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with open(os.path.join(tmp.name, text_module_name()), "wb") as f:
            f.write(content)
        if listed:
            digest = hashlib.sha256(content).hexdigest()
            with open(os.path.join(tmp.name, "SHA256SUMS"), "w", encoding="utf-8") as f:
                f.write(f"{digest}  {text_module_name()}\n")
        if tampered:
            with open(os.path.join(tmp.name, text_module_name()), "wb") as f:
                f.write(content + b"!")
        return tmp.name

    def test_pt2_the_model_is_hash_checked_at_build_and_opened_only_when_first_used(self):
        folder = self.folder()
        with mock.patch.object(text_det, "_open_checked") as opened:
            detector = text_det.TextDetector(models=folder)
            self.assertEqual((detector.sha256, opened.call_count), (hashlib.sha256(b"not a model").hexdigest(), 0))
        for broken, kw in (("missing entry", dict(listed=False)), ("changed file", dict(tampered=True))):
            with self.assertRaises(c.ModelLoadError, msg=broken):
                text_det.TextDetector(models=self.folder(**kw))
        with self.assertRaises(c.ModelLoadError):
            text_det.TextDetector(models=tempfile.gettempdir() + "/no-such-models-folder")

    def test_pt2_a_file_changed_after_the_check_is_never_run(self):
        folder = self.folder()
        detector = text_det.TextDetector(models=folder)
        with open(os.path.join(folder, text_module_name()), "wb") as f:
            f.write(b"another file")
        with self.assertRaises(c.ModelLoadError) as caught:
            detector.detect(np.zeros((64, 64, 3), np.uint8), 0.3)
        self.assertIn("does not match SHA256SUMS", caught.exception.detail)
        self.assertEqual(detector.run_failures, 0)  # a model that cannot load is not a "nothing here"

    def test_pt2_a_picture_too_small_to_hold_text_is_not_sent_to_the_model(self):
        detector = text_det.TextDetector(models=self.folder())
        detector._session = mock.Mock(side_effect=AssertionError("the model was run"))
        self.assertEqual(detector.detect(np.zeros((6, 300, 3), np.uint8), 0.3), [])
        self.assertEqual(detector.run_failures, 0)

    def test_pt2_the_model_is_fed_the_picture_in_bgr_order_as_rapidocr_does(self):
        """PP-OCR detection is run by RapidOCR on the BGR array unchanged: a channel flip costs plates (3 on v2 dev). The blue, green and
        red planes of the picture must reach the network as its channels 0, 1 and 2."""
        detector = text_det.TextDetector(models=self.folder())
        bgr = np.zeros((64, 64, 3), np.uint8)
        bgr[..., 0], bgr[..., 1], bgr[..., 2] = 40, 120, 200  # B, G, R: all different, so any reorder shows
        blob, new_h, new_w = detector._input(bgr)
        t = detector.thresholds
        self.assertEqual(blob.shape, (1, 3, new_h, new_w))
        for channel, value in enumerate((40, 120, 200)):
            self.assertAlmostEqual(float(blob[0, channel].mean()), (value / 255.0 - t.mean) / t.std, places=4)

    def test_pt2_a_run_that_raises_is_counted_and_finds_nothing(self):
        detector = text_det.TextDetector(models=self.folder())
        session = mock.Mock()
        session.get_inputs.return_value = [mock.Mock()]
        session.run.side_effect = RuntimeError("boom")
        detector._session = session
        with self.assertLogs(text_det.log, "WARNING"):
            self.assertEqual(detector.detect(np.zeros((64, 64, 3), np.uint8), 0.3), [])
        self.assertEqual(detector.run_failures, 1)

    def test_pt2_a_probability_map_that_is_not_numbers_is_a_failed_run_never_nothing_found(self):
        for bad in (np.nan, np.inf):
            detector = text_det.TextDetector(models=self.folder())
            prob = np.zeros((1, 1, 96, 96), np.float32)
            prob[0, 0, 20:40, 10:80] = 0.9
            prob[0, 0, 5, 5] = bad
            session = mock.Mock()
            session.get_inputs.return_value = [mock.Mock()]
            session.run.return_value = [prob]
            detector._session = session
            with self.assertLogs(text_det.log, "WARNING"):
                self.assertEqual(detector.detect(np.zeros((64, 64, 3), np.uint8), 0.3), [])
            self.assertEqual(detector.run_failures, 1, bad)

    def test_pt2_recorded_probability_maps_pin_the_threshold_the_unclip_and_the_minimum_side(self):
        """Two 64 x 192 probability maps cut from the real PP-OCRv6 model's output on enlarged vehicle crops (EU public photos): their
        boxes and scores are pinned, so a change of `thresh`, `unclip_ratio` or `min_side` changes what this test sees. The windows were chosen
        as the ones that tell those constants apart."""
        maps = np.load(os.path.join(os.path.dirname(__file__), "data", "text-prob-maps.npz"))["maps"]
        expected = [[(364, 166, 557, 192, 0.875)], [(553, 139, 576, 176, 0.684)]]
        detector = text_det.TextDetector(models=self.folder())
        self.assertEqual(maps.shape, (2, 64, 192))
        for prob, want in zip(maps, expected, strict=True):
            found = detector.decode(prob.astype(np.float32), 576, 192, 0.3)
            self.assertEqual(len(found), len(want))
            for d, (x0, y0, x1, y1, score) in zip(found, want, strict=True):
                self.assertTrue(
                    all(abs(a - b) <= 1 for a, b in zip(d.box(), (x0, y0, x1, y1), strict=True)), (d.box(), want)
                )
                self.assertAlmostEqual(d.score, score, delta=0.002)

    def test_pt2_the_decoding_boxes_regions_of_the_map_and_grows_them(self):
        detector = text_det.TextDetector(models=self.folder())
        prob = np.zeros((96, 192), np.float32)
        prob[40:56, 20:100] = 0.9  # a long line
        prob[10:20, 130:140] = 0.55  # a weaker, shorter one
        prob[70, 150] = 0.95  # a speck: noise
        found = sorted(detector.decode(prob, 384, 192, 0.3), key=lambda d: d.score)  # the map is half the picture
        self.assertEqual(len(found), 2)
        self.assertTrue(0.4 < found[0].score < 0.55 and 0.75 < found[1].score < 0.9, [d.score for d in found])
        long_line = found[1]
        self.assertTrue(
            long_line.x0 < 40 and long_line.x1 > 200 and long_line.y0 < 80 and long_line.y1 > 112, long_line.box()
        )
        self.assertEqual(long_line.kind, c.TEXT)
        self.assertEqual(
            [d.box() for d in detector.decode(prob, 384, 192, 0.6)], [long_line.box()]
        )  # min_score filters

    def test_pt2_the_input_is_a_multiple_of_32_and_never_larger_than_the_cap(self):
        detector = text_det.TextDetector(models=self.folder())
        for shape, want in (((300, 400), (736, 992)), ((900, 1000), (896, 992)), ((100, 5000), (32, 2048))):
            blob, h, w = detector._input(np.zeros((*shape, 3), np.uint8))
            self.assertEqual(blob.shape, (1, 3, h, w))
            self.assertEqual((h % 32, w % 32), (0, 0), shape)
            self.assertLessEqual(max(h, w), 2048)
            self.assertEqual((h, w), want, shape)

    def test_pt2_the_input_keeps_the_bgr_order_of_the_picture(self):
        detector = text_det.TextDetector(models=self.folder())
        img = np.zeros((64, 64, 3), np.uint8)
        img[:, :, 0] = 255  # pure blue in BGR
        blob, _, _ = detector._input(img)
        self.assertGreater(blob[0, 0].min(), 0.9)  # channel 0 of the network input is the picture's channel 0
        self.assertLess(blob[0, 2].max(), -0.9)


def text_module_name() -> str:
    return text_det.TEXT_MODEL


class PT3EveryRegionIsHidden(api_tests.ApiCase):
    def setUp(self):
        super().setUp()
        self.bgr = scene((PLATE_BOX, WHITE), (LOGO_BOX, YELLOW))
        self.photo = self.path("car.png")
        Image.fromarray(self.bgr[:, :, ::-1]).save(self.photo)

    def changed(self, out_path, box) -> bool:
        x0, y0, x1, y1 = box
        out = np.array(Image.open(out_path).convert("RGB"))[:, :, ::-1].astype(int)
        ring = (slice(y0 - 3, y1 + 3), slice(x0 - 3, x1 + 3))
        return np.abs(out[ring] - self.bgr[ring].astype(int)).mean() > 2.0

    def run_photo(self, **options):
        with anonymizer.Anonymizer(anonymizer.Options(**options), modules=[text_module()]) as anon:
            analysis = anon.analyze(self.photo)
            result = anon.render(analysis, self.path(f"out{len(os.listdir(self.dir))}.png"))
        return analysis, result

    def test_pt3_with_unsure_blur_the_plate_and_the_other_writing_on_the_vehicle_are_both_blurred(self):
        analysis, result = self.run_photo(
            unsure="blur"
        )  # the rule of 2.1; the default since 2.2 is "placed" (test_subtle.py)
        verdicts = {i.verdict: i for i in analysis.items}
        self.assertEqual(sorted(verdicts), ["blur", "check"])  # the plate is sure; the logo, scored 0.45, is doubtful
        self.assertTrue(all(i.blur for i in analysis.items))  # privacy first: doubtful is blurred too
        self.assertTrue(self.changed(result.output, PLATE_BOX) and self.changed(result.output, LOGO_BOX))
        self.assertFalse(self.changed(result.output, (10, 10, 30, 30)))  # nothing else was touched
        self.assertEqual((result.warnings, result.visible_check), ([], 0))

    def test_pt3_with_unsure_keep_the_doubtful_text_is_listed_and_left_visible(self):
        analysis, result = self.run_photo(unsure="keep")
        self.assertTrue(self.changed(result.output, PLATE_BOX))
        self.assertFalse(self.changed(result.output, LOGO_BOX))
        self.assertEqual(result.warnings, ["warning.review.check_left_visible"])

    def test_pt3_a_region_below_the_lower_threshold_is_not_found_at_all(self):
        module = text_module(colours=((WHITE, 0.9), (YELLOW, 0.2)))  # 0.2 < tau_lo
        found = module.sweep(self.bgr, plan())
        self.assertEqual(len(found), 1)  # a region still under tau_lo after the union is dropped
        self.assertEqual(
            module.text.min_scores, [module.text_path.merge_floor]
        )  # photos: the detector is asked from the merge floor (2.2, round 2)

    def test_pt3_a_region_below_the_lower_threshold_is_kept_only_as_part_of_a_region_it_touches(self):
        class Stub:  # explicit regions, in the pixels of the enlarged vehicle crop
            run_failures = 0

            def __init__(self, boxes):
                self.boxes, self.min_scores = boxes, []

            def detect(self, bgr, min_score):
                self.min_scores.append(min_score)
                return [c.Detection(*b, s, c.TEXT) for b, s in self.boxes if s >= min_score]

            def close(self):
                pass

        def found(boxes):
            module = TextPlateModule(vehicles=FixedVehicles((VEHICLE_BOX,)), text=Stub(boxes))
            return module, module.sweep(self.bgr, plan())

        plate, frame = (
            ((100, 300, 300, 340), 0.9),
            ((150, 300, 350, 340), 0.2),
        )  # the frame piece overlaps the plate's region by three quarters
        module, both = found([plate, frame])
        _, alone = found([plate])
        _, strip = found([frame])
        self.assertEqual(len(both), 1)
        self.assertGreater(both[0].x1, alone[0].x1)  # the union is wider than the plate's own region...
        self.assertEqual(round(both[0].score, 1), 0.9)  # ...with the better score
        self.assertEqual(strip, [])  # on its own the piece scores under tau_lo and is dropped
        self.assertEqual(module.text.min_scores, [module.text_path.merge_floor])
        video = module.look_at_vehicle(
            self.bgr, module.last_vehicles[0], module.last_vehicles
        )  # video: from tau_lo, as before
        self.assertEqual(module.text.min_scores[-1], module.text_path.tau_lo)
        self.assertEqual(len(video), 1)

    def test_pt3_the_judgement_is_by_score_only(self):
        module = text_module()
        t = module.text_path
        for score, verdict in (
            (t.tau_hi, "blur"),
            (t.tau_hi - 0.01, "check"),
            (t.tau_lo, "check"),
            (t.tau_lo - 0.01, "drop"),
        ):
            det = c.Detection(10, 10, 70, 26, score, c.PLATE, c.SWEEP, True)
            self.assertEqual(module.photo_judgement(det, None).verdict.value, verdict, score)


class PT6Budget(api_tests.ApiCase):
    VEHICLES = [(20 + 90 * i, 60, 100 + 90 * i, 140) for i in range(8)]  # eight vehicles, 80 x 80, in a row

    def setUp(self):
        super().setUp()
        self.bgr = np.full((200, 760, 3), 100, np.uint8)
        for x0, y0, _x1, _y1 in self.VEHICLES:
            self.bgr[y0 + 30 : y0 + 46, x0 + 10 : x0 + 70] = WHITE
        self.photo = self.path("row.png")
        Image.fromarray(self.bgr[:, :, ::-1]).save(self.photo)

    def analyse(self, **options):
        module = text_module(vehicles=self.VEHICLES)
        # the row has 8 vehicles: a small fast budget makes the counting visible; the product's numbers are pinned in the first test
        with (
            mock.patch.dict(c.MAX_VEHICLES, {"fast": 6}),
            anonymizer.Anonymizer(anonymizer.Options(**options), modules=[module]) as anon,
        ):
            return module, anon.analyze(self.photo)

    def test_pt6_the_budget_counts_what_it_does_not_visit_and_the_product_budgets_are_the_pt4_ones(self):
        self.assertEqual(c.MAX_VEHICLES, {"fast": 12, "careful": 40, "deep": 40})
        for detection, visited, skipped in (("fast", 6, 2), ("careful", 8, 0)):
            module, analysis = self.analyse(detection=detection)
            self.assertEqual((module.vehicles_visited, module.vehicles_not_visited), (visited, skipped), detection)
            self.assertEqual(len(analysis.items), visited, detection)  # one plate on each vehicle that was visited
            summary = analysis.summary()
            self.assertEqual((summary["zones_visited"], summary["zones_skipped"]), (visited, skipped), detection)

    def test_pt6_the_largest_vehicles_are_the_ones_visited(self):
        small, large, medium = (20, 60, 100, 140), (120, 60, 280, 220), (300, 60, 420, 190)
        bgr = np.full((300, 500, 3), 100, np.uint8)
        for x0, y0, _x1, _y1 in (small, large, medium):
            bgr[y0 + 30 : y0 + 46, x0 + 10 : x0 + 70] = WHITE
        module = text_module(vehicles=[small, large, medium])
        found = module.sweep(bgr, c.SweepPlan(1600, 640, max_vehicles=2))
        self.assertEqual((module.vehicles_visited, module.vehicles_not_visited), (2, 1))
        centres = sorted((d.x0 + d.x1) / 2 for d in found)
        self.assertEqual(len(centres), 2)
        self.assertTrue(120 < centres[0] < 280 and 300 < centres[1] < 420, centres)  # not the small one at the left

    def test_pt6_the_time_per_crop_and_per_vehicle_stage_is_recorded(self):
        module, _ = self.analyse(detection="careful")
        self.assertEqual(module.calls["text_on_vehicles"], 8)
        self.assertGreaterEqual(module.timings["text_on_vehicles"], 0.0)
        self.assertEqual(module.calls["vehicles"], 1)

    def test_pt6_a_text_model_failure_makes_the_photo_not_checked_never_clean(self):
        module = text_module(vehicles=self.VEHICLES, fail_on=(3,))
        with anonymizer.Anonymizer(modules=[module]) as anon:
            with self.assertRaises(anonymizer.IncompleteAnalysis):
                anon.analyze(self.photo)
        self.assertEqual(module.model_failures, 1)

    def test_pt6_vehicles_too_small_are_counted_in_the_summary_and_survive_a_save(self):
        module = text_module(vehicles=[*self.VEHICLES[:2], (400, 60, 430, 80)])  # two searched, one 30 x 20
        with anonymizer.Anonymizer(anonymizer.Options(detection="careful"), modules=[module]) as anon:
            analysis = anon.analyze(self.photo)
        summary = analysis.summary()
        self.assertEqual((summary["zones_visited"], summary["vehicles_too_small"]), (2, 1))
        saved = self.path("a.json")
        analysis.save(saved)
        self.assertEqual(anonymizer.load_analysis(saved).summary()["vehicles_too_small"], 1)

    def test_pt6_the_counters_are_per_analysis_in_the_summary(self):
        module = text_module(vehicles=self.VEHICLES)
        with (
            mock.patch.dict(c.MAX_VEHICLES, {"fast": 6}),
            anonymizer.Anonymizer(anonymizer.Options(detection="fast"), modules=[module]) as anon,
        ):
            first, second = anon.analyze(self.photo), anon.analyze(self.photo)
        self.assertEqual([a.summary()["zones_skipped"] for a in (first, second)], [2, 2])
        self.assertEqual(module.vehicles_not_visited, 4)  # the module's own counter is the running total


class PT9Report(api_tests.ApiCase):
    def setUp(self):
        super().setUp()
        self.bgr = scene((PLATE_BOX, WHITE), (LOGO_BOX, YELLOW))
        self.photo = self.path("car.png")
        Image.fromarray(self.bgr[:, :, ::-1]).save(self.photo)

    def test_pt9_the_report_separates_plate_shaped_regions_from_other_writing(self):
        with anonymizer.Anonymizer(modules=[text_module()]) as anon:
            analysis = anon.analyze(self.photo)
            result = anon.render(analysis, self.path("out.png"))
        rows = {
            tuple(round(v) for v in analysis.item(r.item_id).params and ()) or r.item_id: r for r in result.report.rows
        }
        by_shape = sorted((r.plate_like, r.reason) for r in rows.values())
        self.assertEqual(by_shape, [(False, "plate.reason.vehicle_text"), (True, "plate.reason.vehicle_text")])
        result.report.to_csv(self.path("r.csv"))
        result.report.to_json(self.path("r.json"))
        with open(self.path("r.csv"), encoding="utf-8-sig") as f:
            self.assertIn("plate_like", f.readline())
        with open(self.path("r.json"), encoding="utf-8") as f:
            self.assertEqual(sorted(r["plate_like"] for r in json.load(f)["items"]), [False, True])

    def test_pt9_other_paths_say_nothing_about_the_shape(self):
        from tests.fakes import FixedModule

        with anonymizer.Anonymizer(modules=[FixedModule(c.PLATE, [(20, 20, 100, 44)])]) as anon:
            analysis = anon.analyze(self.photo)
        self.assertEqual([r.plate_like for r in analysis.report().rows], [None])

    def test_pt9_a_region_wider_than_the_vehicle_is_clipped_to_it_never_dropped(self):
        # B, blind-015: a plate on a small vehicle was lost because its region was wider than the vehicle plus margin. Privacy first:
        # the region is cut to the vehicle with its margin; a stripe across the crop becomes a blur over the vehicle (PT9's accepted cost)
        bgr = scene(((30, 120, 370, 140), WHITE))
        vehicle = (100, 80, 300, 200)
        found = text_module(vehicles=[vehicle]).sweep(bgr, plan())
        self.assertEqual(len(found), 1)
        margin = plates_module.DEFAULT_THRESHOLDS.vehicle_margin
        mx = 200 * margin
        self.assertGreaterEqual(found[0].x0, 100 - mx - 1e-6)
        self.assertLessEqual(found[0].x1, 300 + mx + 1e-6)
        self.assertTrue(found[0].on_vehicle)

    def test_pt9_a_region_whose_centre_is_off_every_vehicle_is_still_not_found(self):
        self.assertIsNone(
            plates_module.clip_to_vehicle(
                c.Detection(500, 500, 600, 520, 0.9, c.TEXT), [c.Detection(0, 0, 100, 100, 0.9, c.VEHICLE)]
            )
        )

    def test_pt9_nothing_found_by_text_lies_off_a_vehicle_by_construction(self):
        wide = [(10, 10, 390, 290)]  # one vehicle covering the picture, a second tiny one: nothing can fall outside
        module = text_module(vehicles=wide + [(0, 0, 30, 20)])
        for d in module.sweep(scene((PLATE_BOX, WHITE), (LOGO_BOX, YELLOW), ((2, 2, 12, 12), WHITE)), plan()):
            self.assertTrue(d.on_vehicle)
            self.assertTrue(plates_module.on_vehicle(d, module.last_vehicles))


if __name__ == "__main__":
    unittest.main()
