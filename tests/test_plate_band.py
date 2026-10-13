"""The plate-position band (PT4 lever, opt-in): on a small vehicle with no text box over the band where a plate usually sits, the band
itself is one more region to blur. Tests with a fake vehicle detector and a fake text detector; the figures are in bench/fit_plate_band.py."""

from __future__ import annotations

import inspect
import json
import os
import unittest
from dataclasses import replace
from unittest import mock

try:
    import numpy as np
    from PIL import Image

    import anonymizer
    from anonymizer import contract as c
    from anonymizer.classes import load as load_classes
    from anonymizer.classes.plates import DEFAULT_TEXT_PATH, TextPathThresholds, TextPlateModule
    from anonymizer.engine import photos
    from tests import test_api as api_tests
    from tests.fakes import ColourTextDetector, FixedVehicles
    from tests.test_plates_text import PLATE_BOX, SIZE, WHITE, plan, scene
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

SMALL = (100, 100, 300, 220)  # 200 x 120: narrower than 400
BAND = (126.0, 151.6, 294.0, 224.8)  # (0.13, 0.43, 0.97, 1.04) of SMALL
BANDED = replace(DEFAULT_TEXT_PATH, band=True)
SUPPRESSING = replace(BANDED, band_text_inside=0.90)  # the suppression is off by default; its mechanism is still tested


def module(vehicles=(SMALL,), band=True, colours=((WHITE, 0.9),), suppress=False, **kw) -> TextPlateModule:
    return TextPlateModule(
        vehicles=FixedVehicles(vehicles),
        text=ColourTextDetector(colours, pad=0.0),
        text_path=(SUPPRESSING if suppress else BANDED) if band else DEFAULT_TEXT_PATH,
        **kw,
    )


def bands_of(found):
    return [d for d in found if d.source == c.BAND]


class BandOff(unittest.TestCase):
    def test_band_is_off_by_default_and_nothing_is_added(self):
        self.assertFalse(DEFAULT_TEXT_PATH.band)
        m = module(band=False)
        self.assertEqual(m.sweep(scene(), plan()), [])
        self.assertEqual((m.bands_added, m.band_area, m.banded_vehicle_area), (0, 0.0, 0.0))

    def test_band_numbers_are_the_fitted_ones(self):
        t = TextPathThresholds()
        self.assertEqual((t.band_box, t.band_max_width, t.band_text_inside), ((0.13, 0.97, 0.43, 1.04), 400.0, 1.01))


class BandPlacement(unittest.TestCase):
    def test_a_small_vehicle_with_no_text_gets_one_band_zone_of_its_own_source(self):
        found = module().sweep(scene(), plan())
        self.assertEqual(len(found), 1)
        d = found[0]
        self.assertEqual((d.kind, d.source, d.on_vehicle), (c.PLATE, c.BAND, True))

    def test_once_the_engine_adds_its_margin_the_blurred_area_is_the_band(self):
        (d,) = module().sweep(scene(), plan())
        blurred = photos.with_margin(d.box(), module().photo_margin, SIZE[0], SIZE[1])
        for got, want in zip(blurred, BAND, strict=True):
            self.assertAlmostEqual(got, want, delta=1.0)

    def test_a_vehicle_of_400_px_or_more_gets_no_band(self):
        self.assertEqual(module(vehicles=[(40, 60, 440, 260)]).sweep(scene(), plan()), [])  # 400 wide
        self.assertEqual(len(module(vehicles=[(40, 60, 439, 260)]).sweep(scene(), plan())), 1)  # 399 wide

    def test_the_right_edge_is_clipped_to_the_picture_too(self):
        right = module(
            vehicles=[(300, 100, 420, 200)]
        )  # the band's right edge is 300 + 0.97 * 120 = 416, past the picture's 400
        (d,) = right.sweep(scene(), plan())
        blurred = photos.with_margin(d.box(), right.photo_margin, SIZE[0], SIZE[1])
        self.assertLessEqual(blurred[2], SIZE[0] + 1e-6)
        self.assertAlmostEqual(blurred[2], SIZE[0], delta=1.0)
        self.assertAlmostEqual(right.band_area, (SIZE[0] - (300 + 0.13 * 120)) * (1.04 - 0.43) * 100, delta=1.0)

    def test_a_vehicle_under_the_minimum_size_gets_no_band_and_is_still_counted_too_small(self):
        m = module(vehicles=[(100, 100, 150, 130)])  # 50 x 30
        self.assertEqual(m.sweep(scene(), plan()), [])
        self.assertEqual(m.vehicles_too_small, 1)

    def test_the_band_is_clipped_to_the_picture(self):
        low = module(
            vehicles=[(100, 205, 300, 299)]
        )  # the band's bottom is 205 + 1.04 * 94 = 302.8, under the picture's 300
        (d,) = low.sweep(scene(), plan())
        blurred = photos.with_margin(d.box(), low.photo_margin, SIZE[0], SIZE[1])
        self.assertLessEqual(blurred[3], SIZE[1] + 1e-6)
        self.assertAlmostEqual(blurred[3], SIZE[1], delta=1.0)
        self.assertAlmostEqual(
            low.band_area, (0.97 - 0.13) * 200 * (SIZE[1] - (205 + 0.43 * 94)), delta=1.0
        )  # the area counted is the clipped one

    def test_every_searchable_small_vehicle_is_banded_even_beyond_the_budget(self):
        vehicles = [(5 + 78 * i, 100, 75 + 78 * i, 160) for i in range(5)]  # five vehicles, 70 x 60
        m = module(vehicles=vehicles)
        with mock.patch.dict(c.MAX_VEHICLES, {"fast": 2}):
            found = m.sweep(scene(), photos.photo_plan("fast"))
        self.assertEqual((m.vehicles_visited, m.vehicles_not_visited), (2, 3))
        self.assertEqual(len(bands_of(found)), 5)

    def test_video_sweeps_and_key_frame_looks_get_no_band(self):
        m = module()
        self.assertEqual(m.sweep(scene(), c.SweepPlan(1600, 640, max_vehicles=0)), [])
        found = m.look_at_vehicle(scene(), c.Detection(*SMALL, 0.9, c.VEHICLE))
        self.assertEqual(bands_of(found), [])


class BandSuppressedByText(unittest.TestCase):
    def test_by_default_text_in_the_band_does_not_stop_it(self):
        found = module().sweep(scene((PLATE_BOX, WHITE)), plan())
        self.assertEqual(len(bands_of(found)), 1)
        self.assertEqual(len(found), 2)

    def test_a_text_region_inside_the_band_means_no_band(self):
        m = module(suppress=True)
        found = m.sweep(scene((PLATE_BOX, WHITE)), plan())  # PLATE_BOX (120..180, 180..196) is inside the band
        self.assertEqual(bands_of(found), [])
        self.assertEqual(len(found), 1)

    def test_a_text_region_mostly_outside_the_band_does_not_stop_it(self):
        m = module(suppress=True)
        found = m.sweep(scene(((270, 170, 310, 186), WHITE)), plan())  # 24 of 40 px inside the band
        self.assertEqual(len(bands_of(found)), 1)
        self.assertEqual(len(found), 2)

    def test_the_boundary_is_at_90_percent_of_the_text_box(self):
        m = module(suppress=True)
        vehicle = [c.Detection(*SMALL, 0.9, c.VEHICLE)]
        for inside, expected in ((0.89, 1), (0.90, 0), (0.91, 0)):
            width = 100.0
            box = (BAND[2] - width * inside, 170.0, BAND[2] - width * inside + width, 190.0)
            text = [c.Detection(*box, 0.9, c.PLATE, c.SWEEP, True)]
            self.assertEqual(len(m._bands(SIZE[0], SIZE[1], vehicle, text)), expected, inside)

    def test_text_on_another_vehicle_does_not_stop_this_vehicles_band(self):
        m = module(vehicles=[SMALL, (20, 10, 90, 60)], suppress=True)
        found = m.sweep(
            scene(((30, 40, 70, 52), WHITE)), plan()
        )  # text on the second vehicle, away from the first one's band
        self.assertEqual(len(bands_of(found)), 1)


class BandCountedAndJudged(api_tests.ApiCase):
    def setUp(self):
        super().setUp()
        self.photo = self.path("small.png")
        Image.fromarray(scene()[:, :, ::-1]).save(self.photo)

    def run_photo(self, m, **options):
        with anonymizer.Anonymizer(anonymizer.Options(**options), modules=[m]) as anon:
            return anon.analyze(self.photo)

    def test_the_band_is_a_plate_item_blurred_even_with_unsure_keep_and_never_a_doubtful_one(self):
        for unsure in ("blur", "keep"):
            analysis = self.run_photo(module(), unsure=unsure)
            (item,) = analysis.items
            self.assertEqual(
                (item.kind, item.verdict, item.reason, item.blur), (c.PLATE, "blur", "plate.reason.position_band", True)
            )
            self.assertEqual(analysis.summary()["doubtful"], 0)

    def test_the_summary_counts_the_bands_and_their_area_share_and_it_survives_a_save(self):
        analysis = self.run_photo(module())
        summary = analysis.summary()
        self.assertEqual(summary["bands_added"], 1)
        self.assertAlmostEqual(summary["band_area_share"], (0.97 - 0.13) * (1.04 - 0.43), delta=0.001)  # 0.51
        saved = self.path("a.json")
        analysis.save(saved)
        again = anonymizer.load_analysis(saved).summary()
        self.assertEqual((again["bands_added"], again["band_area_share"]), (1, summary["band_area_share"]))

    def test_with_the_option_off_the_summary_says_zero(self):
        summary = self.run_photo(module(band=False)).summary()
        self.assertEqual((summary["bands_added"], summary["band_area_share"]), (0, 0.0))

    def test_the_counters_are_per_analysis_and_the_module_keeps_the_running_total(self):
        m = module()
        with anonymizer.Anonymizer(anonymizer.Options(), modules=[m]) as anon:
            first, second = anon.analyze(self.photo), anon.analyze(self.photo)
        self.assertEqual([a.summary()["bands_added"] for a in (first, second)], [1, 1])
        self.assertEqual(m.bands_added, 2)

    def test_the_rendered_photo_is_blurred_over_the_band(self):
        m = module()
        with anonymizer.Anonymizer(anonymizer.Options(), modules=[m]) as anon:
            out = self.path("out.png")
            anon.render(anon.analyze(self.photo), out)
        before = scene().astype(int)
        after = np.array(Image.open(out).convert("RGB"))[:, :, ::-1].astype(int)
        inside, outside = (slice(160, 215), slice(135, 280)), (slice(10, 90), slice(10, 90))
        self.assertGreater(np.abs(after[inside] - before[inside]).mean(), 2.0)
        self.assertLess(np.abs(after[outside] - before[outside]).mean(), 0.5)

    def test_pt10_the_options_say_what_ran_so_off_is_told_from_nothing_banded(self):
        banded = self.run_photo(module())
        self.assertEqual(
            (banded.options["plate_path"], banded.options["vehicle_model"], banded.options["plate_band"]),
            ("text", "yolox", True),
        )
        off = self.run_photo(module(band=False))
        self.assertEqual((off.options["plate_path"], off.options["plate_band"]), ("text", False))
        nothing_found = self.run_photo(module(vehicles=()))
        self.assertEqual((nothing_found.summary()["bands_added"], nothing_found.options["plate_band"]), (0, True))
        saved = self.path("o.json")
        banded.save(saved)
        self.assertTrue(anonymizer.load_analysis(saved).options["plate_band"])

    def test_pt10_the_options_record_the_rfdetr_vehicle_model_and_a_yolo_plate_path(self):
        vehicles = FixedVehicles([SMALL])
        vehicles.name = "rf-detr-nano-384-coco"
        self.assertEqual(self.run_photo(module(band=False)).options["vehicle_model"], "yolox")
        m = TextPlateModule(vehicles=vehicles, text=ColourTextDetector(((WHITE, 0.9),), pad=0.0))
        self.assertEqual(self.run_photo(m).options["vehicle_model"], "rfdetr")
        from tests.fakes import FixedModule

        light = self.run_photo(FixedModule(c.PLATE, [(1, 1, 20, 20)])).options  # no text_path: the v1 plate module
        self.assertEqual((light["plate_path"], light["plate_band"]), ("yolo", False))
        faces_only = self.run_photo(FixedModule(c.FACE, []), plates=False).options
        self.assertEqual((faces_only["plate_path"], faces_only["vehicle_model"], faces_only["plate_band"]), (None,) * 3)


class OptionsReachTheLoad(unittest.TestCase):
    def test_pt10_the_product_defaults_are_the_pt4_configuration(self):
        o = anonymizer.Options()
        self.assertEqual((o.plate_path, o.vehicle_model, o.plate_band), ("text", "rfdetr", None))
        defaults = inspect.signature(load_classes).parameters
        self.assertEqual((defaults["plate_path"].default, defaults["vehicle_model"].default), ("text", "rfdetr"))

    def test_pt10_the_options_are_checked(self):
        for bad in (
            dict(plate_path="ocr"),
            dict(vehicle_model="detr"),
            dict(plate_band=1),
            dict(plate_path="yolo", plate_band=True),
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                anonymizer.Options(**bad)
        anonymizer.Options(plate_path="yolo", plate_band=False)

    def test_pt10_the_anonymizer_hands_the_plate_options_to_the_load(self):
        seen = {}

        def load(kinds, threads, models, **kw):
            seen.update(kw)
            return []

        for options, want in (
            (
                anonymizer.Options(),
                dict(plate_path="text", detection="careful", vehicle_model="rfdetr", plate_band=None, fit="tight"),
            ),
            (
                anonymizer.Options(plate_path="yolo", vehicle_model="yolox", detection="fast"),
                dict(plate_path="yolo", detection="fast", vehicle_model="yolox", plate_band=None, fit="tight"),
            ),
            (
                anonymizer.Options(detection="fast", plate_band=True),
                dict(plate_path="text", detection="fast", vehicle_model="rfdetr", plate_band=True, fit="tight"),
            ),
        ):
            seen.clear()
            with mock.patch("anonymizer.classes.load", load):
                anonymizer.Anonymizer(options).modules()
            self.assertEqual(seen, want)


class BandLoadAndTexts(unittest.TestCase):
    def test_load_needs_the_text_path_for_the_band(self):
        with self.assertRaises(ValueError):
            load_classes([c.PLATE], plate_path="yolo", plate_band=True)

    def test_load_hands_the_band_to_the_text_module_when_asked_and_not_when_refused(self):
        from anonymizer.classes import plates as plates_module

        for band, want in ((True, True), (False, None)):
            with mock.patch.object(plates_module, "TextPlateModule") as built:
                load_classes([c.PLATE], plate_path="text", plate_band=band)
            text_path = built.call_args.kwargs.get("text_path")
            self.assertEqual(text_path.band if text_path else None, want)

    def test_the_band_is_off_by_default_whatever_the_detection_since_2_2(self):
        from anonymizer.classes import plates as plates_module

        cases = (  # 2.1 had it on in careful and deep on the text path: it blurred bare walls and sides of cars (decision of 06/10/2026)
            ("text", "fast", None),
            ("text", "careful", None),
            ("text", "deep", None),
            ("yolo", "careful", None),  # no text path, no band, and no error when it was not asked for
            ("yolo", "deep", None),
        )
        for path, detection, want in cases:
            with (
                mock.patch.object(plates_module, "TextPlateModule") as text_built,
                mock.patch.object(plates_module, "PlateModule") as yolo_built,
                mock.patch("anonymizer.classes.heavy.HeavyFaceModule"),
            ):
                load_classes([c.PLATE], plate_path=path, detection=detection)
            built = text_built if path == "text" else yolo_built
            text_path = built.call_args.kwargs.get("text_path")
            self.assertEqual(text_path.band if text_path else None, want, (path, detection))

    def test_the_band_can_be_refused_or_asked_for_whatever_the_detection(self):
        from anonymizer.classes import plates as plates_module

        for detection, band, want in (("careful", False, None), ("deep", False, None), ("fast", True, True)):
            with mock.patch.object(plates_module, "TextPlateModule") as built:
                load_classes([c.PLATE], plate_path="text", detection=detection, plate_band=band)
            text_path = built.call_args.kwargs.get("text_path")
            self.assertEqual(text_path.band if text_path else None, want, (detection, band))

    def test_the_reason_has_a_text_in_every_language(self):
        folder = os.path.join(os.path.dirname(anonymizer.__file__), "ui", "i18n")
        for name in sorted(os.listdir(folder)):
            if name.endswith(".json"):
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    self.assertTrue(json.load(f).get("plate.reason.position_band"), name)


if __name__ == "__main__":
    unittest.main()
