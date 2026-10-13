"""2.2, "subtle" (the defaults hide only the plate itself and the face itself, tightly and softly): what the
new defaults hide, and how the old rule stays reachable. Fakes only (the figures on real photos are in bench/results/subtle.json)."""

from __future__ import annotations

import unittest
from unittest import mock

try:
    import numpy as np
    from PIL import Image

    import anonymizer
    from anonymizer import contract as c
    from anonymizer.classes import faces as faces_module
    from anonymizer.classes import load as load_classes
    from anonymizer.classes import plates as plates_module
    from anonymizer.engine import blur as b
    from anonymizer.engine import photos
    from tests import test_api as api_tests
    from tests.fakes import ColourTextDetector, FixedModule, FixedVehicles
    from tests.test_plates_text import PLATE_BOX, WHITE, YELLOW, scene, text_module
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

CTX = c.PhotoContext(400, 300)


def plate_item(box, score=0.8):
    return c.Judgement(
        c.Verdict.BLUR if score >= 0.6 else c.Verdict.CHECK, "plate.reason.vehicle_text", {"score": score}
    )


def module_with(vehicle=(0, 0, 400, 200)):
    module = text_module()
    module.last_vehicles = [c.Detection(*vehicle, 0.9, c.VEHICLE)]
    return module


class OptionsOfTwoPointTwo(unittest.TestCase):
    def test_the_defaults_are_subtle_and_the_old_rule_is_one_option_set_away(self):
        o = anonymizer.Options()
        self.assertEqual((o.unsure, o.fit, o.plate_band), ("placed", "tight", None))
        old = anonymizer.Options(unsure="blur", fit="loose", plate_band=True)
        self.assertEqual((old.unsure, old.fit, old.plate_band), ("blur", "loose", True))
        self.assertIs(o.profile, b.SUBTLE_PROFILE)
        self.assertIs(old.profile, b.PROFILE)
        for bad in ("", "wide", None, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                anonymizer.Options(fit=bad)

    def test_the_fit_reaches_the_modules_and_the_band_is_off_unless_asked_for(self):
        with (
            mock.patch.object(plates_module, "TextPlateModule") as text_built,
            mock.patch("anonymizer.classes.heavy.HeavyFaceModule") as face_built,
            mock.patch("anonymizer.classes.rfdetr.load_vehicle_detector"),
        ):
            load_classes([c.FACE, c.PLATE], detection="careful", fit="loose")
        self.assertEqual(text_built.call_args.kwargs["fit"], "loose")
        self.assertEqual(face_built.call_args.kwargs["fit"], "loose")
        self.assertNotIn("text_path", text_built.call_args.kwargs)  # no band
        with self.assertRaises(ValueError):
            load_classes([c.PLATE], fit="wide")

    def test_the_margins_are_small_when_tight_and_the_2_1_ones_when_loose(self):
        self.assertEqual(text_module().photo_margin, plates_module.PHOTO_MARGIN_TIGHT)
        loose = plates_module.TextPlateModule(
            vehicles=FixedVehicles([]), text=ColourTextDetector(((WHITE, 0.9),)), fit="loose"
        )
        self.assertEqual(loose.photo_margin, plates_module.PHOTO_MARGIN)
        self.assertEqual(plates_module.PHOTO_MARGIN, (0.10, 0.18))
        self.assertEqual(
            plates_module.PHOTO_MARGIN_TIGHT, (0.15, 0.25)
        )  # the writing grown to its plate (see the comment at the constant)
        self.assertEqual(faces_module.FaceModule.PHOTO_MARGIN_TIGHT, (0.06, 0.10))


class TheBlurStaysCloseToItsBox(unittest.TestCase):
    PLATE = (40, 40, 140, 62)  # 100 x 22

    def text_plate(self):
        img = np.full((140, 240, 3), 235, np.uint8)
        # a noisy background: a changed pixel is easy to see
        img = np.clip(img.astype(int) + np.random.RandomState(1).randint(-20, 20, img.shape), 0, 255).astype(np.uint8)
        import cv2

        cv2.putText(img, "1234 BCD", (46, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2, cv2.LINE_AA)
        return img

    def reach(self, profile) -> int:
        img = self.text_plate()
        before = img.copy()
        b.blur(img, self.with_margin(), b.RECT, b.GAUSSIAN, 1.0, profile)
        changed = np.abs(img.astype(int) - before.astype(int)).max(axis=2) > 2
        return int(changed.sum())

    def with_margin(self):
        mx, my = plates_module.PHOTO_MARGIN_TIGHT
        x0, y0, x1, y1 = self.PLATE
        return (x0 - (x1 - x0) * mx, y0 - (y1 - y0) * my, x1 + (x1 - x0) * mx, y1 + (y1 - y0) * my)

    def test_the_subtle_profile_changes_less_of_the_picture_than_the_broad_one(self):
        self.assertLess(self.reach(b.SUBTLE_PROFILE), 0.85 * self.reach(b.PROFILE))

    def test_the_subtle_profile_still_hides_the_characters(self):
        for style in b.STYLES:
            for w in (40, 60, 100, 200, 400):
                h = max(8, round(w / 4.5))
                img = np.full((h + 80, w + 80, 3), 235, np.uint8)
                import cv2

                cv2.putText(
                    img,
                    "1234 BCD",
                    (40 + w // 30, 40 + int(h * 0.8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    w / 300 * 1.5,
                    (10, 10, 10),
                    max(1, round(w / 75)),
                    cv2.LINE_AA,
                )
                box = (40, 40, 40 + w, 40 + h)
                mx, my = plates_module.PHOTO_MARGIN_TIGHT
                blurred = (box[0] - w * mx, box[1] - h * my, box[2] + w * mx, box[3] + h * my)
                b.blur(img, blurred, b.RECT, style, b.MIN_STRENGTH, b.SUBTLE_PROFILE)
                self.assertLess(b.residual(img, box), 1.5, f"{style}, {w} px")

    def test_the_default_profile_of_the_function_is_still_the_earlier_applications(self):
        img = self.text_plate()
        again = img.copy()
        b.blur(img, self.PLATE, b.RECT, b.GAUSSIAN, 1.0)
        b.blur(again, self.PLATE, b.RECT, b.GAUSSIAN, 1.0, b.PROFILE)
        np.testing.assert_array_equal(img, again)


class APlateIsPlacedOnAVehicle(unittest.TestCase):
    def test_a_plate_shaped_region_low_on_a_vehicle_is_blurred_by_default(self):
        j = module_with().photo_placement((120, 130, 180, 146), plate_item(None), CTX)  # 60 x 16 on a 400 x 200 vehicle
        self.assertEqual((j.verdict, j.params["placed"]), (c.Verdict.BLUR, 1.0))

    def test_what_sits_high_on_the_vehicle_is_listed_and_left_visible(self):
        j = module_with().photo_placement((120, 20, 180, 36), plate_item(None), CTX)  # the windscreen
        self.assertEqual((j.verdict, j.params["placed"]), (c.Verdict.CHECK, 0.0))

    def test_what_is_big_next_to_the_vehicle_is_not_a_plate(self):
        j = module_with().photo_placement((100, 60, 300, 190), plate_item(None), CTX)  # a door
        self.assertEqual((j.verdict, j.params["placed"]), (c.Verdict.CHECK, 0.0))

    def test_a_badge_is_not_a_plate(self):
        j = module_with().photo_placement((190, 140, 205, 150), plate_item(None), CTX)  # 15 px on 400
        self.assertEqual((j.verdict, j.params["placed"]), (c.Verdict.CHECK, 0.0))

    def test_a_large_region_must_be_wide_like_a_plate_and_a_small_one_is_not_asked(self):
        module = module_with(vehicle=(0, 0, 500, 400))
        wide = module.photo_placement((300, 250, 420, 300), plate_item(None), CTX)  # 120 x 50: 2.4 wide
        round_ = module.photo_placement((300, 250, 405, 365), plate_item(None), CTX)  # 105 x 115: a logo, a wheel hub
        small = module.photo_placement((300, 250, 360, 300), plate_item(None), CTX)  # 60 x 50: too coarse to tell
        self.assertEqual([j.params["placed"] for j in (wide, round_, small)], [1.0, 0.0, 1.0])

    def test_a_region_on_no_vehicle_is_never_placed(self):
        module = module_with(vehicle=(300, 0, 400, 100))
        j = module.photo_placement((120, 130, 180, 146), plate_item(None), CTX)
        self.assertEqual((j.verdict, j.params["placed"]), (c.Verdict.CHECK, 0.0))

    def test_a_doubtful_region_stays_doubtful_and_is_not_blurred_by_default(self):
        j = module_with().photo_placement((120, 130, 180, 146), plate_item(None, score=0.45), CTX)
        self.assertEqual((j.verdict, j.params["placed"]), (c.Verdict.CHECK, 0.0))
        self.assertFalse(photos.blurred_by_default(j.verdict, "placed", j.params))
        self.assertTrue(photos.blurred_by_default(j.verdict, "blur", j.params))

    def test_the_position_band_is_a_prior_and_is_left_as_it_is(self):
        band = c.Judgement(c.Verdict.BLUR, "plate.reason.position_band", {})
        self.assertIs(module_with().photo_placement((120, 20, 180, 36), band, CTX), band)


class FakeReader:
    """Reads what it is told, per call, as (whole crop, crop without margins)."""

    def __init__(self, *readings):
        self.readings = list(readings)
        self.calls = 0

    def read(self, crop):
        self.calls += 1
        return self.readings.pop(0)

    def close(self):
        pass


class TheReaderHasASecondSay(unittest.TestCase):
    """2.2 round 2: a word is not a plate, and a plate the position rule rejected is one (`TextPathThresholds.read_*`)."""

    IMAGE = np.zeros((300, 400, 3), np.uint8)

    def placement(self, box, score, reader, vehicle=(0, 0, 400, 200)):
        module = module_with(vehicle)
        module.reader = reader
        ctx = c.PhotoContext(400, 300, (), self.IMAGE)
        return module.photo_placement(box, plate_item(None, score), ctx)

    def test_a_region_that_reads_as_a_word_is_taken_out_of_the_default_blur(self):
        from anonymizer.classes.platetext import Reading

        word = Reading("RENAULT", 0.98)
        j = self.placement((100, 130, 200, 150), 0.8, FakeReader((word, word)))  # 100 x 20: sits like a plate
        self.assertEqual((j.verdict, j.params["placed"]), (c.Verdict.CHECK, 0.0))

    def test_a_dash_or_a_digit_makes_it_a_plate_and_a_small_region_is_not_read_as_a_word(self):
        from anonymizer.classes.platetext import Reading

        for text in ("WOB-VWMKL", "4314KMJ"):
            r = Reading(text, 0.9)
            j = self.placement((100, 130, 200, 150), 0.8, FakeReader((r, r)))
            self.assertEqual(j.params["placed"], 1.0, text)
        word = Reading("TAXI", 1.0)
        small = FakeReader((word, word))
        j = self.placement((100, 130, 140, 146), 0.8, small)  # 40 px wide: too small for the reader to be a judge
        self.assertEqual((j.params["placed"], small.calls), (1.0, 0))

    def test_a_wide_region_without_two_digits_and_a_roundish_one_that_reads_something_without_digits_are_not_plates(
        self,
    ):
        from anonymizer.classes.platetext import Reading

        vehicle = (0, 0, 500, 400)
        weak = Reading("A1", 0.4)
        wide = self.placement(
            (100, 230, 240, 260), 0.8, FakeReader((weak, weak)), vehicle
        )  # 140 x 30: legible to the reader if it were a plate
        plate = Reading("4314KMJ", 0.9)
        wide_plate = self.placement((100, 230, 240, 260), 0.8, FakeReader((plate, plate)), vehicle)
        emblem = Reading("E", 0.88)
        lion = self.placement((100, 200, 160, 260), 0.8, FakeReader((emblem, emblem)), vehicle)  # 60 x 60
        faint = Reading("E", 0.4)
        faint_one = self.placement(
            (100, 200, 160, 260), 0.8, FakeReader((faint, faint)), vehicle
        )  # a faint reading proves nothing
        self.assertEqual([j.params["placed"] for j in (wide, wide_plate, lion, faint_one)], [0.0, 1.0, 0.0, 1.0])

    def test_a_region_in_which_the_reader_finds_nothing_is_not_blurred_by_default_unless_it_is_small(self):
        from anonymizer.classes.platetext import NOTHING

        wide = self.placement(
            (100, 130, 180, 150), 0.8, FakeReader((NOTHING, NOTHING))
        )  # 80 px: a plate that wide is legible to it
        small = self.placement((100, 130, 140, 146), 0.8, FakeReader((NOTHING, NOTHING)))  # 40 px: nothing can be said
        self.assertEqual((wide.verdict, wide.params["placed"], small.params["placed"]), (c.Verdict.CHECK, 0.0, 1.0))

    def test_a_region_the_position_rule_rejected_is_blurred_when_it_reads_as_a_plate(self):
        from anonymizer.classes.platetext import NOTHING, Reading

        plate = Reading("TV06H", 0.89)
        high = (120, 20, 180, 36)  # the windscreen: rejected by position
        j = self.placement(high, 0.8, FakeReader((NOTHING, plate)))
        self.assertEqual((j.verdict, j.params["placed"]), (c.Verdict.BLUR, 1.0))
        nothing = self.placement(high, 0.8, FakeReader((NOTHING, NOTHING)))
        self.assertEqual((nothing.verdict, nothing.params["placed"]), (c.Verdict.CHECK, 0.0))
        doubtful = self.placement(high, 0.45, FakeReader((NOTHING, plate)))  # a doubtful region that reads as a plate
        self.assertEqual(doubtful.params["placed"], 1.0)

    def test_without_a_reader_or_a_picture_the_position_rule_decides_alone(self):
        low = (120, 130, 180, 146)
        self.assertEqual(self.placement(low, 0.8, None).params["placed"], 1.0)
        module = module_with()
        module.reader = FakeReader()
        j = module.photo_placement(low, plate_item(None, 0.8), CTX)  # CTX holds no picture
        self.assertEqual(j.params["placed"], 1.0)

    def test_the_reader_is_built_with_the_text_path_and_checked_like_a_model(self):
        with (
            mock.patch("anonymizer.classes.plates.TextPlateModule") as text_built,
            mock.patch("anonymizer.classes.rfdetr.load_vehicle_detector"),
            mock.patch("anonymizer.classes.platetext.PlateReader") as reader_built,
        ):
            load_classes([c.PLATE])
        self.assertIs(text_built.call_args.kwargs["reader"], reader_built.return_value)


class ThePlateReaderReadsWithItsModel(unittest.TestCase):
    def reader(self):
        from anonymizer.classes.platetext import PlateReader

        try:
            return PlateReader()
        except c.ModelLoadError as exc:
            raise unittest.SkipTest(f"the reader model is not here: {exc}") from exc

    def test_it_reads_a_drawn_plate_and_a_drawn_word_and_the_judgements_tell_them_apart(self):
        import cv2

        from anonymizer.classes.platetext import best_of, plate_like, word_like

        reader = self.reader()
        try:
            for text, as_plate in (("4314 KMJ", True), ("RENAULT", False)):
                img = np.full((64, 220, 3), 245, np.uint8)
                cv2.putText(img, text, (8, 46), cv2.FONT_HERSHEY_DUPLEX, 1.3, (20, 20, 20), 2, cv2.LINE_AA)
                reading = best_of(reader.read(img))
                self.assertEqual(plate_like(reading, 0.5, 2), as_plate, (text, reading))
                self.assertEqual(word_like(reading, 0.5, 3), not as_plate, (text, reading))
            self.assertEqual(
                best_of(reader.read(np.full((40, 120, 3), 128, np.uint8))).text, ""
            )  # a blank crop reads as nothing
        finally:
            reader.close()

    def test_a_model_that_does_not_match_its_checksum_is_refused(self):
        import tempfile

        from anonymizer.classes.platetext import READER_MODEL, PlateReader

        with tempfile.TemporaryDirectory() as folder:
            with open(f"{folder}/{READER_MODEL}", "wb") as f:
                f.write(b"not a model")
            with open(f"{folder}/SHA256SUMS", "w", encoding="utf-8") as f:
                f.write(f"{'0' * 64}  {READER_MODEL}\n")
            with self.assertRaises(c.ModelLoadError):
                PlateReader(models=folder)


class AFaceIsPlacedOnAPerson(unittest.TestCase):
    def placement(self, box, verdict, persons=(), size=(4000, 3000)):
        face = faces_module.FaceModule.__new__(faces_module.FaceModule)
        face.th = faces_module.ASP_THRESHOLDS
        j = c.Judgement(verdict, "face.reason.verified", {})
        return face.photo_placement(box, j, c.PhotoContext(size[0], size[1], persons))

    def test_a_big_enough_face_is_blurred_whether_or_not_there_is_a_person(self):
        j = self.placement((1000, 1000, 1100, 1120), c.Verdict.BLUR)
        self.assertEqual((j.verdict, j.params["placed"]), (c.Verdict.BLUR, 1.0))

    def test_a_tiny_face_on_no_person_is_a_pot_or_a_plant_and_is_listed_and_left_visible(self):
        j = self.placement((1000, 1000, 1012, 1016), c.Verdict.BLUR)  # 12 px on a 4000 px picture
        self.assertEqual((j.verdict, j.params["placed"]), (c.Verdict.CHECK, 0.0))

    def test_the_same_tiny_face_on_a_person_is_blurred(self):
        person = c.Detection(950, 990, 1100, 1400, 0.9, "person")
        j = self.placement((1000, 1000, 1012, 1016), c.Verdict.BLUR, [person])
        self.assertEqual((j.verdict, j.params["placed"]), (c.Verdict.BLUR, 1.0))

    def test_a_doubtful_face_is_blurred_only_on_the_head_of_a_person(self):
        person = c.Detection(950, 990, 1100, 1400, 0.9, "person")
        head = self.placement((1000, 1000, 1040, 1045), c.Verdict.CHECK, [person])
        feet = self.placement((1000, 1300, 1040, 1345), c.Verdict.CHECK, [person])
        alone = self.placement((1000, 1000, 1040, 1045), c.Verdict.CHECK)
        weak = self.placement(
            (1000, 1000, 1040, 1045), c.Verdict.CHECK, [c.Detection(950, 990, 1100, 1400, 0.3, "person")]
        )
        self.assertEqual([j.params["placed"] for j in (head, feet, alone, weak)], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual({j.verdict for j in (head, feet, alone, weak)}, {c.Verdict.CHECK})

    def test_a_doubtful_face_the_second_look_or_the_sweep_is_sure_of_is_blurred_even_with_no_person(self):
        face = faces_module.FaceModule.__new__(faces_module.FaceModule)
        face.th = faces_module.ASP_THRESHOLDS

        def placed(**params):
            j = c.Judgement(c.Verdict.CHECK, "face.reason.low_verify", params)
            return face.photo_placement((1000, 1000, 1040, 1045), j, c.PhotoContext(4000, 3000, ())).params["placed"]

        self.assertEqual(placed(second_score=0.9, item_score=0.5), 1.0)  # the heavy model agrees
        self.assertEqual(placed(second_score=0.6, item_score=0.75), 1.0)  # the first sweep was sure
        self.assertEqual(placed(second_score=0.6, item_score=0.6), 0.0)  # neither: a pot, a poster
        self.assertEqual(placed(item_score=0.5), 0.0)
        tiny = c.Judgement(c.Verdict.CHECK, "face.reason.low_verify", {"second_score": 0.95, "item_score": 0.8})
        self.assertEqual(  # 20 px on a 8,000 px panorama: a pot or a pole, however sure the models are
            face.photo_placement((1000, 1000, 1020, 1024), tiny, c.PhotoContext(8000, 3000, ())).params["placed"], 0.0
        )

    def test_the_engine_gives_the_class_rule_the_scores_the_zone_has(self):
        seen = {}

        class Faces(FixedModule):
            def photo_placement(self, box, judgement, ctx):
                seen.update(judgement.params)
                return judgement

        photos.decide_photo(
            np.zeros((160, 240, 3), np.uint8), [Faces(c.FACE, [(10, 10, 30, 30)], 0.77, 0.9)], unsure="placed"
        )
        self.assertAlmostEqual(seen["item_score"], 0.77, places=2)

    def test_a_doubtful_face_seen_in_three_sweeps_of_a_video_persists_like_a_face(self):
        face = faces_module.FaceModule.__new__(faces_module.FaceModule)
        face.th = faces_module.ASP_THRESHOLDS
        for views, want in ((2, 0.0), (3, 1.0)):
            stats = c.TrackStats("face", views, views, 0.7, 0.75, 1, 40.0, False)
            self.assertEqual(face.video_judgement(stats).params["placed"], want)


class DecisionLayer(unittest.TestCase):
    def bgr(self):
        return scene((PLATE_BOX, WHITE), ((250, 100, 280, 130), YELLOW))

    def test_placed_blurs_the_plate_and_lists_the_other_writing_visible(self):
        zones = photos.decide_photo(self.bgr(), [text_module()], unsure="placed")
        by_verdict = {z.judgement.verdict.value: z for z in zones}
        self.assertEqual(sorted(by_verdict), ["blur", "check"])  # the plate is sure, the yellow logo scored 0.45
        self.assertEqual([z.blur for z in zones if z.judgement.verdict is c.Verdict.BLUR], [True])
        self.assertEqual([z.blur for z in zones if z.judgement.verdict is c.Verdict.CHECK], [False])

    def test_a_sure_region_that_does_not_sit_like_a_plate_is_demoted_to_doubtful(self):
        high = ((250, 70, 330, 90), WHITE)  # sure (0.9) but at the top of the vehicle
        zones = photos.decide_photo(scene(high), [text_module()], unsure="placed")
        self.assertEqual([(z.judgement.verdict.value, z.blur) for z in zones], [("check", False)])
        old = photos.decide_photo(scene(high), [text_module()], unsure="blur")  # 2.1: no judgement of position
        self.assertEqual([(z.judgement.verdict.value, z.blur) for z in old], [("blur", True)])

    def test_the_old_rules_are_untouched(self):
        for unsure, want in (("blur", [True, True]), ("keep", [True, False])):
            zones = photos.decide_photo(self.bgr(), [text_module()], unsure=unsure)
            self.assertEqual(sorted(z.blur for z in zones), sorted(want), unsure)

    def test_persons_found_once_are_given_to_the_face_rule(self):
        seen = []

        class Faces(FixedModule):
            def photo_placement(self, box, judgement, ctx):
                seen.append(list(ctx.persons))
                return judgement

        person = c.Detection(0, 0, 50, 100, 0.9, "person")
        photos.decide_photo(
            np.zeros((160, 240, 3), np.uint8),
            [Faces(c.FACE, [(10, 10, 30, 30)], 0.9, 0.9)],
            unsure="placed",
            persons=lambda bgr: [person],
        )
        self.assertEqual(seen, [[person]])


class TheOneCallFunctionsFollowTheNewDefaults(api_tests.ApiCase):
    def test_a_doubtful_face_is_left_visible_listed_and_said_and_2_1_blurs_it(self):
        noise = (np.random.RandomState(3).rand(160, 240, 3) * 255).astype(np.uint8)
        photo = self.path("f.png")
        Image.fromarray(noise).save(photo)
        doubted = [FixedModule(c.FACE, [(140, 60, 200, 120)], 0.85, 0.75)]
        with mock.patch("anonymizer.classes.load", lambda kinds, threads, models, **kw: doubted):
            new = anonymizer.anonymize_image(photo, self.path("new.png"))
            old = anonymizer.anonymize_image(
                photo, self.path("old.png"), anonymizer.Options(unsure="blur", fit="loose")
            )
        self.assertEqual((new.warnings, new.visible_check), (["warning.review.check_left_visible"], 1))
        self.assertEqual((old.warnings, old.visible_check), ([], 0))
        out_new = np.array(Image.open(new.output).convert("RGB")).astype(float)
        out_old = np.array(Image.open(old.output).convert("RGB")).astype(float)
        inside = (slice(64, 116), slice(144, 196))
        self.assertGreater(out_new[inside].std(), 0.8 * noise[inside].std())  # untouched
        self.assertLess(out_old[inside].std(), 0.5 * noise[inside].std())  # blurred


if __name__ == "__main__":
    unittest.main()
