"""PF1, PF2, PF3, PF7: what the detectors are not sure about is blurred by default, and the report and the
analysis file say who decided."""

from __future__ import annotations

import csv
import json
import os
import unittest

try:
    import numpy as np
    from PIL import Image

    import anonymizer
    from anonymizer.contract import FACE, PLATE
    from tests import test_api as api_tests
    from tests import test_ui_server as server_tests
    from tests.fakes import FixedModule, doubtful_plate_module
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

FACE_BOX, PLATE_BOX = (140, 60, 200, 120), (20, 20, 100, 44)


class PhotoCase(api_tests.ApiCase):
    def setUp(self):
        super().setUp()
        self.noise = (np.random.RandomState(3).rand(160, 240, 3) * 255).astype(np.uint8)
        self.photo = self.path("f.png")
        Image.fromarray(self.noise).save(self.photo)

    def modules(self):
        # the face is "Not sure" (score 0.85, verify 0.75), the plate is sure
        return [FixedModule(FACE, [FACE_BOX], 0.85, 0.75), FixedModule(PLATE, [PLATE_BOX])]

    def blurred(self, path, box) -> bool:
        x0, y0, x1, y1 = box
        out = np.array(Image.open(path).convert("RGB")).astype(float)
        inside = (slice(y0 + 4, y1 - 4), slice(x0 + 4, x1 - 4))
        return out[inside].std() < 0.5 * self.noise[inside].std()

    def analyse(self, **options):
        options.setdefault(
            "unsure", "blur"
        )  # these tests are the rule of 2.1; the default since 2.2 is tested in test_subtle.py
        with anonymizer.Anonymizer(anonymizer.Options(**options), modules=self.modules()) as anon:
            analysis = anon.analyze(self.photo, self.events.append)
            self.runs = getattr(self, "runs", 0) + 1
            result = anon.render(analysis, self.path(f"out{self.runs}.png"), self.events.append)
        return analysis, result


class PF1PhotoAndVideo(PhotoCase):
    def test_pf1_a_doubtful_photo_item_is_blurred_listed_and_exported_blurred(self):
        analysis, result = self.analyse()
        face = next(i for i in analysis.items if i.kind == FACE)
        self.assertEqual((face.verdict, face.blur, face.decided_by), ("check", True, "engine"))
        self.assertTrue(self.blurred(result.output, FACE_BOX))
        self.assertEqual((result.warnings, result.visible_check), ([], 0))
        analyzed = next(e for e in self.events if isinstance(e, anonymizer.Analyzed))
        self.assertEqual((analyzed.items, analyzed.to_check), (2, 1))  # still counted: Review lists it
        self.assertEqual([b.kind for b in analysis.boxes(0, 0)[0]], [FACE, PLATE])  # blurred boxes include it

    def test_pf1_only_a_person_leaves_a_doubtful_photo_item_visible_and_it_is_said(self):
        with anonymizer.Anonymizer(anonymizer.Options(unsure="blur"), modules=self.modules()) as anon:
            analysis = anon.analyze(self.photo)
            face = next(i for i in analysis.items if i.kind == FACE)
            analysis.set_blur(face.item_id, False)
            result = anon.render(analysis, self.path("out.png"), self.events.append)
        self.assertFalse(self.blurred(result.output, FACE_BOX))
        self.assertEqual((result.warnings, result.visible_check), (["warning.review.check_left_visible"], 1))

    def test_pf1_a_doubtful_video_track_is_blurred_listed_and_exported_blurred(self):
        src = self.clip(audio=False)
        with self.anon_with(doubtful_plate_module()) as anon:
            analysis = anon.analyze(src)
            [item] = analysis.items
            self.assertEqual((item.verdict, item.blur, item.decided_by), ("check", True, "engine"))
            result = anon.render(analysis, self.path("out.mp4"), self.events.append)
        self.assertTrue(api_tests.plate_hidden(result.output, 30))
        self.assertEqual((result.warnings, result.visible_check), ([], 0))
        self.assertEqual(len(analysis.boxes(30, 30)[30]), 1)

    def test_pf1_a_doubtful_video_track_a_person_turned_off_is_exported_visible(self):
        src = self.clip(audio=False)
        with self.anon_with(doubtful_plate_module()) as anon:
            analysis = anon.analyze(src)
            analysis.set_blur(analysis.items[0].item_id, False)
            result = anon.render(analysis, self.path("out.mp4"))
        self.assertFalse(api_tests.plate_hidden(result.output, 30))
        self.assertEqual(result.warnings, ["warning.review.check_left_visible"])

    def test_pf1_the_default_of_the_engine_function_is_privacy_first_too(self):
        from anonymizer.engine import photos

        bgr = np.zeros((160, 240, 3), np.uint8)
        zones = photos.decide_photo(bgr, self.modules())
        self.assertEqual(
            [(z.kind, z.blur, z.judgement.verdict.value) for z in zones], [(FACE, True, "check"), (PLATE, True, "blur")]
        )
        with self.assertRaises(ValueError):
            photos.decide_photo(bgr, self.modules(), unsure="maybe")

    def anon_with(self, module):
        return anonymizer.Anonymizer(modules=[module])


class PF1LocalServer(server_tests.StudioCase):
    modules = [doubtful_plate_module()]

    def test_pf1_the_local_server_lists_a_doubtful_item_blurred_and_exports_it_blurred(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        base = f"/api/clips/{clip_id}"
        [item] = self.api("GET", f"{base}/analysis")["items"]
        self.assertEqual((item["judgement"]["verdict"], item["blur"]), ("check", True))
        folder = os.path.join(self.dir, "out")
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        body = {"clips": [clip_id], "options": {"destination": folder}}
        self.api("POST", "/api/render", body, 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.assertFalse([e for e in reader.events if e["type"] == "job.warning" and "check_left" in e["code"]])
        [name] = [n for n in os.listdir(folder) if n.endswith(".mp4")]
        self.assertTrue(api_tests.plate_hidden(os.path.join(folder, name), 30))


class PF2UnsureOption(PhotoCase):
    def test_pf2_the_default_is_placed_and_only_placed_blur_or_keep_are_accepted(self):
        self.assertEqual(anonymizer.Options().unsure, "placed")
        for good in ("placed", "blur", "keep"):
            self.assertEqual(anonymizer.Options(unsure=good).unsure, good)
        for bad in ("", "hide", "KEEP", None, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                anonymizer.Options(unsure=bad)

    def test_pf2_keep_is_v1s_behaviour_and_blur_is_not(self):
        keep_analysis, keep = self.analyse(unsure="keep")
        face = next(i for i in keep_analysis.items if i.kind == FACE)
        self.assertEqual((face.verdict, face.blur), ("check", False))
        self.assertFalse(self.blurred(keep.output, FACE_BOX))
        self.assertEqual((keep.warnings, keep.visible_check), (["warning.review.check_left_visible"], 1))
        warning = next(e for e in self.events if isinstance(e, anonymizer.JobWarning))
        self.assertEqual(warning.params, {"count": 1})
        blur_analysis, blur = self.analyse(unsure="blur")
        self.assertTrue(next(i for i in blur_analysis.items if i.kind == FACE).blur)
        self.assertTrue(self.blurred(blur.output, FACE_BOX))
        self.assertEqual(blur.warnings, [])

    def test_pf2_the_choice_is_in_the_options_fingerprint_the_analysis_file_and_the_report_header(self):
        for unsure in ("placed", "blur", "keep"):
            analysis, result = self.analyse(unsure=unsure)
            self.assertEqual(analysis.options["unsure"], unsure)
            self.assertEqual(result.report.options["unsure"], unsure)
            saved = self.path(f"{unsure}.json")
            analysis.save(saved)
            with open(saved, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["options"]["unsure"], unsure)
            self.assertEqual(anonymizer.load_analysis(saved).options["unsure"], unsure)
            result.report.to_json(self.path(f"{unsure}-report.json"))
            with open(self.path(f"{unsure}-report.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["options"]["unsure"], unsure)

    def test_pf2_a_video_follows_the_option_too(self):
        src = self.clip(audio=False)
        for unsure, blur in (("placed", True), ("blur", True), ("keep", False)):  # the fake says the track is placed
            with anonymizer.Anonymizer(anonymizer.Options(unsure=unsure), modules=[doubtful_plate_module()]) as anon:
                [item] = anon.analyze(src).items
            self.assertEqual((item.verdict, item.blur), ("check", blur), unsure)


class PF3Report(PhotoCase):
    def rows(self, analysis):
        return {r.kind: r for r in analysis.report().rows}

    def test_pf3_rows_say_who_found_it_what_was_decided_and_by_whom(self):
        analysis, _ = self.analyse()
        rows = self.rows(analysis)
        self.assertEqual(
            [(rows[k].found_by, rows[k].decision, rows[k].decided_by) for k in (FACE, PLATE)],
            [("light", "blurred", "engine")] * 2,
        )
        face = next(i for i in analysis.items if i.kind == FACE)
        analysis.set_blur(face.item_id, False)
        analysis.add_area(0, 0, (180, 130, 230, 150))
        rows = self.rows(analysis)
        self.assertEqual(
            (rows[FACE].blurred, rows[FACE].decision, rows[FACE].decided_by), (False, "left visible", "person")
        )
        self.assertEqual(
            (rows["manual"].found_by, rows["manual"].decision, rows["manual"].decided_by),
            ("manual", "blurred", "person"),
        )

    def test_pf3_csv_json_carry_the_columns_and_the_summary(self):
        analysis, result = self.analyse()
        result.report.to_csv(self.path("r.csv"))
        result.report.to_json(self.path("r.json"))
        with open(self.path("r.csv"), encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue({"found_by", "decision", "decided_by"} <= set(rows[0]))
        self.assertEqual(
            sorted((r["kind"], r["decision"], r["decided_by"]) for r in rows),
            [(FACE, "blurred", "engine"), (PLATE, "blurred", "engine")],
        )
        with open(self.path("r.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["version"], 2)
        self.assertEqual(data["summary"], result.report.summary())
        self.assertEqual(
            [(r["kind"], r["found_by"], r["decision"], r["decided_by"]) for r in data["items"]][0][1:],
            ("light", "blurred", "engine"),
        )

    def test_pf3_the_summary_counts_doubtful_items_by_what_happened_to_them(self):
        analysis, _ = self.analyse()
        want = dict(
            doubtful=1,
            blurred_by_default=1,
            confirmed_by_person=0,
            turned_off_by_person=0,
            left_visible_by_default=0,
            items=2,
            blurred=2,
        )
        counts = analysis.report().summary()
        self.assertEqual({k: counts[k] for k in want}, want)
        self.assertEqual(
            {k: v for k, v in counts.items() if k not in want},
            dict(
                zones_visited=0,
                zones_skipped=0,
                confirmed=0,
                added=0,
                vehicles_too_small=0,
                bands_added=0,
                band_area_share=0.0,
            ),  # the second look and the plate-position band did nothing here
        )
        face = next(i for i in analysis.items if i.kind == FACE)
        analysis.set_blur(face.item_id, True)  # the person looked and confirmed it
        confirmed = analysis.report().summary()
        self.assertEqual({k: confirmed[k] for k in want}, {**want, "blurred_by_default": 0, "confirmed_by_person": 1})
        analysis.set_blur(face.item_id, False)
        self.assertEqual(
            self.privacy_counts(analysis.report()),
            {**want, "blurred_by_default": 0, "turned_off_by_person": 1, "blurred": 1},
        )
        keep, _ = self.analyse(unsure="keep")
        self.assertEqual(
            self.privacy_counts(keep.report()),
            {**want, "blurred_by_default": 0, "left_visible_by_default": 1, "blurred": 1},
        )

    @staticmethod
    def privacy_counts(report) -> dict:
        """The counts of the "Not sure" items: the summary without what the second look and the plate-position band did."""
        second = (
            "zones_visited",
            "zones_skipped",
            "confirmed",
            "added",
            "vehicles_too_small",
            "bands_added",
            "band_area_share",
        )
        return {k: v for k, v in report.summary().items() if k not in second}


class PF7AnalysisFormat3(PhotoCase):
    class Heavy:
        name, version, sha256 = "heavy-face-test", "1", "ab" * 32

        def close(self):
            pass

    def test_pf7_the_analysis_is_format_3_and_a_v1_file_is_refused(self):
        analysis, _ = self.analyse()
        saved = self.path("a.json")
        analysis.save(saved)
        with open(saved, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["version"], 3)
        self.assertEqual(anonymizer.load_analysis(saved).items, analysis.items)
        for old in (1, 2, None):
            data["version"] = old
            with open(saved, "w", encoding="utf-8") as f:
                json.dump(data, f)
            with self.assertRaises(anonymizer.AnonymizerError) as caught:
                anonymizer.load_analysis(saved)
            self.assertEqual(
                (caught.exception.code, caught.exception.params), ("error.analysis.version", {"version": old})
            )

    def test_pf7_the_fingerprint_has_detection_unsure_and_the_name_and_hash_of_every_heavy_model(self):
        modules = self.modules()
        modules[0].heavy_zone = self.Heavy()
        modules[1].heavy_deep = modules[0].heavy_zone  # the same model for two roles is listed once
        with anonymizer.Anonymizer(anonymizer.Options(detection="deep", unsure="keep"), modules=modules) as anon:
            analysis = anon.analyze(self.photo)
        self.assertEqual(
            (analysis.options["detection"], analysis.options["unsure"], analysis.options["models"]),
            ("deep", "keep", [{"name": "heavy-face-test", "sha256": "ab" * 32}]),
        )
        saved = self.path("a.json")
        analysis.save(saved)
        self.assertEqual(anonymizer.load_analysis(saved).options, analysis.options)

    def test_pf7_a_different_privacy_rule_gives_a_different_fingerprint(self):
        blur, _ = self.analyse(unsure="blur")
        keep, _ = self.analyse(unsure="keep")
        self.assertNotEqual(blur.options, keep.options)


if __name__ == "__main__":
    unittest.main()
