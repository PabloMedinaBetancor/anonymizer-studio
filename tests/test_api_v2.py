"""A9-A13: the options, the item fields, the analysis summary, the events, the estimate and the analysis file of
version 2, and the wiring of the second look and deep mode into the photo pipeline (decide_photo, classes.load, the API)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from unittest import mock

try:
    import numpy as np
    from PIL import Image

    import anonymizer
    from anonymizer import classes, contract
    from anonymizer.classes.faces import FaceModule
    from anonymizer.classes.heavy import HeavyFaceModule
    from anonymizer.contract import FACE, Detection
    from anonymizer.engine import deep, photos
    from anonymizer.engine.events import ClipError
    from tests import test_api as api_tests
    from tests.fakes import FixedModule
    from tests.heavy_fakes import TEST_TH, FakeHeavy
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_BOX = (140, 60, 200, 120)


class V2Case(api_tests.ApiCase):
    def setUp(self):
        super().setUp()
        self.noise = (np.random.RandomState(3).rand(160, 240, 3) * 255).astype(np.uint8)
        self.photo = self.path("f.png")
        Image.fromarray(self.noise).save(self.photo)

    def analysis(self, **options):
        modules = [FixedModule(FACE, [FACE_BOX], 0.85, 0.75)]
        with anonymizer.Anonymizer(anonymizer.Options(**{"unsure": "blur", **options}), modules=modules) as anon:
            return anon.analyze(
                self.photo
            )  # the items of these tests are the doubtful face blurred, as in 2.1 ("placed" is in test_subtle.py)


class A9Options(V2Case):
    def test_a9_the_new_options_and_their_defaults(self):
        o = anonymizer.Options()
        self.assertEqual((o.detection, o.heavy_only, o.unsure, o.max_zones), ("careful", False, "placed", None))
        deep = anonymizer.Options(detection="deep", heavy_only=True, max_zones=0)
        self.assertEqual((deep.detection, deep.heavy_only, deep.max_zones), ("deep", True, 0))
        self.assertEqual(anonymizer.Options(detection="fast", unsure="keep").unsure, "keep")

    def test_a9_options_refuse_what_makes_no_sense(self):
        for bad in (
            dict(detection="turbo"),
            dict(detection=None),
            dict(heavy_only=True),  # only deep can leave the light models out
            dict(detection="careful", heavy_only=True),
            dict(max_zones=-1),
            dict(max_zones=2.5),
            dict(max_zones="all"),
            dict(max_zones=True),
            dict(unsure="ask"),
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                anonymizer.Options(**bad)

    def test_a9_the_one_call_default_is_subtle_and_2_1_and_v1_are_one_option_away(self):
        doubted = [FixedModule(FACE, [FACE_BOX], 0.85, 0.75)]  # a doubtful face that its module does not say is placed
        with mock.patch("anonymizer.classes.load", lambda kinds, threads, models, **kw: doubted):
            default = anonymizer.anonymize_image(self.photo, self.path("a.png"))
            v21 = anonymizer.anonymize_image(self.photo, self.path("c.png"), anonymizer.Options(unsure="blur"))
            v1 = anonymizer.anonymize_image(
                self.photo, self.path("b.png"), anonymizer.Options(detection="fast", unsure="keep")
            )
        self.assertEqual(
            (default.warnings, default.report.rows[0].decision), (["warning.review.check_left_visible"], "left visible")
        )
        self.assertEqual((v21.warnings, v21.report.rows[0].decision), ([], "blurred"))
        self.assertEqual(
            (v1.warnings, v1.report.rows[0].decision), (["warning.review.check_left_visible"], "left visible")
        )
        self.assertIn('Options(unsure="blur", fit="loose", plate_band=True)', anonymizer.Options.__doc__)
        self.assertIn('Options(detection="fast", unsure="keep", fit="loose")', anonymizer.Options.__doc__)


class A10Items(V2Case):
    def test_a10_items_carry_found_by_second_score_and_decided_by(self):
        analysis = self.analysis()
        [face] = analysis.items
        self.assertEqual((face.found_by, face.second_score, face.decided_by), ("light", None, "engine"))
        analysis.set_blur(face.item_id, True)
        self.assertEqual(analysis.item(face.item_id).decided_by, "person")
        area = analysis.add_area(0, 0, (10, 10, 60, 40))
        self.assertEqual((area.found_by, area.second_score, area.decided_by), ("manual", None, "person"))

    def test_a10_set_blur_marks_the_person_even_when_the_value_does_not_change(self):
        analysis = self.analysis()
        [face] = analysis.items
        self.assertTrue(face.blur)
        analysis.set_blur(face.item_id, True)
        self.assertEqual((face.blur, face.decided_by), (True, "person"))

    def test_a10_the_summary_has_the_counts_of_the_report_and_of_the_second_look(self):
        analysis = self.analysis()
        summary = analysis.summary()
        self.assertEqual(
            {
                k: summary[k]
                for k in ("doubtful", "blurred_by_default", "zones_visited", "zones_skipped", "confirmed", "added")
            },
            dict(doubtful=1, blurred_by_default=1, zones_visited=0, zones_skipped=0, confirmed=0, added=0),
        )
        analysis.second.update(zones_visited=7, zones_skipped=2, confirmed=3, added=1)  # what the second look will fill
        saved = self.path("a.json")
        analysis.save(saved)
        again = anonymizer.load_analysis(saved).summary()
        self.assertEqual(
            (again["zones_visited"], again["zones_skipped"], again["confirmed"], again["added"]), (7, 2, 3, 1)
        )
        self.assertEqual(again["doubtful"], 1)


class A13AnalysisFile(V2Case):
    def test_a13_the_analysis_file_is_format_3_with_items_that_round_trip(self):
        analysis = self.analysis()
        saved = self.path("a.json")
        analysis.save(saved)
        with open(saved, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["version"], 3)
        self.assertEqual(
            sorted(data["items"][0]),
            sorted(
                [
                    "item_id",
                    "kind",
                    "verdict",
                    "reason",
                    "first_frame",
                    "last_frame",
                    "max_score",
                    "blur",
                    "params",
                    "found_by",
                    "second_score",
                    "decided_by",
                ]
            ),
        )
        self.assertEqual(anonymizer.load_analysis(saved).items, analysis.items)

    def test_a13_the_api_reference_is_regenerated_and_says_the_new_things(self):
        done = subprocess.run(
            [
                sys.executable,
                *(["-s"] if sys.flags.no_user_site else []),
                os.path.join(REPO, "scripts", "api_docs.py"),
                "--check",
            ],
            capture_output=True,
            text=True,
            env=os.environ,
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        reference = os.path.join(REPO, "docs", "reference")
        text = ""
        for name in sorted(os.listdir(reference)):
            with open(os.path.join(reference, name), encoding="utf-8") as f:
                text += f.read()
        needles = (
            "`unsure` (`str`, default `'placed'`)",
            "found_by",
            "decided_by",
            "fingerprint",
            "a hash of its content",
            "summary",
        )
        for needle in needles:
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()


PERSON = (80, 40, 160, 260)  # its head zone is (72, 40, 168, 131): the top 35 %
HEAD_FACE = ((100, 50, 140, 90), 0.95)  # a face in that head zone that only the heavy model sees


def persons(bgr):
    return [Detection(*PERSON, 0.9, "person")]


class WiringCase(api_tests.ApiCase):
    """A plain grey 300 x 400 photo (nothing for the light models to find) with a person in it: the second look has one head zone to visit."""

    def setUp(self):
        super().setUp()
        self.photo = self.path("grey.png")
        Image.fromarray(np.full((300, 400, 3), 120, np.uint8)).save(self.photo)

    def heavy(self, placed=(HEAD_FACE,), **fake) -> HeavyFaceModule:
        module = HeavyFaceModule(heavy_zone=FakeHeavy(placed, **fake), heavy_thresholds=TEST_TH)
        self.addCleanup(module.close)
        return module

    def analyse(self, module, on_event=None, cancel=None, **options):
        with anonymizer.Anonymizer(anonymizer.Options(**options), modules=[module], persons=persons) as anon:
            return anon.analyze(self.photo, on_event, cancel)


class A9TheDetectionModeReallyRuns(WiringCase):
    def test_a9_careful_looks_at_the_head_of_a_person_and_brings_in_a_face_only_the_heavy_model_sees(self):
        module = self.heavy()
        analysis = self.analyse(module, detection="careful")
        (item,) = analysis.items
        self.assertEqual(
            (item.kind, item.verdict, item.blur, item.found_by, item.second_score), (FACE, "blur", True, "second", 0.95)
        )
        self.assertEqual(item.reason, "face.reason.second_found")
        summary = analysis.summary()
        self.assertEqual((summary["zones_visited"], summary["zones_skipped"], summary["added"]), (1, 0, 1))
        self.assertEqual(analysis.options["detection_ran"], "careful")
        (call,) = module.heavy_zone.calls
        self.assertTrue(call[0][0] <= 72 and call[0][2] >= 168, call)  # the region of the head zone, enlarged

    def test_a9_fast_never_calls_the_heavy_model(self):
        module = self.heavy()
        analysis = self.analyse(module, detection="fast")
        self.assertEqual((analysis.items, module.heavy_zone.calls), ([], []))
        self.assertEqual((analysis.options["detection"], analysis.options["detection_ran"]), ("fast", "fast"))

    def test_a9_deep_sweeps_the_whole_picture_and_the_item_says_so(self):
        module = self.heavy()
        analysis = self.analyse(module, detection="deep")
        (item,) = analysis.items
        self.assertEqual((item.found_by, item.verdict), ("deep", "blur"))
        self.assertEqual(analysis.options["detection_ran"], "deep")
        self.assertEqual(analysis.summary()["zones_visited"], deep.call_count(400, 300)["calls"])

    def test_a9_the_options_record_what_ran_not_only_what_was_asked_for(self):
        for options, items in ((dict(detection="careful"), 1), (dict(detection="deep", heavy_only=True), 1)):
            with anonymizer.Anonymizer(anonymizer.Options(**options), modules=[FixedModule(FACE, [FACE_BOX])]) as anon:
                analysis = anon.analyze(self.photo)  # no module has a heavy model: the second look cannot run
            self.assertEqual(
                (analysis.options["detection"], analysis.options["detection_ran"]), (options["detection"], "fast")
            )
            self.assertEqual(len(analysis.items), items)  # and the light result is kept: nothing was left out

    def test_a9_the_fingerprint_lists_the_heavy_model_that_ran(self):
        analysis = self.analyse(self.heavy(), detection="careful")
        self.assertEqual(analysis.options["models"], [{"name": "fake-heavy", "sha256": "0" * 64}])

    def test_a9_classes_load_builds_the_heavy_face_module_only_for_careful_and_deep(self):
        for detection, heavy in (("fast", False), ("careful", True), ("deep", True)):
            (module,) = classes.load([FACE], detection=detection)
            self.addCleanup(module.close)
            self.assertIs(isinstance(module, HeavyFaceModule), heavy, detection)
            self.assertIsInstance(module, FaceModule)
        with self.assertRaises(ValueError):
            classes.load([FACE], detection="turbo")

    def test_a9_careful_without_the_heavy_model_file_raises_and_names_it_and_fast_does_not(self):
        folder = self.path("models")
        os.makedirs(folder)
        source = contract.models_dir(None)
        for name in ("face_detection_yunet_2023mar.onnx", "SHA256SUMS"):
            with open(os.path.join(source, name), "rb") as f, open(os.path.join(folder, name), "wb") as out:
                out.write(f.read())
        for detection, raises in (("careful", True), ("deep", True), ("fast", False)):
            anon = anonymizer.Anonymizer(anonymizer.Options(detection=detection, plates=False), models_dir=folder)
            if raises:
                with self.assertRaises(anonymizer.ModelLoadError) as caught:
                    anon.modules()
                self.assertIn("retinaface_r34.onnx", caught.exception.detail, detection)
            else:
                self.assertEqual([m.kind for m in anon.modules()], [FACE])
            anon.close()


class A9AFailedHeavyLookIsNeverNothingFound(WiringCase):
    def test_a9_a_look_that_fails_makes_the_photo_not_checked_never_clean(self):
        module = self.heavy(fail_on={0})
        with self.assertRaises(anonymizer.IncompleteAnalysis):
            self.analyse(module, detection="careful")
        self.assertEqual(module.model_failures, 1)  # B's probe P1: the failed look on a head zone is counted
        module = self.heavy(fail_on={0})
        with self.assertRaises(ClipError) as raised:
            photos.decide_photo(np.full((300, 400, 3), 120, np.uint8), [module], "blur", "careful", persons)
        self.assertEqual(raised.exception.code, "error.analysis.incomplete")

    def test_a9_a_deep_sweep_that_fails_makes_the_photo_not_checked_too(self):
        with self.assertRaises(anonymizer.IncompleteAnalysis):
            self.analyse(self.heavy(fail_on={0}), detection="deep")

    def test_a9_the_budget_goes_through_decide_photo_and_is_reported(self):
        module = self.heavy()
        tally = photos.SecondTally()
        zones = photos.decide_photo(
            np.full((300, 400, 3), 120, np.uint8), [module], "blur", "careful", persons, max_zones=0, tally=tally
        )
        self.assertEqual((zones, module.heavy_zone.calls), ([], []))
        self.assertEqual((tally.stage, tally.zones_visited, tally.zones_skipped), ("second", 0, 1))

    def test_a9_stop_is_felt_between_the_heavy_calls_of_a_deep_sweep(self):
        module = self.heavy()
        token = anonymizer.CancelToken()
        inner = module.heavy_zone.detect

        def detect_then_stop(*args):
            found = inner(*args)
            token.cancel()
            return found

        module.heavy_zone.detect = detect_then_stop
        big = self.path("big.png")
        Image.fromarray(np.full((1400, 1800, 3), 120, np.uint8)).save(big)  # several tiles
        self.photo = big
        with self.assertRaises(anonymizer.Cancelled):
            self.analyse(module, cancel=token, detection="deep")
        self.assertEqual(len(module.heavy_zone.calls), 1)

    def test_a9_stop_is_felt_between_the_zone_visits_of_the_second_look(self):
        module = self.heavy()
        token = anonymizer.CancelToken()
        inner = module.heavy_zone.detect

        def detect_then_stop(*args):
            found = inner(*args)
            token.cancel()
            return found

        module.heavy_zone.detect = detect_then_stop
        two = lambda bgr: [Detection(*PERSON, 0.9, "person"), Detection(250, 40, 330, 260, 0.9, "person")]  # noqa: E731
        with anonymizer.Anonymizer(anonymizer.Options(detection="careful"), modules=[module], persons=two) as anon:
            with self.assertRaises(anonymizer.Cancelled):
                anon.analyze(self.photo, cancel=token)
        self.assertEqual(len(module.heavy_zone.calls), 1)  # the second head zone was never visited

    def test_a9_what_the_class_rules_dropped_is_not_listed_and_the_second_look_may_bring_it_back(self):
        weak = FixedModule(FACE, [FACE_BOX], 0.3, None)  # a weak detection nothing confirms: DROP
        bgr = np.full((300, 400, 3), 120, np.uint8)
        self.assertEqual(photos.decide_photo(bgr, [weak], "blur", "careful"), [])

    def test_a9_a_picture_over_the_tile_limit_is_refused_with_its_own_error(self):
        refusal = deep.DeepRefused("error.deep.too_large", {"tiles": 300, "limit": 120})
        with mock.patch("anonymizer.api.analyze_photo", side_effect=refusal):
            with self.assertRaises(anonymizer.AnonymizerError) as raised:
                self.analyse(self.heavy(), detection="deep")
        self.assertEqual(
            (raised.exception.code, raised.exception.params), ("error.deep.too_large", {"tiles": 300, "limit": 120})
        )


class A11Events(WiringCase):
    def test_a11_a_careful_photo_reports_its_stages_and_the_zones_in_order(self):
        events = []
        self.analyse(self.heavy(), events.append, detection="careful")
        self.assertIsInstance(events[0], anonymizer.Started)  # the progress of the heavy stage follows Started
        progress = [e for e in events if isinstance(e, anonymizer.Progress)]
        self.assertEqual([(e.stage, e.zones) for e in progress], [("first", None), ("second", {"done": 1, "total": 1})])
        self.assertEqual([e.fraction for e in progress], [0.5, 1.0])
        self.assertIsInstance(events[-1], anonymizer.Analyzed)

    def test_a11_the_fraction_of_the_heavy_stage_follows_the_zones_done(self):
        two = lambda bgr: [Detection(*PERSON, 0.9, "person"), Detection(250, 40, 330, 260, 0.9, "person")]  # noqa: E731
        events = []
        with anonymizer.Anonymizer(
            anonymizer.Options(detection="careful"), modules=[self.heavy(())], persons=two
        ) as anon:
            anon.analyze(self.photo, events.append)
        progress = [e for e in events if isinstance(e, anonymizer.Progress)]
        self.assertEqual([e.fraction for e in progress], [0.5, 0.75, 1.0])
        self.assertEqual([e.zones for e in progress], [None, {"done": 1, "total": 2}, {"done": 2, "total": 2}])

    def test_a11_analyzed_carries_the_zone_counts_and_what_the_second_look_added(self):
        events = []
        self.analyse(self.heavy(), events.append, detection="careful")
        analyzed = next(e for e in events if isinstance(e, anonymizer.Analyzed))
        self.assertEqual(
            (analyzed.items, analyzed.zones_visited, analyzed.zones_skipped, analyzed.second_added), (1, 1, 0, 1)
        )
        events = []
        self.analyse(self.heavy(), events.append, detection="careful", max_zones=0)
        analyzed = next(e for e in events if isinstance(e, anonymizer.Analyzed))
        self.assertEqual(
            (analyzed.items, analyzed.zones_visited, analyzed.zones_skipped, analyzed.second_added), (0, 0, 1, 0)
        )

    def test_a11_the_json_of_the_events_has_the_new_fields(self):
        events = []
        self.analyse(self.heavy(), events.append, detection="careful")
        progress = next(anonymizer.event_json(e) for e in events if isinstance(e, anonymizer.Progress) and e.zones)
        self.assertEqual((progress["stage"], progress["zones"]), ("second", {"done": 1, "total": 1}))
        analyzed = anonymizer.event_json(next(e for e in events if isinstance(e, anonymizer.Analyzed)))
        self.assertEqual(
            {k: analyzed[k] for k in ("zones_visited", "zones_skipped", "second_added")},
            dict(zones_visited=1, zones_skipped=0, second_added=1),
        )

    def test_a11_a_deep_photo_reports_the_deep_stage_and_a_fast_one_reports_none(self):
        events = []
        self.analyse(self.heavy(), events.append, detection="deep")
        stages = {e.stage for e in events if isinstance(e, anonymizer.Progress)}
        self.assertEqual(stages, {"first", "deep"})
        self.assertTrue(all(e.zones is None for e in events if isinstance(e, anonymizer.Progress)))
        events = []
        self.analyse(self.heavy(), events.append, detection="fast")
        self.assertEqual([e for e in events if isinstance(e, anonymizer.Progress)], [])

    def test_a11_a_video_has_no_stage_and_counts_the_heavy_looks_in_its_analyzed(self):
        from tests.fakes import plate_module

        events = []
        with anonymizer.Anonymizer(anonymizer.Options(detection="careful"), modules=[plate_module()]) as anon:
            anon.analyze(self.clip(n=20, audio=False), events.append)
        self.assertEqual({e.stage for e in events if isinstance(e, anonymizer.Progress)}, {None})
        analyzed = next(e for e in events if isinstance(e, anonymizer.Analyzed))
        self.assertEqual((analyzed.zones_visited, analyzed.zones_skipped, analyzed.second_added), (0, 0, 0))


class A9QuestionsOfTheReview(WiringCase):
    """RV-WIRE Q1-Q6: what the report says, a dropped candidate brought back, heavy_only at the API, a person finder that raises."""

    RED = (0, 0, 255)

    def red_photo(self):
        bgr = np.full((300, 400, 3), 120, np.uint8)
        bgr[100:160, 150:200] = self.RED
        path = self.path("red.png")
        Image.fromarray(bgr[:, :, ::-1]).save(path)
        return path

    def test_a9_q1_the_zones_a_photo_left_unvisited_are_a_warning_and_in_the_report_header(self):
        events = []
        crowd = lambda bgr: [Detection(10 + 60 * i, 40, 60 + 60 * i, 200, 0.9, "person") for i in range(5)]  # noqa: E731
        with anonymizer.Anonymizer(
            anonymizer.Options(detection="careful", max_zones=2), modules=[self.heavy(())], persons=crowd
        ) as anon:
            analysis = anon.analyze(self.photo, events.append)
            result = anon.render(analysis, self.path("out.png"))
        (warning,) = [
            e for e in events if isinstance(e, anonymizer.JobWarning) and e.code == "warning.second_look.zones_skipped"
        ]
        self.assertEqual(warning.params["count"], 3)
        self.assertEqual((result.report.summary()["zones_visited"], result.report.summary()["zones_skipped"]), (2, 3))
        saved = self.path("report.json")
        result.report.to_json(saved)
        with open(saved, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["summary"]["zones_skipped"], 3)

    def test_a9_q1_a_photo_with_every_zone_visited_has_no_warning(self):
        events = []
        self.analyse(self.heavy(), events.append, detection="careful")
        self.assertEqual([e for e in events if isinstance(e, anonymizer.JobWarning)], [])

    def test_a9_q2_a_candidate_the_class_rule_dropped_comes_back_when_the_heavy_model_finds_it(self):
        from tests.fakes import SecondColourModule

        weak = SecondColourModule(FACE, self.RED, [self.RED], score=0.5, verify_score=0.1)  # the light rule drops it
        bgr = np.full((300, 400, 3), 120, np.uint8)
        bgr[100:160, 150:200] = self.RED
        self.assertEqual(photos.decide_photo(bgr, [weak], "blur", "fast"), [])
        (zone,) = photos.decide_photo(bgr, [weak], "blur", "careful")
        self.assertEqual(
            (zone.found_by, zone.blur, zone.judgement.reason), ("second", True, "face.reason.second_found")
        )
        self.assertEqual(len(weak.zones), 1)

    def test_a9_q3_heavy_only_leaves_the_light_result_of_a_class_with_a_heavy_model_out(self):
        from tests.fakes import SecondColourModule

        path = self.red_photo()
        for heavy_only, expected in ((False, 1), (True, 0)):
            sure = SecondColourModule(
                FACE, self.RED, [(0, 200, 0)], score=0.9, verify_score=0.9
            )  # the heavy model sees nothing red
            with anonymizer.Anonymizer(
                anonymizer.Options(detection="deep", heavy_only=heavy_only), modules=[sure]
            ) as anon:
                analysis = anon.analyze(path)
            self.assertEqual(len(analysis.items), expected, heavy_only)
            self.assertEqual(analysis.options["heavy_only"], heavy_only)

    def test_a9_q5_a_person_finder_that_raises_is_a_failed_look_with_a_code(self):
        def broken(bgr):
            raise RuntimeError("the finder crashed")

        with anonymizer.Anonymizer(
            anonymizer.Options(detection="careful"), modules=[self.heavy()], persons=broken
        ) as anon:
            with self.assertRaises(anonymizer.IncompleteAnalysis) as raised:
                anon.analyze(self.photo)
        self.assertEqual(raised.exception.code, "error.analysis.incomplete")
        self.assertIn("person finder", raised.exception.detail)

    def test_a9_q6_heavy_only_is_recorded_only_when_a_deep_sweep_ran(self):
        light_only = anonymizer.Anonymizer(
            anonymizer.Options(detection="deep", heavy_only=True), modules=[FixedModule(FACE, [FACE_BOX])]
        )
        with light_only as anon:
            analysis = anon.analyze(self.photo)
        self.assertEqual((analysis.options["heavy_only"], analysis.options["detection_ran"]), (False, "fast"))


class A11Video(WiringCase):
    def test_a11_a_video_with_a_heavy_face_module_gets_its_persons_heads_looked_at_and_the_item_says_who_found_it(self):
        events = []
        module = self.heavy()
        with anonymizer.Anonymizer(anonymizer.Options(detection="careful"), modules=[module], persons=persons) as anon:
            analysis = anon.analyze(self.clip(n=30, audio=False), events.append)
        (item,) = [i for i in analysis.items if i.kind == FACE]
        self.assertEqual((item.found_by, item.verdict, item.blur, item.second_score), ("second", "blur", True, 0.95))
        self.assertGreaterEqual(analysis.summary()["zones_visited"], 1)
        analyzed = next(e for e in events if isinstance(e, anonymizer.Analyzed))
        self.assertEqual(analyzed.zones_visited, analysis.summary()["zones_visited"])
        self.assertGreater(len(module.heavy_zone.calls), 0)
        self.assertEqual(analysis.options["detection_ran"], "careful")


class A12Estimate(WiringCase):
    def estimate(self, path=None, calibration_ms=100.0, module=None, light_ms=250.0, **options):
        options.setdefault("detection", "careful")
        modules = [module or self.heavy()]
        with anonymizer.Anonymizer(
            anonymizer.Options(**options), modules=modules, calibration_ms=calibration_ms, light_ms=light_ms
        ) as anon:
            return anon.estimate(path or self.photo)

    def test_a12_a_fast_photo_costs_its_light_pass_and_visits_no_zone(self):
        self.assertEqual(self.estimate(detection="fast"), anonymizer.Estimate(0.25, 0, 0))

    def test_a12_a_careful_photo_adds_the_expected_zones_at_the_calibrated_time_of_a_call(self):
        self.assertEqual(self.estimate(), anonymizer.Estimate(1.25, 10, 0))  # 0.25 s + 10 zones x 100 ms
        self.assertEqual(self.estimate(max_zones=4), anonymizer.Estimate(0.65, 4, 0))  # the cap bounds it
        self.assertEqual(self.estimate(max_zones=0).zones_expected, 0)

    def test_a12_a_deep_photo_adds_the_calls_and_tiles_of_its_sweep(self):
        counts = deep.call_count(400, 300)
        found = self.estimate(detection="deep")
        self.assertEqual((found.tiles, found.zones_expected), (counts["tiles"], 0))
        self.assertAlmostEqual(found.seconds, 0.25 + counts["calls"] * 0.1, places=2)

    def test_a12_a_picture_over_the_tile_limit_is_refused_before_anything_runs(self):
        huge = anonymizer.MediaInfo("photo", 20000, 12000, 1, 0.0, 0.0, "JPEG", False, False, 1)
        with mock.patch("anonymizer.api.media_info", return_value=huge):
            with self.assertRaises(anonymizer.AnonymizerError) as raised:
                self.estimate(detection="deep")
        self.assertEqual(raised.exception.code, "error.deep.too_large")

    def test_a12_a_video_costs_the_heavy_calls_its_key_frames_may_make_or_its_deep_sweeps(self):
        clip = self.clip(n=60, audio=False)  # 2 s at 30 fps
        found = self.estimate(clip, module=self.heavy())
        self.assertEqual(
            (found.seconds, found.zones_expected, found.tiles), (1.2, 12, 0)
        )  # 6 calls a second x 2 s x 100 ms
        found = self.estimate(clip, detection="deep")
        self.assertEqual((found.seconds, found.zones_expected), (0.2, 2))
        self.assertEqual(self.estimate(clip, detection="fast"), anonymizer.Estimate(0.0, 0, 0))
        with self.assertRaises(anonymizer.AnonymizerError) as raised:
            self.estimate(clip, detection="deep", calibration_ms=5_000_000.0)  # an hour of work and more
        self.assertEqual(raised.exception.code, "error.deep.too_much_work")

    def test_a12_without_a_stored_calibration_the_heavy_model_is_timed_once_and_kept(self):
        module = self.heavy()
        with anonymizer.Anonymizer(anonymizer.Options(detection="careful"), modules=[module]) as anon:
            self.assertIsNone(anon.calibration_ms)
            first = anon.estimate(self.photo)
            self.assertGreater(anon.calibration_ms, 0.0)
            calls = len(module.heavy_zone.calls)
            self.assertEqual(anon.estimate(self.photo), first)
            self.assertEqual(len(module.heavy_zone.calls), calls)  # the second estimate did not run the model again

    def test_a12_without_a_stored_light_time_the_light_pass_is_timed_for_the_size_of_the_photo_and_kept(self):
        module = self.heavy()
        seen = []
        sweep = module.sweep
        module.sweep = lambda bgr, plan: seen.append(bgr.shape[:2]) or sweep(bgr, plan)
        with anonymizer.Anonymizer(anonymizer.Options(detection="fast"), modules=[module], calibration_ms=1.0) as anon:
            self.assertIsNone(anon.light_ms)
            first = anon.estimate(self.photo)
            self.assertEqual(seen[-1], (300, 400))  # a picture of noise of the photo's own size, after a small one
            self.assertLess(seen[0][1], 400)
            swept = len(seen)
            self.assertEqual(anon.estimate(self.photo), first)
            self.assertEqual(len(seen), swept)  # the second estimate of that size did not sweep again
        self.assertGreater(first.seconds, 0.0)

    def test_a12_the_text_detector_of_a_plate_module_adds_the_vehicles_it_is_expected_to_look_into(self):
        from anonymizer.engine import photos

        text = mock.Mock()
        module = mock.Mock(sweep=mock.Mock(return_value=[]), text=text)
        photos.calibrate_light([module], 400, 300)
        self.assertEqual(text.detect.call_count, 1 + 1)  # a warm-up and EXPECTED_VEHICLES timed runs
        self.assertEqual(module.sweep.call_count, 2)

    def test_a12_the_stored_light_time_is_the_one_estimated_whatever_it_is(self):
        self.assertEqual(self.estimate(detection="fast", light_ms=400.0).seconds, 0.4)
        self.assertEqual(self.estimate(light_ms=400.0).seconds, 1.4)
        self.assertAlmostEqual(
            self.estimate(detection="deep", light_ms=400.0).seconds,
            0.4 + deep.call_count(400, 300)["calls"] * 0.1,
            places=2,
        )

    def test_a12_the_timed_light_pass_counts_the_sweep_scaled_by_the_picture_made_and_the_text_runs(self):
        from anonymizer.engine import photos

        shapes = []
        text = mock.Mock()
        module = mock.Mock(sweep=mock.Mock(side_effect=lambda bgr, plan: shapes.append(bgr.shape[:2]) or []), text=text)
        clock = iter([0.0, 1.0, 10.0, 12.0])  # the sweep takes 1 s, the timed text run 2 s
        with mock.patch.object(photos.time, "perf_counter", lambda: next(clock)):
            ms = photos.calibrate_light([module], 16000, 8000)
        self.assertEqual(
            shapes[-1], (4000, 8000)
        )  # noise is never made larger than 8000 px: its time is scaled by the area (4 times)
        self.assertEqual(ms, 4000.0 + photos.EXPECTED_VEHICLES * 2000.0)

    def test_a12_calibrating_a_set_with_no_heavy_model_is_a_model_error(self):
        with anonymizer.Anonymizer(modules=[FixedModule(FACE, [FACE_BOX])]) as anon:
            with self.assertRaises(anonymizer.ModelLoadError):
                anon.calibrate()

    def test_a12_the_hosted_demo_refuses_deep_mode_and_keeps_careful(self):
        module = self.heavy()
        with anonymizer.Anonymizer(
            anonymizer.Options(detection="deep"), modules=[module], demo=True, calibration_ms=1.0, light_ms=1.0
        ) as anon:
            for call in (anon.estimate, anon.analyze):
                with self.assertRaises(anonymizer.AnonymizerError) as raised:
                    call(self.photo)
                self.assertEqual(raised.exception.code, "error.demo.deep")
        self.assertEqual(module.heavy_zone.calls, [])
        self.assertEqual(self.estimate(detection="careful").zones_expected, 10)
