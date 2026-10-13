"""PF4, S11, S14: the Studio starts doubtful items blurred, asks nothing before an export, offers
one confirmed bulk decision, and demo mode refuses Deep."""

from __future__ import annotations

import json
import os
import pathlib
import time
import unittest
from unittest import mock

try:
    import numpy as np
    from PIL import Image

    from anonymizer.contract import FACE, PLATE
    from anonymizer.ui import hub as hubmod
    from anonymizer.ui.settings import InvalidSetting, Settings
    from tests import test_ui_server as server_tests
    from tests.fakes import FixedModule, doubtful_plate_module
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

UI = pathlib.Path(__file__).resolve().parents[2] / "src" / "anonymizer" / "ui"
REMOVED_KEYS = (
    "error.review.pending",
    "export.confirm_visible",
    "export.visible_title",
    "export.render_anyway",
    "export.review_first",
)
LANGUAGES = ("en", "es", "fr", "de", "pt", "it", "zh-Hans")


def ui_sources():
    for path in sorted(UI.rglob("*")):
        if path.suffix in (".py", ".js", ".html", ".json") and path.is_file():
            yield path, path.read_text(encoding="utf-8")


class PF4DoubtfulStartsBlurred(server_tests.StudioCase):
    modules = [doubtful_plate_module()]

    def test_pf4_a_doubtful_item_starts_blurred_and_counts_as_to_check(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        [item] = self.api("GET", f"/api/clips/{clip_id}/analysis")["items"]
        self.assertEqual((item["judgement"]["verdict"], item["blur"]), ("check", True))
        plan = self.api("POST", "/api/render/plan", {"clips": [clip_id], "options": {}})
        self.assertEqual(plan["clips"][0]["to_check"], 1)

    def test_pf4_the_bulk_decision_changes_every_doubtful_item_and_counts_them_as_looked_at(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        base = f"/api/clips/{clip_id}/doubtful"
        kept = self.api("POST", base, {"blur": False})
        self.assertEqual([i["blur"] for i in kept["items"]], [False])
        self.assertEqual(kept["to_check"], 0)
        self.assertFalse(self.api("GET", f"/api/clips/{clip_id}/analysis")["items"][0]["blur"])
        blurred = self.api("POST", base, {"blur": True})
        self.assertEqual([i["blur"] for i in blurred["items"]], [True])
        self.assertTrue(self.api("GET", f"/api/clips/{clip_id}/analysis")["items"][0]["blur"])

    def test_pf4_the_bulk_decision_refuses_anything_but_a_true_or_false(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        base = f"/api/clips/{clip_id}/doubtful"
        for body in ({"blur": "yes"}, {"blur": 1}, {}, {"blur": True, "extra": 1}):
            status, _, data = self.request("POST", base, body)
            self.assertEqual((status, json.loads(data)["code"]), (400, "error.request.invalid"), body)
        self.assertTrue(self.api("GET", f"/api/clips/{clip_id}/analysis")["items"][0]["blur"])

    def test_pf4_nothing_of_the_acknowledge_dialog_is_left_in_the_studio(self):
        for path, text in ui_sources():
            for word in ("acknowledge_visible", "askVisible", *REMOVED_KEYS):
                self.assertNotIn(word, text, f"{path.name} still mentions {word}")
        for lang in LANGUAGES:
            keys = json.loads((UI / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            self.assertFalse([k for k in keys if k.startswith(REMOVED_KEYS)], lang)

    def test_pf4_the_review_screen_asks_for_a_confirmation_before_the_bulk_decision(self):
        review = (UI / "static" / "js" / "views" / "review.js").read_text(encoding="utf-8")
        self.assertIn("bulkDecision({", review)  # the confirm-then-send order is tested in review-logic.test.mjs
        self.assertIn("askConfirm(", review)
        self.assertIn("review.doubtful.keep_visible", review)
        self.assertIn("review.doubtful.blur_all", review)
        self.assertIn("review.doubtful.${kind}_title", review)
        for lang in LANGUAGES:
            keys = json.loads((UI / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            for kind in ("keep", "blur"):
                for part in ("title", "body", "body.one", "confirm"):
                    self.assertIn(f"review.doubtful.{kind}_{part}", keys, lang)


class PF4BulkOnMixedClip(server_tests.StudioCase):
    """A photo with one doubtful face and one plate the engine is sure of: the bulk decision touches only the first."""

    modules = [
        FixedModule(FACE, [(140, 60, 200, 120)], 0.85, 0.75, placed=True),
        FixedModule(PLATE, [(20, 20, 100, 44)]),
    ]

    def photo(self) -> str:
        path = os.path.join(self.dir, "media", "f.png")
        Image.fromarray((np.random.RandomState(3).rand(160, 240, 3) * 255).astype(np.uint8)).save(path)
        return self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]

    def items(self, clip_id: str) -> dict:
        return {i["kind"]: i for i in self.api("GET", f"/api/clips/{clip_id}/analysis")["items"]}

    def test_pf4_keeping_all_doubtful_visible_never_touches_an_item_the_engine_was_sure_of(self):
        clip_id = self.photo()
        self.analyze(clip_id)
        before = self.items(clip_id)
        self.assertEqual(
            (before["face"]["judgement"]["verdict"], before["plate"]["judgement"]["verdict"]), ("check", "blur")
        )
        answer = self.api("POST", f"/api/clips/{clip_id}/doubtful", {"blur": False})
        self.assertEqual([i["kind"] for i in answer["items"]], ["face"])
        after = self.items(clip_id)
        self.assertEqual((after["face"]["blur"], after["face"]["decided_by"]), (False, "person"))
        self.assertEqual((after["plate"]["blur"], after["plate"]["decided_by"]), (True, "engine"))

    def test_pf4_the_bulk_decision_bumps_rev_and_the_report_says_the_person_decided(self):
        clip_id = self.photo()
        self.analyze(clip_id)
        rev = self.api("GET", f"/api/clips/{clip_id}/analysis")["rev"]
        answer = self.api("POST", f"/api/clips/{clip_id}/doubtful", {"blur": False})
        self.assertGreater(answer["rev"], rev)
        report = self.api("GET", f"/api/clips/{clip_id}/report?format=json")
        rows = {row["kind"]: row for row in report["items"]}
        self.assertEqual((rows["face"]["decided_by"], rows["face"]["decision"]), ("person", "left visible"))
        self.assertEqual((rows["plate"]["decided_by"], rows["plate"]["decision"]), ("engine", "blurred"))
        self.assertEqual((report["summary"]["turned_off_by_person"], report["summary"]["blurred_by_default"]), (1, 0))


class S8HeavyOnlyIsNotASetting(unittest.TestCase):
    def test_s8_the_studio_has_no_heavy_only_and_no_deep(self):
        settings = Settings()
        self.assertNotIn("heavy_only", settings.get())
        self.assertFalse(settings.options(detection="careful").heavy_only)
        with self.assertRaises(InvalidSetting):
            settings.update({"heavy_only": True})
        with self.assertRaises(InvalidSetting):
            settings.update({"detection": "deep"})


class S11ExportAsksNothing(server_tests.StudioCase):
    modules = [doubtful_plate_module()]

    def test_s11_the_export_starts_without_an_acknowledgement_and_the_doubtful_item_leaves_blurred(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/render", {"clips": [clip_id], "options": {"destination": folder}}, 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.assertFalse([e for e in reader.events if e["type"] == "job.warning" and "check_left" in e["code"]])
        self.assertEqual(len([n for n in os.listdir(folder) if n.endswith(".mp4")]), 1)

    def test_s11_the_note_says_how_many_doubtful_items_have_no_decision_in_every_language(self):
        for lang in LANGUAGES:
            keys = json.loads((UI / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            self.assertIn("{count}", keys["export.to_check"], lang)
            self.assertNotIn("{count}", keys["export.to_check.one"], lang)
            self.assertIn("1", keys["export.to_check.one"], lang)
        english = json.loads((UI / "i18n" / "en.json").read_text(encoding="utf-8"))
        self.assertIn("goes out as marked", english["export.to_check"])


class S14DemoRefusesDeep(server_tests.StudioCase):
    demo = True

    def upload(self) -> str:
        with open(self.make_clip(), "rb") as f:
            data = f.read()
        status, _, body = self.request("PUT", "/api/upload?name=demo.mp4", raw=data)
        self.assertEqual(status, 200)
        return json.loads(body)["clips"][0]["clip_id"]

    def test_s14_the_demo_accepts_careful_and_fast_and_deep_is_not_a_choice(self):
        clip_id = self.upload()
        status, _, data = self.request(
            "POST", "/api/job/analyze", {"clips": [clip_id], "options": {"detection": "deep"}}
        )
        self.assertEqual((status, json.loads(data)["code"]), (400, "error.request.invalid"))
        self.api("POST", "/api/job/analyze", {"clips": [clip_id], "options": {"detection": "careful"}}, 202)
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.api("POST", "/api/job/analyze", {"clips": [clip_id], "options": {"detection": "fast"}}, 202)
        self.wait_idle()

    def test_s14_the_demo_limits_are_the_v1_ones(self):
        limits = self.api("GET", "/api/state")["demo_limits"]
        self.assertEqual(limits["max_photos"], 10)
        self.assertEqual(limits["max_seconds"], 20)


class S8DetectionControl(server_tests.StudioCase):
    modules = [doubtful_plate_module()]

    def test_s8_the_server_reads_detection_and_refuses_a_made_up_tier_and_deep(self):
        clip_id = self.import_clip()
        self.assertEqual(self.api("GET", "/api/settings")["detection"], "careful")
        for options in ({"detection": "turbo"}, {"detection": "deep"}, {"heavy_only": True}):
            status, _, data = self.request("POST", "/api/job/analyze", {"clips": [clip_id], "options": options})
            self.assertEqual((status, json.loads(data)["code"]), (400, "error.request.invalid"), options)
        self.api("POST", "/api/job/analyze", {"clips": [clip_id], "options": {"detection": "fast"}}, 202)
        self.wait_idle()

    def test_s8_the_costs_are_numbers_only_after_two_tiers_ran_on_the_same_clip(self):
        clip_id = self.import_clip()
        self.assertEqual(self.api("GET", "/api/state")["slowdown"], {"careful": None})
        for detection in ("fast", "careful"):
            self.api("POST", "/api/job/analyze", {"clips": [clip_id], "options": {"detection": detection}}, 202)
            self.wait_idle()
        slowdown = self.api("GET", "/api/state")["slowdown"]
        self.assertGreater(slowdown["careful"], 0)
        self.assertNotIn("deep", slowdown)

    def test_s8_the_page_has_the_control_and_no_invented_cost_and_no_deep(self):
        page = (UI / "static" / "js" / "views" / "import.js").read_text(encoding="utf-8")
        self.assertIn("t(`options.detection.${value}`)", page)
        self.assertIn('const DETECTIONS = ["fast", "careful"];', page)
        self.assertNotIn("heavy", page.lower())
        self.assertIn("costs[value] ?", page)  # a factor is printed only when the server measured one
        for lang in LANGUAGES:
            keys = json.loads((UI / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            self.assertIn("{factor}", keys["options.detection.careful.slower"], lang)
            self.assertNotRegex(keys["options.detection.careful.hint"], r"\d", lang)  # the hint carries no number
            self.assertFalse(
                [
                    k
                    for k in keys
                    if k.startswith(("options.detection.deep", "options.heavy_only")) or k == "analyze.deep_look"
                ],
                lang,
            )

    def test_s8_a_local_studio_does_not_offer_deep(self):
        self.assertNotIn("deep", self.api("GET", "/api/state")["capabilities"])


class S9SecondLookBadge(server_tests.StudioCase):
    modules = [doubtful_plate_module()]

    def test_s9_an_item_says_who_found_it_and_who_decided(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        [item] = self.api("GET", f"/api/clips/{clip_id}/analysis")["items"]
        self.assertEqual((item["found_by"], item["second_score"], item["decided_by"]), ("light", None, "engine"))
        changed = self.api("PATCH", f"/api/clips/{clip_id}/items/{item['item_id']}", {"blur": False})["item"]
        self.assertEqual(changed["decided_by"], "person")

    def test_s9_the_review_screen_has_the_badge_the_score_and_both_filters(self):
        review = (UI / "static" / "js" / "views" / "review.js").read_text(encoding="utf-8")
        for key in ("review.badge.second", "review.second_score", "review.only_check", "review.only_second"):
            self.assertIn(key, review)
        for lang in LANGUAGES:
            keys = json.loads((UI / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            for key in ("review.badge.second", "review.second_score", "review.only_second"):
                self.assertIn(key, keys, lang)


class S10SecondLookStage(server_tests.StudioCase):
    def test_s10_the_stage_and_the_zones_of_the_engine_reach_the_page(self):
        class Progress:  # what the engine emits (A11); the handler keys on the class name
            clip_id, phase, fraction, frame, frames = "c1", "analyze", 0.9, 9, 10
            live = found = None
            stage, zones = "second", {"done": 3, "total": 14}

        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        clip = self.server.session.clip(self.import_clip())
        self.server.session._progress_handler(clip, "analyze")(Progress())
        event = reader.wait_for("clip.progress")
        self.assertEqual((event["stage"], event["zones"]), ("second", {"done": 3, "total": 14}))

    def test_s10_the_analyze_screen_shows_k_of_n_zones_under_the_bar(self):
        page = (UI / "static" / "js" / "views" / "analyze.js").read_text(encoding="utf-8")
        self.assertIn('t("analyze.second_look", { done: zones.done, total: zones.total })', page)
        css = (UI / "static" / "css" / "studio.css").read_text(encoding="utf-8")
        self.assertIn('[data-stage="second"] .box', css)  # the live boxes of the second look have a colour of their own


if __name__ == "__main__":
    unittest.main()


class SubscribeBeforeTheHeaders(server_tests.StudioCase):
    """Review RV-PF2 (anon-medidor, 03/10): the event stream took its starting point after the headers were sent, so a client that
    started a job as soon as the headers arrived could miss job.started."""

    def test_the_stream_is_subscribed_before_the_headers_so_the_first_events_are_not_lost(self):
        original = hubmod.Hub.stream

        def slow_stream(hub, *args, **kwargs):
            time.sleep(
                0.6
            )  # the window a loaded machine opens between the headers and the moment the subscriber is registered
            return original(hub, *args, **kwargs)

        clip_id = self.import_clip()
        with mock.patch.object(hubmod.Hub, "stream", slow_stream):
            reader = server_tests.EventReader(self.port)  # returns when the response headers are in
            self.addCleanup(reader.close)
            self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, expect=202)
            reader.wait_for("job.finished")
        self.wait_idle()
        self.assertEqual(reader.types()[:1], ["job.started"])
