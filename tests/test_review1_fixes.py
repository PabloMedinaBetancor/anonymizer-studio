"""PR-12: the findings H2 and H5 to H12 of the first review, one test each (the review's own cases are named T2 and T4 to T8)."""

from __future__ import annotations

import csv
import json
import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import cv2
    import numpy as np
    from PIL import Image

    import anonymizer
    from anonymizer.classes.faces import FaceModule
    from anonymizer.contract import FACE, PLATE
    from anonymizer.ui import engine as engines
    from anonymizer.ui import session as session_module
    from tests import media
    from tests import test_api as api_tests
    from tests import test_ui_server as server_tests
    from tests.fakes import FixedModule
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


def noise(width=160, height=120) -> np.ndarray:
    return (np.random.RandomState(3).rand(height, width, 3) * 255).astype(np.uint8)


class Thumbnails(api_tests.ApiCase):
    def photo(self, name: str, seed: int) -> str:
        path = self.path(name)
        Image.fromarray((np.random.RandomState(seed).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
        return path

    def test_H2_saving_another_analysis_to_the_same_place_leaves_only_its_own_thumbnails(self):
        first_photo, second_photo = self.photo("a.png", 1), self.photo("b.png", 2)
        saved = self.path("clip.analysis.json")
        two = [FixedModule(PLATE, [(10, 10, 60, 40)]), FixedModule(FACE, [(80, 50, 130, 100)], 0.85, 0.75)]
        with anonymizer.Anonymizer(modules=two) as anon:
            first = anon.analyze(first_photo)
            first.save(saved)
        folder = anonymizer.api.thumbnails_folder(saved)
        self.assertEqual(sorted(os.listdir(folder)), ["0.jpg", "1.jpg"])
        with anonymizer.Anonymizer(modules=[FixedModule(PLATE, [(100, 60, 150, 100)])]) as anon:
            second = anon.analyze(second_photo)
            second.save(saved)
        self.assertEqual(sorted(os.listdir(folder)), ["0.jpg"])
        reloaded = anonymizer.load_analysis(saved)
        self.assertEqual(reloaded.thumbnail_jpeg(0), second.thumbnail_jpeg(0))
        self.assertNotEqual(reloaded.thumbnail_jpeg(0), first.thumbnail_jpeg(0))

    def test_H2_saving_an_analysis_that_was_loaded_from_that_folder_keeps_its_thumbnails(self):
        photo = self.photo("a.png", 1)
        saved = self.path("clip.analysis.json")
        with anonymizer.Anonymizer(modules=[FixedModule(PLATE, [(10, 10, 60, 40)])]) as anon:
            anon.analyze(photo).save(saved)
        again = anonymizer.load_analysis(saved)
        before = again.thumbnail_jpeg(0)
        again.save(saved)
        self.assertEqual(anonymizer.load_analysis(saved).thumbnail_jpeg(0), before)


class Areas(api_tests.ApiCase):
    def setUp(self):
        super().setUp()
        self.photo = self.path("f.png")
        Image.fromarray(noise()).save(self.photo)

    def test_T4_an_area_outside_the_picture_or_not_a_number_is_refused_and_one_partly_outside_is_cut(self):
        with anonymizer.Anonymizer(modules=[FixedModule(PLATE, [])]) as anon:
            analysis = anon.analyze(self.photo)
            for box in (
                (5000, 5000, 5100, 5100),
                (-50, -50, -10, -10),
                (float("nan"), 0, 10, 10),
                (0, 0, float("inf"), 10),
            ):
                with self.assertRaises(ValueError, msg=str(box)):
                    analysis.add_area(0, 0, box)
            self.assertEqual(analysis.items, [])
            area = analysis.add_area(0, 0, (-30, 10, 40, 50))
            result = anon.render(analysis, self.path("out.png"))
        out = np.array(Image.open(result.output))
        self.assertLess(out[15:45, 2:36].std(), 0.5 * noise()[15:45, 2:36].std())
        self.assertEqual([(r.item_id, r.blurred) for r in result.report.rows], [(area.item_id, True)])

    def test_H6_following_the_camera_does_not_go_through_a_change_of_shot(self):
        src = self.path("pan.mp4")
        media.panning_clip(src, n=40)
        with anonymizer.Anonymizer(modules=[FixedModule(FACE, [])]) as anon:
            analysis = anon.analyze(src)
            analysis._video.cuts = {20}
            analysis._video.motions[20] = np.array(
                [[1.0, 0.0, 90.0], [0.0, 1.0, 40.0]]
            )  # what the camera "did" across the cut
            area = analysis.add_area(10, 30, (100, 100, 150, 130), follow="camera")
            boxes = analysis.boxes(10, 30)
        mine = {t: [b.box for b in row if b.item_id == area.item_id][0] for t, row in boxes.items()}
        self.assertEqual(mine[19], mine[20])  # the area stays where it was at the cut
        self.assertNotEqual(mine[10], mine[19])  # and followed the camera before it


class ApiBorders(unittest.TestCase):
    def test_H9_a_failing_face_model_run_is_counted_and_not_taken_as_nothing_found(self):
        def fails(img):
            raise cv2.error("model failed")

        module = FaceModule.__new__(FaceModule)
        module.th = SimpleNamespace(input_multiple=32)
        module._lock = threading.Lock()
        module._input_size = None
        module._net = SimpleNamespace(setInputSize=lambda size: None, detect=fails)
        module.model_failures = 0
        rows = module._infer(np.zeros((40, 40, 3), np.uint8))
        self.assertEqual((rows.shape, module.model_failures), ((0, 15), 1))

    def test_H12_a_file_name_that_a_spreadsheet_would_run_is_written_as_text(self):
        rows = [
            anonymizer.ReportRow(name, 0, "plate", "blur", True, "r", 0.0, 1.0, 5, 0.9)
            for name in ('=HYPERLINK("http://x")', "+1+1", "-2", "@SUM(A1)", "plain.mp4")
        ]
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.csv")
            anonymizer.Report(rows).to_csv(path)
            with open(path, encoding="utf-8-sig", newline="") as f:
                cells = [r["clip"] for r in csv.DictReader(f)]
            anonymizer.Report(rows).to_json(os.path.join(d, "report.json"))
            with open(os.path.join(d, "report.json"), encoding="utf-8") as f:
                kept = [r["clip"] for r in json.load(f)["items"]]
        self.assertEqual(cells, ['\'=HYPERLINK("http://x")', "'+1+1", "'-2", "'@SUM(A1)", "plain.mp4"])
        self.assertEqual(kept[0], '=HYPERLINK("http://x")')  # the JSON keeps the real name


class Server(server_tests.StudioCase):
    def test_T2_the_thumbnail_address_names_the_analysis_and_only_such_an_address_is_cached(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        [item] = self.api("GET", f"/api/clips/{clip_id}/analysis")["items"]
        first = item["thumb_url"]
        self.assertIn("?rev=", first)
        status, response, _ = self.get(first)
        self.assertEqual((status, "max-age" in response.getheader("Cache-Control")), (200, True))
        bare = first.split("?")[0]
        self.assertEqual(self.get(bare)[1].getheader("Cache-Control"), "no-cache")
        self.api("POST", "/api/job/analyze", {"clips": [clip_id], "options": {"gap_fill": "hold"}}, 202)
        self.wait_idle()
        [again] = self.api("GET", f"/api/clips/{clip_id}/analysis")["items"]
        self.assertNotEqual(again["thumb_url"], first)

    def test_T5_a_preview_is_revalidated_and_a_change_of_setting_gives_a_new_picture(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        url = f"/api/clips/{clip_id}/preview?frame=30&view=anonymized&w=320&rev=1"
        status, response, before = self.get(url)
        tag = response.getheader("ETag")
        self.assertEqual((status, response.getheader("Cache-Control")), (200, "private, no-cache"))
        self.assertEqual(self.get(url, headers={"If-None-Match": tag})[0], 304)
        self.api("PUT", "/api/settings", {"blur": "solid"})
        status, response, after = self.get(url, headers={"If-None-Match": tag})
        self.assertEqual(status, 200)
        self.assertNotEqual((response.getheader("ETag"), after), (tag, before))
        self.assertEqual(self.get(url, headers={"If-None-Match": response.getheader("ETag")})[0], 304)

    def test_T5_a_render_takes_the_look_of_the_blur_from_the_request_and_not_from_what_was_last_saved(self):
        settings = self.server.session._render_settings(
            {"blur": "solid", "strength": 1.5, "extra_margin": 0.2, "gap_fill": "hold"}
        )
        o = settings["options"]
        self.assertEqual((o.blur, o.strength, o.extra_margin, o.gap_fill), ("solid", 1.5, 0.2, "hold"))
        self.assertEqual(self.api("GET", "/api/settings")["blur"], "gaussian")  # nothing was saved by asking

    def test_T8_a_folder_import_lists_what_it_did_not_recognise_and_a_file_that_breaks_the_probe_costs_only_itself(
        self,
    ):
        self.make_clip("good.mp4")
        for name in ("a.vob", "b.3g2", "notes.txt", ".hidden"):
            self.write(f"media/{name}", "not media")
        self.make_clip("bad.mp4")
        real = anonymizer.media_info

        def probe(path):
            if os.path.basename(path) == "bad.mp4":
                raise RuntimeError("a library failed on this one")
            return real(path)

        with mock.patch("anonymizer.media_info", probe):
            answer = self.api("POST", "/api/clips", {"paths": [os.path.join(self.dir, "media")]})
        self.assertEqual([c["name"] for c in answer["clips"]], ["good.mp4"])
        by_code = {r["code"]: r for r in answer["rejected"]}
        self.assertEqual(
            set(by_code), {"error.media.unreadable_video", "warning.import.ignored", "warning.import.hidden"}
        )
        self.assertEqual(
            by_code["warning.import.hidden"]["params"], {"count": 1}
        )  # N8: the dot file is said, not skipped in silence
        self.assertEqual(by_code["warning.import.ignored"]["params"], {"count": 3, "extensions": ".3g2, .txt, .vob"})
        self.assertEqual(os.path.basename(by_code["error.media.unreadable_video"]["path"]), "bad.mp4")

    def test_H11_a_render_whose_process_died_leaves_no_hidden_partial_file_behind(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        partial = os.path.join(folder, ".clip_anonymized.mp4.anonymizer-partial")

        def die(self_, clip, analysis, target, on_event, token):
            with open(
                os.path.join(os.path.dirname(target), f".{os.path.basename(target)}.anonymizer-partial"), "wb"
            ) as f:
                f.write(b"frames so far")
            raise engines.EngineFailure("error.engine.crashed", {}, "exit code -9")

        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        with mock.patch.object(engines.LocalEngine, "render", die):
            self.api("POST", "/api/render", {"clips": [clip_id], "options": {"destination": folder}}, 202)
            reader.wait_for("job.finished")
            self.wait_idle()
        self.assertEqual(os.listdir(folder), [])
        self.assertFalse(os.path.exists(partial))

    def test_H12_a_download_name_cannot_split_the_header(self):
        path = self.write("media/x.mp4", "bytes")
        self.server.session.downloads["f1"] = (path, 'a"b\r\nSet-Cookie: injected=1; \u00e9.mp4')
        status, response, _ = self.get("/api/downloads/f1/x.mp4")
        header = response.getheader("Content-Disposition")
        self.assertEqual(status, 200)
        self.assertNotIn("\n", header)
        self.assertNotIn("\r", header)
        self.assertNotIn('b"', header.split("filename*")[0].replace('filename="', "", 1)[:-1] + "x")
        self.assertIn("filename*=UTF-8''", header)
        self.assertIsNone(response.getheader("Set-Cookie"))

    def test_H12_a_name_pattern_cannot_reach_into_attributes_and_both_classes_off_is_an_error_not_a_crash(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        for pattern in ("{name.__class__}", "{name[0]}", "{index.real}", "{name!r}", "{name:>99}"):
            status, _, data = self.request(
                "POST", "/api/render/plan", {"clips": [clip_id], "options": {"name_pattern": pattern}}
            )
            self.assertEqual((status, json.loads(data)["code"]), (400, "error.request.invalid"), pattern)
        for pattern in ("{name}_x", "{name}-{index}", "{name}-{index:03d}"):
            self.api("POST", "/api/render/plan", {"clips": [clip_id], "options": {"name_pattern": pattern}})
        status, _, data = self.request("PUT", "/api/settings", {"faces": False, "plates": False})
        self.assertEqual((status, json.loads(data)["code"]), (400, "error.request.invalid"))
        self.assertTrue(self.api("GET", "/api/settings")["faces"])
        self.api("PUT", "/api/settings", {"faces": False})
        status, _, data = self.request("PUT", "/api/settings", {"plates": False})
        self.assertEqual(status, 400)
        self.assertEqual(
            self.request(
                "POST", "/api/job/analyze", {"clips": [clip_id], "options": {"faces": False, "plates": False}}
            )[0],
            400,
        )

    def test_T4_the_server_refuses_an_area_outside_the_source(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        base = f"/api/clips/{clip_id}"
        for box in ([5000, 5000, 5100, 5100], [-50, -50, -10, -10]):
            status, _, data = self.request("POST", f"{base}/areas", {"frame_from": 0, "frame_to": 5, "box": box})
            self.assertEqual((status, json.loads(data)["code"]), (400, "error.request.invalid"), box)
        self.assertEqual(self.api("GET", f"{base}/analysis")["items"][0]["kind"], "plate")
        self.assertEqual(len(self.api("GET", f"{base}/analysis")["items"]), 1)

    def test_pattern_error_allows_the_documented_fields_only(self):
        self.assertIsNone(session_module.pattern_error("{name}_anonymized"))
        self.assertIsNone(session_module.pattern_error("clip-{index:03d}"))
        self.assertIsNotNone(session_module.pattern_error("{nope}"))
        self.assertIsNotNone(session_module.pattern_error("{name"))


class CheckItems(api_tests.ApiCase):
    def photo(self) -> str:
        path = self.path("f.png")
        Image.fromarray(noise()).save(path)
        return path

    def test_T3_items_the_detectors_doubted_that_stay_visible_are_a_warning_in_the_result_and_in_the_events(self):
        doubted = [FixedModule(FACE, [(80, 50, 130, 100)], 0.85, 0.75), FixedModule(PLATE, [(10, 10, 60, 40)])]
        with anonymizer.Anonymizer(anonymizer.Options(unsure="keep"), modules=doubted) as anon:  # v1's rule
            analysis = anon.analyze(self.photo())
            result = anon.render(analysis, self.path("out.png"), self.events.append)
        self.assertIn("warning.review.check_left_visible", result.warnings)
        self.assertEqual(result.visible_check, 1)  # the count that goes with the code
        [warning] = [e for e in self.events if isinstance(e, anonymizer.JobWarning) and "check_left" in e.code]
        self.assertEqual((warning.params, anonymizer.event_json(warning)["type"]), ({"count": 1}, "job.warning"))
        with anonymizer.Anonymizer(
            anonymizer.Options(unsure="keep"), modules=doubted
        ) as anon:  # decided: on, no warning
            analysis = anon.analyze(self.photo())
            [face] = [i for i in analysis.items if i.kind == FACE]
            analysis.set_blur(face.item_id, True)
            again = anon.render(analysis, self.path("again.png"))
        self.assertNotIn("warning.review.check_left_visible", again.warnings)
        self.assertEqual(again.visible_check, 0)

    def test_T3_the_one_call_functions_warn_too(self):
        doubted = [FixedModule(FACE, [(80, 50, 130, 100)], 0.85, 0.75)]
        with mock.patch("anonymizer.classes.load", lambda kinds, threads, models, **kw: doubted):
            result = anonymizer.anonymize_image(self.photo(), self.path("out.png"), anonymizer.Options(unsure="keep"))
        self.assertEqual(result.warnings, ["warning.review.check_left_visible"])


class ServerCheck(server_tests.StudioCase):
    modules = [FixedModule(FACE, [(140, 60, 200, 120)], 0.85, 0.75, placed=True)]

    def photo(self) -> str:
        path = os.path.join(self.dir, "media", "f.png")
        Image.fromarray(noise(240, 160)).save(path)
        return self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]

    def render(self, clip_id: str, folder: str, options: dict | None = None, **extra):
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        body = {"clips": [clip_id], "options": {"destination": folder, **(options or {})}, **extra}
        self.api("POST", "/api/render", body, 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        return reader

    def test_T3_the_server_exports_not_sure_items_nobody_looked_at_blurred_without_asking(self):
        clip_id = self.photo()
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        plan = self.api("POST", "/api/render/plan", {"clips": [clip_id], "options": {"destination": folder}})
        self.assertEqual(plan["clips"][0]["to_check"], 1)
        status, _, data = self.request(
            "POST", "/api/render", {"clips": [clip_id], "options": {"destination": folder}, "acknowledge_visible": True}
        )
        self.assertEqual((status, json.loads(data)["code"]), (400, "error.request.invalid"))
        self.assertFalse(os.path.exists(folder))
        reader = self.render(clip_id, folder)
        # privacy first: the doubtful face was blurred, so nothing is left visible and there is no warning
        self.assertFalse([e for e in reader.events if e["type"] == "job.warning" and "check_left" in e["code"]])
        face = self.api("GET", f"/api/clips/{clip_id}/analysis")["items"][0]
        self.assertTrue(face["blur"])
        self.api(
            "PATCH", f"/api/clips/{clip_id}/items/{face['item_id']}", {"blur": False}
        )  # looked at, turned off by the person: its decision, and the one warning
        again = self.render(clip_id, os.path.join(self.dir, "again"))
        warning = next(e for e in again.events if e["type"] == "job.warning" and "check_left" in e["code"])
        self.assertEqual((warning["code"], warning["params"]), ("warning.review.check_left_visible", {"count": 1}))

    def test_T6_a_comparison_video_is_announced_as_showing_the_original_and_the_reports_say_so(self):
        self.stop_server()
        self.modules = None
        self.start_server()
        clip_id = self.import_clip()
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        reports = {"csv": True, "json": True, "pdf": False}
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        options = {"destination": folder, "comparison": True, "reports": reports}
        self.api("POST", "/api/render", {"clips": [clip_id], "options": options}, 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        warning = next(e for e in reader.events if e["type"] == "job.warning" and "contains_original" in e["code"])
        self.assertEqual(warning["params"], {"file": "clip_anonymized.compare.mp4"})
        rendered = next(e for e in reader.events if e["type"] == "clip.rendered")
        with open(rendered["report"]["json"], encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(
            data["notes"], [{"code": "report.note.comparison_video", "params": {"file": "clip_anonymized.compare.mp4"}}]
        )
        self.assertEqual(data["items"][0]["kind"], "plate")
        plain = self.render(clip_id, os.path.join(self.dir, "plain"), {"reports": reports})
        self.assertFalse([e for e in plain.events if e["type"] == "job.warning" and "contains_original" in e["code"]])
        with open(
            next(e for e in plain.events if e["type"] == "clip.rendered")["report"]["json"], encoding="utf-8"
        ) as f:
            self.assertNotIn("notes", json.load(f))

    def test_T6_the_pdf_of_a_clip_with_a_comparison_carries_the_note(self):
        from datetime import date

        from anonymizer.ui import pdf

        rows = [
            anonymizer.ReportRow(
                "c.mp4", 0, "plate", "blur", True, "plate.reason.on_vehicle_verified", 0.0, 1.0, 5, 0.9
            )
        ]
        texts = {"report.note.comparison_video": "Comparison with the original ({file}). Do not publish."}
        page = pdf.report_html("c.mp4", rows, texts, True, date(2026, 10, 3), "c.compare.mp4")
        self.assertIn("Comparison with the original (c.compare.mp4). Do not publish.", page)
        self.assertNotIn("Do not publish", pdf.report_html("c.mp4", rows, texts, True, date(2026, 10, 3)))


if __name__ == "__main__":
    unittest.main()
