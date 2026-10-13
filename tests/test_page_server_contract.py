"""The page and the server agree: the requests that the page's own JavaScript builds (tests/ui/capture_bodies.mjs) go to
the real server, and each one does what the page meant. A field in the wrong place, which the server used to ignore, is
now a 400, so a test that sends the page's bodies cannot pass by being ignored (review 2, N1)."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import unittest

import numpy as np

try:
    from PIL import Image

    from anonymizer.contract import FACE
    from tests import test_ui_server as server_tests
    from tests.fakes import FixedModule
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

HERE = pathlib.Path(__file__).resolve().parent


def page_requests() -> list:
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node is not installed")
    done = subprocess.run(
        [node, str(HERE / "ui" / "capture_bodies.mjs")], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout)


def swap(value, names: dict):
    """The requests with the placeholders of the script replaced by this test's clip, item and folder."""
    if isinstance(value, str):
        for name, real in names.items():
            value = value.replace(name, real)
        return value
    if isinstance(value, list):
        return [swap(v, names) for v in value]
    if isinstance(value, dict):
        return {k: swap(v, names) for k, v in value.items()}
    return value


class PageBodiesReachTheServer(server_tests.StudioCase):
    modules = [FixedModule(FACE, [(140, 60, 200, 120)], 0.85, 0.75)]

    def setUp(self):
        super().setUp()
        path = os.path.join(self.dir, "media", "f.png")
        rng = np.random.default_rng(1)
        Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)).save(path)
        self.clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.folder = os.path.join(self.dir, "out")

    def send(self, request: dict, expect: int, **names):
        names = {"CLIP": self.clip_id, "DEST": self.folder, **names}
        path = "/" + swap(request["url"], names)
        return self.api(request["method"], path, swap(request["body"], names), expect)

    def test_every_body_the_page_builds_is_accepted_and_does_what_it_says(self):
        requests = page_requests()

        def find(method: str, url: str) -> dict:
            [one] = [r for r in requests if r["method"] == method and r["url"] == url]
            return one

        self.send(find("PUT", "api/settings"), 200)
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        self.send(find("POST", "api/job/analyze"), 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        plan = self.send(find("POST", "api/render/plan"), 200)
        self.assertEqual(plan["destination"], self.folder)
        self.assertEqual(os.path.basename(plan["clips"][0]["output"]), "f_out.png")
        self.assertEqual(plan["clips"][0]["to_check"], 1)
        # Render anyway, as the page sends it: the one Not sure item was never looked at, and the body says yes
        rendered = self.send(find("POST", "api/render"), 202)
        self.assertEqual(os.path.basename(rendered["outputs"][self.clip_id]), "f_out.png")
        reader.wait_for("job.finished")
        self.wait_idle()
        self.assertTrue(os.path.exists(os.path.join(self.folder, "f_out.png")))
        item = self.api("GET", f"/api/clips/{self.clip_id}/analysis")["items"][0]
        answer = self.send(find("PATCH", "api/clips/CLIP/items/1"), 200, **{"items/1": f"items/{item['item_id']}"})
        self.assertEqual(answer["item"]["blur"], True)
        self.send(find("POST", "api/clips/CLIP/areas"), 201)
        other = os.path.join(self.dir, "media", "g.png")
        Image.fromarray(np.random.default_rng(2).integers(0, 255, (240, 320, 3), dtype=np.uint8)).save(other)
        imported = self.send(find("POST", "api/clips"), 200, PATH=other)
        self.assertEqual([c["name"] for c in imported["clips"]], ["g.png"])

    def test_the_screens_build_their_requests_with_the_functions_the_test_runs(self):
        """views/export.js and views/import.js must call the builders that capture_bodies.mjs calls and keep no body of their own."""
        static = HERE.parent / "src" / "anonymizer" / "ui" / "static" / "js" / "views"
        export_js = (static / "export.js").read_text(encoding="utf-8")
        import_js = (static / "import.js").read_text(encoding="utf-8")
        self.assertIn("renderRequest(state, clips)", export_js)
        self.assertIn("api.render(request.ids, request.options)", export_js)
        self.assertNotIn("acknowledge_visible", export_js)
        self.assertIn("analyzeOptions(state.options)", import_js)
        for js in (export_js, import_js):
            self.assertNotIn("json: {", js)

    def test_a_key_the_server_does_not_read_is_a_400_on_every_endpoint_that_takes_a_body(self):
        # the reader comes before the request: a job that ends first leaves no event for a reader that comes after it
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        self.send({"method": "POST", "url": "api/job/analyze", "body": {"clips": ["CLIP"], "options": {}}}, 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        item = self.api("GET", f"/api/clips/{self.clip_id}/analysis")["items"][0]["item_id"]
        for method, path, body in (
            ("PATCH", f"/api/clips/{self.clip_id}/items/{item}", {"blur": True, "extra": 1}),
            (
                "POST",
                f"/api/clips/{self.clip_id}/areas",
                {"frame_from": 0, "frame_to": 0, "box": [10, 10, 40, 40], "extra": 1},
            ),
            ("POST", "/api/clips", {"paths": [os.path.join(self.dir, "media", "f.png")], "extra": 1}),
        ):
            status, _, data = self.request(method, path, body)
            self.assertEqual((status, json.loads(data)["code"]), (400, "error.request.invalid"), (method, path))

    def test_the_acknowledgement_inside_options_is_refused_loudly_not_ignored(self):
        # the reader comes before the request: a job that ends first leaves no event for a reader that comes after it
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        self.send({"method": "POST", "url": "api/job/analyze", "body": {"clips": ["CLIP"], "options": {}}}, 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        for body in (
            {"clips": [self.clip_id], "options": {"destination": self.folder, "acknowledge_visible": True}},
            {"clips": [self.clip_id], "options": {"destination": self.folder}, "extra": 1},
        ):
            status, _, data = self.request("POST", "/api/render", body)
            self.assertEqual((status, json.loads(data)["code"]), (400, "error.request.invalid"))
        self.assertFalse(os.path.exists(self.folder))


class RestartOnTheSamePort(server_tests.StudioCase):
    """N7: a server restarted with a fixed --port gives the new c1 a version the browser has not cached under the old c1."""

    modules = [FixedModule(FACE, [(140, 60, 200, 120)], 0.85, 0.75)]

    def run_once(self):
        path = os.path.join(self.dir, "media", "f.png")
        Image.fromarray(np.random.default_rng(1).integers(0, 255, (240, 320, 3), dtype=np.uint8)).save(path)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        item = self.api("GET", f"/api/clips/{clip_id}/analysis")["items"][0]
        preview = self.get(f"/api/clips/{clip_id}/preview?frame=0&view=anonymized&w=160")
        return clip_id, item["thumb_url"], preview[1].getheader("ETag")

    def test_the_second_run_does_not_repeat_the_thumbnail_address_or_the_etag_of_the_first(self):
        first = self.run_once()
        self.stop_server()
        self.fixed_port = self.port
        self.start_server()
        second = self.run_once()
        self.assertEqual((first[0], second[0]), ("c1", "c1"))
        self.assertNotEqual(first[1], second[1])
        self.assertIsNotNone(first[2])
        self.assertNotEqual(first[2], second[2])


if __name__ == "__main__":
    unittest.main()
