"""Studio v2.1.1: what the second client QA pass (04/10/2026, qa-anon) found at the server. One test class per bug; each fails on
v2.1 and passes here."""

from __future__ import annotations

import json
import os
import pathlib
import time
import unittest

try:
    import numpy as np
    from PIL import Image

    import anonymizer
    from anonymizer.ui import session as session_module
    from tests import test_ui_server as server_tests
    from tests.fakes import FACE, PLATE, FixedModule, doubtful_plate_module
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


class QaCase(server_tests.StudioCase):
    modules = [FixedModule("face", [])]

    def photo(self, name="ok.jpg", size=(160, 120)) -> str:
        path = os.path.join(self.dir, "media", name)
        Image.fromarray((np.random.RandomState(7).rand(size[1], size[0], 3) * 255).astype(np.uint8)).save(path)
        return path


class EmptyPathIsNotTheWorkingDirectory(QaCase):
    """Bug 1: the Import dialog's button, pressed on the list of places with nothing ticked, sent the empty path, and the server
    read it as its own working directory and imported every photo and video below it."""

    def test_an_empty_or_blank_path_is_refused(self):
        self.photo()
        old = os.getcwd()
        os.chdir(os.path.join(self.dir, "media"))  # a folder with a photo in it: what the empty path used to import
        self.addCleanup(os.chdir, old)
        for paths in ([""], ["   "], [self.photo(), ""]):
            answer = self.api("POST", "/api/clips", {"paths": paths}, expect=400)
            self.assertEqual(answer["code"], "error.request.invalid")
            self.assertEqual(answer["params"], {"field": "paths"})
        self.assertEqual(self.api("GET", "/api/state")["clips"], [])


class AreasDoNotOutliveTheirAnalysis(QaCase):
    """Bug 2: the areas a person drew are kept in a table beside the analysis. A new analysis of the clip (other settings) replaced
    the analysis and kept the table, so an item the engine numbered like a dropped area showed that area's picture."""

    face = FixedModule(FACE, [(10, 10, 60, 60)])
    plate = FixedModule(PLATE, [(80, 70, 140, 100)])
    modules = [face]

    def analyze_with(self, clip_id, options):
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        body = {
            "clips": [clip_id],
            "options": options,
            "discard": True,
        }  # the area drawn on the old analysis would otherwise be asked about
        self.api("POST", "/api/job/analyze", body, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()

    def test_a_new_analysis_drops_the_pictures_of_the_old_areas(self):
        path = self.photo("scene.jpg", size=(160, 120))
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze_with(clip_id, {})
        base = f"/api/clips/{clip_id}"
        self.assertEqual([i["item_id"] for i in self.api("GET", f"{base}/analysis")["items"]], [0])
        area = self.api("POST", f"{base}/areas", {"frame_from": 0, "frame_to": 0, "box": [100, 20, 150, 110]}, 201)
        self.assertEqual(area["item"]["item_id"], 1)
        self.server.session.modules = [self.face, self.plate]
        self.analyze_with(
            clip_id, {"gap_fill": "hold"}
        )  # other settings: a new analysis; the plate is now item 1, as the area was
        items = self.api("GET", f"{base}/analysis")["items"]
        self.assertEqual([(i["item_id"], i["kind"]) for i in items], [(0, "face"), (1, "plate")])
        status, _, picture = self.get(f"{base}/items/1/thumb")
        self.assertEqual(status, 200)
        clip = self.server.session.clips[clip_id]
        self.assertEqual(
            picture, clip.analysis.thumbnail_jpeg(1), "the plate shows the picture of the area that was dropped"
        )
        self.assertEqual(clip.area_boxes, {})


class ModelErrorSaysWhatToDo(unittest.TestCase):
    """Bug 4: a fresh install has no models, and the page said only "A detection model could not be loaded (face)": the folders and
    the way to get the files were in the error's detail, which the page does not show."""

    def test_the_text_in_every_language_names_the_script_and_the_variable_and_keeps_its_placeholder(self):
        folder = pathlib.Path(__file__).resolve().parents[1] / "src" / "anonymizer" / "ui" / "i18n"
        for path in sorted(folder.glob("*.json")):
            text = json.loads(path.read_text(encoding="utf-8"))["error.model.load"]
            for needle in ("{kind}", "scripts/fetch_models.py", "ANONYMIZER_MODELS"):
                self.assertIn(needle, text, f"{path.name}: {text}")


class TranslationsStayInTheirVariety(unittest.TestCase):
    """Bug 5 (the blind back-translation of the v2.1 texts): the Portuguese file is Brazilian (i18n/NOTES.md) but two plate texts
    were European (the Iberian words for plate and for default), and the French file, with curly apostrophes everywhere, had straight ones in six."""

    folder = pathlib.Path(__file__).resolve().parents[1] / "src" / "anonymizer" / "ui" / "i18n"

    def texts(self, lang):
        return json.loads((self.folder / f"{lang}.json").read_text(encoding="utf-8"))

    def test_portuguese_has_no_european_words(self):
        european = (
            "matr\u00edcula",
            "predefini",
            "fichei" + "ro",
            "ecr\u00e3",
            "utiliza" + "dor",
            "fotogra" + "ma",
            "rato ",
        )
        for key, text in self.texts("pt").items():
            for word in european:
                self.assertNotIn(word, text.lower(), f"pt {key}: {text}")

    def test_french_uses_the_curly_apostrophe_throughout(self):
        for key, text in self.texts("fr").items():
            self.assertNotIn("'", text, f"fr {key}: {text}")


class FastDoesNotBlameASecondLook(QaCase):
    """Bug 12: in Fast nothing is looked at a second time, yet the notice for what the budget left out said "The second look ... left 3
    places unchecked" (16 of 21 street photos in the QA run). What Fast counts there is the vehicles beyond the plate search's budget."""

    def codes_for(self, detection):
        clip_id = self.api("POST", "/api/clips", {"paths": [self.photo(f"{detection}.jpg")]})["clips"][0]["clip_id"]
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        session = self.server.session
        warning = anonymizer.JobWarning(clip_id, "warning.second_look.zones_skipped", {"count": 3})
        session._progress_handler(session.clips[clip_id], "analyze", detection)(warning)
        event = reader.wait_for("job.warning")
        self.assertEqual(event["params"], {"count": 3})
        return event["code"]

    def test_fast_names_the_vehicles_and_the_other_levels_keep_the_second_look(self):
        self.assertEqual(self.codes_for("fast"), "warning.vehicles.skipped")
        self.assertEqual(self.codes_for("careful"), "warning.second_look.zones_skipped")


class AnalyzingAgainAsksBeforeItDiscardsADecision(server_tests.StudioCase):
    """Bug 13: other settings and Analyze again replaced the analysis of a clip the person had already reviewed, with every decision and
    every area they had added, without a word. The server now answers 409 until the request says `discard`."""

    modules = [doubtful_plate_module()]

    def reviewed_clip(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        self.api("PATCH", f"/api/clips/{clip_id}/items/0", {"blur": False})
        return clip_id

    def start(self, clip_id, body, expect):
        return self.api("POST", "/api/job/analyze", {"clips": [clip_id], **body}, expect)

    def test_other_settings_ask_first_and_the_decisions_stay_until_the_person_agrees(self):
        clip_id = self.reviewed_clip()
        refused = self.start(clip_id, {"options": {"gap_fill": "hold"}}, 409)
        self.assertEqual(
            (refused["code"], refused["params"]), ("error.analysis.discards", {"count": 1, "decisions": 1})
        )
        self.assertEqual(self.api("GET", "/api/state")["job"], None)
        self.assertFalse(self.api("GET", f"/api/clips/{clip_id}/analysis")["items"][0]["blur"], "the decision was lost")
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        self.start(clip_id, {"options": {"gap_fill": "hold"}, "discard": True}, 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.assertTrue(
            self.api("GET", f"/api/clips/{clip_id}/analysis")["items"][0]["blur"],
            "a new analysis starts from the engine's verdict",
        )

    def test_the_same_settings_or_a_clip_with_no_decision_never_ask(self):
        clip_id = self.reviewed_clip()
        self.start(clip_id, {}, 202)  # same settings: the analysis is reused, with its decision
        self.wait_idle()
        fresh = self.import_clip("fresh.mp4")
        self.analyze(fresh)
        self.start(fresh, {"options": {"gap_fill": "hold"}}, 202)  # nothing reviewed on it
        self.wait_idle()

    def test_discard_must_be_a_boolean(self):
        clip_id = self.reviewed_clip()
        self.assertEqual(self.start(clip_id, {"discard": "yes"}, 400)["params"], {"field": "discard"})


class UploadRefusalsKeepTheServersFoldersToThemselves(server_tests.StudioCase):
    """Bug 14: the refusal of an uploaded file (a corrupt photo, a GIF) carried the path inside the server's work folder, in `path` and in
    the `detail` of the reason: the hosted demo told its visitors where its files are."""

    demo = True

    def refusal(self, name, data):
        status, response, body = self.request(
            "PUT", f"/api/upload?name={name}", raw=data, headers={"Content-Type": "application/octet-stream"}
        )
        self.assertEqual(status, 200, body[:200])
        return body.decode("utf-8"), json.loads(body)["rejected"][0]

    def test_a_corrupt_photo_and_a_gif_are_refused_by_name(self):
        work = self.server.session.work_dir
        for name, data in (
            ("corrupt.jpg", b"\xff\xd8\xff\xe0 not really a photo"),
            ("anim.gif", b"GIF89a" + b"\0" * 40),
        ):
            text, rejected = self.refusal(name, data)
            self.assertNotIn(work, text, name)
            self.assertEqual(rejected["path"], name)
            self.assertEqual(rejected["params"]["name"], name)
        self.assertEqual(os.listdir(os.path.join(work, "uploads")), [], "a refused upload was kept")


class SettingsInAFolderThatCannotBeWritten(server_tests.StudioCase):
    """Bug 17: with a read-only home (or a locked profile) every change of an option answered 500 "internal error" and the page showed
    that notice each time, although the change itself had been applied."""

    def test_the_change_applies_and_the_person_is_told_once_that_it_is_not_kept(self):
        blocker = os.path.join(self.dir, "a-file-where-the-folder-should-be")
        with open(blocker, "w") as f:
            f.write("x")
        self.server.session.settings.folder = os.path.join(blocker, "settings")  # cannot be made: its parent is a file
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        self.assertEqual(self.api("PUT", "/api/settings", {"blur": "solid"})["blur"], "solid")
        self.assertEqual(self.api("PUT", "/api/settings", {"strength": 1.5})["strength"], 1.5)
        self.assertEqual(self.api("GET", "/api/settings")["blur"], "solid")
        reader.wait_for("job.warning")
        time.sleep(0.2)
        warnings = [e["code"] for e in reader.events if e["type"] == "job.warning"]
        self.assertEqual(warnings, ["warning.settings.not_saved"])


class StartingTwice(unittest.TestCase):
    """Bug 19: starting the Studio on a port that is taken (a second launch of the same shortcut) ended in a Python traceback, and so did
    a port that is not a port."""

    def run_main(self, argv):
        import contextlib
        import io

        from anonymizer.ui import server as server_module

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            try:
                code = server_module.main(argv)
            except SystemExit as e:  # argparse
                code = e.code
        return code, err.getvalue()

    def test_a_taken_port_is_said_in_a_sentence_and_a_wrong_port_is_a_usage_error(self):
        import socket

        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen(1)
            code, text = self.run_main(["--no-browser", "--port", str(taken.getsockname()[1])])
        self.assertEqual(code, 2)
        self.assertIn("could not start on port", text)
        self.assertIn("already running", text)
        self.assertNotIn("Traceback", text)
        code, text = self.run_main(["--no-browser", "--port", "99999"])
        self.assertEqual(code, 2)
        self.assertIn("0 to 65535", text)
        self.assertNotIn("Traceback", text)


class OutputNamesThatCannotBeWritten(QaCase):
    """Bug 20: a source whose name is close to the 255 bytes of a file system (240 characters in the QA run) made the export end in
    "The engine stopped unexpectedly", because the render's hidden working name and the output name did not fit."""

    def clip_named(self, length):
        path = self.photo("N" * (length - 4) + ".jpg")
        return self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]

    def test_a_name_that_cannot_be_written_is_blocked_in_the_plan_and_refused_before_the_engine_starts(self):
        limit = session_module.MAX_OUTPUT_NAME - len(
            "_anonymized"
        )  # the longest source name (with its extension) that still fits
        for length, blocked in ((limit, False), (limit + 1, True)):
            clip_id = self.clip_named(length)
            self.analyze(clip_id)
            entry = self.api("POST", "/api/render/plan", {"clips": [clip_id], "options": {}})["clips"][0]
            if blocked:
                self.assertEqual(entry["blocked"]["code"], "error.output.name_too_long")
                self.assertEqual(entry["blocked"]["params"], {"limit": session_module.MAX_OUTPUT_NAME})
                refused = self.api("POST", "/api/render", {"clips": [clip_id], "options": {}}, expect=422)
                self.assertEqual(refused["code"], "error.output.name_too_long")
                self.assertIsNone(self.api("GET", "/api/state")["job"])
            else:
                self.assertIsNone(entry["blocked"])


class ControlCharactersAndNegativeLengths(QaCase):
    """Bug 21: a NUL in an uploaded file's name was a 500, and in a name pattern or a destination it reached the engine and ended in "The
    engine stopped unexpectedly"; a negative Content-Length made the server read until the client hung up."""

    def test_a_nul_is_dropped_from_an_upload_name_and_refused_in_a_pattern_or_destination(self):
        with open(self.photo("src.jpg"), "rb") as f:
            data = f.read()
        status, _, body = self.request(
            "PUT", "/api/upload?name=a%00b.jpg", raw=data, headers={"Content-Type": "application/octet-stream"}
        )
        self.assertEqual(status, 200, body[:200])
        self.assertEqual(json.loads(body)["clips"][0]["name"], "ab.jpg")
        clip_id = json.loads(body)["clips"][0]["clip_id"]
        self.analyze(clip_id)
        for options, field in (
            ({"name_pattern": "x\x00y"}, "name_pattern"),
            ({"destination": "/tmp/a\x00b"}, "destination"),
        ):
            for path in ("/api/render/plan", "/api/render"):
                answer = self.api("POST", path, {"clips": [clip_id], "options": options}, expect=400)
                self.assertEqual(answer["params"], {"field": field})

    def test_a_negative_content_length_is_a_bad_request(self):
        import socket

        for target in ("/api/upload?name=x.jpg", "/api/clips"):
            with socket.create_connection(("127.0.0.1", self.port), timeout=10) as sock:
                sock.sendall(
                    (
                        f"{'PUT' if 'upload' in target else 'POST'} {target} HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n"
                        f"Cookie: anon_session_{self.port}={server_tests.TOKEN}\r\nContent-Length: -5\r\n\r\nabc"
                    ).encode()
                )
                self.assertIn(b" 400 ", sock.recv(300).split(b"\r\n")[0] + b" ")


class WhyARequestIsRefused(QaCase):
    """Bug 24: a folder the person may not read and a page opened from an old bookmark were both "This request is not allowed"."""

    def test_an_unreadable_folder_says_so_and_the_bookmark_hint_is_in_the_other_message(self):
        if os.geteuid() == 0:
            self.skipTest("root reads everything")
        locked = os.path.join(self.dir, "media", "locked")
        os.makedirs(locked)
        os.chmod(locked, 0)
        self.addCleanup(os.chmod, locked, 0o755)
        status, _, body = self.request("GET", "/api/browse?path=" + locked)
        self.assertEqual((status, json.loads(body)["code"]), (403, "error.browse.denied"))
        status, _, body = self.request("GET", "/api/state", cookie=False)
        self.assertEqual((status, json.loads(body)["code"]), (403, "error.request.forbidden"))
        folder = pathlib.Path(__file__).resolve().parents[1] / "src" / "anonymizer" / "ui" / "i18n"
        for path in sorted(folder.glob("*.json")):
            texts = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("error.browse.denied", texts, path.name)
        english = json.loads((folder / "en.json").read_text(encoding="utf-8"))
        self.assertIn("bookmark", english["error.request.forbidden"])


class ThePdfBrowserStaysOffline(unittest.TestCase):
    """Bug 27: the Chrome that prints the PDF report was started with a fresh profile and nothing else, and a fresh Chrome talks to the
    network: in the QA run it looked up and connected to Google's servers on 443 and 80, while the README says nothing leaves the computer."""

    def test_the_command_cuts_the_network(self):
        from unittest import mock

        from anonymizer.ui import pdf

        commands = []

        def run(command, **kwargs):
            commands.append(command)
            raise RuntimeError("the test stops here")

        with mock.patch.object(pdf.subprocess, "run", side_effect=run):
            with self.assertRaises(RuntimeError):
                pdf.html_to_pdf("/usr/bin/browser", "/tmp/page.html", "/tmp/out.pdf")
        for flag in pdf.OFFLINE_FLAGS:
            self.assertIn(flag, commands[0])
        self.assertIn("--proxy-server=socks5://127.0.0.1:1", commands[0])

    def test_a_real_print_makes_no_network_connection(self):
        import re
        import shutil
        import stat
        import tempfile
        from unittest import mock

        from anonymizer.ui import pdf

        browser = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
        if browser is None or shutil.which("strace") is None:
            self.skipTest("needs a Chromium-based browser and strace")
        with tempfile.TemporaryDirectory() as tmp:
            trace = os.path.join(tmp, "connects.txt")
            wrapper = os.path.join(tmp, "traced-browser")
            with open(wrapper, "w") as f:
                f.write(f'#!/bin/sh\nexec strace -f -qq -e trace=connect -o {trace} {browser} --no-sandbox "$@"\n')
            os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IXUSR)
            page = os.path.join(tmp, "page.html")
            with open(page, "w") as f:
                f.write("<html><body><h1>report</h1></body></html>")
            # strace -f slows Chrome on a busy small machine; the product's own bound (90 s) stays what it is
            with mock.patch.object(pdf, "TIMEOUT_SECONDS", 300):
                pdf.html_to_pdf(wrapper, page, os.path.join(tmp, "out.pdf"))
            with open(trace) as f:
                lines = f.read().splitlines()
        if not any("connect(" in line for line in lines):
            self.skipTest("strace could not trace the browser here (no connect call was recorded at all)")
        reached = []
        for line in lines:
            if "AF_INET" not in line:
                continue  # AF_UNIX: the browser's own processes talking to each other
            if re.search(r'127\.0\.0\.1|inet_pton\(AF_INET6, "::1"', line):
                continue  # this computer: the dead proxy that every request is sent to
            if "2001:4860:4860::8888" in line:
                continue  # a UDP connect that asks the kernel for an IPv6 route; nothing is sent
            reached.append(line)
        self.assertEqual(reached, [], "the PDF browser reached out of the computer")


if __name__ == "__main__":
    unittest.main()
