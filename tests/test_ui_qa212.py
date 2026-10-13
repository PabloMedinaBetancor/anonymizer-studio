"""Studio v2.1.2: what the third client QA pass (04/10/2026, qa-anon-3) found at the server. One test class per bug; each fails on
v2.1.1 and passes here."""

from __future__ import annotations

import json
import os
import pathlib
import unittest

try:
    import numpy as np
    from PIL import Image

    from tests import test_ui_server as server_tests
    from tests.fakes import FACE, FixedModule
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


class QaCase(server_tests.StudioCase):
    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def photo(self, name="ok.jpg", size=(160, 120)) -> str:
        path = os.path.join(self.dir, "media", name)
        Image.fromarray((np.random.RandomState(7).rand(size[1], size[0], 3) * 255).astype(np.uint8)).save(path)
        return path

    def reader(self):
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        return reader


class SecondWindowSeesTheChanges(QaCase):
    """Bug 1: a second window (or tab) of one Studio did not see the clips another window imported or removed, nor the decisions
    and the areas it made in Review, until it was reloaded: the server sent no event for any of them."""

    def test_imports_removals_and_decisions_reach_every_open_page(self):
        reader = self.reader()
        path = self.photo()
        clip = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]
        added = reader.wait_for("clips.added")
        self.assertEqual([c["clip_id"] for c in added["clips"]], [clip["clip_id"]])
        self.assertEqual(added["clips"][0]["name"], "ok.jpg")
        self.api("POST", "/api/clips", {"paths": [path]})  # the same file again: nothing new to announce
        self.analyze(clip["clip_id"])
        base = f"/api/clips/{clip['clip_id']}"
        item = self.api("GET", f"{base}/analysis")["items"][0]
        done = self.api("PATCH", f"{base}/items/{item['item_id']}", {"blur": not item["blur"]})
        changed = reader.wait_for("item.changed")
        self.assertEqual(
            (changed["clip_id"], changed["rev"], changed["item"]["blur"]),
            (clip["clip_id"], done["rev"], done["item"]["blur"]),
        )
        area = self.api("POST", f"{base}/areas", {"frame_from": 0, "frame_to": 0, "box": [100, 20, 150, 110]}, 201)
        self.assertEqual(reader.wait_for("area.added")["item"]["item_id"], area["item"]["item_id"])
        self.api("DELETE", f"{base}/areas/{area['item']['item_id']}")
        self.assertEqual(reader.wait_for("area.removed")["item_id"], area["item"]["item_id"])
        self.api("DELETE", base, expect=200)
        self.assertEqual(reader.wait_for("clip.removed")["clip_id"], clip["clip_id"])
        self.assertEqual(reader.types().count("clips.added"), 1)

    def test_a_bulk_decision_reaches_every_open_page(self):
        self.server.session.modules = [FixedModule(FACE, [(10, 10, 60, 60)], 0.85, 0.75)]  # "Not sure"
        clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]
        self.analyze(clip["clip_id"])
        reader = self.reader()
        self.api("POST", f"/api/clips/{clip['clip_id']}/doubtful", {"blur": False})
        self.assertEqual(reader.wait_for("items.changed")["clip_id"], clip["clip_id"])


class TranslationsStayConsistent(unittest.TestCase):
    """Bug 5 (the blind back-translation of the texts of 2.1.1, in all seven languages, each read back into English without its source):
    every text was faithful; what differed was form. The Chinese file addressed the reader with the polite pronoun in thirteen texts and
    the familiar one in five; the Italian file used the curly apostrophe in 31 texts and the straight one in two; the German singular of
    the skipped vehicles said "1 is missing" where the plural says "were left out"."""

    folder = pathlib.Path(__file__).resolve().parents[1] / "src" / "anonymizer" / "ui" / "i18n"

    def texts(self, lang):
        return json.loads((self.folder / f"{lang}.json").read_text(encoding="utf-8"))

    def test_chinese_addresses_the_reader_one_way(self):
        familiar = [k for k, v in self.texts("zh-Hans").items() if "你" in v]
        self.assertEqual(familiar, [])

    def test_italian_uses_the_curly_apostrophe_throughout(self):
        self.assertEqual([k for k, v in self.texts("it").items() if "'" in v], [])

    def test_german_singular_of_the_skipped_vehicles_says_what_the_plural_says(self):
        self.assertIn("ausgelassen", self.texts("de")["warning.vehicles.skipped.one"])


class TheWholeStudioStaysOnThisComputer(unittest.TestCase):
    """The promise of the README is that nothing leaves the computer. The PDF browser has its own guard (test_ui_qa211); this one watches
    the other processes the Studio starts: the server and the engine's worker process, through an import, an analysis, an export and the
    stop, under strace. Every connection and every datagram they make is to this computer."""

    def test_the_server_and_its_worker_reach_only_loopback(self):
        import http.client
        import http.cookiejar
        import json as jsonlib
        import re
        import shutil
        import site
        import subprocess
        import sys
        import tempfile
        import time
        import urllib.request

        if shutil.which("strace") is None:
            self.skipTest("needs strace")
        root = pathlib.Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            trace = os.path.join(tmp, "trace.txt")
            media = os.path.join(tmp, "media")
            os.makedirs(media)
            Image.fromarray((np.random.RandomState(7).rand(120, 160, 3) * 255).astype(np.uint8)).save(
                os.path.join(media, "a.jpg")
            )
            env = dict(os.environ)
            env.update(
                HOME=tmp,
                PYTHONUSERBASE=site.getuserbase(),  # HOME moves, the user site (`pip install --user`) does not
                ANONYMIZER_WORKER_MODULES="tests.fakes:plate_modules",
                PYTHONPATH=os.pathsep.join(filter(None, [str(root / "src"), str(root), os.environ.get("PYTHONPATH")])),
            )
            env.pop("XDG_CONFIG_HOME", None)
            command = ["strace", "-f", "-qq", "-o", trace, "-e", "trace=connect,sendto,sendmsg,execve", sys.executable]
            command += (["-s"] if sys.flags.no_user_site else []) + ["-m", "anonymizer.ui", "--no-browser"]
            server = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=tmp, env=env
            )
            self.addCleanup(lambda: server.poll() is None and server.kill())
            self.addCleanup(server.stdout.close)
            first = server.stdout.readline()
            if "http://" not in first:
                self.skipTest(f"the Studio did not start under strace here: {first!r}")
            url = first.split(": ", 1)[1].strip()
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
            opener.open(url).read()
            base = url.split("/?")[0]

            def call(method, path, body=None):
                request = urllib.request.Request(
                    base + path,
                    data=None if body is None else jsonlib.dumps(body).encode(),
                    method=method,
                    headers={"Content-Type": "application/json"},
                )
                with opener.open(request, timeout=60) as reply:
                    return jsonlib.loads(reply.read() or b"{}")

            def wait():
                end = time.monotonic() + 120
                while call("GET", "/api/state")["job"] is not None and time.monotonic() < end:
                    time.sleep(0.2)

            clip = call("POST", "/api/clips", {"paths": [media]})["clips"][0]["clip_id"]
            call("POST", "/api/job/analyze", {"clips": [clip], "options": {}})
            wait()
            out = os.path.join(tmp, "out")
            call(
                "POST",
                "/api/render",
                {
                    "clips": [clip],
                    "options": {"destination": out, "reports": {"pdf": False, "csv": True, "json": True}},
                },
            )
            wait()
            self.assertTrue(any(name.startswith("a_anonymized") and name.endswith(".jpg") for name in os.listdir(out)))
            try:
                call("POST", "/api/quit")
            except OSError:
                pass
            try:
                server.wait(30)
            except subprocess.TimeoutExpired:
                server.kill()
            with open(trace) as f:
                lines = f.read().splitlines()
        workers = [line for line in lines if "execve(" in line and "anonymizer.ui.worker" in line]
        self.assertTrue(workers, "the trace did not see the worker process start")
        reached = []
        for line in lines:
            if "AF_INET" not in line:
                continue  # AF_UNIX and the like talk inside the computer
            if re.search(r'127\.0\.0\.1|inet_pton\(AF_INET6, "::1"', line):
                continue
            reached.append(line)
        self.assertEqual(reached, [], "a process of the Studio reached out of the computer")


class SettingsAreAnnouncedToEveryWindow(QaCase):
    """Bug 8: the server tells every open page about a change of the settings and whose it was (the page that made it ignores it)."""

    def test_a_change_is_announced_with_the_window_that_made_it(self):
        reader = self.reader()
        request = self.request("PUT", "/api/settings", {"blur": "pixelate"}, headers={"X-Studio-Client": "window-A"})
        self.assertEqual(request[0], 200)
        event = reader.wait_for("settings.changed")
        self.assertEqual((event["settings"]["blur"], event["by"]), ("pixelate", "window-A"))
        before = len(reader.events)
        self.api("PUT", "/api/settings", {"blur": "pixelate"})  # no change: nothing to announce
        self.api("PUT", "/api/settings", {"strength": 1.5})
        later = reader.wait_for("settings.changed", after=before)
        self.assertEqual((later["settings"]["strength"], later["by"]), (1.5, ""))
        self.assertEqual(len([e for e in reader.events if e["type"] == "settings.changed"]), 2)


class AnUnusableSettingsFile(unittest.TestCase):
    """Bug 9: a settings.json that held valid JSON that is not an object (a list, null, a number, a string) made the Studio stop at start with a
    Python traceback, and again at every start until the person found and deleted the file. And one value that was no longer valid (a
    strength out of range, a blur style that no longer exists) made the Studio forget all the others."""

    def load(self, text):
        import tempfile

        from anonymizer.ui import settings

        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "settings.json"), "w", encoding="utf-8") as f:
                f.write(text)
            return settings.Settings(folder, ["en", "es"]).get()

    def test_json_that_is_not_an_object_leaves_the_defaults(self):
        from anonymizer.ui import settings

        for text in ("[]", "null", "1", '"abc"', "true", "[" * 20000):
            self.assertEqual(self.load(text), settings.DEFAULTS, text[:20])

    def test_one_invalid_value_costs_that_setting_and_not_the_others(self):
        values = self.load(
            '{"blur": "pixelate", "strength": 99, "extra_margin": 0.2, "unknown": 1, "language": "xx", "faces": "x", "plates": false}'
        )
        self.assertEqual((values["blur"], values["extra_margin"], values["plates"]), ("pixelate", 0.2, False))
        self.assertEqual(
            (values["strength"], values["language"], values["faces"]), (1.0, None, True)
        )  # an unknown language is none chosen (2.1.6: the page then follows the browser)

    def test_a_file_of_2_1_1_that_chose_deep_falls_back_to_careful_and_keeps_the_rest(self):
        values = self.load('{"detection": "deep", "heavy_only": true, "blur": "solid", "extra_margin": 0.3}')
        self.assertEqual((values["detection"], values["blur"], values["extra_margin"]), ("careful", "solid", 0.3))
        self.assertNotIn("heavy_only", values)

    def test_a_file_that_turns_both_detectors_off_gets_them_back_and_keeps_the_rest(self):
        values = self.load('{"faces": false, "plates": false, "blur": "solid"}')
        self.assertEqual((values["faces"], values["plates"], values["blur"]), (True, True, "solid"))


class TextsThatSayWhatTheStudioDoes(unittest.TestCase):
    """Bug 10: the drop zone and the title of the file picker named videos only (a photo is dropped and imported just the same, and
    photos are what most clients come with), and the English of the Studio spelled the same word two ways: "License plates" in the options,
    "Licence plate" in the PDF report; "Analyze" and "Anonymized" but "colour"."""

    folder = pathlib.Path(__file__).resolve().parents[1] / "src" / "anonymizer" / "ui" / "i18n"

    def texts(self, lang):
        return json.loads((self.folder / f"{lang}.json").read_text(encoding="utf-8"))

    def test_the_places_to_add_files_name_photos_too(self):
        words = {
            "en": "photo",
            "de": "Fo" + "to",
            "es": "fo" + "to",
            "fr": "photo",
            "it": "fo" + "to",
            "pt": "fo" + "to",
            "zh-Hans": "\u7167\u7247",
        }
        for lang, word in words.items():
            for key in ("import.drop", "explorer.title"):
                self.assertIn(word, self.texts(lang)[key], f"{lang} {key}")

    def test_english_is_spelled_one_way(self):
        for key, text in self.texts("en").items():
            for british in ("licence", "colour", "grey", "behaviour", "centre", "analyse"):
                self.assertNotIn(british, text.lower(), key)


class AnOriginalThatWasMovedOrDeleted(QaCase):
    """Bug 11: when the original of an analysed clip was moved, renamed or deleted, every picture the page asks for (thumbnail, previews, the
    picture of an added area) was an "internal error" (500) and the Export screen said "The original file changed after it was analyzed"
    for a file that is not there at all."""

    def test_the_pictures_and_the_plan_say_that_the_file_is_not_there(self):
        path = self.photo("gone.jpg")
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        base = f"/api/clips/{clip_id}"
        area = self.api("POST", f"{base}/areas", {"frame_from": 0, "frame_to": 0, "box": [5, 5, 60, 60]}, 201)["item"][
            "item_id"
        ]
        self.assertEqual(
            self.get(f"{base}/items/{area}/thumb")[0], 200
        )  # made while the file was there, then forgotten again below
        self.server.session.clips[clip_id].area_thumbs.clear()
        os.remove(path)
        for url in (
            f"{base}/thumb",
            f"{base}/preview?frame=0&view=original&w=300",
            f"{base}/preview?frame=0&view=anonymized&w=300",
            f"{base}/items/{area}/thumb",
        ):
            status, _, body = self.get(url)
            answer = json.loads(body)
            self.assertEqual(
                (status, answer["code"], answer["params"]["path"]), (422, "error.source.missing", path), url
            )
        plan = self.api(
            "POST", "/api/render/plan", {"clips": [clip_id], "options": {"destination": os.path.join(self.dir, "out")}}
        )
        self.assertEqual(plan["clips"][0]["blocked"], {"code": "error.source.missing", "params": {"path": path}})
        answer = self.api(
            "POST",
            "/api/render",
            {"clips": [clip_id], "options": {"destination": os.path.join(self.dir, "out")}},
            expect=422,
        )
        self.assertEqual(answer["code"], "error.source.missing")

    def test_a_file_that_changed_is_still_said_to_have_changed(self):
        path = self.photo("changed.jpg")
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        os.utime(path, (1, 1))
        plan = self.api(
            "POST", "/api/render/plan", {"clips": [clip_id], "options": {"destination": os.path.join(self.dir, "out")}}
        )
        self.assertEqual(plan["clips"][0]["blocked"]["code"], "error.source.changed")


class TheInstallTextWorksOnAFreshUbuntu(unittest.TestCase):
    """Bug 12: the README told a fresh machine to start with `python3 -m venv .venv`; on the three machines of the QA pass (Ubuntu 22.04, Python
    3.10, python3-pip installed, python3-venv not, which is how the desktop ships) that stops with "ensurepip is not available" and the README
    had no other way. The way without a venv and without sudo (`pip install --user`) was run from a clean home: it installs, fetches the models,
    and the Studio imports, analyses and exports a photo."""

    def test_the_readme_names_the_package_that_a_venv_needs_and_the_way_without_it(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        for name in ("README.md", "README.template.md"):
            text = (root / name).read_text(encoding="utf-8")
            install = text[text.index("## Install and start") : text.index("## Python API")]
            for needed in ("python3-venv", "--user", "python3 -m anonymizer.ui", "ensurepip"):
                self.assertIn(needed, install, f"{name}: {needed}")


class TheGuideListsEveryColumnOfTheReport(QaCase):
    """Bug 13: the guide described eight columns of the CSV and JSON reports; the files have fourteen."""

    def test_every_column_of_the_csv_is_in_the_guide(self):
        import csv

        clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        self.analyze(clip)
        with open(os.path.join(self.dir, "work", "r.csv"), "wb") as f:
            f.write(self.get(f"/api/clips/{clip}/report?format=csv")[2])
        with open(os.path.join(self.dir, "work", "r.csv"), newline="", encoding="utf-8-sig") as f:
            columns = next(csv.reader(f))
        guide = (pathlib.Path(__file__).resolve().parents[1] / "docs" / "reference" / "files-written.md").read_text(
            encoding="utf-8"
        )
        table = guide
        for column in columns:
            self.assertIn(f"`{column}`", table, column)


class TheTabHasAnIcon(unittest.TestCase):
    """Bug 14: the page had no icon, so the browser asked the server for /favicon.ico (an answer of 404 at every load) and showed its generic
    globe in the tab bar, among a dozen other tabs."""

    def test_the_page_carries_its_own_icon_inside_the_page(self):
        page = (
            pathlib.Path(__file__).resolve().parents[1] / "src" / "anonymizer" / "ui" / "static" / "index.html"
        ).read_text(encoding="utf-8")
        self.assertRegex(page, r'<link rel="icon" href="data:image/svg\+xml,')


class TheStudioDoesNotOfferDeep(QaCase):
    """Deep is not offered in the Studio: it is too slow and too sensitive for publishing work. The Studio offers Fast and
    Careful: no Deep in the choice, the settings, the server's answers, the help, the seven languages, or the guides as a level of the product;
    the Python API keeps `detection="deep"` as an experimental value, and the engine is untouched."""

    root = pathlib.Path(__file__).resolve().parents[1]

    def test_the_settings_and_the_server_do_not_know_deep(self):
        self.assertNotIn("heavy_only", self.api("GET", "/api/settings"))
        for options in ({"detection": "deep"}, {"heavy_only": True}):
            self.assertEqual(self.request("PUT", "/api/settings", options)[0], 400, options)
        state = self.api("GET", "/api/state")
        self.assertNotIn("deep", state["capabilities"])
        self.assertNotIn("deep", state["slowdown"])

    def test_the_page_and_the_texts_have_no_deep(self):
        for path in (self.root / "src" / "anonymizer" / "ui" / "static" / "js").rglob("*.js"):
            if path.name == "mock.js":
                continue
            self.assertNotRegex(
                path.read_text(encoding="utf-8").lower(),
                r"heavy_only|deep_look|detection\.deep|options\.heavy",
                str(path),
            )
        for lang in ("en", "de", "es", "fr", "it", "pt", "zh-Hans"):
            texts = json.loads(
                (self.root / "src" / "anonymizer" / "ui" / "i18n" / f"{lang}.json").read_text(encoding="utf-8")
            )
            self.assertEqual([k for k in texts if "deep" in k and not k.startswith("error.")], [], lang)

    def test_the_guides_do_not_present_deep_as_a_level_of_the_product(self):
        readme = (self.root / "README.md").read_text(encoding="utf-8")
        steps = readme[readme.index("## How it works") : readme.index("## Install and start")]
        self.assertNotIn("Deep", steps)
        self.assertIn("Experimental, not offered: Deep", readme)
        guide = (self.root / "docs" / "concepts" / "how-it-works.md").read_text(encoding="utf-8")
        self.assertIn("is experimental and in the Python API only", guide)
        self.assertIn("the Studio does not offer it", guide)
        self.assertNotIn("Heavy model only", guide)
        for name in ("docs/MEASURED.md",):
            text = (self.root / name).read_text(encoding="utf-8")
            table = text[text.index("| Level |") : text.index("| Paired comparison, all faces")]
            self.assertNotIn("| Deep", table, name)
