"""Studio v2.1.6: what the seventh client QA pass (05/10/2026, qa-anon-7) found at the server. One test class per bug; each fails on
v2.1.5 and passes here."""

from __future__ import annotations

import json
import os
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

    def photo(self, name="ok.jpg", size=(160, 120), folder="media", seed=7) -> str:
        os.makedirs(os.path.join(self.dir, folder), exist_ok=True)
        path = os.path.join(self.dir, folder, name)
        Image.fromarray((np.random.RandomState(seed).rand(size[1], size[0], 3) * 255).astype(np.uint8)).save(path)
        return path


class LanguageIsNoneUntilChosen(QaCase):
    """Bug 1: the setting always held "en", so the page could not tell "never chosen" (follow the browser) from "English chosen"."""

    def test_nobody_has_chosen_and_then_english_is_chosen(self):
        self.assertIsNone(self.api("GET", "/api/settings")["language"])
        self.assertEqual(self.api("PUT", "/api/settings", {"language": "en"})["language"], "en")
        self.assertEqual(self.api("GET", "/api/settings")["language"], "en")
        with open(os.path.join(self.dir, "settings", "settings.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["language"], "en")

    def test_the_choice_is_kept_by_the_next_start_and_none_stays_none(self):
        self.api("PUT", "/api/settings", {"blur": "pixelate"})  # a settings file with no language chosen
        with open(os.path.join(self.dir, "settings", "settings.json"), encoding="utf-8") as f:
            self.assertIsNone(json.load(f)["language"])
        self.stop_server()
        self.start_server()
        self.assertIsNone(self.api("GET", "/api/settings")["language"])
        self.api("PUT", "/api/settings", {"language": "es"})
        self.stop_server()
        self.start_server()
        self.assertEqual(self.api("GET", "/api/settings")["language"], "es")

    def test_the_pdf_is_written_in_the_language_of_the_page(self):
        """No language chosen: the page's own language (what it follows from the browser) goes with the export; a chosen one is the
        setting's when the export names none; a language the server does not have is refused."""
        clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        self.analyze(clip)
        seen = []
        session = self.server.session
        session.texts_for = lambda language: (seen.append(language), {})[1]
        self.browser = None
        bad = self.api("POST", "/api/render/plan", {"clips": [clip], "options": {"language": "xx"}}, expect=400)
        self.assertEqual(bad["params"]["field"], "language")
        self.api("POST", "/api/render/plan", {"clips": [clip], "options": {"language": "es"}})
        render = session._render_settings({"language": "es", "reports": {"pdf": True}})
        self.assertEqual(render["language"], "es")
        self.assertIsNone(session._render_settings({})["language"])
        session.find_browser = lambda: (
            "/bin/true"
        )  # a browser that prints nothing: only the text's language matters here
        for wanted, expected in ((None, "en"), ("es", "es")):
            result = session._make_pdf(session.clip(clip), os.path.join(self.dir, "x.pdf"), None, wanted)
            self.assertFalse(result)  # the stand-in writes no file
            self.assertEqual(seen[-1], expected)
        self.api("PUT", "/api/settings", {"language": "es"})
        session._make_pdf(session.clip(clip), os.path.join(self.dir, "x.pdf"), None, None)
        self.assertEqual(seen[-1], "es")


class StateTellsWhatTheDoorTakes(QaCase):
    """Bug 2 (the server's part): a page that sorts a dropped folder needs the server's own list of formats, not a copy of it."""

    def test_formats_are_the_engines_and_the_reasons_the_doors(self):
        import anonymizer
        from anonymizer.ui import intake

        formats = self.api("GET", "/api/state")["formats"]
        self.assertEqual(set(formats["photo"]), set(anonymizer.PHOTO_EXTENSIONS))
        self.assertEqual(set(formats["video"]), set(anonymizer.VIDEO_EXTENSIONS))
        self.assertEqual(set(formats["unsupported"]), set(anonymizer.UNSUPPORTED_EXTENSIONS))
        for extension, code in formats["unsupported"].items():
            self.assertEqual(code, intake.unsupported_code(f"file{extension}"))
        self.assertEqual(formats["unsupported"][".heic"], "error.media.heic")
        self.assertEqual(formats["output_folder"], "anonymized")


class ExportToSeveralFolders(QaCase):
    """Bug 8: with the clips of two folders and no destination chosen, the plan, the page's "Save to" and its error named the first clip's folder for
    all of them: the files of the second went somewhere the page had not said, and "not writable" named a folder that was fine."""

    def clips_in_two_folders(self):
        a = self.api("POST", "/api/clips", {"paths": [self.photo("a.jpg", folder="one")]})["clips"][0]["clip_id"]
        b = self.api("POST", "/api/clips", {"paths": [self.photo("b.jpg", folder="two")]})["clips"][0]["clip_id"]
        self.analyze(a, b)
        return a, b

    def test_the_plan_says_the_outputs_go_to_two_folders(self):
        a, b = self.clips_in_two_folders()
        plan = self.api("POST", "/api/render/plan", {"clips": [a, b]})
        one, two = (os.path.join(self.dir, f, "anonymized") for f in ("one", "two"))
        self.assertIsNone(plan["destination"])
        self.assertEqual(plan["folders"], [one, two])
        self.assertEqual([os.path.dirname(e["output"]) for e in plan["clips"]], [one, two])
        self.assertEqual((plan["writable"], plan["unwritable"]), (True, []))
        alone = self.api("POST", "/api/render/plan", {"clips": [b]})
        self.assertEqual((alone["destination"], alone["folders"]), (two, [two]))
        chosen = self.api("POST", "/api/render/plan", {"clips": [a, b], "options": {"destination": self.dir}})
        self.assertEqual((chosen["destination"], chosen["folders"]), (self.dir, [self.dir]))

    def test_not_writable_names_the_folder_that_is_not(self):
        a, b = self.clips_in_two_folders()
        locked = os.path.join(self.dir, "two")
        os.chmod(locked, 0o555)
        self.addCleanup(os.chmod, locked, 0o755)
        if os.access(locked, os.W_OK):
            self.skipTest("this account can write to a folder without write permission")
        plan = self.api("POST", "/api/render/plan", {"clips": [a, b]})
        self.assertEqual((plan["writable"], plan["unwritable"]), (False, [os.path.join(locked, "anonymized")]))
        refused = self.api("POST", "/api/render", {"clips": [a, b]}, expect=422)
        self.assertEqual(refused["code"], "error.output.not_writable")
        self.assertEqual(refused["params"]["path"], os.path.join(locked, "anonymized"))


class DestinationWithAnImpossibleName(QaCase):
    """Bug 10: a destination with a folder name of 300 letters was "writable" in the plan, the render started, and then it failed."""

    def test_the_plan_refuses_it_before_the_render_starts(self):
        clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        self.analyze(clip)
        for destination in (os.path.join(self.dir, "a" * 300), os.path.join(self.dir, "ok", "b" * 256, "c")):
            plan = self.api("POST", "/api/render/plan", {"clips": [clip], "options": {"destination": destination}})
            self.assertFalse(plan["writable"], destination[-20:])
            refused = self.api(
                "POST", "/api/render", {"clips": [clip], "options": {"destination": destination}}, expect=422
            )
            self.assertEqual(refused["code"], "error.output.not_writable")
        fine = self.api(
            "POST", "/api/render/plan", {"clips": [clip], "options": {"destination": os.path.join(self.dir, "a" * 200)}}
        )
        self.assertTrue(fine["writable"])


class OpeningAddressKeepsTheRestOfItsQuery(QaCase):
    """Bug 11: the address that opens the Studio, `/?t=<token>&lang=de`, was answered with a redirect to `/`: the language asked for was lost."""

    def test_only_the_token_goes(self):
        import http.client

        def opening(query):
            connection = http.client.HTTPConnection("127.0.0.1", self.port)
            connection.request("GET", f"/?{query}")
            response = connection.getresponse()
            response.read()
            return response.status, response.getheader("Location"), response.getheader("Set-Cookie", "")

        status, location, cookie = opening(f"t={server_tests.TOKEN}&lang=de&x=a%20b")
        self.assertEqual((status, location), (302, "/?lang=de&x=a+b"))
        self.assertIn("anon_session_", cookie)
        self.assertEqual(opening(f"t={server_tests.TOKEN}")[:2], (302, "/"))
        self.assertEqual(opening(f"lang=de&t={server_tests.TOKEN}")[:2], (302, "/?lang=de"))
        wrong = opening("t=wrong&lang=de")
        self.assertEqual(
            (wrong[0], wrong[1], wrong[2]), (200, None, "")
        )  # no redirect and no session for a wrong token


class PublishedFilesNameNoHost(unittest.TestCase):
    """A release review found that the results folder and the README named the computers they were measured on by their host names; a neutral label says which
    machine it was and nothing about whose it is. THIRD-PARTY.md says exactly what the PyAV wheel's FFmpeg is."""

    def test_no_result_names_a_host(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[1]
        for path in sorted((root / "bench" / "results").glob("*.json")) + [
            root / "README.md",
            root / "README.template.md",
        ]:
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"/home/|/Users/", text), path.name)
        labels = set()
        for path in (root / "bench" / "results").glob("*.json"):
            labels.update(re.findall(r'"host": ?"([^"]*)"', path.read_text(encoding="utf-8")))
        self.assertTrue(labels <= {"laptop-a", "laptop-b", "test", "unnamed"}, labels)

    def test_the_benchmark_scripts_write_a_label_and_not_the_computers_name(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        for name in ("bench_cells.py", "bench_detection_speed.py"):
            self.assertNotIn("platform.node", (root / "bench" / name).read_text(encoding="utf-8"), name)

    def test_third_party_says_what_the_pyav_ffmpeg_is(self):
        import pathlib

        text = (pathlib.Path(__file__).resolve().parents[1] / "THIRD-PARTY.md").read_text(encoding="utf-8")
        row = next(line for line in text.splitlines() if line.startswith("| PyAV"))
        for word in ("17.1.0", "LGPL v3 or later", "libx264", "libx265", "GPL"):
            self.assertIn(word, row)
        self.assertNotIn("LGPL-2.1", row)


class MaximumThoroughnessIsNotOfferedByTheStudio(QaCase):
    """A release review found that on video Maximum is not repeatable bit for bit and no better than Standard: what works badly is not shown.
    The Python API keeps the value; the Studio's control, settings and protocol do not."""

    def test_the_settings_and_the_analysis_refuse_it_and_an_old_file_falls_back(self):
        refused = self.api("PUT", "/api/settings", {"thoroughness": "maximum"}, expect=400)
        self.assertEqual(refused["params"]["field"], "thoroughness")
        clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        analysis = self.api(
            "POST", "/api/job/analyze", {"clips": [clip], "options": {"thoroughness": "maximum"}}, expect=400
        )
        self.assertEqual(analysis["params"]["field"], "thoroughness")
        self.assertEqual(self.api("GET", "/api/settings")["thoroughness"], "standard")
        self.api("PUT", "/api/settings", {"blur": "solid"})
        path = os.path.join(self.dir, "settings", "settings.json")
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        saved["thoroughness"] = "maximum"  # what a Studio of 2.1.5 wrote when the person had chosen it
        with open(path, "w", encoding="utf-8") as f:
            json.dump(saved, f)
        self.stop_server()
        self.start_server()
        settings = self.api("GET", "/api/settings")
        self.assertEqual((settings["thoroughness"], settings["blur"]), ("standard", "solid"))

    def test_the_protocol_has_no_slowdown_of_maximum(self):
        self.assertEqual(self.api("GET", "/api/state")["slowdown"], {"careful": None})

    def test_the_python_api_keeps_the_value(self):
        import anonymizer

        self.assertEqual(anonymizer.Options(thoroughness="maximum").thoroughness, "maximum")


class ReadmeNamesNoOtherProduct(unittest.TestCase):
    """A release review found that the README draft named another product, headlined "100%" of the plates and quoted v1 timings."""

    def test_the_published_readme_and_the_card_say_none_of_it(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        for name in ("README.md", "README.template.md"):
            text = (root / name).read_text(encoding="utf-8")
            for word in ("deface", "Celantur", "EgoBlur"):
                self.assertNotIn(word, text, (name, word))
            self.assertNotIn("100% of 51", text)
        card = (root / "docs" / "media" / "benchmark.template.html").read_text(encoding="utf-8")
        self.assertNotIn("v1 measurements", card)
        self.assertNotIn("engine.speed", card)
        self.assertNotIn("of this listing", card)
        self.assertIn("hidden.shipped.k}", card)


class LegendFollowsTheDrawing(unittest.TestCase):
    """Bug 18: in Review, Boxes, the engine draws faces green and plates orange, and the legend, the lanes and the swatches of the
    page said faces orange and plates blue. The page takes the engine's colours; this reads both from their own source."""

    def test_the_stylesheet_has_the_colours_the_engine_draws_with(self):
        import pathlib
        import re

        from anonymizer.engine import preview

        css = (
            pathlib.Path(__file__).resolve().parents[1] / "src" / "anonymizer" / "ui" / "static" / "css" / "studio.css"
        ).read_text(encoding="utf-8")
        for kind in ("face", "plate"):
            blue, green, red = preview.COLOURS[kind]  # BGR
            found = re.search(rf"--{kind}: #([0-9a-fA-F]{{6}})", css).group(1).lower()
            self.assertEqual(found, f"{red:02x}{green:02x}{blue:02x}", kind)

    def test_no_script_keeps_the_old_colours(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "src" / "anonymizer" / "ui" / "static"
        for path in root.rglob("*"):
            if path.suffix in (".js", ".css", ".html"):
                text = path.read_text(encoding="utf-8").lower()
                self.assertNotIn("#f2994a", text, path.name)
                self.assertNotIn("#67c1f5", text, path.name)


if __name__ == "__main__":
    unittest.main()
