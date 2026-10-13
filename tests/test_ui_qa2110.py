"""Anonymizer Studio v2.1.10: what the eleventh client QA pass (06/10/2026, qa-anon-11) found. One test class per entry of CHANGELOG 2.1.10; each fails on
v2.1.9 and passes here."""

from __future__ import annotations

import fractions
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import numpy as np
    from PIL import Image

    import anonymizer
    from anonymizer.ui import server as server_module
    from tests.fakes import FACE, FixedModule, face_module
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

import fetch_models  # noqa: E402

I18N = ROOT / "src" / "anonymizer" / "ui" / "i18n"


def make_server(case):
    tmp = tempfile.TemporaryDirectory()
    case.addCleanup(tmp.cleanup)
    for folder in ("static", "i18n", "work", "settings"):
        os.makedirs(os.path.join(tmp.name, folder))
    with open(os.path.join(tmp.name, "static", "index.html"), "w") as f:
        f.write("<!doctype html><title>Studio</title>")
    with open(os.path.join(tmp.name, "i18n", "en.json"), "w") as f:
        f.write("{}")
    server = server_module.make_server(
        settings_folder=os.path.join(tmp.name, "settings"),
        static_dir=os.path.join(tmp.name, "static"),
        i18n_dir=os.path.join(tmp.name, "i18n"),
        work_dir=os.path.join(tmp.name, "work"),
        modules=[],
        engine="thread",
    )
    case.addCleanup(server.server_close)
    return server


class WindowNeedsADisplay(unittest.TestCase):
    """Entry 6: with the optional pywebview (GTK) installed and no display (an SSH session), `webview.start()` neither raises nor
    returns: the Studio served to nobody and printed nothing. The display is checked first; without one the page goes to the browser
    path and one printed line says so."""

    def run_main(self, env, window):
        server = make_server(self)
        watchdog = threading.Timer(20, server.shutdown)
        watchdog.start()
        self.addCleanup(watchdog.cancel)
        threading.Timer(1, server.shutdown).start()  # the Studio is stopped from outside, as a person would
        err = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {**{k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}, **env},
                clear=True,
            ),
            mock.patch.dict(sys.modules, {"webview": window}),
            mock.patch.object(server_module.webbrowser, "open") as opened,
            mock.patch.object(server_module, "make_server", return_value=server),
            mock.patch.object(server_module, "remove_stale_work_folders"),
            mock.patch.object(server_module, "stop_on_signals"),
            mock.patch.object(server_module, "watch_idle"),
            mock.patch.object(server_module, "user_folder", return_value=None),
            mock.patch("sys.stdout"),
            mock.patch("sys.stderr", err),
        ):
            code = server_module.main([])
        self.assertEqual(code, 0)
        return server, opened, err.getvalue()

    def window(self):
        module = types.ModuleType("webview")
        module.created = []
        module.options = []
        module.screens = [types.SimpleNamespace(x=0, y=0, width=1366, height=768)]

        def create_window(title, url, **options):
            module.created.append(url)
            module.options.append(options)

        module.create_window = create_window

        def start(*args, **options):  # GTK without a display: never returns (the test stops the Studio, then this ends)
            threading.Event().wait(30)

        module.start = start
        return module

    @unittest.skipUnless(sys.platform.startswith("linux"), "the display rule is Linux's")
    def test_no_display_means_the_browser_path_and_one_line_that_says_so(self):
        window = self.window()
        server, opened, said = self.run_main({}, window)
        self.assertEqual(window.created, [], "the window must not be asked for with no display")
        opened.assert_called_once_with(server.url())
        self.assertEqual(len([line for line in said.splitlines() if line.strip()]), 1, said)
        self.assertIn("No display", said)

    @unittest.skipUnless(sys.platform.startswith("linux"), "the display rule is Linux's")
    def test_a_display_keeps_the_window(self):
        window = self.window()
        window.start = lambda *a, **k: None
        server, opened, said = self.run_main({"DISPLAY": ":0"}, window)
        self.assertEqual(window.created, [server.url()])
        opened.assert_not_called()
        self.assertLessEqual(
            window.options[0]["height"], 768 - 100
        )  # the screen of the stand-in is a 1366x768 laptop's

    def test_the_rule_by_system(self):
        with mock.patch.object(sys, "platform", "linux"), mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(server_module.display_available())
            os.environ["WAYLAND_DISPLAY"] = "wayland-0"
            self.assertTrue(server_module.display_available())
        with mock.patch.object(sys, "platform", "darwin"), mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(server_module.display_available())


class WindowFitsTheScreen(unittest.TestCase):
    """Entry 5: the Studio's own window (the optional pywebview) was always 1360x860: on a 1366x768 laptop its lower 100 px, with the Analyze
    button, were below the screen (seen under a virtual 1280x800 screen: the window 1360x860, centred, cut on every side)."""

    def screen(self, width, height):
        return types.SimpleNamespace(screens=[types.SimpleNamespace(x=0, y=0, width=width, height=height)])

    def test_a_laptop_screen_gets_a_window_that_fits_it(self):
        for width, height in ((1366, 768), (1280, 800), (1280, 720), (1024, 768)):
            with self.subTest(screen=(width, height)):
                w, h = server_module.window_size(self.screen(width, height))
                self.assertLessEqual(w, width - 40)
                self.assertLessEqual(h, height - 100)
                self.assertGreaterEqual(h, 560)

    def test_a_big_screen_keeps_the_usual_size(self):
        self.assertEqual(server_module.window_size(self.screen(2560, 1440)), (1360, 860))

    def test_a_screen_that_cannot_be_asked_gets_a_size_every_laptop_holds(self):
        self.assertEqual(server_module.window_size(types.SimpleNamespace()), (1280, 720))
        self.assertEqual(server_module.window_size(types.SimpleNamespace(screens=[])), (1280, 720))


class NoModelsSaidAtStart(unittest.TestCase):
    """Entry 7: a Studio started from a folder with no `models/` (after `pip install` the models stay in the checkout) said nothing: the person found
    out when Analyze failed with "A detection model could not be loaded (face)", not knowing which folder was meant."""

    def test_a_folder_with_no_models_is_said_with_its_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            line = server_module.models_warning(os.path.join(tmp, "nowhere"))
            self.assertIn(os.path.join(tmp, "nowhere"), line)
            self.assertIn("fetch_models.py", line)
            self.assertIn("ANONYMIZER_MODELS", line)
            empty = os.path.join(tmp, "empty")
            os.makedirs(empty)
            self.assertIn(empty, server_module.models_warning(empty))

    def test_a_folder_with_models_says_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "yunet.onnx"), "wb").close()
            self.assertIsNone(server_module.models_warning(tmp))

    def test_the_default_folders_are_named_when_nothing_was_said(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANONYMIZER_MODELS", None)
            cwd = os.getcwd()
            self.addCleanup(os.chdir, cwd)
            os.chdir(tmp)
            with mock.patch("anonymizer.contract._default_model_folders", return_value=[os.path.join(tmp, "models")]):
                line = server_module.models_warning(None)
            self.assertIn(os.path.join(tmp, "models"), line)

    def test_main_prints_it_on_the_terminal(self):
        server = make_server(self)
        threading.Timer(1, server.shutdown).start()
        err = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(server_module, "make_server", return_value=server),
            mock.patch.object(server_module, "remove_stale_work_folders"),
            mock.patch.object(server_module, "stop_on_signals"),
            mock.patch.object(server_module, "watch_idle"),
            mock.patch.object(server_module, "user_folder", return_value=None),
            mock.patch("sys.stdout"),
            mock.patch("sys.stderr", err),
        ):
            code = server_module.main(["--no-browser", "--models", os.path.join(tmp, "none")])
            self.assertEqual(code, 0)
            self.assertIn(os.path.join(tmp, "none"), err.getvalue())


class GpsLabelSaysPhotos(unittest.TestCase):
    """Entry 8: the Import option "Remove GPS and location" does nothing for a video (a video never carries the container's
    metadata); the label did not say it is for photos."""

    WORD = {
        "en": "photo",
        "de": "Fot",
        "es": "fot",
        "fr": "photo",
        "it": "fot",
        "pt": "fot",
        "zh-Hans": "照片",
    }

    def test_every_language_says_it_is_for_photos(self):
        for lang, word in self.WORD.items():
            with self.subTest(lang=lang):
                data = json.loads((I18N / f"{lang}.json").read_text(encoding="utf-8"))
                self.assertIn(word, data["options.remove_gps"])


class FetchModelsOffline(unittest.TestCase):
    """Entry 12: `scripts/fetch_models.py` with no network printed one failed line per model and stopped: no line said what to do."""

    def test_a_closing_line_says_to_check_the_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            (folder / "SHA256SUMS").write_text(("0" * 64) + "  a.onnx\n" + ("1" * 64) + "  b.onnx\n")

            def offline(url, timeout=None):
                raise OSError("Network is unreachable")

            out = io.StringIO()
            code = fetch_models.fetch(
                str(folder),
                sources={"a.onnx": "https://x/a.onnx", "b.onnx": "https://x/b.onnx"},
                opener=offline,
                out=out,
            )
        lines = out.getvalue().strip().splitlines()
        self.assertEqual(code, 1)
        self.assertIn("internet connection", lines[-1])
        self.assertIn("2 of 2", lines[-1])

    def test_nothing_is_added_when_nothing_failed_on_the_way(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            (folder / "SHA256SUMS").write_text(("0" * 64) + "  a.onnx\n")
            out = io.StringIO()
            fetch_models.fetch(str(folder), sources={}, out=out)  # no source: a different failure
        self.assertNotIn("connection", out.getvalue())


class ApiErrorMessage(unittest.TestCase):
    """Entry 9: an `AnonymizerError` carried only its code as its message (`error.source.missing`): a script that printed it, or
    a traceback, told the person nothing."""

    def test_the_message_is_a_sentence_with_its_parameters(self):
        e = anonymizer.AnonymizerError("error.source.missing", {"path": "clips/a.mp4"})
        self.assertEqual(str(e), "clips/a.mp4 was not found.")
        self.assertEqual(e.code, "error.source.missing")
        self.assertEqual(e.params, {"path": "clips/a.mp4"})

    def test_the_detail_follows_the_sentence_and_the_code_stays_what_it_was(self):
        e = anonymizer.UnsupportedMedia(params={"name": "x.bin"}, detail="no extension: unknown")
        self.assertTrue(str(e).startswith("x.bin is not a photo or a video that can be processed."), str(e))
        self.assertTrue(str(e).endswith("(no extension: unknown)"), str(e))
        self.assertNotIn("imported", str(e))  # a word of the Studio, not of the API
        self.assertEqual(e.code, "error.media.unsupported")
        self.assertEqual(e.detail, "no extension: unknown")

    def test_a_photo_given_to_the_video_function_is_a_photo(self):
        with tempfile.TemporaryDirectory() as tmp:
            photo = os.path.join(tmp, "in.jpg")
            Image.fromarray((np.random.RandomState(1).rand(40, 60, 3) * 255).astype(np.uint8)).save(photo)
            with self.assertRaises(anonymizer.UnsupportedMedia) as caught:
                anonymizer.anonymize_video(photo, os.path.join(tmp, "out.mp4"))
            self.assertEqual(str(caught.exception), "in.jpg is not a video.")
            with self.assertRaises(anonymizer.UnsupportedMedia) as caught:
                anonymizer.anonymize_image(os.path.join(tmp, "x.bin"), os.path.join(tmp, "out.jpg"))
            self.assertNotIn("is not a photo.", str(caught.exception))

    def test_a_code_without_a_sentence_and_a_missing_parameter_still_say_something(self):
        self.assertEqual(str(anonymizer.AnonymizerError("error.nothing.like.it")), "error.nothing.like.it")
        self.assertEqual(
            str(anonymizer.AnonymizerError("error.nothing.like.it", detail="d")), "error.nothing.like.it: d"
        )
        self.assertIn("{path}", str(anonymizer.OutputExists()))

    def test_every_error_the_api_names_has_a_sentence(self):
        for cls in (
            anonymizer.UnsupportedMedia,
            anonymizer.ModelLoadError,
            anonymizer.OutputExists,
            anonymizer.SourceChanged,
            anonymizer.Cancelled,
            anonymizer.IncompleteAnalysis,
            anonymizer.AnonymizerError,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertNotEqual(str(cls()), cls.default_code)


class MissingDestinationFolder(unittest.TestCase):
    """Entry 11: `anonymize_image("in.jpg", "no/such/folder/out.jpg")` failed with a `FileNotFoundError` that named the hidden working file
    (`.out.jpg.anonymizer-partial`), a file the person never made; `anonymize_video` said so only after the whole analysis."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.photo = os.path.join(self.dir, "in.jpg")
        Image.fromarray((np.random.RandomState(1).rand(120, 160, 3) * 255).astype(np.uint8)).save(self.photo)
        self.target = os.path.join(self.dir, "no-such-folder", "out.jpg")

    def check(self, e):
        self.assertIsInstance(e, FileNotFoundError)  # still the error a caller already catches
        self.assertEqual(e.filename, os.path.join(self.dir, "no-such-folder"))
        self.assertIn("no-such-folder", str(e))
        self.assertNotIn("anonymizer-partial", str(e))
        self.assertIn("does not exist", str(e))

    def test_render_names_the_folder(self):
        with anonymizer.Anonymizer(modules=[FixedModule(FACE, [(10, 10, 60, 60)])]) as anon:
            analysis = anon.analyze(self.photo)
            with self.assertRaises(FileNotFoundError) as caught:
                anon.render(analysis, self.target)
        self.check(caught.exception)
        self.assertFalse(os.path.exists(os.path.dirname(self.target)))  # nothing was made

    def test_the_one_call_functions_say_it_before_analysing(self):
        with mock.patch.object(anonymizer.Anonymizer, "analyze", side_effect=AssertionError("analysed first")):
            with self.assertRaises(FileNotFoundError) as caught:
                anonymizer.anonymize_image(self.photo, self.target)
            self.check(caught.exception)
            with mock.patch("anonymizer.api.media_of", return_value="video"):
                with self.assertRaises(FileNotFoundError) as caught:
                    anonymizer.anonymize_video(self.photo, self.target)
            self.check(caught.exception)

    def test_a_folder_that_is_a_file_is_said_too(self):
        blocker = os.path.join(self.dir, "blocker")
        open(blocker, "w").close()
        with self.assertRaises(NotADirectoryError) as caught:
            anonymizer.anonymize_image(self.photo, os.path.join(blocker, "out.jpg"))
        self.assertEqual(caught.exception.filename, blocker)

    def test_a_target_in_the_current_folder_and_an_existing_folder_still_work(self):
        out = os.path.join(self.dir, "ok.jpg")
        with anonymizer.Anonymizer(modules=[FixedModule(FACE, [(10, 10, 60, 60)])]) as anon:
            anon.render(anon.analyze(self.photo), out)
        self.assertTrue(os.path.exists(out))
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(self.dir)
        with anonymizer.Anonymizer(modules=[FixedModule(FACE, [(10, 10, 60, 60)])]) as anon:
            anon.render(anon.analyze(self.photo), "relative.jpg")  # no folder part at all
        self.assertTrue(os.path.exists(os.path.join(self.dir, "relative.jpg")))


def anamorphic_clip(path, sar, n=12, width=96, height=72):
    """A clip whose pixels are not square: `sar` is its sample aspect ratio (PAL 720x576 shown as 16:9 is 64:45, HDV 1440x1080 is 4:3)."""
    import av

    with av.open(path, "w") as out:
        vs = out.add_stream("libx264", rate=25)
        vs.width, vs.height, vs.pix_fmt = width, height, "yuv420p"
        vs.codec_context.sample_aspect_ratio = sar
        for t in range(n):
            frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), 40 + 9 * t, np.uint8), format="rgb24")
            frame = frame.reformat(format="yuv420p")
            frame.pts, frame.time_base = t, fractions.Fraction(1, 25)
            for p in vs.encode(frame):
                out.mux(p)
        for p in vs.encode(None):
            out.mux(p)


def sample_aspect_ratio(path):
    import av

    with av.open(path) as container:
        return container.streams.video[0].sample_aspect_ratio


class NonSquarePixelsKeepTheirShape(unittest.TestCase):
    """Entry 1: a video with non-square pixels (PAL 720x576 as 16:9, HDV 1440x1080 as 16:9) left with square ones: its picture played
    squeezed (16:9 became 5:4). The writer never copied the source's sample aspect ratio. Found by writing a clip of each kind of source."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def render(self, src):
        out = os.path.join(self.dir, "out.mp4")
        with anonymizer.Anonymizer(anonymizer.Options(detection="fast"), modules=[face_module()]) as anon:
            anon.render(anon.analyze(src), out)
        return out

    def test_the_output_keeps_the_ratio_of_its_pixels(self):
        for sar in (fractions.Fraction(64, 45), fractions.Fraction(4, 3)):
            with self.subTest(sar=sar):
                src = os.path.join(self.dir, "in.mp4")
                anamorphic_clip(src, sar)
                self.assertEqual(sample_aspect_ratio(src), sar)
                out = self.render(src)
                try:
                    self.assertEqual(sample_aspect_ratio(out), sar)
                finally:
                    os.remove(out)

    def test_the_comparison_video_does_not_mix_ratios(self):
        src = os.path.join(self.dir, "in.mp4")
        anamorphic_clip(src, fractions.Fraction(4, 3))
        out = self.render(src)
        both = os.path.join(self.dir, "both.mp4")
        anonymizer.compare_video(src, out, both)
        self.assertEqual(sample_aspect_ratio(both), fractions.Fraction(4, 3))

    def test_square_pixels_stay_square(self):
        src = os.path.join(self.dir, "in.mp4")
        anamorphic_clip(src, fractions.Fraction(1, 1))
        out = self.render(src)
        self.assertIn(sample_aspect_ratio(out), (None, fractions.Fraction(1, 1), fractions.Fraction(0, 1)))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg makes the rotated clip")
    def test_a_turned_video_turns_the_ratio_over_with_it(self):
        src, turned = os.path.join(self.dir, "in.mp4"), os.path.join(self.dir, "turned.mp4")
        anamorphic_clip(src, fractions.Fraction(4, 3))
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", src, "-c", "copy", "-metadata:s:v:0", "rotate=90", turned], check=True
        )
        out = self.render(turned)
        self.assertEqual(sample_aspect_ratio(out), fractions.Fraction(3, 4))


if __name__ == "__main__":
    unittest.main()
