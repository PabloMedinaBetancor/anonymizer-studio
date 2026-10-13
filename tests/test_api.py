"""A1-A8 and E6, E10, E11, E16: the public API, with stand-in detectors in place of the real models."""

from __future__ import annotations

import csv
import inspect
import json
import os
import re
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

try:
    from PIL import Image

    import anonymizer
    from anonymizer.engine.blur import residual
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

from anonymizer import contract
from anonymizer.contract import FACE, PLATE
from tests import media
from tests.fakes import FixedModule, face_module, plate_module

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PUBLIC = {
    "Analysis": "(source: 'str', media: 'str', fingerprint: 'str', frames: 'int', fps: 'float', width: 'int', height: 'int')",
    "Analyzed": "(clip_id: 'str', items: 'int', to_check: 'int', zones_visited: 'int' = 0, zones_skipped: 'int' = 0, second_added: 'int' = 0) -> None",
    "Anonymizer": "(options: 'Optional[Options]' = None, *, models_dir: 'Optional[str]' = None, threads: 'int' = 0, modules: 'Optional[Sequence[ClassModule]]' = None, persons: 'Optional[Callable]' = None, demo: 'bool' = False, calibration_ms: 'Optional[float]' = None, light_ms: 'Optional[float]' = None)",
    "AnonymizerError": "(code: 'Optional[str]' = None, params: 'Optional[dict]' = None, detail: 'str' = '')",
    "BoxInfo": "(item_id: 'int', kind: 'str', box: 'Box', shape: 'str', blur: 'bool') -> None",
    "CancelToken": "()",
    "Cancelled": "(code: 'Optional[str]' = None, params: 'Optional[dict]' = None, detail: 'str' = '')",
    "Estimate": "(seconds: 'float', zones_expected: 'int', tiles: 'int') -> None",
    "Event": None,
    "Finished": "(clip_id: 'str', status: 'str') -> None",
    "IncompleteAnalysis": "(code: 'Optional[str]' = None, params: 'Optional[dict]' = None, detail: 'str' = '')",
    "Item": "(item_id: 'int', kind: 'str', verdict: 'str', reason: 'str', first_frame: 'int', last_frame: 'int', max_score: 'float', blur: 'bool', params: 'dict' = <factory>, found_by: 'str' = 'light', second_score: 'Optional[float]' = None, decided_by: 'str' = 'engine') -> None",
    "MediaInfo": "(media: 'str', width: 'int', height: 'int', frames: 'int', fps: 'float', duration_s: 'float', codec: 'str', has_audio: 'bool', has_gps: 'bool', size_bytes: 'int') -> None",
    "ItemFound": "(clip_id: 'str', item: 'Item') -> None",
    "ModelLoadError": "(code: 'Optional[str]' = None, params: 'Optional[dict]' = None, detail: 'str' = '')",
    "Options": "(faces: 'bool' = True, plates: 'bool' = True, thoroughness: 'str' = 'standard', gap_fill: 'str' = 'camera', blur: 'str' = 'gaussian', strength: 'float' = 1.0, extra_margin: 'float' = 0.0, remove_gps: 'bool' = True, detection: 'str' = 'careful', heavy_only: 'bool' = False, unsure: 'str' = 'placed', max_zones: 'Optional[int]' = None, plate_path: 'str' = 'text', vehicle_model: 'str' = 'rfdetr', plate_band: 'Optional[bool]' = None, fit: 'str' = 'tight') -> None",
    "OutputExists": "(code: 'Optional[str]' = None, params: 'Optional[dict]' = None, detail: 'str' = '')",
    "Progress": "(clip_id: 'str', phase: 'str', fraction: 'float', frame: 'int', frames: 'int', live: 'Optional[list]' = None, found: 'Optional[dict]' = None, stage: 'Optional[str]' = None, zones: 'Optional[dict]' = None) -> None",
    "Rendered": "(clip_id: 'str', output: 'str', seconds: 'float') -> None",
    "Report": "(rows: 'List[ReportRow]', options: 'dict' = <factory>, second: 'dict' = <factory>) -> None",
    "ReportRow": "(clip: 'str', item_id: 'int', kind: 'str', verdict: 'str', blurred: 'bool', reason: 'str', first_s: 'float', last_s: 'float', frames: 'int', max_score: 'float', found_by: 'str' = 'light', decision: 'str' = 'blurred', decided_by: 'str' = 'engine', plate_like: 'Optional[bool]' = None) -> None",
    "Result": "(output: 'str', report: 'Report', warnings: 'List[str]', seconds: 'float', visible_check: 'int' = 0) -> None",
    "SourceChanged": "(code: 'Optional[str]' = None, params: 'Optional[dict]' = None, detail: 'str' = '')",
    "Started": "(clip_id: 'str', media: 'str', frames: 'int', width: 'int', height: 'int', fps: 'float') -> None",
    "UnsupportedMedia": "(code: 'Optional[str]' = None, params: 'Optional[dict]' = None, detail: 'str' = '')",
    "PHOTO_EXTENSIONS": None,
    "UNSUPPORTED_EXTENSIONS": None,
    "VIDEO_EXTENSIONS": None,
    "JobWarning": "(clip_id: 'str', code: 'str', params: 'dict' = <factory>) -> None",
    "anonymize_image": "(source: 'str', target: 'str', options: 'Optional[Options]' = None, on_event: 'Optional[EventHandler]' = None) -> 'Result'",
    "anonymize_video": "(source: 'str', target: 'str', options: 'Optional[Options]' = None, on_event: 'Optional[EventHandler]' = None, cancel: 'Optional[CancelToken]' = None) -> 'Result'",
    "compare_video": "(original: 'str', anonymized: 'str', target: 'str', captions: 'Tuple[str, str]' = ('Original', 'Anonymized'), on_event: 'Optional[EventHandler]' = None, cancel: 'Optional[CancelToken]' = None) -> 'str'",
    "event_json": "(event: 'Event') -> 'dict'",
    "frame_jpeg": "(path: 'str', frame: 'int' = 0, width: 'int' = 960) -> 'bytes'",
    "load_analysis": "(path: 'str') -> 'Analysis'",
    "media_info": "(path: 'str') -> 'MediaInfo'",
    "thumbnail_jpeg": "(path: 'str', width: 'int' = 240) -> 'bytes'",
}
METHODS = {
    "Anonymizer": ["analyze", "calibrate", "calibrate_light", "close", "estimate", "modules", "render"],
    "Analysis": [
        "add_area",
        "boxes",
        "item",
        "items",
        "preview",
        "remove_area",
        "report",
        "save",
        "set_blur",
        "summary",
        "thumbnail",
        "thumbnail_jpeg",
        "unchecked_frames",
    ],
    "Report": ["summary", "to_csv", "to_json"],
    "CancelToken": ["cancel", "cancelled", "pause", "paused", "resume"],
}


class NoDetection:
    kind, blur_shape = PLATE, "rect"

    def __getattr__(self, name):
        raise AssertionError(f"a detector was used: {name}")


class ApiCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.events = []

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def clip(self, name="clip.mp4", **kw) -> str:
        path = self.path(name)
        media.moving_plate_clip(path, **kw)
        return path

    def anon(self, *modules, **options) -> anonymizer.Anonymizer:
        return anonymizer.Anonymizer(anonymizer.Options(**options), modules=list(modules) or [plate_module()])


def plate_hidden(path: str, t: int, box=None) -> bool:
    box = box or media.plate_box(t)
    inside = (box[0] + 6, box[1] + 6, box[2] - 6, box[3] - 6)
    return residual(media.decoded_frame(path, t), inside) < 1.5


class PublicSurface(unittest.TestCase):
    def test_A1_the_public_names_and_signatures_are_pinned(self):
        self.assertEqual(sorted(anonymizer.__all__), sorted(PUBLIC))
        for name, want in PUBLIC.items():
            obj = getattr(anonymizer, name)
            if want is not None:
                self.assertEqual(str(inspect.signature(obj)), want, name)
        for name, methods in METHODS.items():
            cls = getattr(anonymizer, name)
            public = sorted(m for m, _ in inspect.getmembers(cls) if not m.startswith("_"))
            self.assertEqual(public, sorted(methods), name)

    def test_A1_the_package_is_typed(self):
        self.assertTrue(os.path.isfile(os.path.join(os.path.dirname(anonymizer.__file__), "py.typed")))

    def test_A6_the_studio_ui_imports_only_the_public_package(self):
        ui = os.path.join(REPO, "src", "anonymizer", "ui")
        # siblings inside the ui package are fine; anything above it or inside the engine is not
        private = re.compile(
            r"^\s*(?:from\s+(?:anonymizer\.(?!ui\b)\w+|\.{2,}\w*)\s+import|import\s+anonymizer\.(?!ui\b))", re.M
        )
        for folder, _, files in os.walk(ui):
            for name in files:
                if name.endswith(".py"):
                    with open(os.path.join(folder, name), encoding="utf-8") as f:
                        self.assertIsNone(private.search(f.read()), name)

    def test_A1_options_refuse_values_outside_their_range(self):
        for bad in (
            dict(faces=False, plates=False),
            dict(thoroughness="max"),
            dict(gap_fill="spline"),
            dict(blur="swirl"),
            dict(strength=0.5),
            dict(extra_margin=0.6),
        ):
            with self.assertRaises(ValueError):
                anonymizer.Options(**bad)


class VideoFlow(ApiCase):
    def test_A2_one_call_gives_the_same_file_as_analyze_and_render(self):
        src = self.clip(audio=False)
        one, two = self.path("one.mp4"), self.path("two.mp4")
        with mock.patch("anonymizer.classes.load", lambda kinds, threads, models, **kw: [plate_module()]):
            anonymizer.anonymize_video(src, one)
        with self.anon() as anon:
            anon.render(anon.analyze(src), two)
        with open(one, "rb") as a, open(two, "rb") as b:
            self.assertEqual(a.read(), b.read())
        self.assertTrue(plate_hidden(one, 30))

    def test_A3_E6_decisions_save_reload_and_render_without_detectors(self):
        src = self.clip(audio=False)
        with self.anon() as anon:
            analysis = anon.analyze(src)
        [item] = analysis.items
        self.assertEqual((item.kind, item.verdict, item.blur), (PLATE, "blur", True))
        analysis.set_blur(item.item_id, False)
        saved = self.path("clip.analysis.json")
        analysis.save(saved)
        reloaded = anonymizer.load_analysis(saved)
        self.assertEqual(reloaded.items, analysis.items)
        with anonymizer.Anonymizer(modules=[NoDetection()]) as anon:
            first = anon.render(analysis, self.path("a.mp4"))
            second = anon.render(reloaded, self.path("b.mp4"))
        self.assertFalse(plate_hidden(first.output, 30))
        with open(first.output, "rb") as a, open(second.output, "rb") as b:
            self.assertEqual(a.read(), b.read())

    def test_E6_a_render_from_memory_equals_one_after_a_save_and_a_reload_with_the_blur_on(self):
        src = self.clip(audio=False)
        with self.anon() as anon:
            analysis = anon.analyze(src)
            saved = self.path("blurred.analysis.json")
            analysis.save(saved)
            first = anon.render(analysis, self.path("memory.mp4"))
            second = anon.render(anonymizer.load_analysis(saved), self.path("reloaded.mp4"))
        self.assertTrue(plate_hidden(first.output, 30))
        with open(first.output, "rb") as a, open(second.output, "rb") as b:
            self.assertEqual(a.read(), b.read())

    def test_A4_E10_events_in_order_and_progress_monotonic_per_phase(self):
        src = self.clip(audio=False)
        with self.anon() as anon:
            anon.render(anon.analyze(src, self.events.append), self.path("out.mp4"), self.events.append)
        kinds = [type(e).__name__ for e in self.events]
        order = ["Started", "ItemFound", "Analyzed", "Rendered", "Finished"]
        self.assertEqual([k for k in kinds if k in order], order)
        for phase in ("analyze", "render"):
            fractions = [e.fraction for e in self.events if isinstance(e, anonymizer.Progress) and e.phase == phase]
            self.assertEqual(fractions, sorted(set(fractions)))
            self.assertEqual(fractions[-1], 1.0)
        for e in self.events:
            message = anonymizer.event_json(e)
            self.assertEqual(json.loads(json.dumps(message)), message)
            self.assertEqual(message["v"], 1)
            self.assertIn(
                message["type"],
                {
                    "job.started",
                    "clip.progress",
                    "clip.item",
                    "clip.analyzed",
                    "clip.rendered",
                    "job.warning",
                    "job.finished",
                },
            )

    def test_A4_cancel_stops_quickly_and_leaves_no_output(self):
        src = self.clip(audio=False)
        token = anonymizer.CancelToken()
        stopped_at = []

        def handler(e):
            if isinstance(e, anonymizer.Progress) and e.phase == "render" and e.frame >= 15 and not token.cancelled:
                token.cancel()
                stopped_at.append(time.monotonic())

        out = self.path("out.mp4")
        with self.anon() as anon:
            analysis = anon.analyze(src)
            with self.assertRaises(anonymizer.Cancelled) as caught:
                anon.render(analysis, out, handler, token)
        self.assertLess(time.monotonic() - stopped_at[0], 10.0)  # 2 s on a free machine; slack for a loaded one
        self.assertEqual(caught.exception.code, "error.cancelled")
        self.assertEqual(os.listdir(self.dir), ["clip.mp4"])

    def test_E8_pause_holds_the_work_and_resume_finishes_the_same_file(self):
        src = self.clip(audio=False)
        token = anonymizer.CancelToken()
        held = []

        def handler(e):
            if isinstance(e, anonymizer.Progress) and e.phase == "render" and e.frame >= 15 and not held:
                token.pause()
                held.append(time.monotonic())
                threading.Timer(0.4, token.resume).start()

        with self.anon() as anon:
            analysis = anon.analyze(src)
            plain = anon.render(analysis, self.path("plain.mp4"))
            paused = anon.render(analysis, self.path("paused.mp4"), handler, token)
        self.assertGreaterEqual(time.monotonic() - held[0], 0.4)
        self.assertFalse(token.paused)
        with open(plain.output, "rb") as a, open(paused.output, "rb") as b:
            self.assertEqual(a.read(), b.read())

    def test_E8_cancel_while_paused_stops(self):
        src = self.clip(audio=False)
        token = anonymizer.CancelToken()

        def handler(e):
            if isinstance(e, anonymizer.Progress) and e.phase == "analyze" and e.frame >= 15 and not token.paused:
                token.pause()
                threading.Timer(0.2, token.cancel).start()

        with self.anon() as anon, self.assertRaises(anonymizer.Cancelled):
            anon.analyze(src, handler, token)

    def test_E3_ten_bit_video_comes_out_8_bit_with_a_warning(self):
        src = self.path("hdr.mkv")
        media.ten_bit_clip(src)
        with self.anon() as anon:
            result = anon.render(anon.analyze(src), self.path("out.mp4"), self.events.append)
        from anonymizer.engine import video_io

        out = video_io.probe(result.output)
        self.assertEqual((out.pix_fmt, out.frames, out.width, out.height), ("yuv420p", 12, 160, 120))
        self.assertIn("video.warning.converted_to_8bit", result.warnings)
        self.assertIn(
            "video.warning.converted_to_8bit", [e.code for e in self.events if isinstance(e, anonymizer.JobWarning)]
        )
        a, b = media.decoded_frame(src, 6), media.decoded_frame(result.output, 6)
        self.assertLess(abs(float(a[:40].mean()) - float(b[:40].mean())), 4)
        self.assertTrue(plate_hidden(result.output, 6, (37, 47, 123, 73)))

    def test_E16_an_added_area_follows_the_camera_or_stays_fixed(self):
        src = self.path("pan.mp4")
        media.panning_clip(src)
        plate_at = lambda t: (120 - 3 * t - 3, 97, 220 - 3 * t + 3, 133)  # noqa: E731
        with self.anon(face_module()) as anon:
            analysis = anon.analyze(src)
            self.assertEqual(analysis.items, [])
            area = analysis.add_area(0, 39, plate_at(0), follow="camera")
            self.assertEqual((area.kind, area.verdict), ("manual", "blur"))
            followed = anon.render(analysis, self.path("camera.mp4"))
            analysis.remove_area(area.item_id)
            analysis.add_area(0, 39, plate_at(0), follow="fixed")
            fixed = anon.render(analysis, self.path("fixed.mp4"))
        for t in (0, 20, 30):
            self.assertTrue(plate_hidden(followed.output, t, plate_at(t)), f"frame {t}")
        self.assertFalse(plate_hidden(fixed.output, 30, plate_at(30)))
        self.assertEqual([r.kind for r in followed.report.rows], ["manual"])


class Photos(ApiCase):
    PLATE_BOX, FACE_BOX = (20, 20, 100, 44), (140, 60, 200, 120)

    def setUp(self):
        super().setUp()
        self.noise = (np.random.RandomState(3).rand(160, 240, 3) * 255).astype(np.uint8)
        self.photo = self.path("f.png")
        Image.fromarray(self.noise).save(self.photo)
        self.modules = [FixedModule(FACE, [self.FACE_BOX], 0.85, 0.75), FixedModule(PLATE, [self.PLATE_BOX])]

    def blurred(self, path, box) -> bool:
        x0, y0, x1, y1 = box
        out = np.array(Image.open(path).convert("RGB")).astype(float)
        return out[y0 + 4 : y1 - 4, x0 + 4 : x1 - 4].std() < 0.5 * self.noise[y0 + 4 : y1 - 4, x0 + 4 : x1 - 4].std()

    def test_A3_E11_review_a_photo_and_report_every_item(self):
        with anonymizer.Anonymizer(anonymizer.Options(unsure="keep"), modules=self.modules) as anon:  # v1's rule
            analysis = anon.analyze(self.photo)
            face = next(i for i in analysis.items if i.kind == FACE)
            plate = next(i for i in analysis.items if i.kind == PLATE)
            self.assertEqual((face.verdict, face.blur, plate.verdict, plate.blur), ("check", False, "blur", True))
            analysis.set_blur(face.item_id, True)
            analysis.set_blur(plate.item_id, False)
            analysis.add_area(0, 0, (180, 130, 230, 150))
            result = anon.render(analysis, self.path("out.png"))
        self.assertTrue(self.blurred(result.output, self.FACE_BOX))
        self.assertFalse(self.blurred(result.output, self.PLATE_BOX))
        self.assertTrue(self.blurred(result.output, (180, 130, 230, 150)))
        result.report.to_csv(self.path("report.csv"))
        result.report.to_json(self.path("report.json"))
        with open(self.path("report.csv"), encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(
            sorted((r["kind"], r["blurred"]) for r in rows), [("face", "True"), ("manual", "True"), ("plate", "False")]
        )
        with open(self.path("report.json"), encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)["items"]), len(analysis.items))

    def test_A8_models_load_once_per_anonymizer_and_close_frees_them(self):
        loads, closed = [], []
        module = FixedModule(PLATE, [self.PLATE_BOX])
        module.close = lambda: closed.append(True)

        def load(kinds, threads, models, **kw):
            loads.append(list(kinds))
            return [module]

        for i in range(3):
            Image.fromarray(self.noise).save(self.path(f"p{i}.jpg"))
        with mock.patch("anonymizer.classes.load", load):
            with anonymizer.Anonymizer(anonymizer.Options(faces=False)) as anon:
                for i in range(3):
                    anon.render(anon.analyze(self.path(f"p{i}.jpg")), self.path(f"o{i}.jpg"))
        self.assertEqual((loads, closed), ([[PLATE]], [True]))


def failing_plates(fail_at: int):
    """Plates whose model fails (and says so) on sweep number `fail_at`."""
    module = plate_module()
    sweep = module.sweep

    def failing(bgr, plan):
        if module.calls["sweep"] == fail_at:
            module.model_failures += 1
            module.calls["sweep"] += 1
            return []
        return sweep(bgr, plan)

    module.sweep = failing
    return module


class Errors(ApiCase):
    def test_A5_X3_a_failed_model_run_is_never_written_as_clean(self):
        src = self.clip(n=20, audio=False)
        with anonymizer.Anonymizer(modules=[failing_plates(3)]) as anon:
            analysis = anon.analyze(src)
            self.assertEqual(analysis.unchecked_frames, [6])
            with self.assertRaises(anonymizer.IncompleteAnalysis) as caught:
                anon.render(analysis, self.path("out.mp4"))
        self.assertEqual(caught.exception.params["frames"], 1)
        self.assertFalse(os.path.exists(self.path("out.mp4")))
        Image.new("RGB", (64, 48)).save(self.path("p.jpg"))
        with anonymizer.Anonymizer(modules=[failing_plates(0)]) as anon:
            with self.assertRaises(anonymizer.IncompleteAnalysis):
                anon.analyze(self.path("p.jpg"))

    def test_A5_unsupported_media(self):
        for name, data in (("photo.heic", b"\0" * 32), ("notes.txt", b"hello"), ("broken.jpg", b"not a jpeg")):
            with open(self.path(name), "wb") as f:
                f.write(data)
            with self.assertRaises(anonymizer.UnsupportedMedia) as caught:
                with self.anon() as anon:
                    anon.analyze(self.path(name))
            self.assertEqual(caught.exception.code, "error.media.unsupported")

    def test_A5_model_load_error_names_the_model(self):
        def load(kinds, threads, models, **kw):
            raise contract.ModelLoadError(PLATE, "file missing")

        Image.new("RGB", (32, 32)).save(self.path("a.jpg"))
        with mock.patch("anonymizer.classes.load", load), self.assertRaises(anonymizer.ModelLoadError) as caught:
            anonymizer.anonymize_image(self.path("a.jpg"), self.path("b.jpg"))
        self.assertEqual((caught.exception.code, caught.exception.params), ("error.model.load", {"kind": PLATE}))

    def test_A5_output_exists_source_changed_and_incomplete_analysis(self):
        src = self.clip(n=20, audio=False)
        with self.anon() as anon:
            analysis = anon.analyze(src)
            open(self.path("taken.mp4"), "w").close()
            with self.assertRaises(anonymizer.OutputExists):
                anon.render(analysis, self.path("taken.mp4"))
            analysis.frames -= 1
            analysis._video.frames -= 1
            with self.assertRaises(anonymizer.IncompleteAnalysis):
                anon.render(analysis, self.path("short.mp4"))
            self.assertFalse(os.path.exists(self.path("short.mp4")))
            os.utime(src, (1, 1))
            with self.assertRaises(anonymizer.SourceChanged) as caught:
                anon.render(analysis, self.path("late.mp4"))
        self.assertEqual(caught.exception.code, "error.source.changed")


if __name__ == "__main__":
    unittest.main()
