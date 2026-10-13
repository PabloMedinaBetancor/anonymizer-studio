"""The engine in a worker process: the same results as in-process, Pause and Stop that reach it, and a hung or dead
worker that is killed and reported without hurting the server."""

from __future__ import annotations

import os
import signal
import tempfile
import threading
import time
import unittest
import unittest.mock

try:
    import anonymizer
    from anonymizer.ui import engine as engines
    from tests import media
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

FAKES = "tests.fakes"


def worker_env(factory: str) -> dict:
    return {"ANONYMIZER_WORKER_MODULES": f"{FAKES}:{factory}"}


class EngineCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.source = os.path.join(self.dir, "clip.mp4")
        media.moving_plate_clip(self.source, n=60, audio=True)
        self.events = []

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def engine(self, factory="plate_modules", **options):
        found = engines.ProcessEngine(anonymizer.Options(), work_dir=self.dir, env=worker_env(factory), **options)
        self.addCleanup(found.close)
        return found


class SameAsInProcess(EngineCase):
    def test_analyze_render_and_compare_give_what_the_local_engine_gives(self):
        from tests.fakes import plate_module

        local = engines.LocalEngine(anonymizer.Options(), modules=[plate_module()])
        token = anonymizer.CancelToken()
        want = local.analyze("c1", self.source, lambda e: None, token)
        with self.engine() as worker:
            got = worker.analyze("c1", self.source, self.events.append, token)
            self.assertEqual([i for i in got.items], [i for i in want.items])
            self.assertEqual(got.thumbnail_jpeg(got.items[0].item_id), want.thumbnail_jpeg(want.items[0].item_id))
            self.assertEqual(got.unchecked_frames, want.unchecked_frames)
            got.set_blur(got.items[0].item_id, True)
            result = worker.render("c1", got, self.path("worker.mp4"), self.events.append, token)
            compared = worker.compare(
                "c1", self.source, result.output, self.path("both.mp4"), self.events.append, token
            )
        want.set_blur(want.items[0].item_id, True)  # the same decision on both: the report says who decided
        want_out = local.render("c1", want, self.path("local.mp4"), lambda e: None, token)
        with open(want_out.output, "rb") as a, open(result.output, "rb") as b:
            self.assertEqual(a.read(), b.read())
        self.assertEqual(result.report.rows, want_out.report.rows)
        self.assertEqual(media.probe_size(compared), (640, 240))
        kinds = [type(e).__name__ for e in self.events]
        self.assertIn("ItemFound", kinds)
        self.assertEqual(
            {e.phase for e in self.events if isinstance(e, anonymizer.Progress)}, {"analyze", "render", "compare"}
        )
        self.assertTrue(any(e.live for e in self.events if isinstance(e, anonymizer.Progress) and e.live is not None))

    def test_what_a_library_prints_does_not_corrupt_the_messages(self):
        with self.engine("noisy_plate_modules") as worker:
            analysis = worker.analyze("c1", self.source, self.events.append, anonymizer.CancelToken())
        self.assertEqual(len(analysis.items), 1)

    def test_errors_come_back_as_their_own_classes(self):
        broken = self.path("broken.mp4")
        with open(broken, "wb") as f:
            f.write(b"\0" * 4096)
        with self.engine() as worker:
            with self.assertRaises(anonymizer.UnsupportedMedia) as caught:
                worker.analyze("c1", broken, self.events.append, anonymizer.CancelToken())
            self.assertEqual(caught.exception.code, "error.media.unsupported")
            analysis = worker.analyze("c2", self.source, self.events.append, anonymizer.CancelToken())
            target = self.path("again.mp4")
            worker.render("c2", analysis, target, self.events.append, anonymizer.CancelToken())
            with self.assertRaises(anonymizer.OutputExists):
                worker.render("c2", analysis, target, self.events.append, anonymizer.CancelToken())


class Control(EngineCase):
    def test_stop_reaches_the_worker_and_leaves_no_output(self):
        token = anonymizer.CancelToken()
        started = []
        with self.engine("slow_plate_modules") as worker:

            def on_event(event):
                started.append(event)
                if len(started) == 3:
                    token.cancel()

            with self.assertRaises(anonymizer.Cancelled):
                worker.analyze("c1", self.source, on_event, token)
            # The requirement is that the worker OBEYS Stop (it answers Cancelled itself), not that it does so within
            # some seconds: a worker still being killed for ignoring it is the failure. The 2 s of Stop is measured
            # in-process (test_api A4) and in bench, never against the wall clock of a loaded machine here.
            self.assertNotEqual(worker._killed_for, "cancel")
            self.assertGreaterEqual(len(started), 3)
        self.assertFalse([n for n in os.listdir(self.dir) if n.endswith(".analysis.json")])

    def test_pause_holds_the_worker_without_a_false_silence_and_resume_continues(self):
        token = anonymizer.CancelToken()
        seen = []
        # silent_after must be shorter than the pause for the test to mean anything, and long enough that a loaded
        # machine does not make the first event after start or resume late enough to count as silence.
        with self.engine("slow_plate_modules", silent_after=8.0, watch_every=0.2) as worker:

            def on_event(event):
                seen.append(time.monotonic())
                if len(seen) == 2:
                    token.pause()
                    threading.Timer(10.0, token.resume).start()

            analysis = worker.analyze("c1", self.source, on_event, token)
        gaps = [b - a for a, b in zip(seen, seen[1:], strict=False)]
        self.assertGreater(max(gaps), 9.0)
        self.assertEqual(len(analysis.items), 1)

    def test_a_stop_that_is_not_obeyed_ends_in_a_kill(self):
        token = anonymizer.CancelToken()
        mark = self.path("hang.mark")
        patched = unittest.mock.patch.dict(os.environ, {"ANONYMIZER_TEST_HANG_MARK": mark})
        patched.start()
        self.addCleanup(patched.stop)
        with self.engine("hanging_plate_modules", kill_after_cancel=0.5) as worker:

            def stop_when_stuck():  # a Stop sent before the detector is stuck is simply obeyed: not what is tested
                end = time.monotonic() + 120
                while not os.path.exists(mark) and time.monotonic() < end:
                    time.sleep(0.05)
                token.cancel()

            threading.Thread(target=stop_when_stuck, daemon=True).start()
            began = time.monotonic()
            with self.assertRaises(anonymizer.Cancelled):
                worker.analyze("c1", self.source, self.events.append, token)
            self.assertLess(time.monotonic() - began, 120)  # a hang guard, not a requirement
            worker.proc.wait(timeout=30)  # SIGKILL is sent; the reaping is the kernel's, not instant under load
            self.assertIsNotNone(worker.proc.poll())


class Failures(EngineCase):
    def test_a_silent_worker_is_killed_and_the_reason_is_silence(self):
        with self.engine("hanging_plate_modules", silent_after=1.0, watch_every=0.2) as worker:
            began = time.monotonic()
            with self.assertRaises(engines.EngineFailure) as caught:
                worker.analyze("c1", self.source, self.events.append, anonymizer.CancelToken())
            self.assertEqual(caught.exception.code, "error.engine.silent")
            self.assertLess(time.monotonic() - began, 120)  # a hang guard, not a requirement
            worker.proc.wait(timeout=30)  # SIGKILL is sent; the reaping is the kernel's, not instant under load
            self.assertIsNotNone(worker.proc.poll())
            with self.assertRaises(engines.EngineFailure):
                worker.analyze("c2", self.source, self.events.append, anonymizer.CancelToken())

    def test_a_worker_that_dies_is_reported_with_how_it_died(self):
        with self.engine("slow_plate_modules") as worker:

            def kill(event):
                if isinstance(event, anonymizer.Progress):
                    os.kill(worker.proc.pid, signal.SIGKILL)  # this worker, by its pid

            with self.assertRaises(engines.EngineFailure) as caught:
                worker.analyze("c1", self.source, kill, anonymizer.CancelToken())
        self.assertEqual(caught.exception.code, "error.engine.crashed")
        self.assertIn("exit code -9", caught.exception.detail)

    def test_an_unexpected_exception_in_the_worker_is_a_crash_with_its_traceback(self):
        with self.engine("not_a_function") as worker:
            with self.assertRaises(engines.EngineFailure) as caught:
                worker.analyze("c1", self.source, self.events.append, anonymizer.CancelToken())
        self.assertEqual(caught.exception.code, "error.engine.crashed")
        self.assertIn("not_a_function", caught.exception.detail)


class Events(unittest.TestCase):
    def test_worker_messages_become_the_api_events(self):
        item = anonymizer.Item(1, "plate", "blur", "plate.reason.on_vehicle_confirmed", 3, 9, 0.9, True, {"score": 0.9})
        for event in (
            anonymizer.Progress("a.mp4", "analyze", 0.5, 10, 20, [{"kind": "plate"}], {"plate": 1}),
            anonymizer.ItemFound("a.mp4", item),
            anonymizer.JobWarning("a.mp4", "video.warning.audio_reencoded", {}),
            anonymizer.Started("a.mp4", "video", 20, 320, 240, 25.0),
            anonymizer.Analyzed("a.mp4", 1, 0),
            anonymizer.Rendered("a.mp4", "/tmp/x.mp4", 1.5),
            anonymizer.Finished("a.mp4", "done"),
        ):
            self.assertEqual(engines.event_of(anonymizer.event_json(event)), event)
        self.assertIsNone(engines.event_of({"type": "something.else", "v": 1}))

    def test_an_error_message_becomes_the_class_it_names_and_unknown_ones_the_base(self):
        known = engines.error_of({"kind": "OutputExists", "code": "error.output.exists", "params": {"path": "x"}})
        self.assertIsInstance(known, anonymizer.OutputExists)
        self.assertEqual((known.code, known.params), ("error.output.exists", {"path": "x"}))
        self.assertIs(type(engines.error_of({"kind": "Nonsense", "code": "error.x"})), anonymizer.AnonymizerError)
        self.assertIs(type(engines.error_of({"kind": "Options", "code": "error.x"})), anonymizer.AnonymizerError)


if __name__ == "__main__":
    unittest.main()
