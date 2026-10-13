"""The requirements that are about seconds on a free machine: Stop reaches the work within 2 s. They run only with
ANONYMIZER_TIMING=1, on a machine with nothing else running; the other tests have slack so that a loaded machine does not break the suite
(PR-16). bench/bench_engine.py measures the same on the 1080p clip and writes it to bench/results/engine.json (engine.stop)."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest

if os.environ.get("ANONYMIZER_TIMING") != "1":
    raise unittest.SkipTest("timing requirements run only with ANONYMIZER_TIMING=1 on a free machine")

try:
    import anonymizer
    from anonymizer.engine.events import Cancelled, StopFlag
    from tests import media
    from tests.fakes import plate_module
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

LIMIT_S = 2.0


class StopWithinTwoSeconds(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.source = os.path.join(self.dir, "clip.mp4")
        media.moving_plate_clip(self.source, n=400, audio=False)

    def stop_after(self, run, delay: float) -> float:
        """Seconds from cancel(), called from a timer thread in the middle of the work, to Cancelled."""
        token = anonymizer.CancelToken()
        asked = []

        def stop():
            asked.append(time.monotonic())
            token.cancel()

        threading.Timer(delay, stop).start()
        with self.assertRaises(anonymizer.Cancelled):
            run(token)
        return time.monotonic() - asked[0]

    def test_E8_stop_during_the_analysis_reaches_it_within_two_seconds(self):
        with anonymizer.Anonymizer(modules=[plate_module()]) as anon:
            self.assertLess(self.stop_after(lambda tok: anon.analyze(self.source, None, tok), 0.5), LIMIT_S)

    def test_A4_stop_during_the_render_reaches_it_within_two_seconds(self):
        with anonymizer.Anonymizer(modules=[plate_module()]) as anon:
            analysis = anon.analyze(self.source)
            out = os.path.join(self.dir, "out.mp4")
            self.assertLess(self.stop_after(lambda tok: anon.render(analysis, out, None, tok), 0.3), LIMIT_S)
            self.assertFalse(os.path.exists(out))

    def test_E8_stop_wakes_a_paused_worker_within_two_seconds(self):
        flag = StopFlag()
        flag.pause()
        threading.Timer(0.1, flag.request).start()
        start = time.monotonic()
        with self.assertRaises(Cancelled):
            flag.check()
        self.assertLess(time.monotonic() - start, LIMIT_S)


if __name__ == "__main__":
    unittest.main()
