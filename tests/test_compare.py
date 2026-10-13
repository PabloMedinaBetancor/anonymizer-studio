"""E12: the side-by-side comparison video."""

from __future__ import annotations

import os
import tempfile
import unittest

try:
    import numpy as np

    import anonymizer
    from anonymizer.engine import video_io
    from tests import media
    from tests.fakes import plate_module
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


class Comparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = cls.tmp.name
        cls.source = os.path.join(cls.dir, "source.mp4")
        media.moving_plate_clip(cls.source, n=40, audio=True)
        cls.anonymized = os.path.join(cls.dir, "anonymized.mp4")
        with anonymizer.Anonymizer(modules=[plate_module()]) as anon:
            anon.render(anon.analyze(cls.source), cls.anonymized)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def test_E12_side_by_side_with_the_same_length_and_the_audio(self):
        target = self.path("both.mp4")
        events = []
        self.assertEqual(anonymizer.compare_video(self.source, self.anonymized, target, on_event=events.append), target)
        original, wide = video_io.probe(self.source), video_io.probe(target)
        self.assertEqual((wide.width, wide.height, wide.frames), (original.width * 2, original.height, original.frames))
        self.assertAlmostEqual(wide.duration, original.duration, delta=0.1)
        self.assertTrue(wide.has_audio)
        fractions = [e.fraction for e in events if isinstance(e, anonymizer.Progress)]
        self.assertEqual(
            (fractions == sorted(fractions), fractions[-1], {e.phase for e in events}), (True, 1.0, {"compare"})
        )
        self.assertEqual([n for n in os.listdir(self.dir) if n.startswith(".")], [])

    def test_E12_each_half_is_its_video_and_each_has_a_caption_bar(self):
        target = self.path("halves.mp4")
        anonymizer.compare_video(self.source, self.anonymized, target)
        frame = media.decoded_frame(target, 20).astype(int)
        width = frame.shape[1] // 2
        original = media.decoded_frame(self.source, 20).astype(int)
        hidden = media.decoded_frame(self.anonymized, 20).astype(int)
        bar = frame.shape[0] // 12
        below = slice(bar + 2, None)
        self.assertLess(np.abs(frame[below, :width] - original[below]).mean(), 3)
        self.assertLess(np.abs(frame[below, width:] - hidden[below]).mean(), 3)
        x0, y0, x1, y1 = (int(v) for v in media.plate_box(20))
        inside = (slice(y0 + 6, y1 - 6), slice(x0 + 6, x1 - 6))
        self.assertLess(frame[:, width:][inside].std(), 0.4 * frame[:, :width][inside].std())
        for half in (frame[:bar, :width], frame[:bar, width:]):
            self.assertGreater(half.max(), 200)  # the text
            self.assertLess(np.median(half), 0.6 * np.median(original[:bar]))  # the bar

    def test_E12_nothing_is_overwritten_and_bad_input_is_refused(self):
        existing = self.path("exists.mp4")
        with open(existing, "wb") as f:
            f.write(b"keep")
        with self.assertRaises(anonymizer.OutputExists):
            anonymizer.compare_video(self.source, self.anonymized, existing)
        with open(existing, "rb") as f:
            self.assertEqual(f.read(), b"keep")
        with self.assertRaises(ValueError):
            anonymizer.compare_video(
                self.source, self.anonymized, self.path("x.mp4"), captions=("Original", "Anonymized" + chr(233))
            )
        other = self.path("small.mp4")
        media.moving_plate_clip(other, n=40, width=160, height=120, audio=False)
        with self.assertRaises(anonymizer.AnonymizerError) as caught:
            anonymizer.compare_video(self.source, other, self.path("y.mp4"))
        self.assertEqual(caught.exception.code, "error.output.check_failed")
        shorter = self.path("short.mp4")
        media.moving_plate_clip(shorter, n=20, audio=False)
        with self.assertRaises(anonymizer.AnonymizerError):
            anonymizer.compare_video(self.source, shorter, self.path("z.mp4"))
        self.assertFalse([n for n in os.listdir(self.dir) if n.startswith(".") or n in ("x.mp4", "y.mp4", "z.mp4")])

    def test_E12_cancel_leaves_nothing(self):
        token = anonymizer.CancelToken()
        token.cancel()
        with self.assertRaises(anonymizer.Cancelled):
            anonymizer.compare_video(self.source, self.anonymized, self.path("c.mp4"), cancel=token)
        self.assertFalse([n for n in os.listdir(self.dir) if n.startswith(".") or n == "c.mp4"])


if __name__ == "__main__":
    unittest.main()
