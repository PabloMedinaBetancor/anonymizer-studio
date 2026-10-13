"""E3: video in and out. Ported from the 1.04 tests (video V1, V4, V13-V16 and A5, A6,
A12), at the level of video_io: what a render writes, without the job around it."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

try:
    import av

    from anonymizer.engine import video_io
    from anonymizer.engine.blur import residual
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

from tests import media


def with_sweep_margin(box):
    """The box a render blurs for a plate seen by a sweep: the 1.04 margins 10 % and 18 % per side, at least 3 px."""
    x0, y0, x1, y1 = box
    dx, dy = max((x1 - x0) * 0.10, 3), max((y1 - y0) * 0.18, 3)
    return (x0 - dx, y0 - dy, x1 + dx, y1 + dy)


def yuv_planes(path: str):
    return [video_io.yuv420_planes(frame, rotation) for _, frame, rotation in video_io.frames(path)]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


class VideoCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)


class RewriteKeepsTheClip(VideoCase):
    def test_E3_same_size_frames_audio_and_a_clean_check(self):
        src, dst = self.path("clip.mp4"), self.path("out.mp4")
        media.moving_plate_clip(src)
        info, writer = media.rewrite(src, dst, lambda t: [media.plate_box(t)])
        out = video_io.probe(dst)
        self.assertEqual(
            (out.width, out.height, out.frames, out.has_audio), (info.width, info.height, info.frames, True)
        )
        self.assertEqual(writer.written, 60)
        self.assertEqual(video_io.check_output(dst, info, writer.written, writer.written_duration), [])

    def test_E3_plate_hidden_in_every_frame_and_the_rest_unchanged(self):
        src, plain, blurred = self.path("clip.mp4"), self.path("plain.mp4"), self.path("blurred.mp4")
        media.moving_plate_clip(src)
        media.rewrite(src, plain)
        media.rewrite(src, blurred, lambda t: [with_sweep_margin(media.plate_box(t))])
        plain_planes, blurred_planes = yuv_planes(plain), yuv_planes(blurred)
        for t in range(0, 60, 7):
            box = media.plate_box(t)
            inside = (box[0] + 6, box[1] + 6, box[2] - 6, box[3] - 6)
            self.assertGreater(residual(media.decoded_frame(plain, t), inside), 1.5)
            self.assertLess(residual(media.decoded_frame(blurred, t), inside), 1.5, f"frame {t}")
            # Away from the patch the planes are what the re-encode of the source gives (no colour round trip).
            outside = np.ones(plain_planes[t][0].shape, bool)
            outside[60:175, max(0, int(box[0]) - 40) : int(box[2]) + 40] = False
            for k, (a, b) in enumerate(zip(plain_planes[t], blurred_planes[t], strict=True)):
                mask = outside if k == 0 else outside[::2, ::2]
                self.assertGreaterEqual(psnr(a[mask], b[mask]), 45.0, f"frame {t}, plane {'YUV'[k]}")

    def test_E3_every_audio_track_is_kept(self):
        src, dst = self.path("two.mkv"), self.path("out.mp4")
        media.short_clip(src, 20, "matroska", (("aac", 0.8), ("aac", 0.8)))
        media.rewrite(src, dst)
        with av.open(dst) as c:
            self.assertEqual(len(c.streams.audio), 2)

    def test_E3_reencoded_audio_keeps_its_offset_from_the_video(self):
        # A broadcast .m2ts starts at 10 s; re-encoding the MP2 sound from 0 put it 11 s off (A5).
        src, dst = self.path("tv.m2ts"), self.path("out.mp4")
        media.short_clip(src, 25, "mpegts", (("mp2", 1.0),), start=10.0)
        _, writer = media.rewrite(src, dst)
        self.assertTrue(writer.audio_reencoded)
        before, after = media.stream_starts(src), media.stream_starts(dst)
        self.assertLess(abs((after["audio"] - after["video"]) - (before["audio"] - before["video"])), 0.06)

    def test_E3_longer_sound_than_picture_still_checks_clean(self):
        # The check compares with what was written, not with the container's duration (A6).
        src, dst = self.path("v.mkv"), self.path("out.mp4")
        media.short_clip(src, 20, "matroska", (("aac", 1.4),))
        info, writer = media.rewrite(src, dst)
        self.assertEqual(video_io.check_output(dst, info, writer.written, writer.written_duration), [])

    def test_E3_truncated_source_writes_what_can_be_seen_and_shows_it(self):
        whole, cut, dst = self.path("whole.mp4"), self.path("cut.mp4"), self.path("out.mp4")
        media.short_clip(whole, 50, "mp4", (("aac", 2.0),), options={"movflags": "+faststart"})
        with open(whole, "rb") as f:
            data = f.read()
        with open(cut, "wb") as f:
            f.write(data[: int(len(data) * 0.6)])
        info, writer = media.rewrite(cut, dst)
        self.assertEqual(video_io.check_output(dst, info, writer.written, writer.written_duration), [])
        self.assertLess(writer.written_duration, info.track_duration - 0.5)

    def test_E3_odd_mjpeg_with_pcm_becomes_even_mp4_with_aac_and_same_brightness(self):
        src, dst = self.path("odd.avi"), self.path("out.mp4")
        media.odd_mjpeg_avi(src)
        info, writer = media.rewrite(src, dst)
        out = video_io.probe(dst)
        self.assertTrue(info.odd)
        self.assertEqual((out.width, out.height, out.has_audio), (320, 240, True))
        self.assertTrue(writer.audio_reencoded)
        a, b = media.decoded_frame(src, 10), media.decoded_frame(dst, 10)
        self.assertLess(abs(float(a.mean()) - float(b.mean())), 4)

    def test_E3_full_range_yuv420p_passes_through_with_the_same_brightness(self):
        src, dst = self.path("full.mp4"), self.path("out.mp4")
        media.full_range_mp4(src)
        media.rewrite(src, dst)
        a, b = media.decoded_frame(src, 8), media.decoded_frame(dst, 8)
        self.assertLess(abs(float(a.mean()) - float(b.mean())), 3)

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs the ffmpeg binary to tag a rotation")
    def test_E3_phone_rotation_is_read_upright(self):
        src, rotated = self.path("clip.mp4"), self.path("rotated.mp4")
        media.moving_plate_clip(src, n=10)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                src,
                "-c",
                "copy",
                "-metadata:s:v:0",
                "rotate=90",
                rotated,
            ],
            check=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
        info = video_io.probe(rotated)
        self.assertEqual((info.width, info.height, info.rotation), (240, 320, 90))


LEAVE_EARLY = """
import sys
import numpy as np
import onnxruntime as ort
from anonymizer.engine import video_io

options = ort.SessionOptions()
options.intra_op_num_threads = 2
session = ort.InferenceSession(sys.argv[1], options, providers=["CPUExecutionProvider"])
feed = {session.get_inputs()[0].name: np.zeros((1, 3, 416, 416), np.float32)}
for t, frame, rotation in video_io.frames(sys.argv[2]):
    video_io.bgr_of(frame, rotation)
    session.run(None, feed)
    if t > 5:
        break
print("done")
"""


class LeaveEarly(VideoCase):
    def test_E13_leaving_a_video_half_way_with_a_model_loaded_does_not_hang(self):
        # R7: with decoder frame threads, closing a video half-way hung forever once onnxruntime had started its
        # own threads. Run in a separate process with a time limit, as Stop does.
        from anonymizer.contract import models_dir

        model = os.path.join(models_dir(), "yolox_nano.onnx")
        if not os.path.isfile(model):
            self.skipTest(f"model not here: {model}")
        clip = self.path("clip.mp4")
        media.moving_plate_clip(clip, n=40)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, *(["-s"] if sys.flags.no_user_site else []), "-c", LEAVE_EARLY, model, clip],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
        self.assertEqual((proc.returncode, proc.stdout.strip()), (0, "done"), proc.stderr[-500:])


class CheckOutput(VideoCase):
    def test_E2_written_duration_never_reads_the_closed_input(self):
        # Reading the stream's frame rate after __exit__ read freed memory: wrong durations and segfaults.
        src, dst = self.path("clip.mp4"), self.path("out.mp4")
        media.moving_plate_clip(src, audio=False)
        _, writer = media.rewrite(src, dst)
        writer.vs = writer.ic = None
        self.assertAlmostEqual(writer.written_duration, 2.0, places=3)

    def test_E2_a_frame_count_mismatch_is_reported(self):
        src, dst = self.path("clip.mp4"), self.path("out.mp4")
        media.moving_plate_clip(src, n=20, audio=False)
        info, writer = media.rewrite(src, dst)
        problems = video_io.check_output(dst, info, writer.written + 3, writer.written_duration)
        self.assertTrue(any("frames" in p for p in problems), problems)

    def test_E2_a_file_cut_half_way_is_reported(self):
        src, dst = self.path("clip.mp4"), self.path("out.mp4")
        media.moving_plate_clip(src, n=30)
        info, writer = media.rewrite(src, dst)
        with open(dst, "r+b") as f:
            f.truncate(os.path.getsize(dst) // 2)
        self.assertNotEqual(video_io.check_output(dst, info, writer.written, writer.written_duration), [])


if __name__ == "__main__":
    unittest.main()
