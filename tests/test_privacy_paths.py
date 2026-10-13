"""PR-9: one test per path by which a frame, photo or clip could leave as clean without having been analysed or blurred.
The table of paths and the tests that cover them (these and the older ones) is docs/PRIVACY-PATHS.md."""

from __future__ import annotations

import csv
import json
import os
import signal
import unittest

try:
    import anonymizer
    from anonymizer.engine.files import UNSUPPORTED_EXTENSIONS
    from anonymizer.ui import engine as engines
    from tests import media
    from tests import test_api as api_tests
    from tests import test_ui_engine as engine_tests
    from tests import test_ui_server as server_tests
    from tests.fakes import plate_module
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

DAMAGED_OR_INCOMPLETE = {
    "error.media.damaged",
    "error.analysis.incomplete",
    "error.output.check_failed",
    "error.source.changed",
}


def decoder_raises(path: str) -> bool:
    """Does this machine's FFmpeg raise while decoding the video of `path`? (The engine reports a damaged file when it does.)"""
    import av

    try:
        with av.open(path) as container:
            for _ in container.decode(container.streams.video[0]):
                pass
    except av.error.FFmpegError:
        return True
    return False


# Where in the video data and how: the first is what the tests always did. A scrambled stretch of an entropy-coded stream can decode
# as valid garbage (the encoder of another machine, an aarch64 libx264, puts different bytes in the middle of the file), so
# the next ones damage more of it or somewhere else, until this machine's decoder actually fails.
DAMAGE_ATTEMPTS = (
    (0.5, 400, lambda b: (b * 7 + 13) & 255),
    (0.5, 4000, lambda b: (b * 7 + 13) & 255),
    (0.3, 4000, lambda b: b ^ 0xFF),
    (0.7, 4000, lambda b: b ^ 0xFF),
)


def damage(path: str, keep_stamp: bool = False) -> None:
    """Scramble bytes in the middle of the video data: the container still opens, the decoder fails there. When the decoder of
    this machine takes the first scrambling as valid data, more is scrambled; when nothing makes it fail the test is skipped, as
    there is no damaged clip to test with (the cause of three failures on aarch64: the engine reports what the decoder raises)."""
    stat = os.stat(path)
    with open(path, "rb") as f:
        original = f.read()
    mdat = original.find(b"mdat")
    for fraction, length, change in DAMAGE_ATTEMPTS:
        data = bytearray(original)
        start = mdat + int((len(data) - mdat) * fraction)
        for k in range(start, min(start + length, len(data))):
            data[k] = change(data[k])
        with open(path, "wb") as f:
            f.write(data)
        if decoder_raises(path):
            break
    else:
        with open(path, "wb") as f:
            f.write(original)
        raise unittest.SkipTest("this machine's video decoder takes every damage tried as valid data")
    if keep_stamp:
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def drawn_over(output: str, source: str, t: int, box) -> bool:
    """True when the area is blurred in the output: what is left of its detail is a tenth of the source's."""
    from anonymizer.engine.blur import residual

    inside = (box[0] + 6, box[1] + 6, box[2] - 6, box[3] - 6)
    return residual(media.decoded_frame(output, t), inside) < 0.1 * residual(media.decoded_frame(source, t), inside)


def sha(path: str) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def visible(folder: str) -> list:
    return sorted(n for n in os.listdir(folder) if not n.startswith("."))


def hidden(folder: str) -> list:
    return sorted(n for n in os.listdir(folder) if n.startswith("."))


class Api(api_tests.ApiCase):
    def test_P01_a_saved_analysis_keeps_its_unchecked_frames_and_still_refuses_to_render(self):
        src = self.clip(n=20, audio=False)
        with anonymizer.Anonymizer(modules=[api_tests.failing_plates(3)]) as anon:
            analysis = anon.analyze(src)
            self.assertEqual(analysis.unchecked_frames, [6])
            analysis.save(self.path("a.analysis.json"))
            reloaded = anonymizer.load_analysis(self.path("a.analysis.json"))
            self.assertEqual(reloaded.unchecked_frames, [6])
            with self.assertRaises(anonymizer.IncompleteAnalysis):
                anon.render(reloaded, self.path("out.mp4"))
        self.assertEqual(hidden(self.dir) + [n for n in visible(self.dir) if n.startswith("out")], [])

    def test_P02_a_video_that_stops_being_readable_midway_is_a_typed_error_of_that_clip(self):
        src = self.clip(n=60, audio=False)
        damage(src)
        with self.anon() as anon, self.assertRaises(anonymizer.UnsupportedMedia) as caught:
            anon.analyze(src)
        self.assertEqual(caught.exception.code, "error.media.damaged")
        self.assertEqual(caught.exception.params, {"name": "clip.mp4"})

    def test_P03_a_source_damaged_after_the_analysis_is_never_written_as_clean(self):
        src = self.clip(n=60, audio=False)
        with self.anon() as anon:
            analysis = anon.analyze(src)
            damage(src, keep_stamp=True)  # same size and time: the fingerprint cannot tell
            with self.assertRaises(anonymizer.AnonymizerError) as caught:
                anon.render(analysis, self.path("out.mp4"))
        self.assertIn(caught.exception.code, DAMAGED_OR_INCOMPLETE)
        self.assertEqual(visible(self.dir), ["clip.mp4"])
        self.assertEqual(hidden(self.dir), [])

    def test_P03b_another_file_of_the_same_size_and_time_is_not_the_analysed_source(self):
        src = self.clip(n=60, audio=False)
        with self.anon() as anon:
            analysis = anon.analyze(src)
            stat = os.stat(src)
            with open(src, "r+b") as f:  # same length, the start of the file different, the time put back
                f.seek(100)
                f.write(b"\x00" * 64)
            os.utime(src, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(os.stat(src).st_size, stat.st_size)
            self.assertEqual(os.stat(src).st_mtime_ns, stat.st_mtime_ns)
            with self.assertRaises(anonymizer.SourceChanged):
                anon.render(analysis, self.path("out.mp4"))
        self.assertEqual(visible(self.dir), ["clip.mp4"])

    def test_P04_a_source_replaced_while_it_is_written_is_never_delivered(self):
        src = self.clip(n=60, audio=False)
        touched = []

        def replace_source(event):
            if isinstance(event, anonymizer.Progress) and event.phase == "render" and not touched:
                touched.append(True)
                with open(src, "ab") as f:
                    f.write(b"\0")

        with self.anon() as anon:
            analysis = anon.analyze(src)
            with self.assertRaises(anonymizer.SourceChanged):
                anon.render(analysis, self.path("out.mp4"), replace_source)
        self.assertTrue(touched)
        self.assertEqual(visible(self.dir), ["clip.mp4"])
        self.assertEqual(hidden(self.dir), [])

    def test_P05_a_photo_replaced_while_it_is_written_is_never_delivered(self):
        from unittest import mock

        from PIL import Image

        photo = self.path("p.png")
        Image.new("RGB", (64, 48), (120, 130, 140)).save(photo)
        real = anonymizer.api.process

        def replace_then_write(*args, **kwargs):
            result = real(*args, **kwargs)
            with open(photo, "ab") as f:
                f.write(b"\0")
            return result

        with self.anon() as anon:
            analysis = anon.analyze(photo)
            with mock.patch("anonymizer.api.process", replace_then_write), self.assertRaises(anonymizer.SourceChanged):
                anon.render(analysis, self.path("out.png"))
        self.assertEqual(visible(self.dir), ["p.png"])
        self.assertEqual(hidden(self.dir), [])

    def test_P05b_a_photo_cut_short_is_refused_in_every_format_and_one_damaged_after_the_analysis_is_not_delivered(
        self,
    ):
        import numpy as np
        from PIL import Image

        pixels = (np.random.RandomState(1).rand(120, 160, 3) * 255).astype(np.uint8)
        for ext in (".jpg", ".png", ".webp", ".bmp", ".tif"):
            path = self.path("whole" + ext)
            Image.fromarray(pixels).save(path)
            with open(path, "rb") as f:
                data = f.read()
            cut = self.path("cut" + ext)
            with open(cut, "wb") as f:
                f.write(data[: len(data) // 2])
            with self.assertRaises(anonymizer.UnsupportedMedia, msg=ext), self.anon() as anon:
                anon.analyze(cut)
            with self.anon() as anon:
                analysis = anon.analyze(path)
                stat = os.stat(path)
                with open(path, "wb") as f:
                    f.write(data[: len(data) // 2] + bytes(len(data) - len(data) // 2))
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                try:
                    anon.render(analysis, self.path("out" + ext))
                except anonymizer.AnonymizerError:
                    pass  # refused: nothing may be left under the final name
            if os.path.exists(self.path("out" + ext)):
                with Image.open(self.path("out" + ext)) as written:
                    written.load()  # whatever was written must at least be a whole picture
        self.assertEqual(hidden(self.dir), [])

    def test_P06_every_unsupported_extension_is_refused_and_nothing_is_written(self):
        for ext in UNSUPPORTED_EXTENSIONS:
            path = self.path("file" + ext)
            with open(path, "wb") as f:
                f.write(os.urandom(64))
            with self.assertRaises(anonymizer.UnsupportedMedia, msg=ext), self.anon() as anon:
                anon.analyze(path)
            with self.assertRaises(anonymizer.UnsupportedMedia, msg=ext):
                anonymizer.media_info(path)
        self.assertEqual(len(visible(self.dir)), len(UNSUPPORTED_EXTENSIONS))

    def test_P06b_the_location_of_a_video_is_not_in_the_output_whatever_the_options(self):
        import av

        base = self.clip(name="base.mp4", n=30, audio=False)
        src = self.path("gps.mp4")
        place = "+" + "28.1235" + "-" + "15.4363" + "/"  # ISO 6709, as phones write it
        with av.open(base) as source, av.open(src, "w") as out:
            out.metadata["location"] = place
            out.metadata["com.apple.quicktime.location.ISO6709"] = place
            stream = out.add_stream_from_template(source.streams.video[0])
            stream.metadata["location"] = place
            for packet in source.demux(source.streams.video[0]):
                if packet.dts is not None:
                    packet.stream = stream
                    out.mux(packet)
        self.assertTrue(anonymizer.media_info(src).has_gps)
        for keep in (True, False):
            with self.anon(remove_gps=keep) as anon:
                done = anon.render(anon.analyze(src), self.path(f"out-{keep}.mp4"))
            self.assertFalse(anonymizer.media_info(done.output).has_gps, f"remove_gps={keep}")
            with av.open(done.output) as written:
                keys = {k.lower() for k in written.metadata} | {k.lower() for s in written.streams for k in s.metadata}
            self.assertFalse([k for k in keys if "location" in k], keys)

    def test_P07_a_decision_to_leave_a_plate_visible_shows_in_the_pixels_and_in_the_report_and_back(self):
        src = self.clip(n=60, audio=False)
        with self.anon() as anon:
            analysis = anon.analyze(src)
            [item] = analysis.items
            analysis.set_blur(item.item_id, False)
            left = anon.render(analysis, self.path("left.mp4"))
            analysis.set_blur(item.item_id, True)
            hid = anon.render(analysis, self.path("hid.mp4"))
        self.assertFalse(api_tests.plate_hidden(left.output, 30))
        self.assertTrue(api_tests.plate_hidden(hid.output, 30))
        self.assertEqual([(r.item_id, r.blurred) for r in left.report.rows], [(item.item_id, False)])
        self.assertEqual([(r.item_id, r.blurred) for r in hid.report.rows], [(item.item_id, True)])

    def test_P08_a_manual_area_switched_off_or_removed_is_not_drawn_and_the_report_says_so(self):
        src = self.clip(n=60, audio=False)
        box = media.plate_box(30)
        with self.anon(plate_module()) as anon:
            analysis = anon.analyze(src)
            [found] = analysis.items
            analysis.set_blur(found.item_id, False)
            area = analysis.add_area(0, 59, box, follow="fixed")
            on = anon.render(analysis, self.path("on.mp4"))
            analysis.set_blur(area.item_id, False)
            off = anon.render(analysis, self.path("off.mp4"))
            analysis.remove_area(area.item_id)
            gone = anon.render(analysis, self.path("gone.mp4"))
        self.assertTrue(drawn_over(on.output, src, 30, box))
        self.assertFalse(drawn_over(off.output, src, 30, box))
        self.assertFalse(drawn_over(gone.output, src, 30, box))
        self.assertEqual({r.kind: r.blurred for r in on.report.rows}, {"manual": True, "plate": False})
        self.assertEqual({r.kind: r.blurred for r in off.report.rows}, {"manual": False, "plate": False})
        self.assertEqual({r.kind: r.blurred for r in gone.report.rows}, {"plate": False})


class Worker(engine_tests.EngineCase):
    def test_P09_a_worker_killed_during_the_render_leaves_nothing_under_the_final_name(self):
        target = self.path("out.mp4")
        token = anonymizer.CancelToken()
        with self.engine() as worker:
            analysis = worker.analyze("c1", self.source, self.events.append, token)

            def kill(event):
                if isinstance(event, anonymizer.Progress) and event.phase == "render":
                    os.kill(worker.proc.pid, signal.SIGKILL)  # this worker, by its pid

            with self.assertRaises(engines.EngineFailure) as caught:
                worker.render("c1", analysis, target, kill, token)
        self.assertEqual(caught.exception.code, "error.engine.crashed")
        self.assertFalse(os.path.exists(target))
        self.assertEqual([n for n in visible(self.dir) if n.endswith((".mp4", ".png"))], ["clip.mp4"])
        with self.engine() as again:  # a hidden leftover of the dead one does not get in the way of the next run
            result = again.render("c1", analysis, target, self.events.append, token)
        self.assertTrue(os.path.exists(result.output))
        self.assertEqual(hidden(self.dir), [])


class Server(server_tests.StudioCase):
    def test_P10_a_destination_that_is_the_source_folder_never_replaces_the_original(self):
        from PIL import Image

        photo = os.path.join(self.dir, "media", "p.png")
        Image.new("RGB", (64, 48), (120, 130, 140)).save(photo)
        photo_id = self.api("POST", "/api/clips", {"paths": [photo]})["clips"][0]["clip_id"]
        clip_id = self.import_clip()
        video = self.server.session.clip(clip_id).path
        before = {p: sha(p) for p in (photo, video)}
        self.analyze(clip_id, photo_id)
        folder = os.path.dirname(video)
        options = {
            "destination": folder,
            "name_pattern": "{name}",
            "reports": {"csv": False, "json": False, "pdf": False},
        }
        plan = self.api("POST", "/api/render/plan", {"clips": [clip_id, photo_id], "options": options})
        self.assertEqual(sorted(os.path.basename(c["output"]) for c in plan["clips"]), ["clip-2.mp4", "p-2.png"])
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/render", {"clips": [clip_id, photo_id], "options": options}, 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.assertEqual({p: sha(p) for p in (photo, video)}, before)
        self.assertEqual(visible(folder), ["clip-2.mp4", "clip.mp4", "p-2.png", "p.png"])

    def test_P11_unchecked_frames_that_appear_after_the_plan_still_block_the_render(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        options = {"destination": folder}
        plan = self.api("POST", "/api/render/plan", {"clips": [clip_id], "options": options})
        self.assertIsNone(plan["clips"][0]["blocked"])
        self.server.session.clip(clip_id).analysis._video.unchecked.append(5)
        status, _, data = self.request("POST", "/api/render", {"clips": [clip_id], "options": options})
        self.assertEqual((status, json.loads(data)["code"]), (422, "error.analysis.incomplete"))
        self.assertFalse(os.path.exists(folder))

    def test_P12_a_source_that_changes_after_the_plan_still_blocks_the_render(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        options = {"destination": folder}
        plan = self.api("POST", "/api/render/plan", {"clips": [clip_id], "options": options})
        self.assertIsNone(plan["clips"][0]["blocked"])
        with open(self.server.session.clip(clip_id).path, "ab") as f:
            f.write(b"\0")
        status, _, data = self.request("POST", "/api/render", {"clips": [clip_id], "options": options})
        self.assertEqual((status, json.loads(data)["code"]), (422, "error.source.changed"))
        self.assertFalse(os.path.exists(folder))

    def test_P13_the_decision_to_leave_something_visible_reaches_the_files_and_the_reports(self):
        clip_id = self.import_clip()
        source = self.server.session.clip(clip_id).path
        self.analyze(clip_id)
        [item] = self.api("GET", f"/api/clips/{clip_id}/analysis")["items"]
        reports = {"csv": True, "json": True, "pdf": False}
        written = {}
        for blur in (False, True):
            self.api("PATCH", f"/api/clips/{clip_id}/items/{item['item_id']}", {"blur": blur})
            options = {"destination": os.path.join(self.dir, f"out-{blur}"), "reports": reports}
            reader = server_tests.EventReader(self.port)
            self.addCleanup(reader.close)
            self.api("POST", "/api/render", {"clips": [clip_id], "options": options}, 202)
            written[blur] = reader.wait_for("clip.rendered")
            self.wait_idle()
        x0, y0, x1, y1 = (int(v) for v in media.plate_box(30))
        inside = (slice(y0 + 6, y1 - 6), slice(x0 + 6, x1 - 6))
        original = media.decoded_frame(source, 30)[inside].std()
        for blur, event in written.items():
            [output] = event["outputs"]
            ratio = media.decoded_frame(output, 30)[inside].std() / original
            self.assertEqual(ratio < 0.4, blur, (blur, ratio))
            with open(event["report"]["csv"], encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["blurred"].lower() for r in rows], [str(blur).lower()])
            with open(event["report"]["json"], encoding="utf-8") as f:
                self.assertEqual([i["blurred"] for i in json.load(f)["items"]], [blur])

    def test_P14_every_unsupported_extension_is_listed_as_rejected_and_nothing_is_imported(self):
        paths = []
        for ext in UNSUPPORTED_EXTENSIONS:
            paths.append(self.write("media/file" + ext, "not a picture"))
        answer = self.api("POST", "/api/clips", {"paths": paths})
        self.assertEqual(answer["clips"], [])
        self.assertEqual(sorted(r["path"] for r in answer["rejected"]), sorted(paths))
        # each says which format it is (HEIC, RAW, GIF or another) and what to do; none is the plain "not a supported file"
        self.assertEqual(
            {r["code"] for r in answer["rejected"]},
            {"error.media.heic", "error.media.raw", "error.media.gif", "error.media.format"},
        )

    def test_P15_a_damaged_clip_is_that_clips_error_and_the_other_clip_is_still_analysed(self):
        bad = self.make_clip("bad.mp4", n=60)
        damage(bad)
        good = self.import_clip("good.mp4")
        answer = self.api("POST", "/api/clips", {"paths": [bad]})
        bad_id = answer["clips"][0]["clip_id"]
        reader = self.analyze(bad_id, good)
        errors = [e for e in reader.events if e["type"] == "job.error"]
        self.assertEqual([(e["code"], e["clip_id"]) for e in errors], [("error.media.damaged", bad_id)])
        status = {c["clip_id"]: c["status"] for c in self.api("GET", "/api/state")["clips"]}
        self.assertEqual(status, {bad_id: "failed", good: "analyzed"})


class WorkerServer(server_tests.WorkerCase):
    def test_P16_with_the_engine_in_a_worker_a_damaged_clip_does_not_stop_the_batch(self):
        bad = self.make_clip("bad.mp4", n=60)
        damage(bad)
        good = self.import_clip("good.mp4")
        bad_id = self.api("POST", "/api/clips", {"paths": [bad]})["clips"][0]["clip_id"]
        reader = self.analyze(bad_id, good)
        errors = [e for e in reader.events if e["type"] == "job.error"]
        self.assertEqual([(e["code"], e["clip_id"]) for e in errors], [("error.media.damaged", bad_id)])
        status = {c["clip_id"]: c["status"] for c in self.api("GET", "/api/state")["clips"]}
        self.assertEqual(status, {bad_id: "failed", good: "analyzed"})


class Docs(unittest.TestCase):
    def test_P17_the_table_of_paths_names_only_tests_that_exist(self):
        import glob
        import re

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "docs", "PRIVACY-PATHS.md"), encoding="utf-8") as f:
            named = set(re.findall(r"`(test_\w+)`", f.read()))
        defined = set()
        for path in glob.glob(os.path.join(root, "tests", "**", "test_*.py"), recursive=True):
            with open(path, encoding="utf-8") as f:
                defined.update(re.findall(r"def (test_\w+)\(", f.read()))
        self.assertGreater(len(named), 40)
        self.assertEqual(sorted(named - defined), [])


if __name__ == "__main__":
    unittest.main()
