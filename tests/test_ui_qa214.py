"""Studio v2.1.4: what the fifth client QA pass (05/10/2026, qa-anon-5) found at the server. One test class per bug; each fails on
v2.1.3 and passes here."""

from __future__ import annotations

import json
import os
import unittest
from urllib.parse import quote

try:
    import numpy as np
    from PIL import Image

    from anonymizer.ui import server as server_module
    from tests import test_ui_server as server_tests
    from tests.fakes import FACE, Delayed, FixedModule
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


class QaCase(server_tests.StudioCase):
    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def photo(self, name="ok.jpg", size=(160, 120), folder="media") -> str:
        os.makedirs(os.path.join(self.dir, folder), exist_ok=True)
        path = os.path.join(self.dir, folder, name)
        Image.fromarray((np.random.RandomState(7).rand(size[1], size[0], 3) * 255).astype(np.uint8)).save(path)
        return path


class Vanishing(Delayed):
    """A detector that deletes the files it is told to while it works (a disk unmounted or a folder renamed during an analysis)."""

    def __init__(self, inner, schedule):
        super().__init__(inner, 0.05)
        self.schedule, self.calls_made = dict(schedule), 0

    def sweep(self, *args):
        self.calls_made += 1
        path = self.schedule.get(self.calls_made)  # the n-th photo the detector is given
        if path is not None:
            os.remove(path)
        return super().sweep(*args)


class OriginalGoesAwayDuringABatch(QaCase):
    """Bug 1: an original that vanished while its batch was being analysed (the stamp of the file was read after the engine had read it)
    made the whole job fail with "The engine stopped unexpectedly", left the rest of the batch unanalysed, and left the clip "failed"
    with an analysis that every later "Analyze" found current and would not bring back."""

    def test_a_clip_whose_original_is_gone_before_its_turn_fails_alone(self):
        paths = [self.photo(f"p{n}.jpg") for n in range(4)]
        self.server.session.modules = [Vanishing(self.modules[0], {1: paths[1]})]
        ids = [self.api("POST", "/api/clips", {"paths": [p]})["clips"][0]["clip_id"] for p in paths]
        reader = self.analyze(*ids)
        self.assertEqual(reader.wait_for("job.finished")["status"], "done")
        clips = {c["clip_id"]: c for c in self.api("GET", "/api/clips")["clips"]}
        self.assertEqual([clips[i]["status"] for i in ids], ["analyzed", "failed", "analyzed", "analyzed"])
        self.assertEqual(clips[ids[1]]["failure"]["code"], "error.source.missing")

    def test_an_original_that_goes_while_it_is_being_analysed_does_not_stop_the_job(self):
        paths = [self.photo(f"q{n}.jpg") for n in range(4)]
        self.server.session.modules = [
            Vanishing(self.modules[0], {2: paths[1]})
        ]  # the second photo, once the detector has it
        ids = [self.api("POST", "/api/clips", {"paths": [p]})["clips"][0]["clip_id"] for p in paths]
        reader = self.analyze(*ids)
        self.assertEqual(reader.wait_for("job.finished")["status"], "done")
        clips = {c["clip_id"]: c for c in self.api("GET", "/api/clips")["clips"]}
        self.assertEqual(
            [clips[i]["status"] for i in ids], ["analyzed"] * 4
        )  # what the engine read is what was analysed
        plan = self.api("POST", "/api/render/plan", {"clips": ids, "options": {}})
        self.assertEqual(
            [c["blocked"]["code"] if c["blocked"] else None for c in plan["clips"]],
            [None, "error.source.missing", None, None],  # and the export says which original is not there
        )

    def test_an_analysis_that_still_holds_brings_a_failed_clip_back(self):
        path = self.photo("back.jpg")
        [clip] = self.api("POST", "/api/clips", {"paths": [path]})["clips"]
        self.analyze(clip["clip_id"])
        session = self.server.session
        session.clips[
            clip["clip_id"]
        ].status = "failed"  # as a failed later run left it, with the analysis of the first
        session.clips[clip["clip_id"]].failure = {"code": "error.engine.crashed", "params": {}}
        self.analyze(clip["clip_id"])
        again = self.api("GET", "/api/clips")["clips"][0]
        self.assertEqual((again["status"], again.get("failure")), ("analyzed", None))


class WorkFolderIsNotLeftBehind(unittest.TestCase):
    """Bug 2: a Studio ended by `kill` or by a closed terminal (SIGTERM, SIGHUP) left its work folder in the temporary folder, with the
    face and plate crops of the analysed pictures in it, for good; so did a Studio that was killed outright."""

    def start(self, tmp):
        import site
        import subprocess
        import sys

        # HOME is the empty folder, so the child keeps the user site (a `pip install --user` install) by being told where it is
        env = dict(os.environ, TMPDIR=tmp, HOME=tmp, PYTHONUSERBASE=site.getuserbase())
        process = subprocess.Popen(
            [
                sys.executable,
                *(["-s"] if sys.flags.no_user_site else []),
                "-m",
                "anonymizer.ui",
                "--no-browser",
                "--port",
                "0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self.addCleanup(lambda: (process.poll() is None and process.kill(), process.stdout.close()))
        process.stdout.readline()  # the address: the server is up, its folder made
        return process

    def folders(self, tmp):
        return sorted(n for n in os.listdir(tmp) if n.startswith("anonymizer-studio-"))

    def test_terminate_and_hangup_delete_the_work_folder(self):
        import signal
        import tempfile

        for number in (signal.SIGTERM, signal.SIGHUP):
            with tempfile.TemporaryDirectory() as tmp:
                process = self.start(tmp)
                self.assertEqual(len(self.folders(tmp)), 1)
                process.send_signal(number)
                self.assertEqual(process.wait(30), 0, number)
                self.assertEqual(self.folders(tmp), [], number)

    def test_a_killed_studio_folder_goes_at_the_next_start_and_only_that_one(self):
        import subprocess
        import sys
        import tempfile

        from anonymizer.ui.session import OWNER_FILE, remove_stale_work_folders

        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        with tempfile.TemporaryDirectory() as tmp:
            made = {}
            for key, pid in (("dead", dead.pid), ("alive", os.getpid()), ("unmarked", None), ("junk", "x")):
                folder = os.path.join(tmp, f"anonymizer-studio-{key}")
                os.makedirs(folder)
                if pid is not None:
                    with open(os.path.join(folder, OWNER_FILE), "w") as f:
                        f.write(str(pid))
                made[key] = folder
            other = os.path.join(tmp, "somebody-elses")
            os.makedirs(other)
            removed = remove_stale_work_folders(tmp)
            self.assertEqual(removed, [made["dead"]])
            self.assertEqual(
                sorted(os.listdir(tmp)),
                ["anonymizer-studio-alive", "anonymizer-studio-junk", "anonymizer-studio-unmarked", "somebody-elses"],
            )


class HostedDemoIsSomebodyElsesComputer(QaCase):
    """Bug 3: in the hosted demo a visitor could add any number of videos (only photos were counted), each up to 100 MB, and fill the
    server's disk; a file name of 300 characters was a 500 whose text named the server's folder; and a failure told its visitor the
    server's folders and a traceback (a job that failed, an original that went missing)."""

    demo = True

    def video_bytes(self):
        with open(self.make_clip("v.mp4", n=6), "rb") as f:
            return f.read()

    def test_no_more_than_ten_files_in_all_videos_included(self):
        data = self.video_bytes()
        for n in range(10):
            self.assertEqual(self.request("PUT", f"/api/upload?name=v{n}.mp4", raw=data)[0], 200)
        status, _, body = self.request("PUT", "/api/upload?name=one-more.mp4", raw=data)
        self.assertEqual((status, json.loads(body)["params"]), (422, {"limit": "clips", "max": 10}))
        uploads = os.listdir(os.path.join(self.dir, "work", "uploads"))
        self.assertEqual(len(uploads), 10)  # the refused one is not kept
        self.assertEqual(self.api("GET", "/api/state")["demo_limits"]["max_clips"], 10)

    def test_a_very_long_file_name_is_shortened_and_keeps_its_extension(self):
        data = self.video_bytes()
        for name in ("x" * 300 + ".mp4", chr(233) * 200 + ".mp4"):
            status, _, body = self.request("PUT", "/api/upload?name=" + quote(name), raw=data)
            self.assertEqual(status, 200, body[:300])
            [clip] = json.loads(body)["clips"]
            self.assertTrue(clip["name"].endswith(".mp4") and len(clip["name"].encode()) <= 120, clip["name"])

    def test_a_failure_names_no_folder_of_the_server_and_carries_no_traceback(self):
        with open(self.photo("gone.jpg"), "rb") as f:
            [clip] = json.loads(self.request("PUT", "/api/upload?name=gone.jpg", raw=f.read())[2])["clips"]
        self.analyze(clip["clip_id"])
        os.remove(os.path.join(self.dir, "work", "uploads", "gone.jpg"))  # the visitor's file is gone from the server
        plan = self.api("POST", "/api/render/plan", {"clips": [clip["clip_id"]], "options": {}})
        self.assertEqual(plan["clips"][0]["blocked"], {"code": "error.source.missing", "params": {"path": "gone.jpg"}})
        status, _, body = self.request("GET", f"/api/clips/{clip['clip_id']}/preview?frame=0")
        self.assertEqual(status, 422)
        self.assertNotIn(self.dir, body.decode())
        self.assertEqual(json.loads(body)["params"], {"path": "gone.jpg"})
        # an event: the text and the folder of a failure are not kept for the stream
        self.server.hub.emit(
            "job.error",
            code="error.engine.crashed",
            clip_id="c1",
            params={"path": self.dir + "/work/out"},
            detail=f"Traceback ... {self.dir}",
        )
        last = json.loads(self.server.hub._ring[-1][1])
        self.assertEqual((last["detail"], last["params"]), ("", {"path": "out"}))


class Hardening(QaCase):
    """Bug 4: the Studio sent no Content-Security-Policy, Referrer-Policy or X-Frame-Options, and a connection that sent half a request
    held its thread for ever (a hosted demo on the internet runs out of them)."""

    def test_every_kind_of_answer_carries_the_headers(self):
        clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        self.analyze(clip)
        for path, cookie in (
            ("/", False),
            (f"/?t={server_tests.TOKEN}", False),
            ("/static/app.js", False),
            ("/i18n/en.json", False),
            ("/nope", False),
            ("/api/state", True),
            ("/api/state", False),  # a refusal too
            (f"/api/clips/{clip}/thumb", True),
            (f"/api/clips/{clip}/source", True),
            (f"/api/clips/{clip}/preview?frame=0", True),
            (f"/api/clips/{clip}/analysis.json", True),
        ):
            status, response, _ = self.request("GET", path, cookie=cookie)
            self.assertIn(status, (200, 302, 403, 404), (path, status))
            headers = {k.lower(): v for k, v in response.getheaders()}
            policy = headers.get("content-security-policy", "")
            for must in (
                "default-src 'none'",
                "script-src 'self'",
                "connect-src 'self'",
                "frame-ancestors 'none'",
                "base-uri 'none'",
            ):
                self.assertIn(must, policy, path)
            self.assertNotIn("unsafe-eval", policy)
            self.assertNotRegex(policy, r"script-src[^;]*unsafe-inline")
            self.assertEqual(headers.get("referrer-policy"), "no-referrer", path)
            self.assertEqual(headers.get("x-frame-options"), "DENY", path)
            self.assertEqual(headers.get("x-content-type-options"), "nosniff", path)

    def test_a_connection_that_stops_in_the_middle_of_a_request_is_closed(self):
        import socket
        from unittest import mock

        self.assertTrue(
            10 <= (server_module.Handler.timeout or 0) <= 600, server_module.Handler.timeout
        )  # there is one by default
        with mock.patch.object(server_module.Handler, "timeout", 0.6):
            for half in (
                b"GET /api/state HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nX-a: " % self.port,
                b"PUT /api/upload?name=a.jpg HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Length: 500\r\nCookie: anon_session_%d=%s\r\n\r\nabc"
                % (self.port, self.port, server_tests.TOKEN.encode()),
            ):
                with socket.create_connection(("127.0.0.1", self.port)) as s:
                    s.settimeout(5)
                    s.sendall(half)
                    while s.recv(
                        1000
                    ):  # an answer, or none, and then the server hangs up; without a timeout this waits for ever
                        pass
        self.assertEqual(self.request("GET", "/api/state")[0], 200)  # and it is up


class FullDestinationIsNamedAsSuch(QaCase):
    """Bug 6: a photo that could not be written for want of room was reported as "The camera and descriptive data could not be removed",
    and a write that ran out of space as "Cannot write to ..." (the checks of the photo's own file see a file cut short)."""

    def render_with(self, error, free):
        from collections import namedtuple
        from unittest import mock

        from anonymizer.ui import engine as engine_module

        clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        reader = self.analyze(clip)
        usage = namedtuple("usage", "total used free")
        with (
            mock.patch.object(engine_module.LocalEngine, "render", side_effect=error),
            mock.patch("anonymizer.ui.session.shutil.disk_usage", return_value=usage(10**6, 10**6, free)),
        ):
            self.api("POST", "/api/render", {"clips": [clip], "options": {"reports": {"pdf": False}}}, 202)
            failure = reader.wait_for("job.error")
            reader.wait_for("job.finished")
        self.wait_idle()
        return failure

    def test_a_photo_written_short_on_a_full_folder_says_the_space(self):
        import anonymizer

        failure = self.render_with(
            anonymizer.AnonymizerError("error.photo.metadata_not_removed", {}, "JPEG not understood"), free=0
        )
        self.assertEqual(failure["code"], "error.output.disk_full")
        self.assertIn("media", failure["params"]["path"] + "media")  # the folder, for the message

    def test_with_room_the_engines_own_reason_stands(self):
        import anonymizer

        failure = self.render_with(anonymizer.AnonymizerError("error.photo.metadata_not_removed", {}, "x"), free=10**12)
        self.assertEqual(failure["code"], "error.photo.metadata_not_removed")

    def test_a_write_that_ran_out_of_space_is_not_a_permission_problem(self):
        import errno

        failure = self.render_with(OSError(errno.ENOSPC, "No space left on device"), free=10**12)
        self.assertEqual(failure["code"], "error.output.disk_full")
        failure = self.render_with(OSError(errno.EACCES, "Permission denied"), free=10**12)
        self.assertEqual(failure["code"], "error.output.not_writable")

    def test_no_partial_file_stays_in_the_destination(self):
        """After a write that failed for want of room, nothing that looks like a result may stay in the folder."""
        import errno
        from unittest import mock

        from anonymizer.ui import engine as engine_module

        clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        reader = self.analyze(clip)
        dest = os.path.join(self.dir, "out")

        def render_that_runs_out_of_room(clip_id, analysis, target, on_event, token):
            with open(
                os.path.join(os.path.dirname(target), f".{os.path.basename(target)}.anonymizer-partial"), "wb"
            ) as f:
                f.write(b"half a picture")  # what the engine had written when the disk filled
            raise OSError(errno.ENOSPC, "No space left on device")

        with mock.patch.object(engine_module.LocalEngine, "render", side_effect=render_that_runs_out_of_room):
            self.api(
                "POST",
                "/api/render",
                {"clips": [clip], "options": {"destination": dest, "reports": {"pdf": False}}},
                202,
            )
            self.assertEqual(reader.wait_for("job.error")["code"], "error.output.disk_full")
            reader.wait_for("job.finished")
        self.wait_idle()
        self.assertEqual(os.listdir(dest), [])  # no picture, no report, no hidden partial


class ExportProgressSurvivesAReload(QaCase):
    """Polish of 2.1.3 ("found and not fixed"): a page reloaded in the middle of an export did not show the render's progress or the files of
    the clips already written, though the job and its files were fine."""

    def test_the_summary_of_a_clip_carries_its_export(self):
        import time

        paths = [self.make_clip(f"e{n}.mp4", n=150) for n in range(2)]
        ids = [self.api("POST", "/api/clips", {"paths": [p]})["clips"][0]["clip_id"] for p in paths]
        self.analyze(*ids)
        self.assertNotIn("render", self.api("GET", "/api/clips")["clips"][0])  # nothing exported yet
        dest = os.path.join(self.dir, "out")
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        self.api(
            "POST", "/api/render", {"clips": ids, "options": {"destination": dest, "reports": {"pdf": False}}}, 202
        )
        seen = set()
        end = time.time() + 60
        while time.time() < end and self.api("GET", "/api/state")["job"] is not None:
            for clip in self.api("GET", "/api/clips")["clips"]:
                seen.add((clip["clip_id"], (clip.get("render") or {}).get("phase"), clip["status"]))
        reader.wait_for("job.finished")
        phases = {phase for _, phase, _ in seen}
        self.assertTrue({"queued", "render"} & phases, seen)
        done = [c.get("render") for c in self.api("GET", "/api/clips")["clips"]]
        self.assertEqual([d["phase"] for d in done], ["done", "done"])
        self.assertTrue(all(d["outputs"][0].startswith(dest) and d["report"]["csv"] for d in done), done)

    def test_a_new_analysis_clears_what_was_written(self):
        [path] = [self.make_clip("again.mp4", n=20)]
        [clip] = self.api("POST", "/api/clips", {"paths": [path]})["clips"]
        self.analyze(clip["clip_id"])
        self.api(
            "POST",
            "/api/render",
            {
                "clips": [clip["clip_id"]],
                "options": {"destination": os.path.join(self.dir, "o"), "reports": {"pdf": False}},
            },
            202,
        )
        self.wait_idle()
        self.assertEqual(self.api("GET", "/api/clips")["clips"][0]["render"]["phase"], "done")
        self.analyze_again(clip["clip_id"])
        self.assertNotIn("render", self.api("GET", "/api/clips")["clips"][0])

    def analyze_again(self, clip_id):
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        self.api(
            "POST",
            "/api/job/analyze",
            {"clips": [clip_id], "options": {"gap_fill": "hold"}, "discard": True},
            202,
        )
        reader.wait_for("job.finished")
        self.wait_idle()


class PipeNamedLikeAPhotoDoesNotFreezeTheImport(QaCase):
    """Bug 7: a named pipe (or a socket, or a device) with a photo's name in an imported folder was opened like a file, and opening a pipe
    waits for ever: the import never answered and held the lock that every other import needs."""

    def test_it_is_refused_and_the_rest_of_the_folder_is_imported(self):
        import threading

        self.photo("a.jpg", folder="media/odd")
        os.mkfifo(os.path.join(self.dir, "media", "odd", "pipe.jpg"))
        os.symlink("/nonexistent", os.path.join(self.dir, "media", "odd", "dangling.jpg"))
        answer = {}

        def go():
            answer["r"] = self.request("POST", "/api/clips", {"paths": [os.path.join(self.dir, "media", "odd")]})

        thread = threading.Thread(target=go, daemon=True)
        thread.start()
        thread.join(15)
        if thread.is_alive():  # let the hung request go, so that the server can stop
            with open(os.path.join(self.dir, "media", "odd", "pipe.jpg"), "w") as unblock:
                unblock.write("x")
            thread.join(15)
            self.fail("the import of a folder with a pipe in it did not answer")
        status, _, data = answer["r"]
        body = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual([c["name"] for c in body["clips"]], ["a.jpg"])
        self.assertEqual(
            sorted(os.path.basename(r["path"]) for r in body["rejected"] if r["path"]), ["dangling.jpg", "pipe.jpg"]
        )


class FilesAndFoldersWithoutPermission(QaCase):
    """Bug 8: a photo the person may not read was refused as "damaged or incomplete", and a sub-folder they may not open was skipped in silence
    (the import said nothing of the photos that were in it)."""

    def test_each_is_said_for_what_it_is(self):
        if os.geteuid() == 0:
            self.skipTest("root can read everything")
        self.photo("ok.jpg", folder="media/trip")
        locked = self.photo("secret.jpg", folder="media/trip")
        os.chmod(locked, 0)
        os.makedirs(os.path.join(self.dir, "media", "trip", "private"))
        self.photo("inside.jpg", folder="media/trip/private")
        os.chmod(os.path.join(self.dir, "media", "trip", "private"), 0)
        self.addCleanup(os.chmod, os.path.join(self.dir, "media", "trip", "private"), 0o755)
        answer = self.api("POST", "/api/clips", {"paths": [os.path.join(self.dir, "media", "trip")]})
        self.assertEqual([c["name"] for c in answer["clips"]], ["ok.jpg"])
        said = sorted((os.path.basename(r["path"]), r["code"]) for r in answer["rejected"])
        self.assertEqual(said, [("private", "error.browse.denied"), ("secret.jpg", "error.media.denied")])


class PdfSaysWhoDecidedWhenThePersonOverruled(unittest.TestCase):
    """Polish of 2.1.3: the PDF read "Blurred by default. Please check." next to "Left visible" for an item the person turned off."""

    def rows(self):
        import anonymizer

        return [
            anonymizer.ReportRow(
                "a.jpg", 0, "plate", "check", False, "plate.reason.vehicle_text", 0, 0, 1, 0.4, decided_by="person"
            ),
            anonymizer.ReportRow(
                "a.jpg", 1, "face", "check", True, "face.reason.low_verify", 0, 0, 1, 0.5, decided_by="person"
            ),
            anonymizer.ReportRow("a.jpg", 2, "face", "blur", True, "face.reason.verified", 0, 0, 1, 0.9),
            anonymizer.ReportRow(
                "a.jpg", 3, "manual", "blur", True, "manual.reason.added", 0, 0, 1, 1.0, decided_by="person"
            ),
        ]

    def test_the_reason_column_names_the_person_only_where_they_changed_it(self):
        import json
        from datetime import date

        from anonymizer.ui import pdf

        with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "src", "anonymizer", "ui", "i18n", "en.json"
            ),
            encoding="utf-8",
        ) as f:
            texts = json.load(f)
        page = pdf.report_html("a.jpg", self.rows(), texts, False, date(2026, 10, 5))
        self.assertIn("Please check. You chose to leave it visible.", page)
        self.assertIn("Please check. You chose to blur it.", page)
        self.assertEqual(
            page.count("You chose to"), 2
        )  # not on the engine's own decision, not on an area the person added


class DefaultDestinationIsAFolderPath(QaCase):
    """Bug 9: the default destination of uploaded files accepted a NUL and ten thousand characters; the plan then called it writable and the
    export of every upload failed later with a worker that "crashed"."""

    def test_a_control_character_or_an_absurd_length_is_refused(self):
        for bad in ("a\x00b", "folder\nwith a line feed", "x" * 5000):
            status, _, body = self.request("PUT", "/api/settings", {"default_destination": bad})
            self.assertEqual(
                (status, json.loads(body)["params"]), (400, {"field": "default_destination"}), repr(bad[:20])
            )
        self.assertEqual(self.request("PUT", "/api/settings", {"default_destination": "~/Anonymized photos"})[0], 200)


class BoxesErrorNamesTheFieldAtFault(QaCase):
    """Bug 10: `GET .../boxes?from=0&to=1000000000` answered 400 and named `from` (the same class as the area errors of 2.1.3)."""

    def test_to_is_named_when_it_is_the_one(self):
        clip = self.api("POST", "/api/clips", {"paths": [self.make_clip("b.mp4", n=20)]})["clips"][0]["clip_id"]
        self.analyze(clip)
        for query, field in (
            ("from=0&to=1000000000", "to"),
            ("from=5&to=2", "to"),
            ("from=-5&to=2", "from"),
            ("from=1000&to=1001", "from"),
        ):
            status, _, body = self.request("GET", f"/api/clips/{clip}/boxes?{query}")
            self.assertEqual((status, json.loads(body)["params"]["field"]), (400, field), query)
        self.assertEqual(self.request("GET", f"/api/clips/{clip}/boxes?from=0&to=5")[0], 200)


class JobSummaryCountsAClipThatTheEngineLost(QaCase):
    """Bug 11: when the engine process died the job's summary said `failed: 0` though the clip it was working on had failed."""

    def test_the_clip_is_counted(self):
        from unittest import mock

        from anonymizer.ui import engine as engine_module

        clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        with mock.patch.object(
            engine_module.LocalEngine,
            "analyze",
            side_effect=engine_module.EngineFailure("error.engine.crashed", {}, "exit code -9"),
        ):
            self.api("POST", "/api/job/analyze", {"clips": [clip]}, 202)
            finished = reader.wait_for("job.finished")
        self.wait_idle()
        self.assertEqual(
            (finished["status"], finished["summary"]["failed"], finished["summary"]["analyzed"]), ("failed", 1, 0)
        )
        self.assertEqual(self.api("GET", "/api/clips")["clips"][0]["failure"]["code"], "error.engine.crashed")


class SourceNameWithAControlCharacter(QaCase):
    """Polish of 2.1.3 ("found and not fixed"): a source file name with a line feed in it (legal on Linux) gave an output name with one."""

    def test_the_output_name_has_none(self):
        path = os.path.join(self.dir, "media", "two\nlines.jpg")
        Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(path)
        clip = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip)
        plan = self.api(
            "POST", "/api/render/plan", {"clips": [clip], "options": {"destination": os.path.join(self.dir, "o")}}
        )
        self.assertEqual(os.path.basename(plan["clips"][0]["output"]), "two_lines_anonymized.jpg")


class DemoClipFailureNamesNoFolder(QaCase):
    """Bug 3, the clip's own summary: the reason a clip failed (`failure.params.path`) carried the server's folder in the hosted demo."""

    demo = True

    def test_the_failure_of_a_clip_whose_upload_went_missing(self):
        with open(self.photo("lost.jpg"), "rb") as f:
            [clip] = json.loads(self.request("PUT", "/api/upload?name=lost.jpg", raw=f.read())[2])["clips"]
        os.remove(os.path.join(self.dir, "work", "uploads", "lost.jpg"))
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": [clip["clip_id"]]}, 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        failed = self.api("GET", "/api/clips")["clips"][0]
        self.assertEqual(failed["failure"], {"code": "error.source.missing", "params": {"path": "lost.jpg"}})
        self.assertNotIn(self.dir, json.dumps(self.api("GET", "/api/state")))
