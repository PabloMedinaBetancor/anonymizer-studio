"""Studio v2.1.5: what the sixth client QA pass (05/10/2026, qa-anon-6) found at the server. One test class per bug; each fails on
v2.1.4 and passes here."""

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


class CsvReportOpensInExcel(QaCase):
    """Bug 1: the CSV report was UTF-8 with no byte-order mark, and Excel (a double click) showed a file name with an accent wrongly."""

    def test_the_csv_of_the_api_and_of_the_studio_start_with_the_mark(self):
        import anonymizer

        row = anonymizer.ReportRow(
            "\u00d1u d\u00eda.jpg", 0, "face", "blur", True, "face.reason.verified", 0, 0, 1, 0.9
        )
        path = os.path.join(self.dir, "work", "r.csv")
        anonymizer.Report([row]).to_csv(path)
        with open(path, "rb") as f:
            raw = f.read()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(raw.count(b"\xef\xbb\xbf"), 1)
        self.assertIn("\u00d1u d\u00eda.jpg", raw.decode("utf-8-sig"))
        clip = self.api("POST", "/api/clips", {"paths": [self.photo("\u00d1and\u00fa d\u00eda.jpg")]})["clips"][0][
            "clip_id"
        ]
        self.analyze(clip)
        status, response, data = self.get(f"/api/clips/{clip}/report?format=csv")
        self.assertEqual(status, 200)
        self.assertTrue(data.startswith(b"\xef\xbb\xbfkind,") or data.startswith(b"\xef\xbb\xbf"), data[:40])
        self.assertIn("\u00d1and\u00fa d\u00eda.jpg", data.decode("utf-8-sig"))

    def test_the_csv_written_by_an_export_starts_with_the_mark(self):
        path = self.photo("\u00d1u d\u00eda.jpg")
        clip = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip)
        out = os.path.join(self.dir, "out")
        reader = server_tests.EventReader(self.port)
        self.addCleanup(reader.close)
        self.api(
            "POST",
            "/api/render",
            {"clips": [clip], "options": {"destination": out, "reports": {"csv": True, "json": False, "pdf": False}}},
            202,
        )
        reader.wait_for("job.finished")
        [name] = [n for n in os.listdir(out) if n.endswith(".csv")]
        with open(os.path.join(out, name), "rb") as f:
            self.assertTrue(f.read().startswith(b"\xef\xbb\xbf"))


class PercentIsWrittenAsThePageWritesIt(unittest.TestCase):
    """Bug 2: the PDF wrote "83 %" in every language, the lists of the page "83%" (English, Spanish, Italian, Portuguese, Chinese) or
    "83 %" (French, German): one report read two ways."""

    def test_the_pdf_follows_the_language_of_the_page(self):
        from datetime import date

        import anonymizer
        from anonymizer.ui import pdf

        row = anonymizer.ReportRow("a.jpg", 0, "face", "blur", True, "face.reason.verified", 0, 0, 1, 0.834)
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "anonymizer", "ui", "i18n")
        for language in ("en", "es", "fr", "de", "pt", "it", "zh-Hans"):
            with open(os.path.join(folder, f"{language}.json"), encoding="utf-8") as f:
                texts = json.load(f)
            page = pdf.report_html("a.jpg", [row], texts, False, date(2026, 10, 5))
            written = texts["review.cert.sure"].replace("{percent}", "83")
            self.assertIn(f"<td class=n>{written}</td>", page, language)


class TwoNamesOfOneFileAreOneClip(QaCase):
    """Bug 3: a link to a photo, next to the photo, imported as two clips of one file (two analyses, two outputs, two rows in every report)."""

    def test_a_symlink_and_a_hard_link_are_not_a_second_clip(self):
        path = self.photo("real.jpg")
        folder = os.path.dirname(path)
        os.symlink(path, os.path.join(folder, "link.jpg"))
        os.link(path, os.path.join(folder, "hard.jpg"))
        answer = self.api("POST", "/api/clips", {"paths": [folder]})
        self.assertEqual(len(answer["clips"]), 1, [c["name"] for c in answer["clips"]])
        self.assertEqual(answer["rejected"], [])
        again = self.api("POST", "/api/clips", {"paths": [os.path.join(folder, "link.jpg")]})
        self.assertEqual([c["clip_id"] for c in again["clips"]], [answer["clips"][0]["clip_id"]])
        self.assertEqual(len(self.api("GET", "/api/clips")["clips"]), 1)

    def test_two_different_photos_are_still_two_clips(self):
        self.photo("a.jpg", seed=1)
        path = self.photo("b.jpg", seed=2)
        answer = self.api("POST", "/api/clips", {"paths": [os.path.dirname(path)]})
        self.assertEqual(len(answer["clips"]), 2)


class SessionCookieIsSecureBehindTls(QaCase):
    """Bug 4: behind a proxy that ends TLS (the hosted demo) the session cookie had no Secure flag."""

    def cookie(self, **headers):
        status, response, _ = self.request("GET", f"/?t={server_tests.TOKEN}", cookie=False, headers=headers)
        self.assertEqual(status, 302)
        return response.getheader("Set-Cookie")

    def test_secure_only_when_the_proxy_says_https(self):
        self.assertNotIn("Secure", self.cookie())
        self.assertIn("; Secure", self.cookie(**{"X-Forwarded-Proto": "https"}))
        self.assertIn("; Secure", self.cookie(**{"X-Forwarded-Proto": "HTTPS, http"}))
        self.assertNotIn("Secure", self.cookie(**{"X-Forwarded-Proto": "http"}))
        self.assertIn("HttpOnly", self.cookie(**{"X-Forwarded-Proto": "https"}))


class AnalysisSaysWhatItWasMadeWith(QaCase):
    """Bug 5: "Analyze 8 clips" counted the clips that were already analysed with these settings (the server skips them at once)."""

    def test_the_summary_and_the_event_carry_the_settings(self):
        clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        reader = self.analyze(clip)
        event = next(e for e in reader.events if e["type"] == "clip.analyzed")
        [summary] = self.api("GET", "/api/clips")["clips"]
        expected = {"faces", "plates", "thoroughness", "detection", "gap_fill"}
        self.assertEqual(set(summary["analysis"]["settings"]), expected)
        self.assertEqual(event["settings"], summary["analysis"]["settings"])

    def test_the_settings_are_gone_when_the_file_changed(self):
        path = self.photo()
        clip = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip)
        self.photo(size=(200, 150))  # the file is not the one that was analysed
        [summary] = self.api("GET", "/api/clips")["clips"]
        self.assertIsNone(summary["analysis"]["settings"])


class FoldersThatAreLinks(QaCase):
    """Bug 6: a link to a folder inside a folder that was imported was skipped in silence, with the photos in it (the list of files
    of the Import dialog showed it with its count)."""

    def test_the_photos_behind_a_linked_folder_are_imported_and_a_loop_is_not_walked_for_ever(self):
        real = os.path.join(self.dir, "real")
        top = os.path.join(self.dir, "top")
        os.makedirs(os.path.join(real, "inner"))
        os.makedirs(top)
        self.photo("a.jpg", folder="real", seed=1)
        self.photo("b.jpg", folder="real/inner", seed=2)
        self.photo("own.jpg", folder="top", seed=3)
        os.symlink(real, os.path.join(top, "linked"))
        os.symlink(top, os.path.join(top, "loop"))
        answer = self.api("POST", "/api/clips", {"paths": [top]})
        self.assertEqual(sorted(c["name"] for c in answer["clips"]), ["a.jpg", "b.jpg", "own.jpg"])
        self.assertEqual(answer["rejected"], [])


class OutputNamesDifferBeyondTheCase(unittest.TestCase):
    """Bug 7: two outputs that differ only in the case of their letters were two names to the Studio and one file to Windows, macOS and
    the USB disks people copy to: the second clip failed at its turn with "a file with that name exists" (the engine never replaces a
    file), after the plan had promised both."""

    def setUp(self):
        import tempfile

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name

    def test_a_name_that_differs_in_case_or_in_accent_form_is_taken(self):
        from anonymizer.ui.session import free_name

        first = free_name(os.path.join(self.folder, "IMG_1_anonymized.jpg"))
        second = free_name(os.path.join(self.folder, "img_1_anonymized.jpg"), [first])
        self.assertEqual(os.path.basename(second), "img_1_anonymized-2.jpg")
        composed, decomposed = "caf\u00e9.jpg", "cafe\u0301.jpg"
        third = free_name(os.path.join(self.folder, decomposed), [os.path.join(self.folder, composed)])
        self.assertEqual(os.path.basename(third), "cafe\u0301-2.jpg")

    def test_a_file_already_there_in_other_case_is_taken_too(self):
        from anonymizer.ui.session import free_name

        with open(os.path.join(self.folder, "Photo_anonymized.jpg"), "wb") as f:
            f.write(b"x")
        self.assertEqual(
            os.path.basename(free_name(os.path.join(self.folder, "photo_anonymized.jpg"))), "photo_anonymized-2.jpg"
        )

    def test_plain_names_are_left_alone(self):
        from anonymizer.ui.session import free_name

        path = os.path.join(self.folder, "plain.jpg")
        self.assertEqual(free_name(path), path)
        self.assertEqual(
            free_name(os.path.join(self.folder, "no", "such", "folder.jpg")),
            os.path.join(self.folder, "no", "such", "folder.jpg"),
        )


class StudioSaysItsVersion(unittest.TestCase):
    """Bug 10: `python3 -m anonymizer.ui --version` was "unrecognized arguments" (the first thing a person types to see what they have)."""

    def test_the_flag_prints_the_version_and_stops(self):
        import contextlib
        import io

        import anonymizer
        from anonymizer.ui import server

        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as stopped:
            server.main(["--version"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(out.getvalue().strip(), f"Anonymizer Studio {anonymizer.__version__}")


class PhotoWithTooManyPixelsIsNotCalledDamaged(QaCase):
    """Bug 11: a valid photo with more pixels than Pillow's safety limit allows (a 200 MP phone photo) was refused as "damaged or
    incomplete"; it is too big, and the person is told so, with its size and the limit."""

    def test_the_refusal_names_the_size_and_the_limit(self):
        path = os.path.join(self.dir, "media", "panorama.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.new("L", (20000, 20000)).save(path)  # 400 MP, a few kilobytes
        answer = self.api("POST", "/api/clips", {"paths": [path]})
        self.assertEqual(answer["clips"], [])
        [refusal] = answer["rejected"]
        self.assertEqual(refusal["code"], "error.media.too_large")
        self.assertEqual(refusal["params"]["name"], "panorama.png")
        self.assertEqual(refusal["params"]["megapixels"], 400)
        self.assertEqual(refusal["params"]["limit"], 179)

    def test_a_damaged_photo_is_still_called_damaged(self):
        path = os.path.join(self.dir, "media", "broken.jpg")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"not a photo")
        [refusal] = self.api("POST", "/api/clips", {"paths": [path]})["rejected"]
        self.assertEqual(refusal["code"], "error.media.unreadable_photo")


class EventStreamTakesAbsurdLastEventId(QaCase):
    """Bug 13: `Last-Event-ID` with thousands of digits (Python's limit on integer strings) answered "internal error" (500), the class of
    fault that 2.1.3 closed in the addresses of items."""

    def test_a_number_of_five_thousand_digits_opens_the_stream_like_none(self):
        import http.client

        for value in ("9" * 5000, "9" * 300, "7"):
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            self.addCleanup(conn.close)
            conn.request(
                "GET",
                "/api/events",
                headers={"Cookie": f"anon_session_{self.port}={server_tests.TOKEN}", "Last-Event-ID": value},
            )
            response = conn.getresponse()
            self.assertEqual(
                (response.status, response.getheader("Content-Type")), (200, "text/event-stream"), len(value)
            )
            conn.close()


class RangeHeaderWithAbsurdNumbers(QaCase):
    """Bug 13 (same fault, other header): `Range: bytes=0-<5,000 digits>` on the source of a clip was a 500 as well."""

    def test_a_range_of_five_thousand_digits_is_a_range_not_an_internal_error(self):
        clip = self.api("POST", "/api/clips", {"paths": [self.photo("range.jpg")]})["clips"][0]["clip_id"]
        size = os.path.getsize(self.photo("range.jpg"))
        huge = "9" * 5000
        for spec, status in ((f"bytes=0-{huge}", 206), (f"bytes=-{huge}", 206), (f"bytes={huge}-", 416)):
            code, response, data = self.get(f"/api/clips/{clip}/source", headers={"Range": spec})
            self.assertEqual(code, status, (spec[:20], data[:100]))
            if code == 206:
                self.assertEqual(len(data), size)


class ClipListSurvivesImportsAtTheSameTime(QaCase):
    """Bug 14 (found by 8 threads of random calls for seven minutes): `GET /api/clips` walked the list of clips with no lock while another
    request added or removed one, and now and then answered "dictionary changed size during iteration" (500)."""

    def test_asking_for_the_list_while_clips_come_and_go_never_fails(self):
        import threading
        import time

        files = [self.photo(f"r{n}.jpg", seed=n, size=(16, 16)) for n in range(4)]
        crowd = [self.photo(f"crowd{n}.jpg", seed=n, size=(16, 16), folder="crowd") for n in range(150)]
        self.api(
            "POST", "/api/clips", {"paths": [os.path.dirname(crowd[0])]}
        )  # a long list: the walk over it takes a while
        from unittest import mock

        from anonymizer.ui.session import Session

        original = Session.summary

        def slow(
            session, clip
        ):  # the walk over a long list of clips, made long enough for another request to come in the middle of it
            time.sleep(0.0005)
            return original(session, clip)

        patch = mock.patch.object(Session, "summary", slow)
        patch.start()
        self.addCleanup(patch.stop)
        stop = threading.Event()
        failures = []

        def churn():
            while not stop.is_set():
                for path in files:
                    answer = self.api("POST", "/api/clips", {"paths": [path]})
                    for clip in answer["clips"]:
                        self.request("DELETE", f"/api/clips/{clip['clip_id']}")

        worker = threading.Thread(target=churn, daemon=True)
        worker.start()
        try:
            for _ in range(12):
                status, _, data = self.get("/api/clips")
                if status != 200:
                    failures.append((status, data[:120]))
        finally:
            stop.set()
            worker.join(30)
        self.assertEqual(failures, [])


class ThumbnailOfAFileThatIsGone(QaCase):
    """Bug 15 (found by a race test of the hosted demo: two 500 in 60,000 requests): the picture of a clip whose original is gone, asked for
    before it was ever made, was "internal error" (the engine says "unsupported: FileNotFoundError"); every other picture of such a clip
    says the original is missing (2.1.4, entry 1)."""

    def test_the_thumbnail_says_the_original_is_missing(self):
        for path in (self.photo("going.jpg"), self.make_clip("going.mp4")):
            clip = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
            os.remove(path)
            status, _, data = self.get(f"/api/clips/{clip}/thumb")
            self.assertEqual(status, 422, (path, data[:200]))
            self.assertEqual(json.loads(data)["code"], "error.source.missing")

    def test_a_file_that_goes_while_the_picture_is_being_made_is_just_as_missing(self):
        """The engine opens the file twice (its header, then its frame): when the file goes in between, it says "unsupported:
        FileNotFoundError", which was a 500."""
        from unittest import mock

        import anonymizer

        path = self.photo("midway.jpg")
        clip = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]

        def goes_midway(target, width):
            os.remove(target)
            raise anonymizer.UnsupportedMedia(
                detail=f"FileNotFoundError: [Errno 2] No such file or directory: {target!r}"
            )

        with mock.patch("anonymizer.thumbnail_jpeg", goes_midway):
            status, _, data = self.get(f"/api/clips/{clip}/thumb")
        self.assertEqual((status, json.loads(data)["code"]), (422, "error.source.missing"))

    def test_the_thumbnail_of_a_file_that_is_there_is_a_picture(self):
        clip = self.api("POST", "/api/clips", {"paths": [self.photo("here.jpg")]})["clips"][0]["clip_id"]
        status, response, data = self.get(f"/api/clips/{clip}/thumb")
        self.assertEqual((status, response.getheader("Content-Type")), (200, "image/jpeg"))


class TwoUploadsOfOneNameAreTwoFiles(QaCase):
    """Bug 18: the name of an uploaded file was chosen, and the file made later, in two steps: two uploads of one name at the same moment (two
    windows, a script) were given one file and wrote into it together."""

    def test_the_second_upload_gets_its_own_name_even_when_the_first_is_slow(self):
        import threading
        import time
        from unittest import mock

        from anonymizer.ui import session as session_module

        real = session_module.free_name

        def slow(
            path, *args
        ):  # the time between choosing a name and making the file, made long enough for the other upload to come
            chosen = real(path, *args)
            time.sleep(0.05)
            return chosen

        targets = []
        gate = threading.Barrier(2)

        def take():
            gate.wait()
            targets.append(self.server.session.upload_target("same.jpg"))

        with mock.patch.object(session_module, "free_name", slow):
            threads = [threading.Thread(target=take) for _ in range(2)]
            [t.start() for t in threads]
            [t.join(30) for t in threads]
        self.assertEqual(len(targets), 2)
        self.assertEqual(len(set(targets)), 2, targets)


class FileSystemWithOneInodeForEveryFile(QaCase):
    """Guard for entry 3: a file system that gives all its files one inode number (some network and FUSE ones) must not turn a folder into one
    clip: two files are the same file only when device, inode, size and time all agree."""

    def test_files_that_share_an_inode_number_but_not_a_size_stay_separate(self):
        from unittest import mock

        folder = os.path.join(self.dir, "media", "oneinode")
        self.photo("a.jpg", folder="media/oneinode", seed=1, size=(160, 120))
        self.photo("b.jpg", folder="media/oneinode", seed=2, size=(200, 150))
        real = os.stat

        def same_inode(path, *args, **kwargs):
            result = real(path, *args, **kwargs)
            if str(path).startswith(folder):
                return os.stat_result((*result[:1], 4242, *result[2:]))  # st_ino is the second field
            return result

        with mock.patch("anonymizer.ui.session.os.stat", same_inode):
            answer = self.api("POST", "/api/clips", {"paths": [folder]})
        self.assertEqual(sorted(c["name"] for c in answer["clips"]), ["a.jpg", "b.jpg"])
