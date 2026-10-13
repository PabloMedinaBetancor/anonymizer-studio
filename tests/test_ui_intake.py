"""A file the Studio cannot use is refused when it is imported, with the reason a person can act on (QA report of 04/10/2026,
should-fix 3 and 4): a truncated photo used to be accepted and then stopped the analysis, and every refusal said only
"is not a supported file"."""

from __future__ import annotations

import json
import os
import unittest

try:
    import numpy as np
    from PIL import Image

    from tests import test_ui_server as server_tests
    from tests.fakes import FixedModule
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

FORMAT_CODES = {
    "phone.heic": "error.media.heic",
    "phone.heif": "error.media.heic",
    "camera.dng": "error.media.raw",
    "camera.CR2": "error.media.raw",
    "anim.gif": "error.media.gif",
    "scan.jp2": "error.media.format",
    "layers.psd": "error.media.format",
    "logo.svg": "error.media.format",
    "notes.txt": "error.media.unsupported",
}


class IntakeCase(server_tests.StudioCase):
    modules = [FixedModule("face", [])]

    def photo(self, name="ok.jpg", size=(160, 120)) -> str:
        path = os.path.join(self.dir, "media", name)
        Image.fromarray((np.random.RandomState(7).rand(size[1], size[0], 3) * 255).astype(np.uint8)).save(path)
        return path

    def put(self, name: str, data: bytes) -> str:
        path = os.path.join(self.dir, "media", name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def rejected(self, *paths):
        answer = self.api("POST", "/api/clips", {"paths": list(paths)})
        return answer["clips"], answer["rejected"]


class RefusedAtTheDoor(IntakeCase):
    def test_a_truncated_photo_is_refused_at_import_and_never_becomes_a_clip(self):
        good = self.photo("good.jpg")
        with open(good, "rb") as f:
            whole = f.read()
        broken = self.put("corrupt.jpg", whole[: len(whole) // 2])
        clips, rejected = self.rejected(broken)
        self.assertEqual(clips, [])
        self.assertEqual(
            [(r["code"], r["params"]["name"]) for r in rejected], [("error.media.unreadable_photo", "corrupt.jpg")]
        )
        self.assertEqual(self.api("GET", "/api/state")["clips"], [])

    def test_a_good_photo_next_to_a_broken_one_still_imports(self):
        good = self.photo("good.jpg")
        with open(good, "rb") as f:
            broken = self.put("corrupt.jpg", f.read()[:300])
        clips, rejected = self.rejected(broken, good)
        self.assertEqual([c["name"] for c in clips], ["good.jpg"])
        self.assertEqual([r["code"] for r in rejected], ["error.media.unreadable_photo"])

    def test_an_empty_file_says_it_is_empty(self):
        clips, rejected = self.rejected(self.put("empty.jpg", b""), self.put("empty.mp4", b""))
        self.assertEqual(clips, [])
        self.assertEqual([r["code"] for r in rejected], ["error.media.empty", "error.media.empty"])

    def test_text_in_a_file_named_mp4_says_it_is_not_a_readable_video(self):
        clips, rejected = self.rejected(self.put("notvideo.mp4", b"this is plain text, not a movie" * 50))
        self.assertEqual(clips, [])
        self.assertEqual([r["code"] for r in rejected], ["error.media.unreadable_video"])

    def test_a_truncated_video_is_refused(self):
        path = self.make_clip("cut.mp4")
        with open(path, "rb") as f:
            data = f.read()
        self.put("cut.mp4", data[: len(data) // 3])
        clips, rejected = self.rejected(path)
        self.assertEqual((clips, [r["code"] for r in rejected]), ([], ["error.media.unreadable_video"]))

    def test_an_animation_with_several_images_is_refused_with_its_count(self):
        path = os.path.join(self.dir, "media", "moving.webp")
        frames = [Image.new("RGB", (40, 30), color) for color in ("red", "green", "blue")]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=100)
        clips, rejected = self.rejected(path)
        self.assertEqual(clips, [])
        self.assertEqual((rejected[0]["code"], rejected[0]["params"]["count"]), ("error.photo.several_images", 3))

    def test_every_unsupported_format_says_which_it_is(self):
        for name, code in FORMAT_CODES.items():
            clips, rejected = self.rejected(self.put(name, b"x" * 64))
            self.assertEqual(clips, [], name)
            self.assertEqual([r["code"] for r in rejected], [code], name)
            self.assertEqual(rejected[0]["params"]["name"], name)
        self.assertEqual(rejected[0]["params"]["extension"], ".txt")

    def test_the_reasons_reach_the_person_in_every_language(self):
        ui = os.path.join(os.path.dirname(server_tests.__file__), "..", "src", "anonymizer", "ui", "i18n")
        for language in ("en", "es", "fr", "de", "pt", "it", "zh-Hans"):
            with open(os.path.join(ui, f"{language}.json"), encoding="utf-8") as f:
                texts = json.load(f)
            for code in set(FORMAT_CODES.values()) | {
                "error.media.empty",
                "error.media.unreadable_photo",
                "error.media.unreadable_video",
            }:
                self.assertIn("{name}", texts[code], (language, code))
            self.assertIn("{extension}", texts["error.media.format"], language)

    def test_the_upload_route_refuses_a_broken_photo_and_leaves_no_file_behind(self):
        with open(self.photo("good.jpg"), "rb") as f:
            good = f.read()
        status, _, data = self.request(
            "PUT", "/api/upload?name=cut.jpg", raw=good[:200], headers={"Content-Type": "application/octet-stream"}
        )
        answer = json.loads(data)
        self.assertEqual(
            (status, answer["clips"], [r["code"] for r in answer["rejected"]]),
            (200, [], ["error.media.unreadable_photo"]),
        )
        uploads = os.path.join(self.dir, "work", "uploads")
        self.assertEqual(os.listdir(uploads) if os.path.isdir(uploads) else [], [])

    def test_the_picker_gets_the_reason_of_each_greyed_file(self):
        folder = os.path.join(self.dir, "media")
        self.put("anim.gif", b"GIF89a")
        self.put("phone.heic", b"x")
        self.photo("ok.jpg")
        listing = self.api("GET", f"/api/browse?path={folder}")
        by_name = {e["name"]: e for e in listing["entries"]}
        self.assertEqual(by_name["anim.gif"]["reason"], "error.media.gif")
        self.assertEqual(by_name["phone.heic"]["reason"], "error.media.heic")
        self.assertNotIn("reason", by_name["ok.jpg"])


if __name__ == "__main__":
    unittest.main()
