"""E10: every error, warning and reason key the code can send exists in en.json, says something, and only uses the
placeholders its params carry."""

from __future__ import annotations

import json
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src", "anonymizer")
EN = os.path.join(SRC, "ui", "i18n", "en.json")
PREFIXES = (
    "error.",
    "warning.",
    "video.warning.",
    "photo.warning.",
    "review.warning.",
    "manual.reason.",
    "plate.reason.",
    "face.reason.",
    "comparison.",
    "report.",
)
KEY = re.compile(r"\"((?:" + "|".join(re.escape(p) for p in PREFIXES) + r")[a-z_0-9.]+)\"")
CLIENT_SIDE = {
    "error.network",
    # the page picks the one for the file's state when the server says "error.clip.state" with that status (i18n.js)
    *(
        f"error.clip.state.{state}"
        for state in ("analyzing", "queued", "rendering", "imported", "stopped", "failed", "idle")
    ),
    "warning.export.stopped",  # the page's own line when the person stops an export
    "warning.import.same_reason",  # the page's own line for many files refused for one reason
}
PLACEHOLDER = re.compile(r"\{(\w+)\}")
PARAMS = {
    "error.deep.too_much_work": {"minutes", "limit"},
    "error.analysis.version": {"version"},
    "error.deep.too_large": {"tiles", "limit"},
    "error.demo.limit": {"limit", "max"},
    "error.engine.silent": {"minutes"},
    "error.media.damaged": {"name"},
    "error.media.unsupported": {"name"},
    "error.media.heic": {"name"},
    "error.media.raw": {"name"},
    "error.media.gif": {"name"},
    "error.media.format": {"name", "extension"},
    "error.media.empty": {"name"},
    "error.media.too_large": {"name", "megapixels", "limit"},
    "error.media.denied": {"name"},
    "error.media.unreadable_photo": {"name"},
    "error.media.unreadable_video": {"name"},
    "error.model.load": {"kind"},
    "error.output.exists": {"path"},
    "error.output.name_too_long": {"limit"},
    "error.output.not_writable": {"path"},
    "error.output.disk_full": {"path"},
    "error.photo.not_enough_memory": {"megapixels", "needed_mb", "free_mb"},
    "error.photo.several_images": {"count"},
    "error.browse.denied": set(),
    "error.request.invalid": {"field"},
    "error.source.missing": {"path"},
    "error.upload.too_large": {"max_mb"},
    "report.generated": {"date"},
    "report.note.comparison_video": {"file"},
    "report.summary": {"items", "blurred", "visible"},
    "warning.analysis.unchecked": {"frames"},
    "warning.comparison.contains_original": {"file"},
    "warning.comparison.stopped": {"name"},
    "warning.import.hidden": {"count"},
    "warning.import.ignored": {"count", "extensions"},
    "warning.review.check_left_visible": {"count"},
    "error.analysis.discards": {"count", "decisions"},
    "warning.second_look.zones_skipped": {"count"},
    "warning.vehicles.skipped": {"count"},
}


def keys_in_code() -> set:
    found = set()
    for folder, subs, files in os.walk(SRC):
        subs[:] = [s for s in subs if s not in ("static", "i18n", "__pycache__")]
        for name in files:
            if name.endswith(".py"):
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    found.update(KEY.findall(f.read()))
    return found


def base_key(key: str) -> str:
    return re.sub(r"\.(one|other|few|many|two|zero)$", "", key)


class EngineKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(EN, encoding="utf-8") as f:
            cls.texts = json.load(f)
        cls.used = keys_in_code()

    def test_E10_every_key_the_code_sends_is_in_en_json(self):
        self.assertTrue(self.used)
        missing = sorted(k for k in self.used if k not in self.texts)
        self.assertEqual(missing, [], "add these to src/anonymizer/ui/i18n/en.json")

    def test_E10_no_engine_key_in_en_json_is_unused(self):
        engine = {base_key(k) for k in self.texts if k.startswith(PREFIXES)} - CLIENT_SIDE
        self.assertEqual(sorted(engine - self.used), [])

    def test_E10_texts_say_something_with_only_their_own_placeholders(self):
        for key in sorted(self.used):
            text = self.texts[key]
            self.assertTrue(text.strip(), key)
            self.assertNotRegex(text, r"[^\x00-\x7f]", f"{key} has non-ASCII characters in the English text")
            self.assertEqual(set(PLACEHOLDER.findall(text)), PARAMS.get(key, set()), key)

    def test_E10_plural_forms_keep_the_same_placeholders(self):
        for key, text in self.texts.items():
            if key.endswith(".one") and key.startswith(PREFIXES):
                self.assertIn(base_key(key), self.texts, key)
                self.assertTrue(set(PLACEHOLDER.findall(text)) <= set(PLACEHOLDER.findall(self.texts[base_key(key)])))


if __name__ == "__main__":
    unittest.main()
