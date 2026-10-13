"""The Studio server with the real detectors on a public photo and on a short clip made from it. Off unless
ANONYMIZER_HEAVY=1 (models and bench data are not in the repo; see anonimizador-v2.md for where this runs)."""

from __future__ import annotations

import fractions
import os
import shutil
import unittest

from tests.test_ui_server import EventReader, StudioCase

HEAVY = os.environ.get("ANONYMIZER_HEAVY") == "1"
EU = os.path.join(os.environ.get("ANONYMIZER_BENCH_DATA", ""), "plates", "openalpr-eu")
PHOTO = "eu1.jpg"
REAL_PLATE_REASONS = {
    "plate.reason.on_vehicle_verified",
    "plate.reason.on_vehicle_confirmed",
    "plate.reason.unconfirmed",
    "plate.reason.off_vehicle",
    "plate.reason.vehicle_text",  # the text path (the default)
    "plate.reason.position_band",
}  # the stand-in detectors use other keys, so these show the real model ran


def clip_from_photo(photo: str, target: str, frames: int = 16, fps: int = 8) -> None:
    import av
    import cv2

    image = cv2.imread(photo)
    height, width = image.shape[:2]
    scale = 720 / width if width > 720 else 1.0
    image = cv2.resize(image, (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2), interpolation=cv2.INTER_AREA)
    with av.open(target, "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = image.shape[1], image.shape[0], "yuv420p"
        stream.options = {"crf": "18", "preset": "ultrafast"}
        for t in range(frames):
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = t, fractions.Fraction(1, fps)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


@unittest.skipUnless(HEAVY, "set ANONYMIZER_HEAVY=1 to run the tests that load the models")
class WithRealModules(StudioCase):
    real = True

    def setUp(self):
        super().setUp()
        source = os.path.join(EU, PHOTO)
        if not os.path.isfile(source):
            self.skipTest(f"{PHOTO} is not in {EU}")
        self.photo = os.path.join(self.dir, "media", PHOTO)
        shutil.copy(source, self.photo)

    def blurred_somewhere(self, original, result, box) -> bool:
        x0, y0, x1, y1 = (int(v) for v in box)
        return result[y0:y1, x0:x1].std() < 0.6 * original[y0:y1, x0:x1].std()

    def test_a_photo_through_the_server_finds_the_plate_and_the_render_hides_it(self):
        import cv2

        [clip] = self.api("POST", "/api/clips", {"paths": [self.photo]})["clips"]
        reader = self.analyze(clip["clip_id"])
        items = self.api("GET", f"/api/clips/{clip['clip_id']}/analysis")["items"]
        plates = [i for i in items if i["kind"] == "plate"]
        self.assertTrue(plates, [e for e in reader.events if e["type"] in ("job.error", "job.warning")])
        self.assertTrue({p["judgement"]["reason"] for p in plates} <= REAL_PLATE_REASONS)
        for item in plates:
            self.api("PATCH", f"/api/clips/{clip['clip_id']}/items/{item['item_id']}", {"blur": True})
        boxes = self.api("GET", f"/api/clips/{clip['clip_id']}/boxes?from=0&to=0")["frames"]["0"]
        folder = os.path.join(self.dir, "out")
        self.api("POST", "/api/render", {"clips": [clip["clip_id"]], "options": {"destination": folder}}, 202)
        self.wait_idle()
        [output] = [os.path.join(folder, n) for n in os.listdir(folder) if n.endswith(".jpg")]
        original, result = cv2.imread(self.photo), cv2.imread(output)
        self.assertTrue(any(self.blurred_somewhere(original, result, b["box"]) for b in boxes if b["kind"] == "plate"))

    def test_a_short_clip_through_the_server_with_the_real_modules(self):
        clip_path = os.path.join(self.dir, "media", "street.mp4")
        clip_from_photo(self.photo, clip_path)
        [clip] = self.api("POST", "/api/clips", {"paths": [clip_path]})["clips"]
        reader = self.analyze(clip["clip_id"])
        analyzed = next(e for e in reader.events if e["type"] == "clip.analyzed")
        self.assertGreaterEqual(analyzed["items"], 1)
        found = self.api("GET", f"/api/clips/{clip['clip_id']}/analysis")["items"]
        self.assertTrue({i["judgement"]["reason"] for i in found} <= REAL_PLATE_REASONS)
        self.assertEqual(analyzed["unchecked_frames"], 0)
        base = f"/api/clips/{clip['clip_id']}"
        for item in self.api("GET", f"{base}/analysis")["items"]:
            self.api("PATCH", f"{base}/items/{item['item_id']}", {"blur": True})
        status, _, data = self.get(f"{base}/preview?frame=5&view=anonymized&w=400")
        self.assertEqual((status, data[:3]), (200, b"\xff\xd8\xff"))
        folder = os.path.join(self.dir, "out")
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/render", {"clips": [clip["clip_id"]], "options": {"destination": folder}}, 202)
        reader.wait_for("job.finished", timeout=300)
        self.wait_idle()
        self.assertTrue(os.path.exists(os.path.join(folder, "street_anonymized.mp4")))
        self.assertNotIn("job.error", reader.types())


if __name__ == "__main__":
    unittest.main()
